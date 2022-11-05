
import datetime
import numpy as np
from tqdm import notebook
from functools import partial
from typing import List, Tuple, Union

import jax
import jax.numpy as jnp
from jax import nn, lax, grad, tree_multimap, value_and_grad, jit

import optax
from optax._src import base
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_X_y

# The probabilities are clipped to __EPS__ for stability.
__EPS__ = 1e-12
learning_rate = 0.01
n_gradient_steps = 500

class EloRatingNet:
    """
    Train Elo rating model using recurrent neural network.
    Parameters
    ----------
    n_teams: int
        Number of team to rate

    allow_draw: bool
        If True the probability of a draw if also computed
    """

    def __init__(self, n_teams: int, allow_draw: bool=True):
        self.allow_draw = allow_draw
        self.n_teams = n_teams

    def init_params(self):
        """Initial set parameters. This is the set of learnable parameters."""
        return dict(
            alpha=0.0,
            beta=1.0,
            gamma=0.2,
            kappa=0.1,
            # 1000 is arbitrary
            r0=jnp.array([1000.0 for k in range(self.n_teams)]),)


@jit
def predict_proba(params: dict, teamA_rating: float, teamB_rating: float, allow_draw: bool) -> jnp.DeviceArray:
    """
    Predict the probability of team A wins, draw and team B wins.

    Args:
        params:  the dictionnary of learnable parameters.
        teamA_rating: the current rating of team A
        teamB_rating: the current rating of team B
        allow_draw: If True the probability of a draw is computed.

    Returns:
        The probabilities of team A wins, draw and team B wins.

    """
    rho = (teamA_rating - teamB_rating) * params["beta"]
    gamma = params["gamma"] * allow_draw
    alpha = params["alpha"]

    pA = jnp.clip(nn.sigmoid(rho - gamma + alpha), __EPS__, 1 - __EPS__)
    pB = jnp.clip(nn.sigmoid(-rho - gamma), __EPS__, 1 - __EPS__)
    pD = jnp.clip((1.0 - pA - pB) * allow_draw, __EPS__, 1 - __EPS__)

    return jnp.array([pA, pD, pB]) / jnp.sum(jnp.array([pA, pD, pB]))

@jit
def get_target(scores: tuple) -> float:
    """return the target given the scores"""
    return 1 * (scores[0] < scores[1]) + 0.5 * (scores[0] == scores[1])

@jit
def get_log_loss(y: float, probabilities: jnp.DeviceArray) -> float:
    """return the log loss given the target and probabilities."""
    A_win = y == 0
    draw = y == 0.5
    B_win = y == 1
    return jnp.log(probabilities).dot(jnp.array([A_win, draw, B_win]))

@jit
def update_ratings(
    params: dict,
    teamA_idx: int,
    teamB_idx: int,
    y: float,
    rating: jnp.DeviceArray,
    allow_draw: bool,
) -> jnp.DeviceArray:
    """
    Compute the update step in the Elo rating system.

    Args:
        params: The model parameters.
        teamA_idx: Positional index of team A in the rating vector.
        teamB_idx: Positional index of team B in the rating vector.
        y: The match result, 0 means team A wins, 1 means team B wins amd 0.5 means draw.
        rating: The current rating array.
        allow_draw: If True the probability of a draw is computed.

    Returns:
        The updated rating.

    """
    # get the current team's rating and model parameters
    teamA_rating = rating[teamA_idx]
    teamB_rating = rating[teamB_idx]
    kappa = params["kappa"]
    is_draw_result = y == 0.5

    # predict the match
    pA, _, pB = predict_proba(
        params=params,
        teamA_rating=teamA_rating,
        teamB_rating=teamB_rating,
        allow_draw=allow_draw,
    )

    # compute the rating update
    rho_D = is_draw_result * kappa * (pA - pB)
    rho_A = (~is_draw_result) * kappa * (1 - y - pA) - rho_D
    rho_B = (~is_draw_result) * kappa * (y - pB) + rho_D

    # update the rating
    rating = rating.at[teamA_idx].add(rho_A)
    rating = rating.at[teamB_idx].add(rho_B)

    return rating


@jit
def f(carry: dict, x: dict, keep_rating: bool = True, allow_draw: bool = True) -> Tuple[dict, List]:
    """
    Inner function for lax scan. The function update the rating.

    Args:
        carry: The dictionary of carrying parameters. It contains the model parameters and 
               the current rating vector.
        x: The match data.
        keep_rating: If true the rating is outputted to keep tracking it in time. Let it to 
                     False if you are not interrested by tracking them.

    Returns:
        The carry updated and a list that contains the log loss and the rating. 
        If keep_rating is True the rating output is nan.

    """

    # recover current rating and model parameters
    rating = carry["rating"]
    params = carry["params"]

    # get team index for the match and the result
    teamA_idx, teamB_idx = x["team_index"]
    scores = x["scores"]

    # predict the result, the actual result and compute the loss
    probabilities = predict_proba(
        params, rating[teamA_idx], rating[teamB_idx], allow_draw
    )
    y = get_target(scores)
    loss = get_log_loss(y, probabilities)

    # update the rating
    carry["rating"] = update_ratings(
        params, teamA_idx, teamB_idx, y, rating, allow_draw=allow_draw
    )

    if keep_rating:
        # we output the loss and the rating so at the en of the scan procedure we
        # get back all the loss and all the rating for each t.
        return carry, [loss, rating]
    else:
        return carry, [loss, jnp.nan]

@jit
def scan_dataset(params: dict, dataset: dict) -> dict:
    """
    Predict and rate the entire dataset given the paramters.

    Args:
        params: The model parameters.
        dataset: The dataset of matches.

    Returns:
        A dictionary that contains the last carry, the loss history and the rating history.

    """

    carry = dict()
    carry["params"] = params
    carry["rating"] = params["r0"]  # initial rating
    carry, output = lax.scan(f, carry, dataset)

    return {
        "carry": carry,
        "loss_history": output[0],
        "rating_history": output[1],
    }

@jit
def negative_average_log_loss(params: dict, dataset: dict) -> float:
    """
    Compute the negative average log-loss of each match over the dataset.
    This is the loss we will optimise.

    Args:
        params: The model parameters.
        dataset: The dataset of matches.

    Returns:
        The negative average log-loss

    """
    output = scan_dataset(params, dataset)
    
    # loss_history is the list of all log-loss coming from each call of f made by scan.
    return -jnp.mean(output["loss_history"])

def optimise(params: dict, optimiser: base.GradientTransformation, dataset: dict) -> dict:
    opt_state = optimiser.init(params)

    @jit
    def gradient_step(params, opt_state, dataset):
        loss_value, grads = value_and_grad(negative_average_log_loss)(params, dataset)
        updates, opt_state = optimiser.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    for i in range(n_gradient_steps):
        params, opt_state, loss_value = gradient_step(params, opt_state, dataset)
        if i % 100 == 0:
            print(f"step {i}, loss: {loss_value}")

    return params


class EloDataset:
    def __init__(
        self,
        valid_fration=0.2,
        test_fraction=0.2,
        time=None,
        valid_date=None,
        test_date=None,
    ):
        """
        Dataset object to train the Elo rating model. The data are split in three sets:
        the train set where the model is fitted, the validation set used for early 
        stopping and the test set. Sets are split using the fraction of the total 
        dataset size if no time and dates are provided.

        Parameters
        ----------

        test_fraction: double
            Define the size of the test data in % of the dataset total size.

        valid_fration: double
            Define the size of the validation data in % of the dataset total size.

        time: array of datetime (optional)
            Time index for matches.

        valid_date: date (optional)
            Date to split between train (before it) and validation set.

        test_date: date (optional)
            Date to split between validation (before it) and test set.

        """

        self.time = time
        self.test_date = test_date
        self.valid_date = valid_date
        self.test_fraction = test_fraction
        self.valid_fration = valid_fration

    def encode_teams(self, team_names):
        """Encode team name."""
        self.le_ = LabelEncoder().fit(list(team_names[:, 0]) + list(team_names[:, 1]))
        team_index = np.zeros_like(team_names)
        team_index[:, 0] = self.le_.transform(team_names[:, 0])
        team_index[:, 1] = self.le_.transform(team_names[:, 1])
        self.n_teams_ = len(self.le_.classes_)
        self.team_index_ = np.array(team_index).astype(np.int32)

    def prepare_data(self, team_names, scores):
        """
        Prepare the data for training.

        Parameters
        ----------
        team_names: string array of size (n_matches,2)
            Array of team names with names of team A in column 0 and names of team B in column 1.

        scores: array of size (n_matches,2)
            Array of goals score by Team A (column 0) and team B (column 1)

        Returns
        -------

        """
        team_names, scores = check_X_y(
            team_names, scores, dtype=None, multi_output=True
        )
        self.scores_ = np.array(scores).astype(np.float32)
        self.encode_teams(team_names)

        if self.time is None:
            time = np.array(range(len(scores)))
            self.test_date = None
            self.valid_date = None
        else:
            time = self.time
        if self.test_date is None:
            test_date = datetime.datetime.today()
        else:
            test_date = self.test_date

        if self.valid_date is not None:
            assert time is not None
            test_date = test_date
            valid_date = self.valid_date
            split_type_ = "time"
        else:
            test_idx = int((1 - self.test_fraction) * len(time))
            train_idx = int((1 - self.test_fraction - self.valid_fration) * len(time))
            test_date = time[test_idx]
            valid_date = time[train_idx]
            split_type_ = "fraction"
        print(f"The dataset contains {len(scores)} matches and {self.n_teams_} teams.")
        self.train_index_ = time < valid_date
        self.valid_index_ = (time > valid_date) & (time < test_date)
        self.test_index_ = time > test_date
        print(
            f"The split is done given {split_type_}. \
            The train size is {int(sum(self.train_index_))}, \
            validation size is {int(sum(self.valid_index_))} and \
            test size is {int(sum(self.test_index_))}."
        )

    def get_train(self):
        '''get the training dataset'''
        train_set = {
            "team_index": self.team_index_[self.train_index_, :],
            "scores": self.scores_[self.train_index_, :],
        }
        return train_set
    
    def get_dataset(self):
        '''get the full dataset'''
        assert hasattr(
            self, "scores_"
        ), "split_train_test(team_names, scores) needs to be call first."
        return {"team_index": self.team_index_, "scores": self.scores_}

class EloRatingNet:
    """
    Train Elo rating model using recurrent neural network.

    Parameters
    ----------

    allow_draw: bool
        If True the probability of a draw if also computed
    """

    def __init__(self, allow_draw=True):
        self.allow_draw = allow_draw
        self._loss_path = None
        self._ratings = None
        self._best_params = None

    @property
    def loss_path(self) -> pd.DataFrame:
        """return the loss path."""
        return self._loss_path

    @property
    def best_params(self) -> dict:
        """return the set of best parameters."""
        return self._best_params

    @property
    def ratings(self) -> pd.DataFrame:
        """return the last rating."""
        return self._ratings

    def init_params(self, n_teams):
        """Set the initial parameters. This is the set of learnable parameters."""
        return dict(
            beta=1.0,
            gamma=0.1,
            kappa=0.1,
            alpha=0.0,
            r0=jnp.array([1000.0 for k in range(n_teams)]),
        )

    @staticmethod
    @partial(jax.jit)
    def _predict_proba(
        params: dict, teamA_rating: float, teamB_rating: float, allow_draw: bool
    ) -> jnp.DeviceArray:
        """
        Predict the probability of team A wins, draw and team B wins.

        Args:
            params:  the dictionary of learnable parameters.
            teamA_rating: the current rating of team A
            teamB_rating: the current rating of team B
            allow_draw: If True the probability of a draw is computed.

        Returns:
            The probabilities of team A wins, draw and team B wins.

        """
        rho = (teamA_rating - teamB_rating) * params["beta"]
        gamma = (params["gamma"]) * allow_draw
        alpha = params["alpha"]

        pA = jnp.clip(nn.sigmoid(rho - gamma + alpha), __EPS__, 1 - __EPS__)
        pB = jnp.clip(nn.sigmoid(-rho - gamma), __EPS__, 1 - __EPS__)
        pD = jnp.clip(nn.relu(1.0 - pA - pB) * allow_draw, __EPS__, 1 - __EPS__)

        return jnp.array([pA, pD, pB]) / sum(jnp.array([pA, pD, pB]))

    @staticmethod
    @partial(jax.jit)
    def get_target(scores: tuple) -> float:
        """return the target given the scores"""
        return 1 * (scores[0] < scores[1]) + 0.5 * (scores[0] == scores[1])

    @staticmethod
    @partial(jax.jit)
    def get_log_loss(y: float, probabilities: jnp.DeviceArray) -> float:
        """return the log loss given the target and probabilities."""
        A_win = y == 0
        draw = y == 0.5
        B_win = y == 1
        return jnp.log(probabilities).dot(jnp.array([A_win, draw, B_win]))

    @partial(jax.jit, static_argnums=(0,))
    def update_ratings(
        self,
        params: dict,
        teamA_idx: int,
        teamB_idx: int,
        y: float,
        rating: jnp.DeviceArray,
        allow_draw: bool,
    ) -> jnp.DeviceArray:
        """
        Compute the update step in the Elo rating system.

        Args:
            params: The model parameters.
            teamA_idx: Positional index of team A in the rating vector.
            teamB_idx: Positional index of team B in the rating vector.
            y: The match result, 0 means team A wins, 
               1 means team B wins amd 0.5 means draw.
            rating: The current rating array.
            allow_draw: If True the probability of a draw is computed.

        Returns:
            The updated rating.

        """
        # get the current team's rating and model parameters
        teamA_rating = rating[teamA_idx]
        teamB_rating = rating[teamB_idx]
        kappa = params["kappa"]
        is_draw_result = y == 0.5

        # predict the match
        pA, _, pB = self._predict_proba(
            params=params,
            teamA_rating=teamA_rating,
            teamB_rating=teamB_rating,
            allow_draw=allow_draw,
        )

        # compute the rating update
        rho_D = is_draw_result * kappa * (pA - pB)
        rho_A = (~is_draw_result) * kappa * (1 - y - pA) - rho_D
        rho_B = (~is_draw_result) * kappa * (y - pB) + rho_D

        # update the rating
        rating = rating.at[teamA_idx].add(rho_A)
        rating = rating.at[teamB_idx].add(rho_B)

        return rating

    @partial(jax.jit, static_argnums=(0,))
    def f(self, carry: dict, x: dict, keep_rating: bool = False) -> Tuple[dict, List]:
        """
        Inner function for lax scan. The function update the rating.

        Args:
            carry: The dictionary of carrying parameters. It contains 
                   the model parameters and the current rating vector.
            x: The match data.
            keep_rating: If true the rating is outputted to keep tracking 
                         it in time. Let it false if you are not interested 
                         by tracking them.

        Returns:
            The carry updated and a list that contains the log loss and the rating. 
            If keep_rating is True the rating output is nan.

        """

        # recover current rating and model parameters
        rating = carry["rating"]
        params = carry["params"]

        # get team index for the match and the result
        teamA_idx, teamB_idx = x["team_index"]
        scores = x["scores"]

        # predict the result, the actual result and compute the loss
        probabilities = self._predict_proba(
            params, rating[teamA_idx], rating[teamB_idx], self.allow_draw
        )
        y = self.get_target(scores)
        loss = self.get_log_loss(y, probabilities)

        # update the rating
        carry["rating"] = self.update_ratings(
            params, teamA_idx, teamB_idx, y, rating, allow_draw=self.allow_draw
        )

        if not self._loss_mode:
            # we output the loss and the rating so at the end of the scan procedure we
            # get back all the loss and all the rating for each t.
            return carry, [loss, rating, probabilities]
        else:
            return carry, [loss, jnp.nan, jnp.nan]

    @partial(jax.jit, static_argnums=(0,))
    def scan_dataset(
        self, params: dict, dataset: dict, keep_rating: bool = False
    ) -> dict:
        """
        Predict and rate the entire dataset given the parameters.

        Args:
            params: The model parameters.
            dataset: The dataset of matches.

        Returns:
            A dictionary that contains the last carry, the loss history 
            and the rating history.

        """

        carry = dict()
        carry["params"] = params
        carry["rating"] = jnp.array(params["r0"], copy=True)
        carry, output = lax.scan(self.f, carry, dataset)

        return {
            "carry": carry,
            "loss_history": output[0],
            "rating_history": output[1],
            "probabilities_history": output[2],
        }

    @partial(jax.jit, static_argnums=(0,))
    def negative_average_log_loss(self, params: dict, dataset: dict) -> float:
        """
        Compute the negative average log-loss over the dataset. This is the loss 
        to optimise.

        Args:
            params: The model parameters.
            dataset: The dataset of matches.
            initial_rating: Init the rating. Usefull in eval mode to start from the last 
                            optimised rating.

        Returns:
            The negative average log-loss

        """
        self._loss_mode = True
        output = self.scan_dataset(params, dataset)
        self._loss_mode = False
        return -jnp.mean(output["loss_history"])

    def fit_parameters(
        self,
        elo_dataset: EloDataset,
        optimiser: base.GradientTransformation,
        max_step: int = 10000,
        early_stopping: int = 100,
        verbose: int = 50,
    ):
        """
        Fit the model parameters with optimiser.

        Args:
            elo_dataset: a EloDataset class that contains the data.
            optimiser: an Optax gradient descent optimiser
            max_step: maximum number of gradient step
            early_stopping: stop training if the validation loss stopped decreasing 
                            since that number of steps.
            verbose: print losses every verbose number of step.

        """

        params = self.init_params(elo_dataset.n_teams_)

        opt_state = optimiser.init(params)

        train_data = elo_dataset.get_train()
        full_data = elo_dataset.get_dataset()

        loss_path = []
        min_loss = 1e5
        stopping = 0
        self.grads_ = []  # keep gradient history

        @jax.jit
        def gradient_step(params, opt_state, dataset):
            loss_value, grads = jax.value_and_grad(self.negative_average_log_loss)(
                params, dataset
            )
            updates, opt_state = optimiser.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_value, grads

        for i in notebook.tqdm(range(max_step)):
            
            # gradient descent of the train set
            params, opt_state, train_loss, grads = gradient_step(
                params, opt_state, train_data
            )
            self.grads_.append(grads)
            
            # scan the full dataset
            output = self.scan_dataset(params, full_data)
            
            # compute the validation and test losses
            valid_loss = -jnp.mean(output["loss_history"][elo_dataset.valid_index_])
            test_loss = -jnp.mean(output["loss_history"][elo_dataset.test_index_])

            if i % verbose == 0:
                print(
                    f"step {i}, \
                    train loss: {train_loss:.4f}, \
                    validation loss: {valid_loss:.4f}, \
                    test loss: {test_loss:.4f}"
                )

            # stopping rules
            if min_loss > valid_loss:
                min_loss = valid_loss
                stopping = 0
                best_params = params.copy()
                best_validation_loss = min_loss
                _train_loss = train_loss
                _test_loss = test_loss

            else:
                stopping += 1

            if stopping > early_stopping:
                n_iter = i - stopping
                print(
                    f"Optimal stopping at step: {i-early_stopping}. \
                    Best validation loss: {best_validation_loss:.4f}, \
                    train loss: {_train_loss:.4f}, \
                    test loss: {_test_loss:.4f}"
                )
                break
            # keep tracking of the losses
            loss_path = loss_path + [jnp.array([train_loss, valid_loss])]

        if i == max_step - 1:
            print(
                "Maximum number of step reached. Consider increasing the number of step."
            )
        
        # scan the full dataset with the best set of parameters
        output = self.scan_dataset(best_params, full_data)
        
        # keep informations
        self._loss_path = pd.DataFrame(
            loss_path, columns=["train_loss", "valid_loss"]
        ).astype(float)
        self._best_params = best_params
        self._output = output
        
        # format nicely the last ratings
        self._time = elo_dataset.time
        self._teams_names = elo_dataset.le_.classes_
        self._rating_dict = dict(zip(self._teams_names, output["carry"]["rating"]))
        last_time = self._time[-1]
        self._ratings = (
            pd.DataFrame([self._rating_dict], index=[last_time])
            .astype(float)
            .T.sort_values(last_time, ascending=False)
        )


    def predict_proba(self, teamA: str, teamB: str) -> dict:
        """
        Predict the probability of wining for each team. 
        
        Parameters
        ----------
        teamA: string
            Name of teamA
        teamB: string
            Name of teamB
        Returns
        -------
        A dict containing the probabilities.
        """
        assert (
            self.ratings is not None
        ), "The model should be fitted before making prediction."
        teamA_rating = self._rating_dict[teamA]
        teamB_rating = self._rating_dict[teamB]
        pA, pD, pB = self._predict_proba(
            self.best_params,
            teamA_rating=teamA_rating,
            teamB_rating=teamB_rating,
            allow_draw=self.allow_draw,
        )
        return {f"{teamA}": float(pA), "Draw": float(pD), f"{teamB}": float(pB)}

    def plot_rating_history(self, team_names: list = None):
        """plot the rating history of teams"""
        rating_history = pd.DataFrame(
            self._output["rating_history"],
            columns=self._teams_names,
            index=pd.DatetimeIndex(self._time),
        ).astype(float)
        if team_names is not None:
            rating_history = rating_history[team_names]
        rating_history.plot(xlabel="time", ylabel="rating", figsize=(11, 7))
