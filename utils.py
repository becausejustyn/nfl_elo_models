
# import numpy as np


def calc_expected_score(team_rating, opp_team_rating):
    return 1 / (1 + 10**((opp_team_rating - team_rating) / 400))

def calc_new_rating(team_rating, observed_score, expected_score, k_factor = 20):
    return team_rating + k_factor * (observed_score - expected_score)