import numpy as np

def logit_qbr(meta_games, lamda=None):
    """
    Calculate the logistic quantal best response with Gambit.
    :param meta_games: payoff matrix
    :param lamda: logit parameter, inversely related to the level of error.
    :return:
    """