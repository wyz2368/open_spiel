import numpy as np
import copy
import os
import itertools
from collections import OrderedDict
# from  open_spiel.python.algorithms.psro_v2.eval_utils import regret, strategy_regret
# from open_spiel.python.algorithms.nash_solver.replicator_dynamics_solver import replicator_dynamics

# a = set()
# a.add((1,2,3,1,2,4,4,5))
# a.add((1,2,3))
# a.add((1,2,3,1,2,4,4,5))
#
# print(a)

def check_completeness(subgame):
    """
    Check if a subgame is complete. If not, simulate missing entries.
    :param subgame:
    :return:
    """
    nan_lable = np.isnan(subgame[0])
    print("nan_table", nan_lable)
    if np.any(nan_lable):
        nan_position = list(np.where(nan_lable == 1))
        print("nan_position", nan_position)
        for profile in zip(*nan_position):
            print("profile", profile)
        return True
    return False


subgame = [np.array([[[ 0.32 , -2.625,    np.nan,    np.nan],
        [   np.nan, -2.59 , -2.175, -3.055],
        [   np.nan, -3.45 , -1.755, -3.085],
        [   np.nan, -4.06 , -2.66 , -3.175]],

       [[   np.nan,  1.84 ,    np.nan,  0.97 ],
        [ 0.48 , -1.89 , -3.925, -3.86 ],
        [-1.485, -2.945, -4.375, -2.095],
        [ 2.29 , -2.42 , -0.07 ,  0.23 ]],

       [[   np.nan,  1.88 ,  2.21 ,    np.nan],
        [   np.nan,    np.nan,  2.145, -1.025],
        [ 1.945,  0.935,  1.135, -0.42 ],
        [   np.nan, -0.995, -0.795, -2.505]],

       [[ 2.17 , -0.73 ,  2.535,  1.305],
        [ 2.1  ,  0.875, -1.48 ,  1.325],
        [ 1.425,  0.4  ,  5.235, -0.81 ],
        [ 1.325,  0.225,  1.765, -1.13 ]]]), np.array([[[-0.485, -3.045,    np.nan,    np.nan],
        [   np.nan, -0.5  , -1.315,  0.535],
        [   np.nan,  3.465,  1.015, -1.435],
        [   np.nan,  1.75 ,  5.625, -1.145]],

       [[   np.nan, -2.68 ,    np.nan, -2.13 ],
        [ 2.54 ,  2.025,  0.86 ,  0.69 ],
        [ 4.14 ,  5.365,  2.23 ,  0.655],
        [-0.24 ,  0.205,  0.685, -1.315]],

       [[   np.nan, -3.725, -1.305,    np.nan],
        [   np.nan,    np.nan, -4.505,  0.605],
        [ 0.055,  3.4  , -0.275, -1.55 ],
        [   np.nan, -1.38 ,  3.72 , -0.01 ]],

       [[-1.09 , -1.67 , -2.485, -3.405],
        [-0.275,  0.745,  0.58 ,  0.2  ],
        [-1.225, -2.525, -1.435, -1.445],
        [ 0.605, -3.345,  0.84 , -0.255]]]), np.array([[[ 0.165,  5.67 ,    np.nan,    np.nan],
        [   np.nan,  3.09 ,  3.49 ,  2.52 ],
        [   np.nan, -0.015,  0.74 ,  4.52 ],
        [   np.nan,  2.31 , -2.965,  4.32 ]],

       [[   np.nan,  0.84 ,    np.nan,  1.16 ],
        [-3.02 , -0.135,  3.065,  3.17 ],
        [-2.655, -2.42 ,  2.145,  1.44 ],
        [-2.05 ,  2.215, -0.615,  1.085]],

       [[   np.nan,  1.845, -0.905,    np.nan],
        [   np.nan,    np.nan,  2.36 ,  0.42 ],
        [-2.   , -4.335, -0.86 ,  1.97 ],
        [   np.nan,  2.375, -2.925,  2.515]],

       [[-1.08 ,  2.4  , -0.05 ,  2.1  ],
        [-1.825, -1.62 ,  0.9  , -1.525],
        [-0.2  ,  2.125, -3.8  ,  2.255],
        [-1.93 ,  3.12 , -2.605,  1.385]]])]


subgame_idx = [[0, 0, 1, 1], [1, 0, 0, 1], [0, 1, 0, 1]]

def get_complete_meta_game(subgame, subgame_idx):
    """
    Returns the subgame given the subgame index.
    """
    selector = []
    for i in range(3):
        selector.append(list(np.where(np.array(subgame_idx[i]) == 1)[0]))
    complete_subgame = [subgame[i][np.ix_(*selector)] for i in range(3)]
    return complete_subgame


complete_subgame = get_complete_meta_game(subgame, subgame_idx)
print(complete_subgame)
flag = check_completeness(complete_subgame)
print(flag)
a = tuple([0, 0, 1])
print(subgame[0][a])