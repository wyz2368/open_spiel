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

# def check_completeness(subgame):
#     """
#     Check if a subgame is complete. If not, simulate missing entries.
#     :param subgame:
#     :return:
#     """
#     nan_lable = np.isnan(subgame[0])
#     print("nan_table", nan_lable)
#     if np.any(nan_lable):
#         nan_position = list(np.where(nan_lable == 1))
#         print("nan_position", nan_position)
#         for profile in zip(*nan_position):
#             print("profile", profile)
#
#
# subgame = [np.array([[[ 1.95 ,    np.nan],
#             [-1.815, -1.2  ]],
#
#            [[-0.885,  2.575],
#             [ 0.275,  0.19 ]]]),
#            np.array([[[-0.31 ,    np.nan],
#             [ 3.115,  0.485]],
#
#            [[-2.075, -3.085],
#             [-2.505,  0.945]]]), np.array([[[-1.64 ,    np.nan],
#             [-1.3  ,  0.715]],
#
#            [[ 2.96 ,  0.51 ],
#             [ 2.23 , -1.135]]])]
#
# check_completeness(subgame)

a = np.random.randint(0, 10, (5,5))
print("matrix:", a)
subgame = [a, -a]

subgame_idx = [[1,1,0,0,0],[1,0,0,0,1]]

def get_complete_meta_game(subgame, subgame_idx):
    """
    Returns the subgame given the subgame index.
    """
    selector = []
    for i in range(2):
        selector.append(list(np.where(np.array(subgame_idx[i]) == 1)[0]))
    complete_subgame = [subgame[i][np.ix_(*selector)] for i in range(2)]
    return complete_subgame


complete_subgame = get_complete_meta_game(subgame, subgame_idx)
print(complete_subgame)