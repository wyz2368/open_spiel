import numpy as np
import copy
import os
import itertools
# from  open_spiel.python.algorithms.psro_v2.eval_utils import regret, strategy_regret
# from open_spiel.python.algorithms.nash_solver.replicator_dynamics_solver import replicator_dynamics

# def find_all_combinations(beneficial_dev_pol):
#     """
#     Find all possible subgame index.
#     :param beneficial_dev_pol: [[1,2,3], [2,4], [3]]
#     :return:
#     """
#     all_combinations = []
#     for ele in beneficial_dev_pol:
#         combinations = []
#         for i in range(1, len(ele) + 1):
#             comb = list(itertools.combinations(ele, i))
#             for set in comb:
#                 combinations.append(set)
#         all_combinations.append(combinations)
#     all_subgames = itertools.product(*all_combinations)
#     return list(all_subgames)
#
# a = [[1,2,3], [2,4], [3]]
# print(find_all_combinations(a))

a = [[0,1], [1, 0]]
print(np.sum(a))