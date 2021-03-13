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


subgame = [np.array([[[ 1.95 ,    np.nan],
            [-1.815, -1.2  ]],

           [[-0.885,  2.575],
            [ 0.275,  0.19 ]]]),
           np.array([[[-0.31 ,    np.nan],
            [ 3.115,  0.485]],

           [[-2.075, -3.085],
            [-2.505,  0.945]]]), np.array([[[-1.64 ,    np.nan],
            [-1.3  ,  0.715]],

           [[ 2.96 ,  0.51 ],
            [ 2.23 , -1.135]]])]

print("NAN:", subgame[0][0,0,1])

check_completeness(subgame)