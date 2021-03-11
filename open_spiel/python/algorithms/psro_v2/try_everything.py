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


subgame = [np.array([[np.nan, 2],
                     [2, np.nan]])]

check_completeness(subgame)