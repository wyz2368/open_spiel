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

def verification_encoding(subgame_idx):
    """
    Translate subgame index to a tuple. For example, [[0,0,1],[0,1,0]]->(0,0,1,0,1,0)
    :param subgame_idx:
    :return:
    """
    flat_list = [item for sublist in subgame_idx for item in sublist]
    return tuple(flat_list)

a = [[0,0,1],[0,1,0]]
print(verification_encoding(a))