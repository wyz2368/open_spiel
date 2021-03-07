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

a = [[1,2,3], [2,3,4]]
b = [[2,3,4], [1,2,3]]
print(list(itertools.product(*a)))