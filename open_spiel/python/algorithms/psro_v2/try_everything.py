import numpy as np
import copy
import os
import itertools
# from  open_spiel.python.algorithms.psro_v2.eval_utils import regret, strategy_regret
from open_spiel.python.algorithms.nash_solver.replicator_dynamics_solver import replicator_dynamics

meta_game = [np.array([[[-4.07]], [[-4.08]]]),
             np.array([[[1.705]], [[2.785]]]),
             np.array([[[2.365]], [[1.295]]])]

print(np.shape(meta_game[2]))
ne = replicator_dynamics(meta_game)
print("NE:", ne)