import numpy as np
import copy
import os
import itertools
# from  open_spiel.python.algorithms.psro_v2.eval_utils import regret, strategy_regret

eq = [np.array([0.2, 0.8]), np.array([0.1, 0.9])]
ne_subgame_nonzero = [ele[ele >= 0.5] for ele in eq]
print(ne_subgame_nonzero)