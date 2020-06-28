import numpy as np
import copy
import os
# from  open_spiel.python.algorithms.psro_v2.eval_utils import regret, strategy_regret

meta_games = [np.array([1,2]), np.array([1,2])]
b = np.take(meta_games, 1, axis=1)
res = b > np.array([0,3])
print(np.expand_dims(res, axis=1))