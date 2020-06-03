import numpy as np
import copy
import os
# from  open_spiel.python.algorithms.psro_v2.eval_utils import regret, strategy_regret

meta_game = np.array([[1,2,3,4,5],
                          [2,3,4,5,6],
                          [3,4,5,6,7],
                          [4,5,6,7,8],
                          [5,6,7,8,9]])
a = np.delete(meta_game, [0, 4], axis=0)
a = np.delete(a, [0, 4], axis=1)
print(a)