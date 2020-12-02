import numpy as np
import copy
import os
# from  open_spiel.python.algorithms.psro_v2.eval_utils import regret, strategy_regret

a = np.array([1,2,3,4,5])
b = np.array([1,2,3,4,5])
c = np.outer(a, b)
d = np.reshape(c, -1)
print(d)