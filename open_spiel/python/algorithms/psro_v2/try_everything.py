import numpy as np
import copy
import os
import itertools
# from  open_spiel.python.algorithms.psro_v2.eval_utils import regret, strategy_regret

a = [[0,0,0,1,1], [0,0,0,1,1]]
cum_sum = [np.cumsum(ele) for ele in a]
print(cum_sum)