import numpy as np
import copy
import os
# from  open_spiel.python.algorithms.psro_v2.eval_utils import regret, strategy_regret

class Dog():
    def __init__(self):
        self.bark = 1

a = [Dog() for _ in range(4)]
print(list(np.delete(a, [0,1])))