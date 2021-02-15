import numpy as np
import copy
import os
import itertools
# from  open_spiel.python.algorithms.psro_v2.eval_utils import regret, strategy_regret

class Dog():
    @property
    def zero(self):
        return np.zeros(4)

    def happy(self):
        a = self.zero
        print(a)

dog = Dog()
dog.happy()
