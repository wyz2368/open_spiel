import numpy as np
import copy
import os
# from  open_spiel.python.algorithms.psro_v2.eval_utils import regret, strategy_regret

class Dog():
    def __init__(self):
        self.a = 3
        self.b = 2

def add(dog):
    dog.a = 5

a = Dog()
add(a)
print(a.a)