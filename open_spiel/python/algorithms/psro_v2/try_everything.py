import numpy as np
import copy
import cloudpickle

class Dog():
    def __init__(self):
        self.bark = False


a = Dog

b = cloudpickle.dumps(a)
print(b)

c = cloudpickle.loads(b)
print(c)