import numpy as np
import copy
#
# def lagging_mean(li,lag=3):
#   """
#   Calcualte the lagging mean of list given
#   Params:
#     li : the one dimensional list to be processed
#     lag: length of moving average
#   """
#   if len(li) <= lag:
#     return list(np.cumsum(li)/(np.arange(len(li))+1))
#   else:
#     first_half = np.cumsum(li[0:lag-1])/(np.arange(lag-1)+1)
#     ret = np.cumsum(li)
#     ret[lag:] = ret[lag:] - ret[:-lag]
#     second_half = ret[lag - 1:] / lag
#     return first_half.tolist() + second_half.tolist(), 1
#
# a = [1,2,3,4,5,6,7]
# p = lagging_mean(a)
# print(p)

a = 3

class Happy():
    def __init__(self, a):
        self._a = a

    def add(self):
        self._a += 3
        print(self._a)

b1 = Happy(a)
b2 = Happy(a)
print(b1._a is b2._a, b1._a is a)