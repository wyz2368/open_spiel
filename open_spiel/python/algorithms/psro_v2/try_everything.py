import numpy as np
import copy
import os
# from  open_spiel.python.algorithms.psro_v2.eval_utils import regret, strategy_regret

def smoothing_kl(p, q, eps=0.001):
  p = smoothing(p, eps)
  q = smoothing(q, eps)
  return np.sum(p * np.log(p / q))


def smoothing(p, eps):
  p = np.array(p, dtype=np.float)
  zeros_pos_p = np.where(p == 0)[0]
  num_zeros = len(zeros_pos_p)
  x = eps * num_zeros / (len(p) - num_zeros)
  for i in range(len(p)):
    if i in zeros_pos_p:
      p[i] = eps
    else:
      p[i] -= x
  return p

def kl_divergence(p, q):
  return np.sum(np.where(p != 0, p * np.log(p / q), 0))


a = np.array([1])
print(np.append(a, [0,0,0]))