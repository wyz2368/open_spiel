import numpy as np
import copy
# from  open_spiel.python.algorithms.psro_v2.eval_utils import regret, strategy_regret


# def softmax(x, temperature=1/5):
#     return np.exp(x / temperature)/np.sum(np.exp(x / temperature))

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def smoothing_kl(p, q, eps=0.001):
    p = smoothing(p, eps)
    q = smoothing(q, eps)
    print(p, q)
    return np.sum(p * np.log(p / q))


def smoothing(p, eps):
    zeros_pos_p = np.where(p == 0)[0]
    num_zeros = len(zeros_pos_p)
    x = eps * num_zeros / (len(p) - num_zeros)
    for i in range(len(p)):
        if i in zeros_pos_p:
            p[i] = eps
        else:
            p[i] -= x
    return p


b = np.array([1., 0., 0., 0., 0., 0.])
a = np.array([0., 1., 0., 0., 0., 0.])

print(np.sum(a * np.log(a / b)))


