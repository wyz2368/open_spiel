import numpy as np
import copy
from  open_spiel.python.algorithms.psro_v2.eval_utils import regret, strategy_regret


# def softmax(x, temperature=1/5):
#     return np.exp(x / temperature)/np.sum(np.exp(x / temperature))

# x = np.array([0.3,0,0])
# print(softmax(x))

HT_p1_meta_game = np.array([[4, 1], [3, 2]])
HT_p2_meta_game = np.array([[4, 3], [1, 2]])
HT_meta_games = [HT_p1_meta_game, HT_p2_meta_game]

print(regret(HT_meta_games, 1))