import numpy as np
import copy

from open_spiel.python.algorithms.psro_v2.eval_utils import strategy_regret

# (1) Matching Pennies
MP_p1_meta_game = np.array([[1, -1], [-1, 1]])
MP_p2_meta_game = np.array([[-1, 1], [1, -1]])
MP_meta_games = [MP_p1_meta_game, MP_p2_meta_game]

#(2) Battle of Sexes
BOS_p1_meta_game = np.array([[3, 0], [0, 2]])
BOS_p2_meta_game = np.array([[2, 0], [0, 3]])
BOS_meta_games = [BOS_p1_meta_game, BOS_p2_meta_game]

#(3) Bar Crowding Game (3 players)
BC_p1_meta_game = np.array([[[-1, 2],[1, 1]], [[2, 0], [1, 1]]])
BC_p2_meta_game = np.array([[[-1, 1],[2, 1]], [[2, 1], [0, 1]]])
BC_p3_meta_game = np.array([[[-1, 2],[2, 0]], [[1, 1], [1, 1]]])
BC_meta_games = [BC_p1_meta_game, BC_p2_meta_game, BC_p3_meta_game]

ne, dev, regret = strategy_regret(BOS_meta_games)


print(ne)
print(dev)
print(regret)