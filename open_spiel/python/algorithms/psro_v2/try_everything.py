import numpy as np
import copy
from open_spiel.python.algorithms.psro_v2.eval_utils import regret, strategy_regret
from open_spiel.python.algorithms.nash_solver.general_nash_solver import nash_solver


BOS_p1_meta_game = np.array([[-1, 0], [0, 5]])
BOS_p2_meta_game = np.array([[-1, 0], [0, 5]])
BOS_meta_games = [BOS_p1_meta_game, BOS_p2_meta_game]

print(strategy_regret(BOS_meta_games, 1))

