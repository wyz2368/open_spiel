import numpy as np
import copy
from open_spiel.python.algorithms.psro_v2.eval_utils import regret, strategy_regret
# from open_spiel.python.algorithms.nash_solver.general_nash_solver import nash_solver
from open_spiel.python.algorithms.psro_v2.se_example import print_beneficial_deviation_analysis


BOS_p1_meta_game = np.array([[-1, 0], [0, 5]])
BOS_p2_meta_game = np.array([[-1, 0], [0, 5]])
BOS_meta_games = [BOS_p1_meta_game, BOS_p2_meta_game]
last_BOS_meta_games = [BOS_p1_meta_game[:1,:1], BOS_p2_meta_game[:1,:1]]
nash = [np.array([1]), np.array([1])]




print(print_beneficial_deviation_analysis(last_BOS_meta_games, BOS_meta_games, nash, True))

