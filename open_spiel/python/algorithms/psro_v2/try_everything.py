import numpy as np
import copy
import os
# from  open_spiel.python.algorithms.psro_v2.eval_utils import regret, strategy_regret

def weighted_NE_strategy(checkpoint_dir=None, gamma=0.4):
  BOS_p1_meta_game = np.array([[3, 0], [0, 2]])
  BOS_p2_meta_game = np.array([[2, 0], [0, 3]])
  meta_games = [BOS_p1_meta_game, BOS_p2_meta_game]
  num_players = len(meta_games)
  NE_list = [[np.array([1]), np.array([1])]]
  if len(NE_list) == 0:
    return [np.array([1.])] * num_players

  num_used_policies = len(NE_list[-1][0])

  num_strategies = len(meta_games[0])
  equilibria = [np.array([0., 1.]), np.array([0., 1.])]

  result = [np.zeros(num_strategies)] * num_players
  for player in range(num_players):
    for i, NE in enumerate(NE_list):
      print(gamma ** (num_used_policies - i))
      result[player][:len(NE[player])] += NE[player] * gamma ** (num_used_policies - i)
    result[player] += equilibria[player]
    result[player] /= np.sum(result[player])

  return result

print([np.array([1.])] * 2)