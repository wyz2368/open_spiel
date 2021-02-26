"""
Running replicator dynamics until regret threshold is met within the empirical game.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from open_spiel.python.algorithms.projected_replicator_dynamics import _projected_replicator_dynamics_step
from open_spiel.python.algorithms.psro_v2.eval_utils import dev_regret, dev_regret_general


def controled_replicator_dynamics(payoff_tensors,
                                  regret_threshold,
                                  prd_initial_strategies=None,
                                  prd_iterations=int(1e5),
                                  prd_dt=1e-3,
                                  prd_gamma=0,
                                  average_over_last_n_strategies=None,
                                  num_players=2,
                                  **unused_kwargs):

  """The Control Replicator Dynamics algorithm.
  Running replicator dynamics until regret threshold is met within the empirical game.

  Args:
    payoff_tensors: List of payoff tensors for each player.
    prd_initial_strategies: Initial list of the strategies used by each player,
      if any. Could be used to speed up the search by providing a good initial
      solution.
    prd_iterations: Number of algorithmic steps to take before returning an
      answer.
    prd_dt: Update amplitude term.
    prd_gamma: Minimum exploratory probability term.
    average_over_last_n_strategies: Running average window size for average
      policy computation. If None, use the whole trajectory.
    **unused_kwargs: Convenient way of exposing an API compatible with other
      methods with possibly different arguments.

  Returns:
    PRD-computed strategies.
  """
  number_players = len(payoff_tensors)
  # Number of actions available to each player.
  action_space_shapes = payoff_tensors[0].shape

  # If no initial starting position is given, start with uniform probabilities.
  new_strategies = prd_initial_strategies or [
      np.ones(action_space_shapes[k]) / action_space_shapes[k]
      for k in range(number_players)
  ]

  # Random pick one as initial strategy.
  # new_strategies = []
  # for player in range(number_players):
  #     new_strategy = np.zeros(action_space_shapes[player])
  #     selected_idx = np.random.choice(np.arange(action_space_shapes[player]))
  #     new_strategy[selected_idx] = 1
  #     new_strategies.append(new_strategy)


  average_over_last_n_strategies = average_over_last_n_strategies or prd_iterations

  meta_strategy_window = []
  for i in range(prd_iterations):
    new_strategies = _projected_replicator_dynamics_step(
        payoff_tensors, new_strategies, prd_dt, prd_gamma, use_approx=False)

    if i >= prd_iterations - average_over_last_n_strategies:
      meta_strategy_window.append(new_strategies)

    if i > 1e4:
        # return average_new_strategies
        average_new_strategies = np.mean(meta_strategy_window, axis=0)
        nash_list = [average_new_strategies[i] for i in range(number_players)]

        # Regret Control
        if num_players == 2:
            current_regret = dev_regret(payoff_tensors, nash_list)
        else:
            current_regret = dev_regret_general(payoff_tensors, nash_list)
        if current_regret < regret_threshold:
            break

  print("Inner Iter#:", i)

  return nash_list


