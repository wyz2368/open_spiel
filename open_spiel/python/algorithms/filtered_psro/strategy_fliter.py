import numpy as np
from absl import flags

from open_spiel.python.algorithms.psro_v2.utils import alpharank_strategy

FLAGS = flags.FLAGS

def strategy_filter(solver, method=FLAGS.filtering_method):
    if method == "alpharank":
        marginals, _ = alpharank_strategy(solver, return_joint=True)
        return alpharank_filter(solver._meta_games, solver._policies, marginals)
    elif method == "trace":
        raise NotImplementedError
    else:
        return solver._meta_games, solver._policies


def alpharank_filter(meta_games,
                     policies,
                     marginals,
                     size_threshold=FLAGS.strategy_set_size):
    """
    Use alpharank to filter out the transient strategies in the empirical game.
    :param meta_games: PSRO meta_games
    :param policies: a list of list of strategies, one per player.
    :param marginals: a list of list of marginal alpharank distribution, one per player.
    :param size_threshold: maximum size for each meta game. (unused)
    :return:
    """
    # TODO:add skip functionality.
    num_str, _ = np.shape(meta_games[0])
    if num_str <= size_threshold:
        return meta_games, policies
    num_players = len(meta_games)
    filtered_idx_list = []
    for player in range(num_players):
        lowest_ranked_str = np.argmin(marginals[player])
        filtered_idx_list.append([lowest_ranked_str])

    for player in range(num_players):
        # filter meta_games.
        for dim in range(num_players):
            filtered_idx = filtered_idx_list[dim]
            meta_games[player] = np.delete(meta_games[player], filtered_idx, axis=dim)
        # filter policies.
        policies[player] = np.delete(policies[player], filtered_idx_list[player])
        policies[player] = list(policies[player])

    print("Strategies filtered:")
    num_str_players = []
    for player in range(num_players):
        print("Player " + str(player) + ":", filtered_idx_list[player])
        num_str_players.append(len(policies[player]))
    print("Number of strategies after filtering:", num_str_players)

    return meta_games, policies




def alpharank_filter_test():
    meta_game = np.array([[1,2,3,4,5],
                          [2,3,4,5,6],
                          [3,4,5,6,7],
                          [4,5,6,7,8],
                          [5,6,7,8,9]])
    meta_games = [meta_game, -meta_game]

    policies = [[1,2,3,4,5], [6,7,8,9,10]]
    marginals = [np.array([0.001, 0.3, 0.3, 0.3, 0.009]), np.array([0.3, 0.001, 0.001, 0.3, 0.001])]
    meta_games, policies = alpharank_filter(meta_games,
                                            policies,
                                            marginals)
    print("meta_games:", meta_games)
    print("policies:", policies)
