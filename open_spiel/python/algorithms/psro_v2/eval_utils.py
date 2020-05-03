import numpy as np
from open_spiel.python.algorithms.nash_solver.general_nash_solver import nash_solver

def regret(meta_games, subgame_index):
    """
    Calculate the regret based on a complete payoff matrix for PSRO.
    Assume all players have the same number of policies.
    :param meta_games: meta_games in PSRO
    :param subgame_index: the subgame to evaluate.
    :return: a list of regret, one for each player.
    """
    num_policy = np.shape(meta_games[0])[0]
    num_players = len(meta_games)
    if num_policy == subgame_index:
        print("The subgame is same as the full game. Return zero regret.")
        return np.zeros(num_players)

    index = [slice(0, subgame_index) for _ in range(num_players)]
    submeta_games = [subgame[tuple(index)] for subgame in meta_games]
    nash = nash_solver(submeta_games, solver="gambit")

    nash_payoffs = []
    deviation_payoffs = []

    for current_player in range(num_players):
        meta_game = submeta_games[current_player]
        for dim in range(num_players):
            newshape = -np.ones(num_players, dtype=np.int64)
            newshape[dim] = len(nash[dim])
            meta_game = np.reshape(nash[dim], newshape=newshape) * meta_game

        nash_payoff = np.sum(meta_game)
        nash_payoffs.append(nash_payoff)

    extended_nash = []
    for dist in nash:
        ex_nash = np.zeros(num_policy)
        ex_nash[:len(dist)] = dist
        extended_nash.append(ex_nash)

    for current_player in range(num_players):
        _meta_game = meta_games[current_player]
        for player in range(num_players):
            if current_player == player:
                continue
            newshape = -np.ones(num_players, dtype=np.int64)
            newshape[player] = num_policy
            _meta_game = np.reshape(extended_nash[player], newshape=newshape) * _meta_game

        axis = np.delete(np.arange(num_players), current_player)
        deviation = np.max(np.sum(_meta_game, axis=tuple(axis)))
        deviation_payoffs.append(deviation)

    regret = np.maximum(np.array(deviation_payoffs)-np.array(nash_payoffs), 0)

    return regret


class SElogs(object):
    def __init__(self,
                 slow_oracle_period,
                 fast_oracle_period,
                 meta_strategy_methods,
                 heuristic_list):

        self.slow_oracle_period = slow_oracle_period
        self.fast_oracle_period = fast_oracle_period
        self.meta_strategy_methods = meta_strategy_methods
        self.heuristic_list = heuristic_list

        self._slow_oracle_iters = []
        self._fast_oracle_iters = []

        self.regrets = []
        self.nashconv = []

        self._meta_probs_history = []

    def update_regrets(self, regrets):
        self.regrets.append(regrets)

    def get_regrets(self):
        return self.regrets

    def update_nashconv(self, nashconv):
        self.nashconv.append(nashconv)

    def get_nashconv(self):
        return self.nashconv

    def update_slow_iters(self, iter):
        self._slow_oracle_iters.append(iter)

    def get_slow_iters(self):
        return self._slow_oracle_iters

    def update_fast_iters(self, iter):
        self._fast_oracle_iters.append(iter)

    def get_fast_iters(self):
        return self._fast_oracle_iters

    def update_meta_probs(self, probs):
        self._meta_probs_history.append(probs)

    def get_meta_probs(self):
        return self._meta_probs_history


def strategy_regret(meta_games):
    """
        Calculate the strategy regret based on a complete payoff matrix for PSRO.
        This function only works for two-player games.
        Assume all players have the same number of policies.
        :param meta_games: meta_games in PSRO
    """
    num_players = len(meta_games)

    nash = nash_solver(meta_games, solver="gambit")

    nash_payoffs = []
    dev_payoffs = []
    regrets = []

    nash_p1 = nash[0]
    nash_p1 = np.reshape(nash_p1, newshape=(len(nash_p1),1))
    nash_p2 = nash[1]

    # Calculate the NE payoff and deviation payoff for each player.
    for player in range(num_players):
        meta_game = meta_games[player]
        if player == 0:
            dev = np.reshape(np.sum(meta_game * nash_p2, axis=1), -1)
        elif player == 1:
            dev = np.reshape(np.sum(nash_p1 * meta_game, axis=0), -1)
        else:
            raise ValueError("Only work for two-player games.")

        dev_payoffs.append(dev)
        nash_payoff = np.sum(nash_p1 * meta_game * nash_p2)
        nash_payoffs.append(nash_payoff)

    # Calculate the regret of each strategy.

    for i, ne in enumerate(nash_payoffs):
        regrets.append(ne - dev_payoffs[i])

    return nash_payoffs, dev_payoffs, regrets