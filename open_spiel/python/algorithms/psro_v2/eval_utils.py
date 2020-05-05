import numpy as np
from open_spiel.python.algorithms.nash_solver.general_nash_solver import nash_solver
from open_spiel.python.algorithms.psro_v2 import meta_strategies


def regret(meta_games, subgame_index, subgame_ne=None):
    """
    Calculate the regret based on a complete payoff matrix for PSRO
    Assume all players have the same number of policies
    :param meta_games: meta_games in PSRO
    :param subgame_index: the subgame to evaluate. Redundant when subgame_ne is supplied
    :param: subgame_ne: subgame nash equilibrium vector.
    :return: a list of regret, one for each player.
    """
    num_policy = np.shape(meta_games[0])[0]
    num_players = len(meta_games)
    if num_policy == subgame_index:
        print("The subgame is same as the full game. Return zero regret.")
        return np.zeros(num_players)
    num_new_pol = num_policy - subgame_index

    index = [list(np.arange(subgame_index)) for _ in range(num_players)]
    submeta_games = [ele[np.ix_(*index)] for ele in meta_games]
    nash = nash_solver(submeta_games, solver="gambit") if not subgame_ne else subgame_ne
    prob_matrix = meta_strategies.general_get_joint_strategy_from_marginals(nash)
    this_meta_prob = [np.append(nash[i],[0 for _ in range(num_new_pol)]) for i in range(num_players)]

    nash_payoffs = []
    deviation_payoffs = []

    for i in range(num_players): 
        ne_payoff = np.sum(submeta_games[i]*prob_matrix)
        # iterate through player's new policy
        dev_payoff = []
        for j in range(num_new_pol):
            dev_prob = this_meta_prob.copy()
            dev_prob[i] = np.zeros(num_policy)
            dev_prob[i][subgame_index+j] = 1
            new_prob_matrix = meta_strategies.general_get_joint_strategy_from_marginals(dev_prob)
            dev_payoff.append(np.sum(meta_games[i]*new_prob_matrix))
        deviation_payoffs.append(dev_payoff-ne_payoff)
        nash_payoffs.append(ne_payoff)
    
    regret = np.maximum(np.max(deviation_payoffs,axis=1),0)
    return regret

def strategy_regret(meta_games, subgame_index, ne=None, subgame_ne=None):
    """
        Calculate the strategy regret based on a complete payoff matrix for PSRO.
        strategy_regret of player equals to nash_payoff in meta_game - fix opponent nash strategy, player deviates to subgame_nash
        Assume all players have the same number of policies.
        :param meta_games: meta_games in PSRO
        :param subgame_index: subgame to evaluate, redundant if subgame_nash supplied
        :param: nash: equilibrium vector
        :param: subgame_ne: equilibrium vector
        :return: a list of regret, one for each player.

    """
    num_players = len(meta_games)
    num_new_pol = np.shape(meta_games[0])[0] - subgame_index

    ne = nash_solver(meta_games, solver="gambit") if not ne else ne
    subgame_ne = nash_solver(meta_games[:subgame_index, :subgame_index], solver="gambit") if not subgame_ne else subgame_ne
    nash_prob_matrix = meta_strategies.general_get_joint_strategy_from_marginals(ne)

    regrets = []
    for i in range(num_players):
        ne_payoff = np.sum(meta_games[i]*nash_prob_matrix)
        dev_prob = ne.copy()
        dev_prob[i] = list(np.append(subgame_ne[i],[0 for _ in range(num_new_pol)]))
        dev_prob_matrix = meta_strategies.general_get_joint_strategy_from_marginals(dev_prob)
        subgame_payoff = np.sum(meta_games[i]*dev_prob_matrix)
        regrets.append(ne_payoff-subgame_payoff)

    return regrets


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


#def regret(meta_games, subgame_index):
#    """
#    Calculate the regret based on a complete payoff matrix for PSRO.
#    Assume all players have the same number of policies.
#    :param meta_games: meta_games in PSRO
#    :param subgame_index: the subgame to evaluate.
#    :return: a list of regret, one for each player.
#    """
#    num_policy = np.shape(meta_games[0])[0]
#    num_players = len(meta_games)
#    if num_policy == subgame_index:
#        print("The subgame is same as the full game. Return zero regret.")
#        return np.zeros(num_players)
#
#    index = [slice(0, subgame_index) for _ in range(num_players)]
#    submeta_games = [subgame[tuple(index)] for subgame in meta_games]
#    nash = nash_solver(submeta_games, solver="gambit")
#
#    nash_payoffs = []
#    deviation_payoffs = []
#
#    for current_player in range(num_players):
#        meta_game = submeta_games[current_player]
#        for dim in range(num_players):
#            newshape = -np.ones(num_players, dtype=np.int64)
#            newshape[dim] = len(nash[dim])
#            meta_game = np.reshape(nash[dim], newshape=newshape) * meta_game
#
#        nash_payoff = np.sum(meta_game)
#        nash_payoffs.append(nash_payoff)
#
#    extended_nash = []
#    for dist in nash:
#        ex_nash = np.zeros(num_policy)
#        ex_nash[:len(dist)] = dist
#        extended_nash.append(ex_nash)
#
#    for current_player in range(num_players):
#        _meta_game = meta_games[current_player]
#        for player in range(num_players):
#            if current_player == player:
#                continue
#            newshape = -np.ones(num_players, dtype=np.int64)
#            newshape[player] = num_policy
#            _meta_game = np.reshape(extended_nash[player], newshape=newshape) * _meta_game
#
#        axis = np.delete(np.arange(num_players), current_player)
#        deviation = np.max(np.sum(_meta_game, axis=tuple(axis)))
#        deviation_payoffs.append(deviation)
#
#    regret = np.maximum(np.array(deviation_payoffs)-np.array(nash_payoffs), 0)
#
#    return regret




#def strategy_regret(meta_games, subgame_index):
#    """
#        Calculate the strategy regret based on a complete payoff matrix for PSRO.
#        This function only works for two-player games.
#        Assume all players have the same number of policies.
#        :param meta_games: meta_games in PSRO
#    """
#    num_players = len(meta_games)
#
#    nash = nash_solver(meta_games, solver="gambit")
#    subgame_nash = nash_solver(meta_games[:subgame_index, :subgame_index], solver="gambit")
#
#    regrets = []
#
#    nash_p1 = nash[0]
#    nash_p1 = np.reshape(nash_p1, newshape=(len(nash_p1),1))
#    nash_p2 = nash[1]
#
#    sub_nash_p1 = subgame_nash[0]
#    sub_nash_p2 = subgame_nash[1]
#
#    # Calculate the NE payoff and deviation payoff for each player.
#    for player in range(num_players):
#        meta_game = meta_games[player]
#        if player == 0:
#            dev = np.reshape(np.sum(meta_game * nash_p2, axis=1), -1)
#            subgame_payoffs = np.sum(sub_nash_p1 * dev[:subgame_index])
#        elif player == 1:
#            dev = np.reshape(np.sum(nash_p1 * meta_game, axis=0), -1)
#            subgame_payoffs = np.sum(sub_nash_p2 * dev[:subgame_index])
#        else:
#            raise ValueError("Only work for two-player games.")
#
#        nash_payoff = np.sum(nash_p1 * meta_game * nash_p2)
#        regrets.append(nash_payoff - subgame_payoffs)
#
#    return regrets
#
