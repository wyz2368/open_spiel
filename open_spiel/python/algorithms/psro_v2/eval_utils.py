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
    index = [list(np.arange(subgame_index)) for _ in range(num_players)]
    submeta_games = [ele[np.ix_(*index)] for ele in meta_games]
    subgame_ne = nash_solver(submeta_games, solver="gambit") if not subgame_ne else subgame_ne
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


