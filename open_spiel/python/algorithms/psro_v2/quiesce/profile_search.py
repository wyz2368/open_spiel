# Implement backward profile search which is an alternative to quiesce.

import itertools
import copy
import numpy as np
import time
import functools
print = functools.partial(print, flush=True)

from open_spiel.python import policy
from open_spiel.python.algorithms.psro_v2 import utils
from open_spiel.python.algorithms.psro_v2 import psro_v2
from open_spiel.python.algorithms.psro_v2 import meta_strategies
from open_spiel.python.algorithms.nash_solver import controled_RD
from open_spiel.python.algorithms.nash_solver.general_nash_solver import normalize_ne


# TODO: test symmetric game, as self.symmetric flags changes self.policies and self.num_players
# TODO: incomplete meta_game may be called in other part of strategy exploration. Please check
class PSROQuiesceSolver(psro_v2.PSROSolver):
    """
    quiesce class, incomplete information nash finding
    """

    def _initialize_policy(self, initial_policies):
        self._policies = [[] for k in range(self._num_players)]
        self._new_policies = [([initial_policies[k]] if initial_policies else
                               [policy.UniformRandomPolicy(self._game)])
                              for k in range(self._num_players)]
        self.eq_quiesce = None
        self.eq_quiesce_RD = None
        self.eq_quiesce_nonmargin = None
        self.eq_quiesce_RD_nonmargin = None

    def _initialize_game_state(self):
        effective_payoff_size = self._game_num_players
        self._meta_games = [
            np.array(utils.empty_list_generator(effective_payoff_size))
            for _ in range(effective_payoff_size)
        ]
        super(PSROQuiesceSolver, self).update_empirical_gamestate(seed=None)
        # self.update_complete_ind([0 for _ in range(self._game_num_players)], add_sample=True)
        self.number_profile_sampled = 1 #TODO: print out this.

    def update_meta_strategies(self):
        """Recomputes the current meta strategy of each player.
        Given new payoff tables, we call self._meta_strategy_method to update the
        meta-probabilities.
        """
        if self._meta_strategy_str in ['nash', 'general_nash', 'prd']:
            start_time = time.time()
            self._meta_strategy_probabilities, self._non_marginalized_probabilities = self.inner_loop(regret_threshold=self.quiesce_regret_threshold,
                                                                                                      regularization=self.RD_regularization,
                                                                                                      regularization_regret=self.RD_regret_threshold)
            return time.time() - start_time  # Return quiesce running time
        else:
            super(PSROQuiesceSolver, self).update_meta_strategies()
            return 0

    def update_empirical_gamestate(self, seed=None):
        """Given new agents in _new_policies, update meta_games through simulations.
        For quiesce, only update the meta game grid, but does not need to fill in values.
        If filling in value required, use parent class method.

        Args:
          seed: Seed for environment generation.

        Returns:
          Meta game payoff matrix.
        """
        if seed is not None:
            np.random.seed(seed=seed)
        assert self._oracle is not None

        # Concatenate both lists.
        updated_policies = [
            self._policies[k] + self._new_policies[k]
            for k in range(self._num_players)
        ]

        # Each metagame will be (num_strategies)^self._num_players.
        # There are self._num_player metagames, one per player.
        total_number_policies = [
            len(updated_policies[k]) for k in range(self._num_players)
        ]

        # number_older_policies = [
        #     len(self._policies[k]) for k in range(self._num_players)
        # ]

        # Initializing the matrix with nans to recognize unestimated states.
        meta_games = [
            np.full(tuple(total_number_policies), np.nan)
            for k in range(self._num_players)
        ]

        # Filling the matrix with already-known values.
        older_policies_slice = tuple(
            [slice(len(self._policies[k])) for k in range(self._num_players)])
        for k in range(self._num_players):
            meta_games[k][older_policies_slice] = self._meta_games[k]

        self._meta_games = meta_games
        self._policies = updated_policies
        # self.update_complete_ind(number_older_policies, add_sample=False)
        return meta_games

    @property
    def get_complete_meta_game(self):
        """
        Returns the maximum complete game matrix
        in the same form as empirical game
        """
        selector = []
        for i in range(self._game_num_players):
            selector.append(list(np.where(np.array(self._complete_ind[i]) == 1)[0]))
        complete_subgame = [self._meta_games[i][np.ix_(*selector)] for i in range(self._game_num_players)]
        return complete_subgame

    def inner_loop(self, regret_threshold=0.0, support_threshold=0.005, regularization=False, regularization_regret=0.5):
        """
        Find equilibrium in the incomplete self._meta_games through iteratively augment the maximum complete subgame by sampling. Symmetric game could have insymmetric nash equilibrium, so uses self._game_num_players instead of self._num_players
        Returns:
            Equilibrium support, non_margianlized profile probability
        :param regret_threshold: controls regret in quiesce search.
        :param support_threshold: controls minimal support in NE of the empirical game.
        :param regularization: controls whether run RD in the subgame containing the NE.
        :param regularization_regret: controls when RD stops.
        """
        found_confirmed_eq = False
        # TODO: change to replicator with regularization for online target.(Low implementation priority.)
        # NE_solver = 'replicator' if self._num_players > 2 else 'gambit'
        NE_solver = 'gambit'

        num_strategies = [len(self._policies[k]) for k in range(self._game_num_players)]
        # Complete index has a different meaning from the one in quiesce.py .
        # Here complete index keeps strategy set of the subgame, i.e., which strategy each player has.
        self._complete_ind = [[0 for _ in range(num_strategies[player])] for player in range(self._game_num_players)]
        # Simulate the newest strategy profile and mark it.
        newest_profile = [len(self._policies[k]) - 1 for k in range(self._game_num_players)]
        self.sample_pure_policy_to_empirical_game(newest_profile)
        for player in range(self._game_num_players):
            self._complete_ind[player][-1] = 1

        while not found_confirmed_eq:
            maximum_subgame = self.get_complete_meta_game
            ne_subgame = meta_strategies.general_nash_strategy(solver=self, return_joint=False, NE_solver=NE_solver,
                                                               game=maximum_subgame, checkpoint_dir=self.checkpoint_dir)

            # cumsum: index ne_subgame with self._complete_ind
            cum_sum = [np.cumsum(ele) for ele in self._complete_ind]
            ne_support_index = []
            for i in range(self._game_num_players):
                ne_support_index_p = []
                for j in range(len(self._complete_ind[i])):
                    if self._complete_ind[i][j] == 1 and ne_subgame[i][cum_sum[i][j] - 1] >= support_threshold:
                        ne_support_index_p.append(j)
                assert len(ne_support_index_p) != 0
                ne_support_index.append(ne_support_index_p)

            # ne_subgame: non-zero equilibrium support, [[0.1,0.5,0.4],[0.2,0.4,0.4]]
            ne_subgame_nonzero = [np.array(ele) for ele in ne_subgame]
            ne_subgame_nonzero = [ele[ele >= support_threshold] for ele in ne_subgame_nonzero]

            # get players' payoffs in nash equilibrium
            ne_payoffs = self.get_mixed_payoff(ne_support_index, ne_subgame_nonzero)

            # all possible deviation payoffs (Only get all deviations and corresponding payoff rather than choosing the maximal one.)
            dev_pol, dev_payoffs = self.schedule_deviation(ne_support_index, ne_subgame_nonzero)

            # TODO: add sampling scheme, not find all deviation payoff.
            # check max deviations and sample full subgame where beneficial deviation included
            dev = []
            maximum_subgame_index = [list(np.where(np.array(ele) == 1)[0]) for ele in self._complete_ind]
            for i in range(self._game_num_players):
                if not len(dev_payoffs[i]) == 0 and max(dev_payoffs[i]) - regret_threshold > ne_payoffs[i]:
                    pol = dev_pol[i][np.argmax(dev_payoffs[i])]
                    new_subgame_sample_ind = copy.deepcopy(maximum_subgame_index)
                    maximum_subgame_index[i].append(pol) # all other player's policies have to sample previous players' best deviation
                    new_subgame_sample_ind[i] = [pol]
                    # add best deviation into subgame and sample the corresponding profiles.
                    for profile in itertools.product(*new_subgame_sample_ind):
                        self.sample_pure_policy_to_empirical_game(profile)
                    dev.append(i)
                    # update complete index.
                    self._complete_ind[i][pol] = 1

            found_confirmed_eq = (len(dev) == 0)
            # debug: check maximum subgame remains the same
            # debug: check maximum game reached

        eq = []
        policy_len = [len(self._policies) for _ in range(self._game_num_players)] if self.symmetric_game else [len(ele)
                                                                                                               for ele
                                                                                                               in
                                                                                                               self._policies]
        for p in range(self._game_num_players):
            eq_p = np.zeros([policy_len[p]], dtype=float)
            np.put(eq_p, ne_support_index[p], ne_subgame_nonzero[p])
            eq.append(eq_p)

        self.eq_quiesce = normalize_ne(eq)
        self.eq_quiesce_nonmargin = meta_strategies.get_joint_strategy_from_marginals(eq)


        # return confirmed nash equilibrium
        # If True, first run quiesce to find the subgame containing a NE. Than run RD to regularize.
        if True: #regularization:
            # ne_subgame = controled_RD.controled_replicator_dynamics(maximum_subgame,
            #                                                         regret_threshold=regularization_regret)
            ne_subgame = meta_strategies.general_nash_strategy(solver=self, return_joint=False, NE_solver="replicator",
                                                               game=maximum_subgame, checkpoint_dir=self.checkpoint_dir)

            cum_sum = [np.cumsum(ele) for ele in self._complete_ind]
            ne_support_index = []
            for i in range(self._game_num_players):
                ne_support_index_p = []
                for j in range(len(self._complete_ind[i])):
                    if self._complete_ind[i][j] == 1 and ne_subgame[i][cum_sum[i][j] - 1] >= 0:
                        ne_support_index_p.append(j)
                assert len(ne_support_index_p) != 0
                ne_support_index.append(ne_support_index_p)

            # ne_subgame: non-zero equilibrium support, [[0.1,0.5,0.4],[0.2,0.4,0.4]]
            ne_subgame_nonzero = [np.array(ele) for ele in ne_subgame]
            ne_subgame_nonzero = [ele[ele >= 0] for ele in ne_subgame_nonzero]

            eq = []
            policy_len = [len(self._policies) for _ in range(self._game_num_players)] if self.symmetric_game else [len(ele)
                                                                                                                   for ele
                                                                                                                   in
                                                                                                           self._policies]
            for p in range(self._game_num_players):
                eq_p = np.zeros([policy_len[p]], dtype=float)
                np.put(eq_p, ne_support_index[p], ne_subgame_nonzero[p])
                eq.append(eq_p)

            self.eq_quiesce_RD = normalize_ne(eq)
            self.eq_quiesce_RD_nonmargin = meta_strategies.get_joint_strategy_from_marginals(eq)

        if regularization:
            return self.eq_quiesce_RD, self.eq_quiesce_RD_nonmargin
        else:
            return self.eq_quiesce, self.eq_quiesce_nonmargin

    def schedule_deviation(self, eq, eq_sup):
        """
        Sample all possible deviation from eq
        Return a list of best deviation for each player
        if none for a player, return None for that player
        Params:
          eq     : list of list, where equilibrium is.[[1],[0]]: an example of 2x2 game
          eq_sup : list of list, contains equilibrium support for each player
        Returns:
          dev_pol: list of list, position of policy sampled for each player
          devs   : list of list, deviation payoff of policy sampled
        """
        devs = []
        dev_pol = []
        # Only get all deviations and corresponding payoff rather than choosing the maximal one.
        for p in range(self._game_num_players):
            # check all possible deviations
            dev = []
            possible_dev = list(np.where(np.array(self._complete_ind[p]) == 0)[0])
            iter_eq = copy.deepcopy(eq)  # TODO: eq support is eq.
            iter_eq[p] = possible_dev
            for pol in itertools.product(*iter_eq):
                self.sample_pure_policy_to_empirical_game(pol)  # Simulate payoff and fill in payoff matrix.

            for pol in possible_dev:
                stra_li, stra_sup = copy.deepcopy(eq), copy.deepcopy(eq_sup)
                stra_li[p] = [pol]
                stra_sup[p] = np.array([1.0])
                dev.append(self.get_mixed_payoff(stra_li, stra_sup)[p])
            devs.append(dev)
            dev_pol.append(possible_dev)
        return dev_pol, devs

    def get_mixed_payoff(self, strategy_list, strategy_support):
        """
        Check if the payoff exists for the profile given. If not, return False
        Params:
          strategy_list    : list of list, policy index for each player
          strategy_support : list of list, policy support probability for each player
        Returns:
          payoffs          : payoff for each player in the profile
        """
        if np.any(np.isnan(self._meta_games[0][np.ix_(*strategy_list)])):  # If submatrix misses entries, return False.
            return False

        # Calculate a probability matrix and multiple it by payoff matrix elementwise.
        meta_game = [ele[np.ix_(*strategy_list)] for ele in self._meta_games]
        prob_matrix = meta_strategies.general_get_joint_strategy_from_marginals(strategy_support)
        payoffs = []
        for i in range(self._num_players):
            payoffs.append(np.sum(meta_game[i] * prob_matrix))
        return payoffs



    def sample_pure_policy_to_empirical_game(self, policy_indicator):
        """
        Simulate payoff data and fill it to data grid(self.meta_game)
        Params:
          policy_indicator: 1 dim list, containing poicy to sample for each player, e.g., [3, 4], not an indicator function.
        Returns:
          Bool            : True if data successfully added, False is data already there
        """
        if not np.isnan(self._meta_games[0][tuple(policy_indicator)]):
            return False

        self.number_profile_sampled += 1  # Record how many profiles are sampled.

        if self.symmetric_game:
            estimated_policies = [self._policies[policy_indicator[i]] for i in range(self._game_num_players)]
        else:
            estimated_policies = [self._policies[i][policy_indicator[i]] for i in range(self._game_num_players)]

        utility_estimates = self.sample_episodes(estimated_policies, self._sims_per_entry)
        for k in range(self._game_num_players):
            self._meta_games[k][tuple(policy_indicator)] = utility_estimates[k]
        return True

    def check_completeness(self, subgame):
        """
        Check if a subgame is complete. If not, simulate missing entries.
        :param subgame:
        :return:
        """
        nan_lable = np.isnan(subgame[0])
        if np.any(nan_lable):
            nan_position = list(np.where(nan_lable == 1))
            for profile in zip(*nan_position):
                self.sample_pure_policy_to_empirical_game(profile)

    def get_another_probs(self):
        if self.RD_regularization:
            return self.eq_quiesce
        else:
            return self.eq_quiesce_RD