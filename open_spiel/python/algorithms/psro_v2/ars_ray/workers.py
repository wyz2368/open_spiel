'''
Parallel implementation of the Augmented Random Search method.
Horia Mania --- hmania@berkeley.edu
Aurelia Guy
Benjamin Recht 
'''

import numpy as np
import ray

from open_spiel.python.algorithms.psro_v2.ars_ray.shared_noise import *
from open_spiel.python.algorithms.psro_v2.ars_ray.utils import rewards_combinator
from open_spiel.python.algorithms.psro_v2 import utils
from open_spiel.python.algorithms.psro_v2 import rl_policy
import pyspiel
from open_spiel.python import rl_environment

@ray.remote
class Worker(object):
    def __init__(self):
        self.value = 1
    # def __init__(self,
    #              env_name,
    #              env_seed,
    #              deltas=None,
    #              slow_oracle_kargs=None,
    #              fast_oracle_kargs=None
    #              ):
    #     # initialize rl environment.
    #
    #     self._env_name = env_name

        # game = pyspiel.load_game_as_turn_based(env_name,
        #                                        {"players": pyspiel.GameParameter(
        #                                            2)})
        # self._env = rl_environment.Environment(game)
        #
        # # Each worker gets access to the shared noise table
        # # with independent random streams for sampling
        # # from the shared noise table.
        # self.deltas = SharedNoiseTable(deltas, env_seed + 7)
        #
        # self._policies = [[] for _ in range(self._num_players)]
        # self._slow_oracle_kargs = slow_oracle_kargs
        # self._fast_oracle_kargs = fast_oracle_kargs
        # self._delta_std = self._fast_oracle_kargs['noise']
        #
        # self.strategy_sampler = utils.sample_strategy

class Worker1(object):
    """ 
    Object class for parallel rollout generation.
    """

    def __init__(self,
                 env,
                 env_seed,
                 deltas=None,
                 slow_oracle_kargs=None,
                 fast_oracle_kargs=None
                 ):

        # initialize rl environment.

        self._num_players = env.num_players
        game = pyspiel.load_game_as_turn_based(env.name,
                                               {"players": pyspiel.GameParameter(
                                                   env.num_players)})
        self._env = rl_environment.Environment(game)

        # Each worker gets access to the shared noise table
        # with independent random streams for sampling
        # from the shared noise table. 
        self.deltas = SharedNoiseTable(deltas, env_seed + 7)

        self._policies = [[] for _ in range(self._num_players)]
        self._slow_oracle_kargs = slow_oracle_kargs
        self._fast_oracle_kargs = fast_oracle_kargs
        self._delta_std = self._fast_oracle_kargs['noise']

        self.strategy_sampler = utils.sample_strategy


    def sample_episode(self,
                       unused_time_step,
                       agents,
                       is_evaluation=False,
                       noise=None,
                       chosen_player=None):
        """
        Sample an episode and get the cumulative rewards. Notice that we do not
        update the agents during this sampling.
        :param unused_time_step: placeholder for openspiel.
        :param agents: a list of policies, one per player.
        :param is_evaluation: evaluation flag.
        :param noise: noise to be added to current policy.
        :param live_agent_id: id of the agent being trained.
        :return: a list of returns, one per player.
        """

        time_step = self._env.reset()
        cumulative_rewards = 0.0

        while not time_step.last():
            if time_step.is_simultaneous_move():
                action_list = []
                for i, agent in enumerate(agents):
                    if i == chosen_player:
                        output = agent.step(time_step,
                                            is_evaluation=is_evaluation,
                                            noise=noise)
                    else:
                        output = agent.step(time_step, is_evaluation=is_evaluation)
                    action_list.append(output.action)
                time_step = self._env.step(action_list)
                cumulative_rewards += np.array(time_step.rewards)
            else:
                player_id = time_step.observations["current_player"]

                agent_output = agents[player_id].step(
                    time_step, is_evaluation=is_evaluation)
                action_list = [agent_output.action]
                time_step = self._env.step(action_list)
                cumulative_rewards += np.array(time_step.rewards)

        # No agents update at this step. This step may not be necessary.
        if not is_evaluation:
            for agent in agents:
                agent.step(time_step)

        return cumulative_rewards


    def do_sample_episode(self,
                          probabilities_of_playing_policies,
                          chosen_player,
                          num_rollouts = 1,
                          is_evaluation = False):
        """ 
        Generate multiple rollouts using noisy policies.
        """

        rollout_rewards = [[] for _ in range(self._num_players)]
        deltas_idx = []

        for _ in range(num_rollouts):

            agents = self.sample_agents(probabilities_of_playing_policies, chosen_player)

            if is_evaluation:
                deltas_idx.append(-1)
                reward = self.sample_episode(None, agents, is_evaluation)
                for i, rew in enumerate(reward):
                    rollout_rewards[i].append(rew)
                
            else:
                # The idx marks the beginning of a sequence of noise with length dim.
                # Refer to shared_noise.py
                idx, delta = self.deltas.get_delta(agents[chosen_player].get_weights().size)
             
                delta = (self._delta_std * delta).reshape(agents[chosen_player].get_weights().shape)
                deltas_idx.append(idx)

                # compute reward used for positive perturbation rollout. List, one reward per player.
                pos_reward = self.sample_episode(None, agents, is_evaluation, delta, chosen_player)

                # compute reward used for negative pertubation rollout. List, one reward per player.
                neg_reward = self.sample_episode(None, agents, is_evaluation, delta, chosen_player)

                # a list of lists, one per player. For each player, a list contains the positive
                # rewards and negative rewards in a format [[pos rew, neg rew],
                #                                           [pos rew, neg rew]]
                #, one row per noise.
                rollout_rewards = rewards_combinator(rollout_rewards, pos_reward, neg_reward)

                            
        return {'deltas_idx': deltas_idx, 'rollout_rewards': rollout_rewards}

    def freeze_all(self):
        """Freezes all policies within policy_per_player.

        Args:
          policies_per_player: List of list of number of policies.
        """
        for policies in self._policies:
            for pol in policies:
                pol.freeze()

    def sync_total_policies(self, extra_policies_weights, policies_types):
        for player in range(self._num_players):
            for i, policy_type in enumerate(policies_types[player]):
                new_pol = self.best_responder(policy_type, player)
                new_pol.set_weights(extra_policies_weights[player][i])
                self._policies[player].append(new_pol)


    def get_num_policies(self):
        return len(self._policies[0])

    def best_responder(self, policy_type, player):
        if policy_type == "DQN":
            agent_class = rl_policy.DQNPolicy
            assert self._slow_oracle_kargs is not None
            new_pol = agent_class(self._env, player, **self._slow_oracle_kargs)
        elif policy_type == "PG":
            agent_class = rl_policy.PGPolicy
            assert self._slow_oracle_kargs is not None
            new_pol = agent_class(self._env, player, **self._slow_oracle_kargs)
        elif policy_type == "ARS_parallel":
            agent_class = rl_policy.ARSPolicy_parallel
            new_pol = agent_class(self._env, player, **self._fast_oracle_kargs)
        else:
            raise ValueError("Agent class not supported in workers")

        return new_pol

    def sample_agents(self, probabilities_of_playing_policies, chosen_player):
        agents = self.strategy_sampler(self._policies, probabilities_of_playing_policies)
        agents[chosen_player] = self._policies[chosen_player][-1]

        return agents
