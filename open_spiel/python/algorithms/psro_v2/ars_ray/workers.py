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
import dill as cloudpickle

@ray.remote
class Worker(object):
    """ 
    Object class for parallel rollout generation.
    """

    def __init__(self,
                 env,
                 env_seed,
                 deltas=None
                 ):

        # initialize rl environment.
        import pyspiel
        from open_spiel.python import rl_environment


        game = pyspiel.load_game_as_turn_based(env.name,
                                               {"players": pyspiel.GameParameter(
                                                   env.num_players)})
        self._env = rl_environment.Environment(game)

        # Each worker gets access to the shared noise table
        # with independent random streams for sampling
        # from the shared noise table. 
        self.deltas = SharedNoiseTable(deltas, env_seed + 7)


    def sample_episode(self,
                       unused_time_step,
                       agents,
                       is_evaluation=False,
                       noise=None,
                       live_agent_id=None):
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
                    if i == live_agent_id:
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


    def do_sample_episode(self, agents, num_rollouts = 1, is_evaluation = False):
        """ 
        Generate multiple rollouts using noisy policies.
        """

        rollout_rewards = [[] for _ in agents]
        deltas_idx = []
        agents = cloudpickle.loads(agents)

        # Assume only one agent is active.
        # Get the index, policy and noise coefficient of the active agent.
        for i, agent in enumerate(agents):
            if not agent.is_frozen():
                live_agent_id = i
                active_policy = agent._polcy.get_weights()
                delta_std = agent._polcy._noise
                break

        for _ in range(num_rollouts):

            if is_evaluation:
                deltas_idx.append(-1)
                reward = self.sample_episode(None, agents, is_evaluation)
                for i, rew in enumerate(reward):
                    rollout_rewards[i].append(rew)
                
            else:
                # The idx marks the beginning of a sequence of noise with length dim.
                # Refer to shared_noise.py
                idx, delta = self.deltas.get_delta(active_policy.size)
             
                delta = (delta_std * delta).reshape(active_policy.shape)
                deltas_idx.append(idx)

                # compute reward used for positive perturbation rollout. List, one reward per player.
                pos_reward = self.sample_episode(None, agents, is_evaluation, live_agent_id, delta)

                # compute reward used for negative pertubation rollout. List, one reward per player.
                neg_reward = self.sample_episode(None, agents, is_evaluation, live_agent_id, delta)

                # a list of lists, one per player. For each player, a list contains the positive
                # rewards and negative rewards in a format [[pos rew, neg rew],
                #                                           [pos rew, neg rew]]
                #, one row per noise.
                rollout_rewards = rewards_combinator(rollout_rewards, pos_reward, neg_reward)

                            
        return {'deltas_idx': deltas_idx, 'rollout_rewards': rollout_rewards}
