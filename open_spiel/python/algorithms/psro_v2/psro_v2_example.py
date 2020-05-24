# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Example running PSRO on OpenSpiel Sequential games.

To reproduce our ICLR paper, run this script with
  - `game_name` in ['kuhn_poker', 'leduc_poker']
  - `n_players` in [2, 3, 4, 5]
  - `meta_strategy_method` in ['alpharank', 'uniform', 'nash', 'prd']
  - `rectifier` in ['', 'rectified']

The other parameters keeping their default values.
"""

import time
import datetime
import os
import sys
from absl import app
from absl import flags
import numpy as np
import pickle

import pyspiel
import random

import tensorflow.compat.v1 as tf
from tensorboardX import SummaryWriter
import logging
logging.disable(logging.INFO)
import functools
print = functools.partial(print, flush=True)

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import get_all_states
from open_spiel.python.algorithms import policy_aggregator
from open_spiel.python.algorithms.psro_v2 import best_response_oracle
from open_spiel.python.algorithms.psro_v2 import psro_v2
from open_spiel.python.algorithms.psro_v2 import rl_oracle
from open_spiel.python.algorithms.psro_v2 import rl_policy
from open_spiel.python.algorithms.psro_v2 import strategy_selectors
from open_spiel.python.algorithms.psro_v2.quiesce.quiesce import PSROQuiesceSolver
from open_spiel.python.algorithms.psro_v2 import meta_strategies
from open_spiel.python.algorithms.psro_v2.quiesce import quiesce_sparse
from open_spiel.python.algorithms.psro_v2.utils import set_seed
from open_spiel.python.algorithms.psro_v2.eval_utils import save_strategies, save_nash, save_pkl


FLAGS = flags.FLAGS
# Game-related
flags.DEFINE_string("game_name", "kuhn_poker", "Game name.")
flags.DEFINE_integer("n_players", 2, "The number of players.")
flags.DEFINE_list("game_param",None,"game parameters") #game_param=v=1,"vmodule=a=0,b=2"
# PSRO related
flags.DEFINE_string("meta_strategy_method", "general_nash",
                    "Name of meta strategy computation method.")
flags.DEFINE_integer("number_policies_selected", 1,
                     "Number of new strategies trained at each PSRO iteration.")
flags.DEFINE_integer("sims_per_entry", 1000,
                     ("Number of simulations to run to estimate each element"
                      "of the game outcome matrix."))

flags.DEFINE_integer("gpsro_iterations", 150,
                     "Number of training steps for GPSRO.")
flags.DEFINE_bool("symmetric_game", False, "Whether to consider the current "
                  "game as a symmetric game.")
flags.DEFINE_bool("quiesce", False, "Whether to use quiece")
flags.DEFINE_bool("sparse_quiesce", False, "whether to use sparse matrix quiesce implementation")

# Rectify options
flags.DEFINE_string("rectifier", "",
                    "(No filtering), 'rectified' for rectified.")
flags.DEFINE_string("training_strategy_selector", "probabilistic",
                    "Which strategy selector to use. Choices are "
                    " - 'top_k_probabilities': select top "
                    "`number_policies_selected` strategies. "
                    " - 'probabilistic': Randomly samples "
                    "`number_policies_selected` strategies with probability "
                    "equal to their selection probabilities. "
                    " - 'uniform': Uniformly sample `number_policies_selected` "
                    "strategies. "
                    " - 'rectified': Select every non-zero-selection-"
                    "probability strategy available to each player.")

# General (RL) agent parameters
flags.DEFINE_string("oracle_type", "DQN", "Choices are DQN, PG (Policy "
                    "Gradient), BR (exact Best Response) or ARS(Augmented Random Search)")
flags.DEFINE_integer("number_training_episodes", int(1e4), "Number training "
                     "episodes per RL policy. Used for PG and DQN")
flags.DEFINE_float("self_play_proportion", 0.0, "Self play proportion")
flags.DEFINE_integer("hidden_layer_size", 256, "Hidden layer size")
flags.DEFINE_integer("batch_size", 32, "Batch size")
flags.DEFINE_float("sigma", 0.0, "Policy copy noise (Gaussian Dropout term).")
flags.DEFINE_string("optimizer_str", "adam", "'adam' or 'sgd'")
flags.DEFINE_integer("n_hidden_layers", 4, "# of hidden layers")
flags.DEFINE_float("discount_factor",0.999,"discount factor")

# Policy Gradient Oracle related
flags.DEFINE_string("loss_str", "qpg", "Name of loss used for BR training.")
flags.DEFINE_integer("num_q_before_pi", 8, "# critic updates before Pi update")
flags.DEFINE_float("entropy_cost", 0.001, "Self play proportion")
flags.DEFINE_float("critic_learning_rate", 1e-2, "Critic learning rate")
flags.DEFINE_float("pi_learning_rate", 1e-3, "Policy learning rate.")

# DQN
flags.DEFINE_float("dqn_learning_rate", 1e-2, "DQN learning rate.")
flags.DEFINE_integer("update_target_network_every", 500, "Update target "
                     "network every [X] steps")
flags.DEFINE_integer("learn_every", 10, "Learn every [X] steps.")

#ARS
flags.DEFINE_float("ars_learning_rate", 0.02, "ARS learning rate.")
flags.DEFINE_integer("num_directions", 16, "Number of exploration directions.")
flags.DEFINE_integer("num_best_directions", 16, "Select # best directions.")
flags.DEFINE_float("noise", 0.03, "Coefficient of Gaussian noise.")

#ARS_parallel
flags.DEFINE_integer("num_workers", 4, "Number of workers for parallel ars.")


# General
flags.DEFINE_string("root_result_folder",'root_result',"root directory of saved results")
flags.DEFINE_bool("sbatch_run",False,"whether to redirect standard output to checkpoint directory")
flags.DEFINE_integer("seed", None, "Seed.")
flags.DEFINE_bool("local_launch", False, "Launch locally or not.")
flags.DEFINE_bool("verbose", True, "Enables verbose printing and profiling.")
flags.DEFINE_bool("log_train", False,"log training reward curve")
flags.DEFINE_bool("compute_exact_br",True,"compute and log exp with exact br. If false, than use combined game at the last iteration to evaluate")


def init_pg_responder(sess, env):
  """Initializes the Policy Gradient-based responder and agents."""
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]

  agent_class = rl_policy.PGPolicy

  agent_kwargs = {
      "session":sess,
      "info_state_size": info_state_size,
      "num_actions": num_actions,
      "loss_str": FLAGS.loss_str,
      "loss_class": False,
      "hidden_layers_sizes": [FLAGS.hidden_layer_size] * FLAGS.n_hidden_layers,
      "batch_size": FLAGS.batch_size,
      "entropy_cost": FLAGS.entropy_cost,
      "critic_learning_rate": FLAGS.critic_learning_rate,
      "pi_learning_rate": FLAGS.pi_learning_rate,
      "num_critic_before_pi": FLAGS.num_q_before_pi,
      "optimizer_str": FLAGS.optimizer_str,
      "additional_discount_factor": FLAGS.discount_factor
  }
  oracle = rl_oracle.RLOracle(
      env,
      agent_class,
      agent_kwargs,
      number_training_episodes=FLAGS.number_training_episodes,
      self_play_proportion=FLAGS.self_play_proportion,
      sigma=FLAGS.sigma)
  agents = [
      agent_class(  # pylint: disable=g-complex-comprehension
          env,
          player_id,
          **agent_kwargs)
      for player_id in range(FLAGS.n_players)
  ]
  for agent in agents:
    agent.freeze()
 
  agent_kwargs_save = {key:val for key,val in agent_kwargs.items() if key!="session" }
  agent_kwargs_save["policy_class"] = "PG"
  return oracle, agents, agent_kwargs_save


def init_br_responder(env):
  """Initializes the tabular best-response based responder and agents."""
  random_policy = policy.TabularPolicy(env.game)
  oracle = best_response_oracle.BestResponseOracle(
      game=env.game, policy=random_policy)
  agents = [random_policy.__copy__() for _ in range(FLAGS.n_players)]
  agent_kwargs = {}
  agent_kwargs["policy_class"] = "BR"
  return oracle, agents, agent_kwargs


def init_dqn_responder(sess, env):
  """Initializes the Policy Gradient-based responder and agents."""
  state_representation_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]

  agent_class = rl_policy.DQNPolicy
  agent_kwargs = {
      "session": sess,
      "state_representation_size": state_representation_size,
      "num_actions": num_actions,
      "hidden_layers_sizes": [FLAGS.hidden_layer_size] * FLAGS.n_hidden_layers,
      "batch_size": FLAGS.batch_size,
     "learning_rate": FLAGS.dqn_learning_rate,
      "update_target_network_every": FLAGS.update_target_network_every,
      "learn_every": FLAGS.learn_every,
      "optimizer_str": FLAGS.optimizer_str,
      "discount_factor": FLAGS.discount_factor
  }
  oracle = rl_oracle.RLOracle(
      env,
      agent_class,
      agent_kwargs,
      number_training_episodes=FLAGS.number_training_episodes,
      self_play_proportion=FLAGS.self_play_proportion,
      sigma=FLAGS.sigma)

  agents = [
      agent_class(  # pylint: disable=g-complex-comprehension
          env,
          player_id,
          **agent_kwargs)
      for player_id in range(FLAGS.n_players)
  ]
  for agent in agents:
    agent.freeze()

  agent_kwargs_save = {key:val for key,val in agent_kwargs.items() if key!="session" }
  agent_kwargs_save["policy_class"] = "DQN"
  return oracle, agents, agent_kwargs_save


def init_ars_responder(sess, env):
  """
  Initializes the ARS responder and agents.
  :param sess: A fake sess=None
  :param env: A rl environment.
  :return: oracle and agents.
  """
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]
  agent_class = rl_policy.ARSPolicy
  agent_kwargs = {
    "session": None,
    "info_state_size": info_state_size,
    "num_actions": num_actions,
    "learning_rate": FLAGS.ars_learning_rate,
    "nb_directions": FLAGS.num_directions,
    "nb_best_directions": FLAGS.num_directions,
    "noise": FLAGS.noise
  }
  oracle = rl_oracle.RLOracle(
    env,
    agent_class,
    agent_kwargs,
    number_training_episodes=FLAGS.number_training_episodes,
    self_play_proportion=FLAGS.self_play_proportion,
    sigma=FLAGS.sigma)

  agents = [
    agent_class(
      env,
      player_id,
      **agent_kwargs)
    for player_id in range(FLAGS.n_players)
  ]
  for agent in agents:
    agent.freeze()

  agent_kwargs_save = {key:val for key,val in agent_kwargs.items() if key!="session" }
  agent_kwargs_save["policy_class"] = "ARS"
  return oracle, agents, agent_kwargs_save


def print_beneficial_deviation_analysis(last_meta_game, meta_game, last_meta_prob, verbose=False):
  """
  Function to check whether players have found policy of beneficial deviation in current meta_game compared to the last_meta_game
  Args:
    last_meta_game: List of list of meta_game (One array per game player). The meta game to compare against
    meta_game: List of list of meta_game (One array per game player). Current iteration's meta game. Same length with last_meta_game, and each element in meta_game has to include all entries in last_meta_game's corresponding elements
    last_meta_prob: nash equilibrium of last g_psro_iteration. List of list. Last iteration
  Returns:
    a list of length num_players, indicating the number of beneficial deviations for each player from last_meta_prob
  """
  num_player = len(last_meta_prob)
  num_new_pol = [ meta_game[0].shape[i]-len(last_meta_prob[i]) for i in range(num_player)]
  num_pol = [ meta_game[0].shape[i] for i in range(num_player)]
  prob_matrix = meta_strategies.general_get_joint_strategy_from_marginals(last_meta_prob)
  this_meta_prob = [np.append(last_meta_prob[i],[0 for _ in range(num_new_pol[i])]) for i in range(num_player)]
  beneficial_deviation = [0 for _ in range(num_player)]
  for i in range(num_player): 
    ne_payoff = np.sum(last_meta_game[i]*prob_matrix)
    # iterate through player's new policy
    for j in range(num_new_pol[i]):
      dev_prob = this_meta_prob.copy()
      dev_prob[i] = np.zeros(num_pol[i])
      dev_prob[i][len(last_meta_prob[i])+j] = 1
      new_prob_matrix = meta_strategies.general_get_joint_strategy_from_marginals(dev_prob)
      dev_payoff = np.sum(meta_game[i]*new_prob_matrix)
      if ne_payoff < dev_payoff:
        beneficial_deviation[i] += 1

  if verbose:
    print("\n---------------------------\nBeneficial Deviation :")
    for p in range(len(beneficial_deviation)):
      print('player '+str(p)+':',beneficial_deviation[p])
  return beneficial_deviation

def init_ars_parallel_responder(sess, env):
  """
  Initializes the ARS responder and agents.
  :param sess: A fake sess=None
  :param env: A rl environment.
  :return: oracle and agents.
  """
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]
  agent_class = rl_policy.ARSPolicy_parallel
  agent_kwargs = {
    "session": None,
    "info_state_size": info_state_size,
    "num_actions": num_actions,
    "learning_rate": FLAGS.ars_learning_rate,
    "nb_directions": FLAGS.num_directions,
    "nb_best_directions": FLAGS.num_directions,
    "noise": FLAGS.noise
  }

  oracle = rl_oracle.RLOracle(
    env,
    agent_class,
    agent_kwargs,
    number_training_episodes=FLAGS.number_training_episodes,
    self_play_proportion=FLAGS.self_play_proportion,
    sigma=FLAGS.sigma,
    num_workers=FLAGS.num_workers,
    ars_parallel=True
  )

  agents = [
    agent_class(
      env,
      player_id,
      **agent_kwargs)
    for player_id in range(FLAGS.n_players)
  ]
  for agent in agents:
    agent.freeze()
  agent_kwargs["policy_class"] = "ARS_parallel"
  return oracle, agents, agent_kwargs



def print_policy_analysis(policies, game, verbose=False):
  """Function printing policy diversity within game's known policies.

  Warning : only works with deterministic policies.
  Args:
    policies: List of list of policies (One list per game player)
    game: OpenSpiel game object.
    verbose: Whether to print policy diversity information. (True : print)

  Returns:
    List of list of unique policies (One list per player)
  """
  states_dict = get_all_states.get_all_states(game, np.infty, False, False)
  unique_policies = []
  for player in range(len(policies)):
    cur_policies = policies[player]
    cur_set = set()
    for pol in cur_policies:
      cur_str = ""
      for state_str in states_dict:
        if states_dict[state_str].current_player() == player:
          pol_action_dict = pol(states_dict[state_str])
          max_prob = max(list(pol_action_dict.values()))
          max_prob_actions = [
              a for a in pol_action_dict if pol_action_dict[a] == max_prob
          ]
          cur_str += "__" + state_str
          for a in max_prob_actions:
            cur_str += "-" + str(a)
      cur_set.add(cur_str)
    
    unique_policies.append(cur_set)
  if verbose:
    print("\n---------------------------\nPolicy Diversity :")
    for player, cur_set in enumerate(unique_policies):
      print("Player {} : {} unique policies.".format(player, len(cur_set)))
  print("")
  return unique_policies

def save_at_termination(solver, file_for_meta_game):
    with open(file_for_meta_game,'wb') as f:
        pickle.dump(solver.get_meta_game(), f)

def gpsro_looper(env, oracle, agents, writer, quiesce=False, checkpoint_dir=None, seed=None):
  """Initializes and executes the GPSRO training loop."""
  sample_from_marginals = True  # TODO(somidshafiei) set False for alpharank
  training_strategy_selector = FLAGS.training_strategy_selector or strategy_selectors.probabilistic_strategy_selector
  
  if not quiesce:
    solver = psro_v2.PSROSolver
  elif FLAGS.sparse_quiesce:
    solver = quiesce_sparse.PSROQuiesceSolver
  else:
    solver = PSROQuiesceSolver

  g_psro_solver = solver(
      env.game,
      oracle,
      initial_policies=agents,
      training_strategy_selector=training_strategy_selector,
      rectifier=FLAGS.rectifier,
      sims_per_entry=FLAGS.sims_per_entry,
      number_policies_selected=FLAGS.number_policies_selected,
      meta_strategy_method=FLAGS.meta_strategy_method,
      prd_iterations=50000,
      prd_gamma=1e-10,
      sample_from_marginals=sample_from_marginals,
      symmetric_game=FLAGS.symmetric_game,
      checkpoint_dir=checkpoint_dir)
  
  last_meta_prob = [np.array([1]) for _ in range(FLAGS.n_players)]
  last_meta_game = g_psro_solver.get_meta_game()
  #atexit.register(save_at_termination, solver=g_psro_solver, file_for_meta_game=checkpoint_dir+'/meta_game.pkl')
  start_time = time.time()
  for gpsro_iteration in range(1,FLAGS.gpsro_iterations+1):
    if FLAGS.verbose:
      print("\n===========================\n")
      print("Iteration : {}".format(gpsro_iteration))
      print("Time so far: {}".format(time.time() - start_time))
    train_reward_curve = g_psro_solver.iteration(seed=seed)
    meta_game = g_psro_solver.get_meta_game()
    meta_probabilities = g_psro_solver.get_meta_strategies()
    nash_meta_probabilities = g_psro_solver.get_nash_strategies()
    policies = g_psro_solver.get_policies()
   
    if FLAGS.verbose:
      # print("Meta game : {}".format(meta_game))
      print("Probabilities : {}".format(meta_probabilities))
      print("Nash Probabilities : {}".format(nash_meta_probabilities))
    
    if FLAGS.compute_exact_br: #simple extensive form calculate best response
      aggregator = policy_aggregator.PolicyAggregator(env.game)
      aggr_policies = aggregator.aggregate(
          range(FLAGS.n_players), policies, nash_meta_probabilities)
      
      print('found aggregated probabiolities')

      exploitabilities, expl_per_player = exploitability.nash_conv(
          env.game, aggr_policies, return_only_nash_conv=False)

      print('calculated exploitabilities')
      for p in range(len(expl_per_player)):
        writer.add_scalar('player'+str(p)+'_exp',expl_per_player[p],gpsro_iteration)
      writer.add_scalar('exp',exploitabilities,gpsro_iteration)
      if FLAGS.verbose:
        print("Exploitabilities : {}".format(exploitabilities))
        print("Exploitabilities per player : {}".format(expl_per_player))

      unique_policies = print_policy_analysis(policies, env.game, FLAGS.verbose)
      for p, cur_set in enumerate(unique_policies):
        writer.add_scalar('p'+str(p)+'_unique_p',len(cur_set),gpsro_iteration)

    else: # use combined game to evaluate exp, thus nash_ne for every iteration should be saved
      save_nash(nash_meta_probabilities, gpsro_iteration, checkpoint_dir)

    if gpsro_iteration % 5 ==0:
      save_at_termination(solver=g_psro_solver, file_for_meta_game=checkpoint_dir+'/meta_game.pkl')
      save_strategies(solver=g_psro_solver, checkpoint_dir=checkpoint_dir)
    
    beneficial_deviation = print_beneficial_deviation_analysis(last_meta_game, meta_game, last_meta_prob, FLAGS.verbose)
    last_meta_prob, last_meta_game = meta_probabilities, meta_game
    for p in range(len(beneficial_deviation)):
      writer.add_scalar('p'+str(p)+'_beneficial_dev',int(beneficial_deviation[p]),gpsro_iteration)
    writer.add_scalar('beneficial_devs',sum(beneficial_deviation),gpsro_iteration)

    if FLAGS.log_train and (gpsro_iteration<=10 or gpsro_iteration%5==0):
      for p in range(len(train_reward_curve)):
        for p_i in range(len(train_reward_curve[p])):
          writer.add_scalar('player'+str(p)+'_'+str(gpsro_iteration),train_reward_curve[p][p_i],p_i)
    
def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
 
  if FLAGS.seed is None:
    seed = np.random.randint(low=0, high=1e5)
  else:
    seed = FLAGS.seed
  set_seed(seed)
  
  checkpoint_dir = FLAGS.game_name
  if FLAGS.game_name in ['laser_tag']: # games where parameter does not have num_players
    #game_param = {'zero_sum': pyspiel.GameParameter(False)}
    game_param_raw = {'zero_sum': False}
  elif FLAGS.game_name in ['markov_soccer']:
    game_param_raw = {'horizon':20}
  else:
    game_param_raw = {"players": FLAGS.n_players}
  if FLAGS.game_param is not None:
    for ele in FLAGS.game_param:
      ele_li = ele.split("=")
      game_param_raw[ele_li[0]] = int(ele_li[1])
      checkpoint_dir += '_'+ele_li[0]+'_'+ele_li[1]
    checkpoint_dir += '_'

  #game_param = {}
  #for key, val in game_param_raw.items():
  #  game_param[key] = pyspiel.GameParameter(val)
  #game = pyspiel.load_game_as_turn_based(FLAGS.game_name, game_param)

  game_param_str = FLAGS.game_name+"("
  for key,val in game_param_raw.items():
    game_param_str += key+"="+str(val)+","
  game_param_str = game_param_str[:-1]+")"
  game = pyspiel.load_game(game_param_str)

  env = rl_environment.Environment(game,seed=seed)
  env.reset()

  if not os.path.exists(FLAGS.root_result_folder):
    os.makedirs(FLAGS.root_result_folder)
  checkpoint_dir += str(FLAGS.n_players)+'_sims_'+str(FLAGS.sims_per_entry)+'_it_'+str(FLAGS.gpsro_iterations)+'_ep_'+str(FLAGS.number_training_episodes)+'_or_'+FLAGS.oracle_type+'_heur_'+FLAGS.meta_strategy_method
  if FLAGS.oracle_type == 'ARS':
    oracle_flag_str = '_arslr_'+str(FLAGS.ars_learning_rate)+'_arsn_'+str(FLAGS.noise)+'_arsnd_'+str(FLAGS.num_directions)+'_arsbd_'+str(FLAGS.num_best_directions)
  elif FLAGS.oracle_type == 'BR':
    oracle_flag_str = ''
  else:
    oracle_flag_str = '_hl_'+str(FLAGS.hidden_layer_size)+'_bs_'+str(FLAGS.batch_size)+'_nhl_'+str(FLAGS.n_hidden_layers)
    if FLAGS.oracle_type == 'DQN':
      oracle_flag_str += '_dqnlr_'+str(FLAGS.dqn_learning_rate)+'_tnuf_'+str(FLAGS.update_target_network_every)+'_lf_'+str(FLAGS.learn_every)
    else:
      oracle_flag_str += '_ls_'+str(FLAGS.loss_str)+'_nqbp_'+str(FLAGS.num_q_before_pi)+'_ec_'+str(FLAGS.entropy_cost)+'_clr_'+str(FLAGS.critic_learning_rate)+'_pilr_'+str(FLAGS.pi_learning_rate)

  checkpoint_dir = checkpoint_dir + oracle_flag_str+'_se_'+str(seed)+'_'+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
  checkpoint_dir = os.path.join(os.getcwd(),FLAGS.root_result_folder, checkpoint_dir)
                                
  writer = SummaryWriter(logdir=checkpoint_dir+'/log')
  if FLAGS.sbatch_run:
    sys.stdout = open(checkpoint_dir+'/stdout.txt','w+')

  env_kwargs = {"game_name":FLAGS.game_name, "param": game_param_raw}
  strategy_path = os.path.join(checkpoint_dir, 'strategies')
  if not os.path.exists(strategy_path):
    os.makedirs(strategy_path)
  save_pkl(env_kwargs, strategy_path+'/env.pkl')

  # Initialize oracle and agents
  with tf.Session() as sess:
    if FLAGS.oracle_type == "DQN":
      oracle, agents, agent_kwargs = init_dqn_responder(sess, env)
    elif FLAGS.oracle_type == "PG":
      oracle, agents, agent_kwargs = init_pg_responder(sess, env)
    elif FLAGS.oracle_type == "BR":
      oracle, agents, agent_kwargs = init_br_responder(env)
    elif FLAGS.oracle_type == "ARS":
      oracle, agents, agent_kwargs = init_ars_responder(sess, env)
    elif FLAGS.oracle_type == "ARS_parallel":
      oracle, agents, agent_kwargs = init_ars_parallel_responder(sess, env)
    # sess.run(tf.global_variables_initializer())
    save_pkl(agent_kwargs, strategy_path+'/kwargs.pkl')

    gpsro_looper(env, oracle, agents, writer, quiesce=FLAGS.quiesce, checkpoint_dir=checkpoint_dir, seed=seed)

  writer.close()

if __name__ == "__main__":
  app.run(main)
