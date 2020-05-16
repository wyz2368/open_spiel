from open_spiel.python.algorithms.nash_solver.general_nash_solver import nash_solver
from open_spiel.python.algorithms.psro_v2 import meta_strategies
from open_spiel.python.algorithms.psro_v2 import rl_policy
from open_spiel.python import policy
import pandas as pd
import numpy as np
import itertools
import pickle
import os

#def regret(meta_games, subgame_index, subgame_ne=None):
#    """
#    Calculate the regret based on a complete payoff matrix for PSRO
#    Assume all players have the same number of policies
#    :param meta_games: meta_games in PSRO
#    :param subgame_index: the subgame to evaluate. Redundant when subgame_ne is supplied
#    :param: subgame_ne: subgame nash equilibrium vector.
#    :return: a list of regret, one for each player.
#    """
#    num_policy = np.shape(meta_games[0])[0]
#    num_players = len(meta_games)
#    if num_policy == subgame_index:
#        print("The subgame is same as the full game. Return zero regret.")
#        return np.zeros(num_players)
#    num_new_pol = num_policy - subgame_index
#
#    index = [list(np.arange(subgame_index)) for _ in range(num_players)]
#    submeta_games = [ele[np.ix_(*index)] for ele in meta_games]
#    nash = nash_solver(submeta_games, solver="gambit") if not subgame_ne else subgame_ne
#    prob_matrix = meta_strategies.general_get_joint_strategy_from_marginals(nash)
#    this_meta_prob = [np.append(nash[i],[0 for _ in range(num_new_pol)]) for i in range(num_players)]
#
#    nash_payoffs = []
#    deviation_payoffs = []
#
#    for i in range(num_players): 
#        ne_payoff = np.sum(submeta_games[i]*prob_matrix)
#        # iterate through player's new policy
#        dev_payoff = []
#        for j in range(num_new_pol):
#            dev_prob = this_meta_prob.copy()
#            dev_prob[i] = np.zeros(num_policy)
#            dev_prob[i][subgame_index+j] = 1
#            new_prob_matrix = meta_strategies.general_get_joint_strategy_from_marginals(dev_prob)
#            dev_payoff.append(np.sum(meta_games[i]*new_prob_matrix))
#        deviation_payoffs.append(dev_payoff-ne_payoff)
#        nash_payoffs.append(ne_payoff)
#    
#    regret = np.maximum(np.max(deviation_payoffs,axis=1),0)
#    return regret

def regret(meta_games, subgame_index, subgame_ne=None, start_index=0):
    """
    Calculate the regret based on a complete payoff matrix for PSRO
    Assume all players have the same number of policies
    :param meta_games: meta_games in PSRO
    :param subgame_index: the subgame to evaluate. Redundant when subgame_ne is supplied
    :param start_index: starting index for the subgame
    :param: subgame_ne: subgame nash equilibrium vector.
    :return: a list of regret, one for each player.
    """
    num_policy = np.shape(meta_games[0])[0]
    num_players = len(meta_games)
    if num_policy == subgame_index-start_index:
        print("The subgame is same as the full game. Return zero regret.")
        return np.zeros(num_players)

    num_new_pol_back = num_policy - subgame_index

    index = [list(np.arange(start_index,subgame_index)) for _ in range(num_players)]
    submeta_games = [ele[np.ix_(*index)] for ele in meta_games]
    nash = nash_solver(submeta_games, solver="gambit") if not subgame_ne else subgame_ne
    prob_matrix = meta_strategies.general_get_joint_strategy_from_marginals(nash)
    this_meta_prob = [np.concatenate([0 for _ in range(start_index)], nash[i], [0 for _ in range(num_new_pol_back)]) for i in range(num_players)]

    nash_payoffs = []
    deviation_payoffs = []

    for i in range(num_players): 
        ne_payoff = np.sum(submeta_games[i]*prob_matrix)
        # iterate through player's new policy
        dev_payoff = []
        for j in range(start_index + num_new_pol_back):
            dev_prob = this_meta_prob.copy()
            dev_prob[i] = np.zeros(num_policy)
            if j < start_index:
                dev_prob[i][j] = 1
            else:
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

def sample_episodes(env, agents, number_epsiodes=1):
    """
    sample pure strategy payoff in an env
    Params:
        agents : a list of length num_player
        env    : open_spiel environment
    Returns:
        a list of length num_player containing players' strategies
    """
    cumulative_rewards = np.zeros(len(agents))

    for _ in range(number_episodes):
      time_step = env.reset()
      cumulative_rewards = 0.0
      while not time_step.last():
        if time_step.is_simultaneous_move():
          action_list = []
          for agent in agents:
            output = agent.step(time_step, is_evaluation=True)
            action_list.append(output.action)
          time_step = env.step(action_list)
          cumulative_rewards += np.array(time_step.rewards)
        else:
          player_id = time_step.observations["current_player"]
          agent_output = agents[player_id].step(time_step, is_evaluation=False)
          action_list = [agent_output.action]
          time_step = env.step(action_list)
          cumulative_rewards += np.array(time_step.rewards)

    return cumulative_rewards/number_episodes

def rollout(env, strategies, strategy_support, sims_per_entry=1000):
    """
    Evaluate player's mixed strategy with support in env.
    Params:
        env              : an open_spiel env
        strategies       : list of list, each list containing a player's strategy
        strategy_support : mixed_strategy support probability vector
        sims_per_entry   : number of episodes for each pure strategy profile to sample
    Return:
        a list of players' payoff
    """
    num_players = len(strategies)
    num_strategies = [len(ele) for ele in strategies]
    prob_matrix = meta_strategies.general_get_joint_strategy_from_marginals(strategy_support)
    payoff_tensor = np.zeros([num_players]+num_strategies)

    for ind in itertools.product(*[np.arange(ele) for ele in num_strategies]):
        strat = [strategies[i][ind[i]] for i in range(num_players)]
        pure_payoff = sample_episodes(env, strat, sims_per_entry)
        payoff_tensor[tuple([...]+list(ind))] = pure_payoff

    return [np.sum(payoff_tensor[i]*prob_matrix) for i in range(num_players)]

def calculate_combined_game(strat_and_meta_game_li, combined_game_save_path, sims_per_entry=1000, env_file_name="env.pkl", strategy_kwarg_file_name="kwarg.pkl", meta_game_file_name="meta_game.pkl"):
    """
    Calculate combined game
    Params:
        strat_and_meta_game_li : a list of checkpoint directory 
        combined_game_save_path: a parent directory where calculated combined game is to be saved
    Assumes directory structure to strategies are
    --meta_game.pkl (meta_game_file_name)
    --strategies (params given)
        --kwarg.pkl (strategy_kwarg_file_name)
        --env.pkl   (env_file_name)
        --player_0
            1.pkl
            2.pkl
            ...
        --player_1
        ...
    """
    num_runs = len(strat_and_meta_game_li)

    # process strategy dir. 
    strategies = [] # a list of length num_runs. Each sub list contains number of player sublist
    meta_games = []
    for i in range(num_runs): # traverse all runs
        meta_games.append(load_pkl(os.path.join(strat_and_meta_game_li[i],meta_game_file_name)))
        strat_root_dir = strat_and_meta_game_li[i] + "/strategies"
        player_dir = [x for x in os.listdir(strat_root_dir) if os.path.isdir(os.path.join(strat_root_dir,x))]

        # initialize game once
        if i == 0:
            env_kwargs = load_pkl(os.path.join(strat_root_dir, env_file_name))
            game = pyspiel.load_game_as_turn_based(env_kwargs["game_name"], env_kwargs["param"])
        # load strategy kwargs: each run might have different strategy type and kwargs
        strategy_kwargs = load_pkl(os.path.join(strat_root_dir, strategy_kwarg_file_name))
        strategy_type = strategy_kwargs["policy_class"]
        strategy_kwargs = strategy_kwargs["strategy_kwargs"]

        strategy_run = []
        for p in len(player_dir):  # traverse players from each run
            strategy_player = []
            strat_player_dir = os.path.join(strat_root_dir,player_dir[p])
            num_strategy = len(os.listdir(strat_player_dir))
            for weight_file in range(num_strategy)+1: # traverse the weight file
                weight = load_pkl(os.path.join(strat_player_dir,str(weight_file)+'.pkl'))
                strategy = load_strategy(strategy_type, strategy_kwargs, p, weight)
                strategy_player.append(strategy)
            strategy_run.append(strategy_player)
        strategies.append(strategy_run)
    
    # build the meta_game matrix
    num_player = len(strategies[0])
    # strategy_num: np array of size: num_runs* num_player
    strategy_num = np.array([[len(ele[i]) for i in range(num_player)] for ele in strategies])
    combined_game = [np.zeros(list(strategy_num.sum(axis=0)))*np.nan for _ in range(num_player)]
    
    # fill in the existing meta_game values
    for runs in range(num_runs):
        meta_game_start_index = strategy_num[:runs].sum(axis=0)
        meta_game_end_index = strategy_num[:runs+1].sum(axis=0)
        index = [slice(meta_game_start_index[j],meta_game_end_index[j]:1) \
            for j in range(num_player)]
        for p in range(num_player):
            combined_game[index] = meta_games[i]
    
    # reshape strategies in terms of combined game
    strategies_combined = [list(itertools.chain.from_iterable(ele[p] for ele in strategies)) \
        for p in range(num_player)]
    # sample the missing entries
    it = np.nditer(combined_game[0],flags=['multi_index'])
    while not it.finished:
        index = it.multi_index
        if not np.isnan(combined_game[0][index]):
            agents = [strategies_combined[p][index[p]] for p in range(num_player)]
            rewards = sample_episode(env, agents, sims_per_entry)
            for p in range(num_player):
                combined_game[p][index] = rewards[p]
        _ = it.iternext()

    save_pkl(combined_game, os.path.join(combined_game_save_path,"combined_game.pkl"))

    return combined_game 

def calculate_regret_from_combined_game(ne_dir, strat_start, combined_game_path):
    """
    WARNING: assumes automatically that all strategies represneted by ne_dir
             is in the combined_game, ranged from strat_start and consecutive. 
    Calculate the exploitability of each iteration, the number of iterations equaling
    to the number of files in the ne_dir.
    Params:
        ne_dir             : directory that contains and only contains int.pkl, absolute path
        combined_game_path : the file that contains the last combined game
    Returns:
        save exp file to exp.csv at the same directory level with combined_game_path
        return exp
    """
    combined_game = load_pkl(combined_game_path)
    exploitability = [[],[]]
    for fi in os.listdir(ne_dir):
        subgame_index = int(fi.split('.pkl')[0])+1+strat_start
        file_name = os.path.join(ne_dir, fi)
        eq = load_pkl(file_name)
        # calculate regret
        regrets = regret(combined_game, subgame_index, subgame_ne=eq, start_index=strat_start)
        exploitability[0].append(sum(regrets))
        exploitability[1].append(subgame_index)
    
    # write the exploitabilities to csv file
    result_file = '/'.join(combined_game_path.split('/')[:-1]) + 'exp.csv'
    exploitability = [x for _, x in sorted(zip(exploitability[1],exploitability[0]))]
    df = pd.DataFrame(exploitability)
    df.to_csv(result_file,index=False)
    
    return exploitability

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


def smoothing_kl(p, q, eps=0.001):
    p = smoothing(p, eps)
    q = smoothing(q, eps)
    return np.sum(p * np.log(p / q))


def smoothing(p, eps):
    zeros_pos_p = np.where(p == 0)[0]
    num_zeros = len(zeros_pos_p)
    x = eps * num_zeros / (len(p) - num_zeros)
    for i in range(len(p)):
        if i in zeros_pos_p:
            p[i] = eps
        else:
            p[i] -= x
    return p


def isExist(path):
    """
    Check if a path exists.
    :param path: path to check.
    :return: bool
    """
    return os.path.exists(path)

def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if isExists:
        raise ValueError(path + " already exists.")
    else:
        os.makedirs(path)
        print(path + " has been created successfully.")

def save_pkl(obj,path):
    """
    Pickle a object to path.
    :param obj: object to be pickled.
    :param path: path to save the object
    """
    with open(path,'wb') as f:
        pickle.dump(obj,f)

def load_pkl(path):
    """
    Load a pickled object from path
    :param path: path to the pickled object.
    :return: object
    """
    if not isExist(path):
        raise ValueError(path + " does not exist.")
    with open(path,'rb') as f:
        result = pickle.load(f)
    return result


def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def save_nash(nash_prob, iteration, checkpoint_dir):
    """
    Save nash probabilities
    """
    current_path = os.path.join(checkpoint_dir,'nash_prob/')
    if not isExist(current_path):
        mkdir(current_path)
    save_pkl(nash_prob, current_path+str(iteration)+'.pkl')

def save_strategies(solver, checkpoint_dir):
    """
    Save all strategies.
    """
    num_players = solver._num_players
    for player in range(num_players):
        current_path = os.path.join(checkpoint_dir, 'strategies/player_' + str(player) + "/")
        if not isExist(current_path):
            mkdir(current_path)
        for i, policy in enumerate(solver.get_policies()[player]):
            if isExist(current_path + str(i+1) + '.pkl'):
                continue
            save_pkl(policy.get_weights(), current_path + str(i+1) + '.pkl')

def load_strategy(strategy_type, strategy_kwargs, env, player_id, strategy_weight):
    """
    Load Strategies. If initialization required, initialize
    """
    if strategy_type == "BR":
        agent_class = policy.TabularPolicy(env.game)
    elif strategy_type == "ARS":
        agent_class = rl_policy.ARSPolicy
    elif strategy_type == "DQN":
        agent_class = rl_policy.DQNPolicy
    elif strategy_type == "PG":
        agent_class = rl_policy.PGPolicy
    elif strategy_type == "ARS_parallel":
        agent_class = rl_policy.ARSPolicy_parallel
    else:
        raise NotImplementedError
  agent = agent_class(env, player_id, **strategy_kwargs)
  agent.set_weight(strategy_weight)
  agent.freeze()

  return agent
