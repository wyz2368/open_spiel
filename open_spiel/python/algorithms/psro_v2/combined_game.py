import pyspiel
from open_spiel.python import rl_environment
from open_spiel.python.algorithms.psro_v2.utils import set_seed
from open_spiel.python.algorithms.psro_v2.eval_utils import *

import os
from absl import app
from absl import flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("seed", None, "seed")
flags.DEFINE_bool("only_combine_nash", False, "only combine nash in combined game")
flags.DEFINE_string("root_result_folder","","root result folder that contains the experiments runs to be calculated into combined game")
flags.DEFINE_integer("sims_per_entry", 1000,
                     ("Number of simulations to run to estimate each element"
                      "of the game outcome matrix."))

def calculate_combined_game(checkpoint_dirs,
                            combined_game_save_path,
                            only_combine_nash=True,
                            sims_per_entry=1000,
                            seed=4,
                            env_file_name="env.pkl",
                            strategy_kwarg_file_name="kwargs.pkl",
                            meta_game_file_name="meta_game.pkl"):
    """
    Calculate combined game
    Params:
        checkpoint_dirs         : a list of checkpoint directory 
        combined_game_save_path : parent directory to save combined game
        only_combine_nash       : only combine strategies with nash equilibrium support
    Assumes directory structure to strategies are
    --meta_game.pkl (meta_game_file_name)
    --nash_prob
        --1.pkl
        --2.pkl
        ...
    --strategies (params given)
        --kwarg.pkl (strategy_kwarg_file_name)
        --env.pkl   (env_file_name)
        --player_0
            1.pkl
            2.pkl
            ...
        --player_1
        ...
    Return:
        combined game
    """
    num_runs = len(checkpoint_dirs)
    
    # process strategy dir. 
    strategies = [] # a list of length num_runs. Each sub list contains number of player sublist
    meta_games = []
    for i in range(num_runs): # traverse all runs
        meta_games.append(load_pkl(os.path.join(checkpoint_dirs[i],meta_game_file_name)))
        strat_root_dir = checkpoint_dirs[i] + "/strategies"
        player_dir = [x for x in os.listdir(strat_root_dir) \
            if os.path.isdir(os.path.join(strat_root_dir,x))]

        # initialize game & rl environment once
        if i == 0:
            env_kwargs = load_pkl(os.path.join(strat_root_dir, env_file_name))
            for k,v in env_kwargs["param"] .items():
                env_kwargs["param"][k] = pyspiel.GameParameter(v)
            game = pyspiel.load_game_as_turn_based(env_kwargs["game_name"], env_kwargs["param"])
            env = rl_environment.Environment(game, seed=seed)
            env.reset()

        
        # load strategy kwargs: each run might have different strategy type and kwargs
        strategy_kwargs = load_pkl(os.path.join(strat_root_dir, strategy_kwarg_file_name))
        strategy_type = strategy_kwargs.pop("policy_class")
        
        # load all strategies
        strategy_run = []
        for p in range(len(player_dir)):  # traverse players from each run
            strategy_player = []
            strat_player_dir = os.path.join(strat_root_dir,player_dir[p])
            num_strategy = len(os.listdir(strat_player_dir))

            # find the final nash equilibrium, adjust meta_games accordingly
            # nash is saved every iteration, yet strategies are saved every five iterations
            # extract the ne file that corresponds to number of strategies
            if only_combine_nash and p==0:
                nash_root_dir = checkpoint_dirs[i] + "/nash_prob"
                final_nash_file = os.path.join(nash_root_dir, str(num_strategy-1)+".pkl")
                nash_ne = load_pkl(final_nash_file)
                none_zero_index = [[i for i in range(len(ele)) if ele[i]>0.001] \
                    for ele in nash_ne]
                meta_games[i] = [ele[np.ix_(*none_zero_index)] for ele in meta_games[i]]

            for weight_file in range(1,num_strategy+1): # traverse the weight file
                if only_combine_nash and nash_ne[p][weight_file-1] < 0.001:
                    continue # only combine nash strategies
                weight = load_pkl(os.path.join(strat_player_dir,str(weight_file)+'.pkl'))
                strategy = load_strategy(strategy_type, strategy_kwargs, env, p, weight)
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
        index = [slice(meta_game_start_index[j],meta_game_end_index[j],1) \
            for j in range(num_player)]
        for p in range(num_player):
            combined_game[runs][tuple(index)] = meta_games[runs][p]
    
    # reshape strategies in terms of combined game
    strategies_combined = [list(itertools.chain.from_iterable(ele[p] for ele in strategies)) \
        for p in range(num_player)]
    # sample the missing entries
    it = np.nditer(combined_game[0],flags=['multi_index'])
    while not it.finished:
        index = it.multi_index
        if not np.isnan(combined_game[0][index]):
            agents = [strategies_combined[p][index[p]] for p in range(num_player)]
            rewards = sample_episodes(env, agents, sims_per_entry)
            for p in range(num_player):
                combined_game[p][index] = rewards[p]
        _ = it.iternext()

    save_pkl(combined_game, os.path.join(combined_game_save_path,"combined_game_se_"+str(seed)+'_neonly_'+str(only_combine_nash)+".pkl"))

    return combined_game 

def calculate_regret_from_combined_game(ne_dir, strat_start, strat_end, combined_game_path):
    """
    WARNING: assumes automatically that all strategies represneted by ne_dir
             is in the combined_game, ranged from strat_start and consecutive. This is 99% likely not the case 
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

def main(argv):
    if len(argv) > 1:
      raise app.UsageError("Too many command-line arguments.")
  
    assert os.path.isdir(FLAGS.root_result_folder), "no such directory"
    dir_content = os.listdir(FLAGS.root_result_folder)
    assert len(dir_content)>1, "directory provided cannot make combined game"
    checkpoint_dirs = [os.path.join(FLAGS.root_result_folder, x) \
        for x in os.listdir(FLAGS.root_result_folder) \
        if os.path.isdir(os.path.join(FLAGS.root_result_folder, x))]
  
    seed = FLAGS.seed if FLAGS.seed else np.random.randint(low=0, high=1e5)
    set_seed(seed)

    combined_game = calculate_combined_game(checkpoint_dirs,
                                            FLAGS.root_result_folder,
                                            only_combine_nash=FLAGS.only_combine_nash,
                                            sims_per_entry=FLAGS.sims_per_entry,
                                            seed=seed)

if __name__ == "__main__":
    app.run(main)
