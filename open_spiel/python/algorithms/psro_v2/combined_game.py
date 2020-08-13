"""
The script calculate combined game and evaluates ne performance of each run or evaluate iteration regrets of every run. Assumes a standard file structure. Please copy all relevant experiment runs into a root folder and supply the root folder through FLAGS.
--root_result_folder
    --experiment1
        |--meta_game.pkl
        |--nash_prob
        |  |--1.pkl
        |  |--2.pkl
        |  |--...
        |--strategies
        |  |--kwarg.pkl
        |  |--env.pkl(env args here for policy initialization)
        |  |--player_0
        |  |  |--1.pkl
        |  |  |--2.pkl
        |  |  |--...
        |  |--player_1
        |  |--...

    --experiment2
    ...
    # The following is the directory structure after running command1
    --combined_game_datetime
        |--run_0_1.sh
        |--run_0_2.sh
        |--run_1_2.sh
        |--run_0_1
        |  |-- simlink experiment 0
        |  |-- simlink experiment 1
        |--run_0_2
        |  |-- simlink experiment 0
        |  |-- simlink experiment 2

The following documents the procedure for calculating combined game and evaluating nash conv using combined game
1. 
python combined_game.py --break_into_subcombine_game=True --base_slurm=<base_slurm_file> --num_evaluation_episodes=<number of desired strategy in each run-1> --root_result_folder=<root_result_folder_location>
This creates a combined_game experiment folder in <root_result_folder_location> and create job file and their working directories for pairwise runs, as indicated in the chart above. The purpose is that one huge combined game is too large to load and too slow to simulate. Advise to break up a huge combined game into multiple pair-wise combined game when each player has >150 strategies
2.
after checking the sh file, modify the batch job file and submit it to greatlakes
3.
python combined_game.py --gather_subgame=True --num_run=<number of experiments> --num_evaluation_episodes=<number of desired strategy in each run> --root_result_folder=<combined_game_datetime folder>
This essentially combines the pairwise combined game into the full combined game and save it in <combined_game_datetime> folder
4.
python combined_game.py --num_evaluation_episodes=<the number before-1> --evaluate_nash=<True/False> --evaluate_iters=<True/False> --combined_game_path=<The path above> --root_result_folder=<root_result_folder_location>
This finally evaluates the nash conv or the final nash performance of experiments and save them to csv in the original folder. Note the <num_evaluation_episode> is the previous one minus one, this value is the real epsiode number, previous ones are all number of strategies
"""

import pyspiel
from open_spiel.python import rl_environment
from open_spiel.python.algorithms.psro_v2.utils import set_seed
from open_spiel.python.algorithms.psro_v2.eval_utils import *
from open_spiel.python.algorithms.psro_v2.abstract_meta_trainer import sample_episode

import os
import psutil
import datetime
import time
import shutil
import itertools
from tqdm import tqdm
import functools
print = functools.partial(print, flush=True)
from absl import app
from absl import flags
FLAGS = flags.FLAGS


# scripts breaking big combined game into subgames
flags.DEFINE_bool("break_into_subcombine_game",False,"generate scripts for subcombined game")
flags.DEFINE_string("base_slurm","/home/qmaai/se/open_spiel/open_spiel/python/algorithms/psro_v2/combine_game_base_slurm.sh","position of base slurm")

# scripts combining subgames into a big combined game
flags.DEFINE_bool("gather_subgame",False,"gather subcombine game into a full combined game")
flags.DEFINE_integer("num_run",10,"number of runs in sub combined game")

# calculate combined game only with ne strategies
flags.DEFINE_integer("num_evaluation_episodes",150,"number of strategies to be considered each run")
flags.DEFINE_bool("only_combine_nash", False, "only combine nash in combined game")
flags.DEFINE_bool("evaluate_nash",True, "evaluate regret of final nash for all runs against meta_game")
flags.DEFINE_bool("evaluate_iters",False, "evaluate regret curve for all runs against meta_game")

flags.DEFINE_integer("seed", None, "seed")
flags.DEFINE_string("root_result_folder","","root result folder that contains the experiments runs to be calculated into combined game")
flags.DEFINE_string("combined_game_path","","provide combined game pkl if there already is an existing combined game, in this case, only need to calculate exploitabilities of each run")
flags.DEFINE_integer("sims_per_entry", 1000,
                     ("Number of simulations to run to estimate each element"
                      "of the game outcome matrix."))

ZERO_THRESHOLD = 0.001

def load_env(strat_root_dir, env_file_name, seed=None):
    env_kwargs = load_pkl(os.path.join(strat_root_dir, env_file_name))
    env_kwargs_str = env_kwargs["game_name"]+"("
    for k,v in env_kwargs["param"].items():
      env_kwargs_str += k+"="+str(v)+","
    env_kwargs_str = env_kwargs_str[:-1]+")"
    game = pyspiel.load_game(env_kwargs_str)
    env = rl_environment.Environment(game, seed=seed)
    env.reset()
    return env

def calculate_combined_game(checkpoint_dirs,
                            combined_game_save_path,
                            episode=150,
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
        episode                 : number of episodes/strategies to evaluate in each run
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
    #cg0=load_pkl("/home/qmaai/se/open_spiel/open_spiel/python/algorithms/psro_v2/slurm_scripts/combined_game/0_2_leduc_poker2_sims_1000_it_250_ep_10000_or_DQN_heur_uniform_hl_256_bs_32_nhl_4_dqnlr_0.01_tnuf_500_lf_10_se_43912_2020-07-02_18-22-38/meta_game.pkl")
    #cg1=load_pkl("/home/qmaai/se/open_spiel/open_spiel/python/algorithms/psro_v2/slurm_scripts/combined_game/1_0_leduc_poker2_sims_1000_it_250_ep_10000_or_DQN_heur_uniform_hl_256_bs_32_nhl_4_dqnlr_0.01_tnuf_500_lf_10_se_21939_2020-07-02_18-29-02/meta_game.pkl")
    #cg2=load_pkl("/home/qmaai/se/open_spiel/open_spiel/python/algorithms/psro_v2/slurm_scripts/combined_game/2_1_leduc_poker2_sims_1000_it_250_ep_10000_or_DQN_heur_nash_hl_256_bs_32_nhl_4_dqnlr_0.01_tnuf_500_lf_10_se_36619_2020-07-02_18-22-39/meta_game.pkl")

    process = psutil.Process(os.getpid())
    num_runs = len(checkpoint_dirs)
    
    # process strategy dir. 
    strategies = [] # a list of length num_runs. Each sub list contains number of player sublist
    meta_games = []
    indexes = []  # directories are not read in numerical sequence
    for i in range(num_runs): # traverse all runs
        
        # assume directories are all labeled with the first char
        if checkpoint_dirs[i].split('/')[-1][0].isnumeric():
          indexes.append(int(checkpoint_dirs[i].split('/')[-1][0]))
        else:
          indexes.append(i)
        strat_root_dir = checkpoint_dirs[i] + "/strategies"
        player_dir = [x for x in os.listdir(strat_root_dir) \
            if os.path.isdir(os.path.join(strat_root_dir,x))]

        # initialize game & rl environment once
        if i == 0:
            env = load_env(strat_root_dir, env_file_name, seed)

        # load strategy kwargs: each run might have different strategy type and kwargs
        strategy_kwargs = load_pkl(os.path.join(strat_root_dir, strategy_kwarg_file_name))
        strategy_type = strategy_kwargs.pop("policy_class")

        # load existing meta games, sloppiness and only work for 2 player game
        meta_game = load_pkl(os.path.join(checkpoint_dirs[i],meta_game_file_name))
        meta_games.append([ele[:episode,:episode] for ele in meta_game])
        
        # load all strategies
        strategy_run = []
        for p in tqdm(range(len(player_dir))):  # traverse players from each run
            strategy_player = []
            strat_player_dir = os.path.join(strat_root_dir,player_dir[p])
            num_strategy = len(os.listdir(strat_player_dir))
            num_strategy = num_strategy if num_strategy < episode else episode

            # find the final nash equilibrium, adjust meta_games accordingly
            # nash is saved every iteration, yet strategies are saved every five iterations
            # extract the ne file that corresponds to number of strategies
            if only_combine_nash and p==0:
                nash_root_dir = checkpoint_dirs[i] + "/nash_prob"
                final_nash_file = os.path.join(nash_root_dir, str(num_strategy-1)+".pkl")
                nash_ne = load_pkl(final_nash_file)
                none_zero_index = [[i for i in range(len(ele)) if ele[i] > ZERO_THRESHOLD] \
                    for ele in nash_ne]
                meta_games[i] = [ele[np.ix_(*none_zero_index)] for ele in meta_games[i]]
            
            exist_previous_loaded_weights = False
            for weight_file in range(1,num_strategy+1): # traverse the weight file, first strategy weight saved in 1.pkl instead of 0.pkl
                if only_combine_nash and nash_ne[p][weight_file-1] < ZERO_THRESHOLD:
                    continue # only combine nash strategies
                weight = load_pkl(os.path.join(strat_player_dir,str(weight_file)+'.pkl'))
                if not exist_previous_loaded_weights:
                    strategy = load_strategy(strategy_type, strategy_kwargs, env, p, weight)
                    exist_previous_loaded_weights = True
                else: #copy previous strategy structure
                    strategy = strategy.copy_with_weights_frozen(weight)

                strategy_player.append(strategy)
            strategy_run.append(strategy_player)
            print("finished loading run {} player {}".format(i,p),end=" GB: ")
            print(process.memory_info().rss/1024/1024/1024)
        strategies.append(strategy_run)

    print("finished loading all possible strategies ",end="GB: ")
    
    # sort the strategies and meta_games according to index
    strategies = [x for _,x in sorted(zip(indexes,strategies))]
    meta_games = [x for _,x in sorted(zip(indexes,meta_games))]

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
            combined_game[p][tuple(index)] = meta_games[runs][p]
    print("finished filling in all existing values",end=" GB: ")
    print(process.memory_info().rss/1024/1024/1024)

    # reshape strategies in terms of combined game
    strategies = [list(itertools.chain.from_iterable(ele[p] for ele in strategies)) \
        for p in range(num_player)]
    print("reshaping strategies",end=" GB: ")
    print(process.memory_info().rss/1024/1024/1024)

    # sample the missing entries
    it = np.nditer(combined_game[0],flags=['multi_index'])
    
    pbar = tqdm(total=combined_game[0].size-num_runs*meta_games[0][0].size)
    while not it.finished:
        index = it.multi_index
        if np.isnan(combined_game[0][index]):
            agents = [strategies[p][index[p]] for p in range(num_player)]
            
            # ensure env is not passed into function to save memory
            #rewards = np.zeros(num_player)
            #for _ in range(sims_per_entry):
            #  rewards += sample_episode(env._game.new_initial_state(),
            #                            agents).reshape(-1)
            #rewards /= sims_per_entry

            rewards = sample_episodes(env, agents, sims_per_entry)

            for p in range(num_player):
                combined_game[p][index] = rewards[p]
            pbar.update(1)
        _ = it.iternext()
    pbar.close()

    save_path =  os.path.join(combined_game_save_path,"combined_game_se_"+str(seed)+'_neonly_'+str(only_combine_nash)+".pkl")
    save_pkl(combined_game, save_path)
    print("combined game {} only combine NE saved at {}".format(only_combine_nash,save_path), end=" GB: ")
    print(process.memory_info().rss/1024/1024/1024)

    return combined_game, strategy_num


def calculate_regret_from_combined_game(ne_dir,
                                        combined_game,
                                        strat_start,
                                        strat_end,
                                        checkpoint_dir):
    """
    Calculate the exploitability of all possible iterations
    The number of max iterations evaluatable depends on strat_start and strat_end
    Assume that each player has the same number of strategies in subgame
    Params:
        ne_dir         : directory that contains and only contains int.pkl, absolute path
        strat_start    : starting index of this subgame in combined game
        strat_end      : ending index of this subgame in combined game
        checkpoint_dir : directory to save the exploitability curve
    Returns:
        save exp file to exp.csv under checkpoint_dir
        return exp
    """
    exploitability = [[],[]]
    # ne.pkl saving iteration ahead of meta_games.pkl
    # so not all ne.pkl's strategies are in the meta_games,
    # consequently not in combined_game
    max_gpsro_iter = strat_end - strat_start
    for fi in os.listdir(ne_dir):
        gpsro_iter = int(fi.split('.pkl')[0])
        if gpsro_iter > max_gpsro_iter:
            continue        
        subgame_index = gpsro_iter + strat_start
        eq = load_pkl(os.path.join(ne_dir, fi))
        # calculate regret
        regrets = regret(combined_game, subgame_index, subgame_ne=eq, start_index=strat_start)
        exploitability[0].append(sum(regrets))
        exploitability[1].append(gpsro_iter)
    
    # exploitability[1] carries gpsro_iteration
    # sort exploitibility according to iteration
    exploitability = [x for _, x in sorted(zip(exploitability[1],exploitability[0]))]

    result_file = os.path.join(checkpoint_dir, 'exp_combined_game_epsiode'+str(max_gpsro_iter)+'.csv')
    df = pd.DataFrame(exploitability)
    df.to_csv(result_file,index=False)
    print("NashConv saved to", result_file) 

    return exploitability


def calculate_iterations_regret_over_meta_game(checkpoint_dirs, combined_game, episode=None):
    """
    Go over all experiments and calculate regret curves for each run
    assumes that combined game include all strategies inside each run
    """
    num_player = len(combined_game)
    regret_curves = []
    current_index = 0
    for di in checkpoint_dirs:
        snum = check_strategies_num_in_combined_game(di, episode, True)[0]
        ne_dir = os.path.join(di,"nash_prob")
        regret_curve = calculate_regret_from_combined_game(ne_dir,
                                                           combined_game,
                                                           current_index,
                                                           current_index + snum-1,
                                                           di)
        current_index += snum
        regret_curves.append(regret_curve)

    return regret_curves

def check_strategies_num_in_combined_game(checkpoint_dir,
                                          episode=None,
                                          only_return_strategy_length=False):
    """
    From meta game length, decide gpsro_iteration. Then decide which ne.pkl to use,
    and eventually decide how many strategies of subgame are in the meta_game. 
    Params:
        episode      : number of iterations in meta_game
    Return:
        strats_num   : number of strategies. A list of len num_player
        strats_num_ne: number of ne strategies
        strats_ne    : list of list, index of ne support strategies for each player
    """
    meta_game = load_pkl(os.path.join(checkpoint_dir, "meta_game.pkl"))
    gpsro_it = meta_game[0].shape[0]-1 # -1 because of the initial policy
    gpsro_it = episode if episode and episode<gpsro_it else gpsro_it
    strats_num = np.ones(len(meta_game),dtype=int)*(gpsro_it+1)
    if only_return_strategy_length:
        return strats_num

    ne_folder = os.path.join(checkpoint_dir, "nash_prob")
    ne = load_pkl(os.path.join(ne_folder,str(gpsro_it)+".pkl"))
    strats_ne = [[i for i in range(len(ele)) if ele[i] > ZERO_THRESHOLD] for ele in ne]
    strats_num_ne = np.array([len(ele) for ele in strats_ne],dtype=int)

    return strats_num, strats_num_ne, strats_ne, ne
    

def calculate_final_ne_performance_over_meta_game(checkpoint_dirs,
                                                  combined_game,
                                                  episode=None,
                                                  combined_game_only_contains_ne=True):
    """
    Evaluate regret of nash equilibrium in each experiment file, against combined game
    Tricky part is combined game could contain only ne/ all strategies
    Params:
        checkpoint_dirs : a list of checkpoint dir, one for each experiment
        combined_game   : list of list, each one for one player
        episode         : gpsro_iteration to stop
        combined_game_only_contains_ne: combined game only contains nash equilibrium strategies of experiment runs or not
    """
    num_player = len(combined_game)
    current_index = np.zeros(num_player,dtype=int)
    exploitabilities = []
    for di in checkpoint_dirs:
        snum, snum_ne, stra_ne, ne = check_strategies_num_in_combined_game(di,episode=episode)
        if combined_game_only_contains_ne:
            ne = [np.array([x for x in ele if x > ZERO_THRESHOLD]) for ele in ne]
            subgame_index = current_index + snum_ne - 1
        else:
            subgame_index = current_index + snum - 1

        #try:
        exploitabilities.append(regret(combined_game,
                                       subgame_index,
                                       subgame_ne=ne,
                                       start_index=current_index))
        current_index += snum_ne if combined_game_only_contains_ne else snum
        print('finished calculating for dir',di)

    return np.array(exploitabilities).sum(axis=1)

def break_into_subcombine_games(directory, base_slurm, num_evaluation_episodes=150, sims_per_entry=1000):
  """
  Console to create job to break down combined game into several smaller combined games. It generates slurm scripts for sbatch to run. Note the usage of symlink here because we do not want to copy multiple copies of runs 

  Params:
    directory               : root directory where all example runs are stored.
    base_slurm              : the base slurm file to append command to
    num_evaluation_episodes : number of iterations in a run for combined game
    sims_per_entry          : number of simulation for one strategy profile
  """
  runs = []
  # rename folder
  for run in os.listdir(directory):
    if os.path.isdir(os.path.join(directory,run)) and 'heur' in run:
      if run[0].isnumeric():
        runs.append(run)
      else:
        folder = os.path.join(directory,str(len(runs))+'_'+run)
        runs.append(str(len(runs))+'_'+run)
        os.rename(os.path.join(directory,run),folder)
  
  index = [int(x.split('_')[0]) for x in runs]
  runs = [x for _,x in sorted(zip(index,runs))]
        
  # create experiment folder
  folder = os.path.join(directory,'combined_game'+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
  os.mkdir(folder)
  # create symbolic link folder combinatorially
  for index in list(itertools.combinations(range(len(runs)),2)):
    pair_folder = os.path.join(folder,'runs_'+str(index[0])+'_'+str(index[1]))
    os.mkdir(pair_folder)
    ln_runs = [os.path.join(pair_folder,runs[index[0]]),os.path.join(pair_folder,runs[index[1]])]
    os.symlink(os.path.join(directory,runs[index[0]]),ln_runs[0])
    os.symlink(os.path.join(directory,runs[index[1]]),ln_runs[1])
    # create job scripts
    commands = "python /home/qmaai/se/open_spiel/open_spiel/python/algorithms/psro_v2/combined_game.py --evaluate_nash=False --sims_per_entry="+str(sims_per_entry)+" --num_evaluation_episodes="+str(num_evaluation_episodes)+" --root_result_folder="+pair_folder
    target = pair_folder+'.sh'
    shutil.copyfile(base_slurm, target)
    with open(target,'a') as file:
      file.write("#SBATCH --output="+pair_folder+'.out'+'\n')
      file.write("module load python3.6-anaconda/5.2.0\n")
      file.write("cd ${SLURM_SUBMIT_DIR}\n")
      file.write(commands)

def gather_subgame_matrix_into_combined_game_matrix(directory, episode, num_run):
  """
  combine sub combined games with restricted indexes
  assume that the subgame is formed by two players and square matrix
  """
  total_size = episode * num_run
  combined_game = [np.ones([total_size,total_size])*np.nan for _ in range(2)]
  for run in os.listdir(directory):
    run_dir = os.path.join(directory,run)
    if os.path.isdir(run_dir) and 'runs_' in run:
        ind1,ind2 = int(run.split('_')[1]),int(run.split('_')[2])
        if ind1>=num_run or ind2>=num_run:
          continue
        matrix_file = [x for x in os.listdir(run_dir) if 'pkl' in x]
        meta_game = load_pkl(run_dir+'/'+matrix_file[0])
        p_stra_len = int(meta_game[0].shape[0]/2)   # assume that subgame has 2 parts
        print()
        print("filling out run",ind1,ind2)
        for p in range(2):
          combined_game[p][ind1*episode:(1+ind1)*episode,\
              ind1*episode:(1+ind1)*episode] = meta_game[p][0:episode,0:episode]
          combined_game[p][ind1*episode:(1+ind1)*episode,\
              ind2*episode:(1+ind2)*episode] = meta_game[p][0:episode,p_stra_len:episode+p_stra_len]
          combined_game[p][ind2*episode:(1+ind2)*episode,\
              ind1*episode:(1+ind1)*episode] = meta_game[p][p_stra_len:p_stra_len+episode,0:episode]
          combined_game[p][ind2*episode:(1+ind2)*episode,\
              ind2*episode:(1+ind2)*episode] = meta_game[p][p_stra_len:p_stra_len+episode,p_stra_len:p_stra_len+episode]
  assert not np.any(np.isnan(combined_game[0])), 'filling is run'
  save_path = os.path.join(directory,'combined_game_episode'+str(episode)+'.pkl')
  save_pkl(combined_game,save_path)
  print("combined game saved at {}".format(save_path))


#def gather_subgame_matrix_into_combined_game_matrix(directory, episode, num_run):
#  """
#  combine sub combined games into one large combined game matrix
#  Params:
#    directory     : root experiments folder, like "combined_gamedatetime"
#    episode       : number of episodes for each player
#    num_run       : number of runs to combine
#  Note that in fact both episode and num_run could be automatically filterd out.
#  I am just sloppy here and specify it by hand
#  """
#  # for checking whether combined game combined maintains the same structure
#  # as the originla meta_game in each run
#  #cg0=load_pkl("/home/qmaai/se/open_spiel/open_spiel/python/algorithms/psro_v2/slurm_scripts/combined_game/0_2_leduc_poker2_sims_1000_it_250_ep_10000_or_DQN_heur_uniform_hl_256_bs_32_nhl_4_dqnlr_0.01_tnuf_500_lf_10_se_43912_2020-07-02_18-22-38/meta_game.pkl")
#  #cg1=load_pkl("/home/qmaai/se/open_spiel/open_spiel/python/algorithms/psro_v2/slurm_scripts/combined_game/1_0_leduc_poker2_sims_1000_it_250_ep_10000_or_DQN_heur_uniform_hl_256_bs_32_nhl_4_dqnlr_0.01_tnuf_500_lf_10_se_21939_2020-07-02_18-29-02/meta_game.pkl")
#  #cg2=load_pkl("/home/qmaai/se/open_spiel/open_spiel/python/algorithms/psro_v2/slurm_scripts/combined_game/2_1_leduc_poker2_sims_1000_it_250_ep_10000_or_DQN_heur_nash_hl_256_bs_32_nhl_4_dqnlr_0.01_tnuf_500_lf_10_se_36619_2020-07-02_18-22-39/meta_game.pkl")
#  total_size = episode * num_run
#  combined_game = [np.ones([total_size,total_size])*np.nan for _ in range(2)]
#  for run in os.listdir(directory):
#    run_dir = os.path.join(directory,run)
#    if os.path.isdir(run_dir) and 'runs_' in run:
#        ind1,ind2 = int(run.split('_')[1]),int(run.split('_')[2])
#        if ind1>=num_run or ind2>=num_run:
#          continue
#        matrix_file = [x for x in os.listdir(run_dir) if 'pkl' in x]
#        meta_game = load_pkl(run_dir+'/'+matrix_file[0])
#        print()
#        print("filling out run",ind1,ind2)
#        print("{}:{},{}:{}--{}:{},{}:{}".format(ind1*episode,(1+ind1)*episode,ind1*episode,(1+ind1)*episode,0,episode,0,episode))
#        print("{}:{},{}:{}--{}:{},{}:{}".format(ind1*episode,(1+ind1)*episode,ind2*episode,(1+ind2)*episode,0,episode,episode,meta_game[0].shape[0]))
#        print("{}:{},{}:{}--{}:{},{}:{}".format(ind2*episode,(1+ind2)*episode,ind1*episode,(1+ind1)*episode,episode,meta_game[0].shape[0],0,episode))
#        print("{}:{},{}:{}--{}:{},{}:{}".format(ind2*episode,(1+ind2)*episode,ind2*episode,(1+ind2)*episode,episode,meta_game[0].shape[0],episode,meta_game[0].shape[0]))
#        for p in range(2):
#          combined_game[p][ind1*episode:(1+ind1)*episode,\
#              ind1*episode:(1+ind1)*episode] = meta_game[p][0:episode,0:episode]
#          combined_game[p][ind1*episode:(1+ind1)*episode,\
#              ind2*episode:(1+ind2)*episode] = meta_game[p][0:episode,episode:]
#          combined_game[p][ind2*episode:(1+ind2)*episode,\
#              ind1*episode:(1+ind1)*episode] = meta_game[p][episode:,0:episode]
#          combined_game[p][ind2*episode:(1+ind2)*episode,\
#              ind2*episode:(1+ind2)*episode] = meta_game[p][episode:,episode:]
#  assert not np.any(np.isnan(combined_game[0])), 'filling is run'
#  save_path = os.path.join(directory,'combined_game_episode'+str(episode)+'.pkl')
#  save_pkl(combined_game,save_path)
#  print("combined game saved at {}".format(save_path))

def main(argv):
    if len(argv) > 1:
      raise app.UsageError("Too many command-line arguments.")
    
    assert os.path.isdir(FLAGS.root_result_folder), "no such directory"

    run_names = [x for x in os.listdir(FLAGS.root_result_folder) if \
        os.path.isdir(os.path.join(FLAGS.root_result_folder, x)) and \
        'combined_game' not in x] #erase the calculate combined_game folder
    checkpoint_dirs = [os.path.join(FLAGS.root_result_folder, x) for x in run_names]
    # checkpoint_dirs might be ordered, try to order if first char is int
    if run_names[0][0].isnumeric():
      index = [int(x[0]) for x in run_names]
      checkpoint_dirs = [x for _,x in sorted(zip(index,checkpoint_dirs))]
    
    seed = FLAGS.seed if FLAGS.seed else np.random.randint(low=0, high=1e5)
    set_seed(seed)
    
    num_strategies = FLAGS.num_evaluation_episodes + 1
    # automatically append order integer in checkpoint directories
    if FLAGS.break_into_subcombine_game:
        break_into_subcombine_games(FLAGS.root_result_folder, FLAGS.base_slurm, FLAGS.num_evaluation_episodes, FLAGS.sims_per_entry)
        return

    if FLAGS.gather_subgame:
        gather_subgame_matrix_into_combined_game_matrix(FLAGS.root_result_folder,
            FLAGS.num_evaluation_episodes, FLAGS.num_run)
        return

    if FLAGS.combined_game_path == "":
        dir_content = os.listdir(FLAGS.root_result_folder)
        assert len(dir_content)>1, "directory provided cannot make combined game"
             
        # calculate combined game
        combined_game,strat_num = calculate_combined_game(checkpoint_dirs,
                                                FLAGS.root_result_folder,
                                                episode=num_strategies,
                                                only_combine_nash=FLAGS.only_combine_nash,
                                                sims_per_entry=FLAGS.sims_per_entry,
                                                seed=seed)
        combined_game_only_contains_ne = FLAGS.only_combine_nash
    else: # combined game supplied, no need to calculate again
        # combined game loaded in order of indexes
        combined_game = load_pkl(FLAGS.combined_game_path)
        combined_game_only_contains_ne = '_neonly_True' in FLAGS.combined_game_path
        print("loaded combined game from", FLAGS.combined_game_path)

    if FLAGS.evaluate_nash:
        
        exp = calculate_final_ne_performance_over_meta_game(checkpoint_dirs,\
            combined_game,\
            FLAGS.num_evaluation_episodes,\
            combined_game_only_contains_ne=combined_game_only_contains_ne)
        print("nash exp ne only:", exp) 
        path = os.path.join(FLAGS.root_result_folder, "nash_exp_neonly_"+str(combined_game_only_contains_ne)+'_episode_'+str(FLAGS.num_evaluation_episodes+1)+".pkl")
        save_pkl(exp, path)
        print("final nash exp performance saved to",path)
    
    if FLAGS.evaluate_iters:
        assert not combined_game_only_contains_ne,\
            "combined game doesn't contain all strategies"
        calculate_iterations_regret_over_meta_game(checkpoint_dirs,
                                                   combined_game,
                                                   FLAGS.num_evaluation_episodes)
    
if __name__ == "__main__":
    app.run(main)
