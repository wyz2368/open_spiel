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
"""

import pyspiel
from open_spiel.python import rl_environment
from open_spiel.python.algorithms.psro_v2.utils import set_seed
from open_spiel.python.algorithms.psro_v2.eval_utils import *

import os
from absl import app
from absl import flags
FLAGS = flags.FLAGS

# calculate combined game only with ne strategies
flags.DEFINE_bool("only_combine_nash", False, "only combine nash in combined game")
flags.DEFINE_bool("evaluate_nash",True, "evaluate regret of final nash for all runs against meta_game")
flags.DEFINE_bool("evaluate_iters",False, "evaluate regret curve for all runs against meta_game")

flags.DEFINE_integer("seed", None, "seed")
flags.DEFINE_string("root_result_folder","","root result folder that contains the experiments runs to be calculated into combined game")
flags.DEFINE_string("combine_game_path","","provide combined game pkl if there already is an existing comnbined game, in this case, only need to calculate exploitabilities of each run")
flags.DEFINE_integer("sims_per_entry", 1000,
                     ("Number of simulations to run to estimate each element"
                      "of the game outcome matrix."))

ZERO_THRESHOLD = 0.001

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
                none_zero_index = [[i for i in range(len(ele)) if ele[i] > ZERO_THRESHOLD] \
                    for ele in nash_ne]
                meta_games[i] = [ele[np.ix_(*none_zero_index)] for ele in meta_games[i]]

            for weight_file in range(1,num_strategy+1): # traverse the weight file
                if only_combine_nash and nash_ne[p][weight_file-1] < ZERO_THRESHOLD:
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
    
    save_path =  os.path.join(combined_game_save_path,"combined_game_se_"+str(seed)+'_neonly_'+str(only_combine_nash)+".pkl")
    save_pkl(combined_game, save_path)
    print("combined game {} only combine NE saved at {}".format(only_combine_nash,save_path))

    return combined_game, strategy_num


def calculate_regret_from_combined_game(ne_dir,
                                        combined_game
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
    max_gpsro_iter = strat_end - strat_start - 1
    for fi in os.listdir(ne_dir):
        gpsro_iter = int(fi.split('.pkl')[0])
        if gpsro_iter > gpsro_iter:
            continue        
        subgame_index = gpsro_iter + 1 + strat_start
        eq = load_pkl(os.path.join(ne_dir, fi))
        # calculate regret
        regrets = regret(combined_game, subgame_index, subgame_ne=eq, start_index=strat_start)
        exploitability[0].append(sum(regrets))
        exploitability[1].append(subgame_index)
    
    # exploitability[1] carries gpsro_iteration
    # sort exploitibility according to iteration
    exploitability = [x for _, x in sorted(zip(exploitability[1],exploitability[0]))]

    result_file = os.path.join(checkpoint_dir, 'exp.csv')
    df = pd.DataFrame(exploitability)
    df.to_csv(result_file,index=False)
    print("NashConv saved to", result_file) 

    return exploitability


def calculate_iterations_regret_over_meta_game(checkpoint_dirs, combined_game):
    """
    Go over all experiments and calculate regret curves for each run
    assumes that combined game include all strategies inside each run
    """
    num_player = len(combined_game)
    regret_curves = []
    current_index = 0
    for di in checkpoint_dirs:
        snum = check_strategies_num_in_combined_game(di, True)
        ne_dir = os.path.join(di,"nash_prob")
        regret_curve = calculate_regret_from_combined_game(ne_dir,
                                                           combined_game,
                                                           current_index,
                                                           current_index + snum,
                                                           di)
        current_index += snum
        regret_curves.append(regret_curve)

    return regret_curves

def check_strategies_num_in_combined_game(checkpoint_dir,
                                          only_return_strategy_length=False):
    """
    From meta game length, decide gpsro_iteration. Then decide which ne.pkl to use,
    and eventually decide how many strategies of subgame are in the meta_game
    Return:
        strats_num   : number of strategies. A list of len num_player
        strats_num_ne: number of ne strategies
        strats_ne    : list of list, index of ne support strategies for each player
    """
    meta_game = load_pkl(os.path.join(checkpoint_dir, "meta_game.pkl"))
    gpsro_it = meta_game[0].shape[0][0]-1 # -1 because of the initial policy
    strats_num = np.ones(len(meta_game))*(gpsro_it+1)
    if only_return_strategy_length:
        return strats_num

    ne_folder = os.path.join(checkpoint_dir, "nash_prob")
    ne = load_pkl(os.path.join(ne_folder,str(gpsro_it)+".pkl"))
    strats_ne = [[i for i in range(len(ele)) if ele[i] > ZERO_THRESHOLD] for ele in ne]
    strats_num_ne = np.array([len(ele) for ele in strats])

    return strats_num, strats_num_ne, strats_ne, ne
    

def calculate_final_ne_performance_over_meta_game(checkpoint_dirs,
                                                  combined_game,
                                                  combined_game_only_contains_ne=True):
    """
    Evaluate regret of nash equilibrium in each experiment file, against combined game
    Tricky part is combined game could contain only ne/ all strategies
    Params:
        checkpoint_dirs : a list of checkpoint dir, one for each experiment
        combined_game   : list of list, each one for one player
        combined_game_only_contains_ne: combined game only contains nash equilibrium strategies of experiment runs or not
    """
    num_player = len(combined_game)
    current_index = np.zeros(num_player)
    exploitabilities = []
    for di in checkpoint_dirs:
        snum, snum_ne, stra_ne, ne = check_strategies_num_in_combined_game(di)
        if combined_game_only_contains_ne:
            ne = [np.array([x for x in ele if x > ZERO_THRESHOLD]) for ele in ne]
            subgame_index = current_index + snum_ne
        else:
            subgame_index = current_index + snum
        exploitabilities.append(regret(combined_game,
                                       subgame_index,
                                       subgame_ne=ne,
                                       start_index=current_index))
        current_index += snum_ne if combined_game_only_contains_ne else snum

    return np.array(exploitabilities).sum(axis=1)


def main(argv):
    if len(argv) > 1:
      raise app.UsageError("Too many command-line arguments.")
  
    assert os.path.isdir(FLAGS.root_result_folder), "no such directory"

    checkpoint_dirs = [os.path.join(FLAGS.root_result_folder, x) \
        for x in os.listdir(FLAGS.root_result_folder) \
        if os.path.isdir(os.path.join(FLAGS.root_result_folder, x))]
    
        seed = FLAGS.seed if FLAGS.seed else np.random.randint(low=0, high=1e5)
        set_seed(seed)


    # combined game supplied, no need to calculate again
    if FLAGS.combined_game_path == "":
        dir_content = os.listdir(FLAGS.root_result_folder)
        assert len(dir_content)>1, "directory provided cannot make combined game"
             
        # calculate combined game
        combined_game = calculate_combined_game(checkpoint_dirs,
                                                FLAGS.root_result_folder,
                                                only_combine_nash=FLAGS.only_combine_nash,
                                                sims_per_entry=FLAGS.sims_per_entry,
                                                seed=seed)
        combined_game_only_contains_ne = FLAGS.only_combine_nash
    else:
        combined_game = load_pkl(FLAGS.combined_game_path)
        combined_game_only_contains_ne = '_neonly_True' in FLAGS.combined_game_path
    
    if FLAGS.evaluate_nash:
        
        exp = calculate_final_ne_performance_over_meta_game(checkpoint_dirs,\
            combined_game,\
            combined_game_only_contains_ne=combined_game_only_contains_ne)
        
        save_pkl(exp,os.path.join(FLAGS.root_result_folder,\
            "nash_exp_neonly_"+combined_game_only_contains_ne+".pkl"))
    
    if FLAGS.evaluate_iters:
        assert not combined_game_only_contains_ne,\
            "combined game doesn't contain all strategies"
        calculate_iterations_regret_over_meta_game(checkpoint_dirs, combined_game)
    
if __name__ == "__main__":
    app.run(main)
