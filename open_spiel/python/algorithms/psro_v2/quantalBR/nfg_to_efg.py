import numpy as np
from open_spiel.python.algorithms.nash_solver import subproc
import os
import pickle
import itertools
import logging

"""
This script connects meta-games with gambit. It translates a meta-game to am EFG format 
that gambit-logit could recognize to find the QRE.
Gambit file format: http://www.gambit-project.org/gambit16/16.0.0/formats.html 
"""

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

# This functions help to translate meta_games into gambit nfg.
def product(shape, axes):
    prod_trans = tuple(zip(*itertools.product(*(range(shape[axis]) for axis in axes))))

    prod_trans_ordered = [None] * len(axes)
    for i, axis in enumerate(axes):
        prod_trans_ordered[axis] = prod_trans[i]
    return zip(*prod_trans_ordered)

def encode_gambit_file_qre(meta_games, checkpoint_dir=None):
    """
    Encode a meta-game to nfg file that gambit can recognize.
    :param meta_games: A meta-game (payoff tensor) in PSRO.
    """
    num_players = len(meta_games)
    num_strategies = np.shape(meta_games[0])
    if checkpoint_dir is None:
        gambit_DIR = os.path.dirname(os.path.realpath(__file__)) + '/efg'
    else:
        gambit_DIR = checkpoint_dir + '/efg'
    gambit_NFG = gambit_DIR + '/payoffmatrix.efg'

    # Write header
    with open(gambit_NFG, "w") as nfgFile:
        nfgFile.write('EFG 2 R "Empirical Game" ')
        name_players = '{ "p1"'
        for i in range(2, num_players+1):
            name_players += " " + "\"" + 'p' + str(i) + "\""
        name_players += ' }'
        nfgFile.write(name_players)
        # Write strategies
        nfgFile.write("\n")
        nfgFile.write("\"\"" + '\n\n')

        # Write outcomes
        first_player_str = "p \"\" 1 1 \"(1, 1)\"" + " { "
        for i in range(num_strategies[0]):
            first_player_str += "\"" + str(i+1) + "\" "
        first_player_str += "} 0"
        nfgFile.write(first_player_str)
        nfgFile.write("\n")

        outcome_idx = 1
        action_name = " { "
        for k in range(num_strategies[1]):
            action_name += "\"" + str(k+1) + "\" "
        action_name += "} "

        for i in range(num_strategies[0]):
            second_player_str = "p \"\" 2 2 "
            added_player2 = False
            for j in range(num_strategies[1]):
                if not added_player2:
                    second_player_str += "\"" + str((2, i)) + "\"" + action_name + " 0 "
                    nfgFile.write(second_player_str)
                    nfgFile.write("\n")
                    added_player2 = True
                terminal_str = "t \"\" " + str(outcome_idx) + " \"Outcome " + str(outcome_idx) + "\" { " + str(meta_games[0][i,j]) + ", " + str(meta_games[1][i,j]) + " }"
                nfgFile.write(terminal_str)
                outcome_idx += 1
                nfgFile.write("\n")

def gambit_analysis_qre(timeout, checkpoint_dir=None):
    """
    Call a subprocess and run gambit to find all NE.
    :param timeout: Maximum time for the subprocess.
    :param method: The gamebit command line method.
    """
    if checkpoint_dir is None:
        gambit_DIR = os.path.dirname(os.path.realpath(__file__)) + '/nfg'
    else:
        gambit_DIR = checkpoint_dir + '/nfg'
    gambit_EFG = gambit_DIR + '/payoffmatrix.efg'

    if not isExist(gambit_EFG):
        raise ValueError(".efg file does not exist!")
    command_str = "gambit-logit" + " -q " + gambit_EFG + " -d 8 > " + gambit_DIR + "/qre.txt"
    subproc.call_and_wait_with_timeout(command_str, timeout)

def decode_gambit_file_qre(meta_games, mode="all", checkpoint_dir=None):
    """
    Decode the results returned from gambit to a numpy format used for PSRO.
    :param meta_games: A meta-game in PSRO.
    :param mode: "all", "pure", "one" options
    :param max_num_nash: the number of NE considered to return
    :return: a list of NE
    """
    if checkpoint_dir is None:
        gambit_DIR = os.path.dirname(os.path.realpath(__file__)) + '/efg'
    else:
        gambit_DIR = checkpoint_dir + '/efg'

    nash_DIR = gambit_DIR + '/qre.txt'
    if not isExist(nash_DIR):
        raise ValueError("qre.txt file does not exist!")
    num_lines = file_len(nash_DIR)

    logging.info("Number of QRE (lambda) is ", num_lines)

    shape = np.shape(meta_games[0])
    slice_idx = []
    pos = 0
    for i in range(len(shape)):
        slice_idx.append(range(pos, pos + shape[i]))
        pos += shape[i]

    equilibria = []
    with open(nash_DIR,'r') as f:
        for _ in np.arange(num_lines):
            equilibrim = []
            nash = f.readline()
            if len(nash.strip()) == 0:
                continue
            nash = nash[3:]
            nash = nash.split(',')
            new_nash = []
            for j in range(len(nash)):
                new_nash.append(convert(nash[j]))

            new_nash = np.array(new_nash)
            new_nash = np.round(new_nash, decimals=8)
            for idx in slice_idx:
                equilibrim.append(new_nash[idx])
            equilibria.append(equilibrim)

    if mode == "all" or mode == "pure":
        return equilibria
    elif mode == "one":
        return equilibria[0]
    else:
        logging.info("mode is beyond all/pure/one.")


def do_gambit_analysis_qre(meta_games, mode, timeout = 600, checkpoint_dir=None):
    """
    Combine encoder and decoder.
    :param meta_games: meta-games in PSRO.
    :param mode: "all", "pure", "one" options
    :param timeout: Maximum time for the subprocess
    :param method: The gamebit command line method.
    :param method_pure_ne: The gamebit command line method for finding pure NE.
    :return: a list of NE.
    """
    if checkpoint_dir is None:
        gambit_DIR = os.path.dirname(os.path.realpath(__file__)) + '/efg'
    else:
        gambit_DIR = checkpoint_dir + '/efg'
    if not isExist(gambit_DIR) and not checkpoint_dir is None:
        mkdir(gambit_DIR)

    if np.shape(meta_games[0]) == (1,1):
        return [np.array([1.]), np.array([1.])]

    encode_gambit_file_qre(meta_games, checkpoint_dir)
    while True:
        gambit_analysis_qre(timeout, checkpoint_dir)
        # If there is no pure NE, find mixed NE.
        nash_DIR = gambit_DIR + '/qre.txt'
        if not isExist(nash_DIR):
            raise ValueError("qre.txt file does not exist!")

        equilibria = decode_gambit_file_qre(meta_games, mode, checkpoint_dir=checkpoint_dir)
        if len(equilibria) != 0:
            break
        timeout += 120
        if timeout > 7200:
            logging.warning("Gambit has been running for more than 2 hour.!")
        logging.warning("Timeout has been added by 120s.")
    logging.info('gambit_analysis done!')
    return equilibria


def convert(s):
    """
    Convert a probability string to a float number.
    :param s: probability string.
    :return: a float probability.
    """
    try:
        return float(s)
    except ValueError:
        num, denom = s.split('/')
        return float(num) / float(denom)


def file_len(fname):
    """
    Get the number of lines in a text file. This is used to count the number of NE found by gamebit.
    :param fname: file name.
    :return: number of lines.
    """
    num_lines = sum(1 for line in open(fname))
    return num_lines



