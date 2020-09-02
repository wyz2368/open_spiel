"""
This document is specifically writen for 2 player Kuhn poker extracting the 64 strategies into a meta game. When modifying this document, consider:
    1. change player number in construct_meta_game and simulate_strategy
    2. The only turns that Nature act is when dealing private hands to players. When changing into other games where player reveal the three cards in other hold'em games, simulate_strategy has to be changed
    game = pyspiel.load_game("leduc_poker") #468 info state per player. too many
"""

from open_spiel.python.algorithms import generate_playthrough
from open_spiel.python.algorithms import get_all_states
import pyspiel
import itertools
import functools
import math
import numpy as np
import pickle as pkl


def summarize_infostates(game, num_player=2, num_actions=2):
    num_player = num_player
    info_states = [[] for _ in range(num_player)]
    init_states = []  # all possible states after chance nodes assign
    states = get_all_states.get_all_states(game, depth_limit=-1, include_terminals=True, include_chance_states=False,
                                           to_string=lambda s: s.history_str())
    # extract all information state
    for his, state in states.items():
        if not state.is_player_node():
            continue
        cur_p = state.current_player()
        info_states[cur_p].append(state.information_state_string())

        if len(his.split(' ')) == num_player:
            init_states.append(state)

    info_states = [list(set(ele)) for ele in info_states]
    print('info states for players', info_states)

    # Generate Strategies for info states, mapping from states into actions
    num_actions = num_actions
    strategies = [list(range(0, math.floor(math.pow(num_actions, len(ele))))) for ele in info_states]
    return info_states, strategies, init_states


def strategy_at_infoset(infoset, strategy, infostate, player):
    """
    Return what strategy play at this infoset
    Input:
        infoset    :   all possible information_state_string for this game, for both player
        strategy   :   index of strategy
        infostate  :   a OpenSpiel game State's information_state_string
        player     :   1 or 2, the acting player
    """
    # integer to binary
    s = "{0:06b}".format(strategy)
    index = infoset[player].index(infostate)
    return int(s[index])


def construct_meta_game(game, strategies, init_states, strategy_at_infostate):
    """
    Only works for two players
    game         :   a pyspiel game
    strategies   :   a list of list. each one a list ranging from 0 to max_strategy
    init_states  :   after chance nodes deal private cards, all possible start states
    strategy_at_infostate  : a function. input strategy, infostate, output action
    """
    num_players = len(strategies)
    num_strategies = [len(ele) for ele in strategies]
    meta_game = [np.zeros(num_strategies) * np.nan for _ in range(num_players)]
    for s1, s2 in list(itertools.product(*strategies)):
        expected_payoff = simulate_strategy(game, [s1, s2], init_states, strategy_at_infostate)
        meta_game[0][s1, s2] = expected_payoff[0]
        meta_game[1][s1, s2] = expected_payoff[1]
    return meta_game


# simulate payoff matrix for strategies
def simulate_strategy(game, strategies, init_states, func):
    """
    Harvest an accurate average payoff with the two strategies, considering traversing
    all possible private cards dealt by nature
    game        :       the pyspiel game
    strategies  :       the index of player's strategy, a length two list
    init_states :       a set of possible openspiel state after chance deals private cards
    func        :       given a strategy index and an infostate, output action
    """
    payoff = np.array([0, 0], dtype=int)
    for root in init_states:  # traverse game tree
        node = root
        while not node.is_terminal():
            assert not node.is_chance_node(), "Doesn't exist chance nodes in kuhn's poker after private hands are dealt"
            player = node.current_player()
            action = func(strategies[player], node.information_state_string(), player)
            assert action in node.legal_actions(), "action not legal!"
            node = node.child(action)
        payoff = payoff + node.returns()

    return payoff / len(init_states)


def expand_efg_to_nfg(game_name, num_player, num_actions, result_path):
    game = pyspiel.load_game(game_name)
    info_states, strategies, init_states = summarize_infostates(game, num_player, num_actions)
    eva_strategy_func = functools.partial(strategy_at_infoset, info_states)
    meta_game = construct_meta_game(game, strategies, init_states, eva_strategy_func)
    with open(result_path, 'wb') as f:
        pkl.dump([meta_game, info_states], f)
    print('saved results to', result_path)


if __name__ == "__main__":
    expand_efg_to_nfg(game_name='kuhn_poker',
                      num_player=2,
                      num_actions=2,
                      result_path="kuhn_meta_game.pkl")