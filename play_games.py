import torch
from torch import nn
import numpy as np
import copy
import os
import pickle
from time import sleep
import game.state as state
import game.game as game
import model

import sys
sys.setrecursionlimit(10000)

def get_feat_nn(state):
    my_map = state.return_maps()
    summap0 = my_map[0][1] 
    summap0 = summap0.flatten()
    summap1 = my_map[1][1].flatten()
    n_feat = 2 * len(summap0) + 9
    x = np.zeros(n_feat)
    x[0] = np.min([np.sum(my_map[0][0].flatten()), 10]) / 10
    x[1] = np.min([np.sum(my_map[1][0].flatten()), 10]) / 10
    x[2] = np.sum(my_map[0][1].flatten()) / 10
    x[3] = np.sum(my_map[1][1].flatten()) / 10
    x[4] = np.sum(my_map[0][3].flatten()) / 10
    x[5] = np.sum(my_map[1][3].flatten()) / 10
    distancesum = 0
    otherhome = state.players[1][0][0].position
    for i, unit in enumerate(state.players[0][1]):
        d = (np.abs(unit.position[0] - otherhome[0]) + np.abs(unit.position[1] - otherhome[1]))
        if d == 0:
            distancesum = 1e3
        else:
            distancesum += unit.size / d
    x[6] = distancesum / 10
    distancesum = 0
    otherhome = state.players[0][0][0].position
    for i, unit in enumerate(state.players[1][1]):
        d = (np.abs(unit.position[0] - otherhome[0]) + np.abs(unit.position[1] - otherhome[1]))
        if d == 0:
            distancesum = 1e3
        else:
            distancesum += unit.size / d
    x[7] = distancesum / 10
    x[8] = state.to_play
    x[9:9 + len(summap0)] = summap0 / 5
    x[9 + len(summap1):] = summap1 / 5
    return x

def v3(state, net):
    x = get_feat_nn(state)
    x = torch.tensor(x, dtype=torch.float32)
    value, probs = net(x)
    if state.done:
        value = state.victory
    return value, probs, x

def q(state, action):
    newstate = copy.deepcopy(state)
    newstate.step(action)
    return v(newstate)

def load_ref(name='ref'):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

n_pix = 4
eps = 0.0

_, my_model = load_ref(name='ref_best')

n_games = 10000
for i in range(n_games):
    my_game = game.Game(n_pix, q, v3, randomize=True, rand_frac=0.8, random_start=True)
    try:
        my_game.mcts_vs_mcts(my_model, my_model, eps=eps, n_sim=150, replay_folder='replays_uni2')
    except RecursionError:
        print('Encountered a recursion error')
        continue
    # replay, victory, _ = load_game(gameid)
    replay = my_game.replay
    victory = my_game.state.victory
    # print(my_game.state.victory)
    # print(my_game.turn)
    # print(len(replay))
    n_replay = len(replay) - 1
    # victory = (victory + 1) * 0.5
    print('length of game: ', n_replay + 1, ' score: ', victory)