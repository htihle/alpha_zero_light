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

def conv_features(state):
    my_map = state.return_maps()

    x_map = np.zeros((7, n_pix, n_pix))
    x_map[0] = my_map[0][0]
    x_map[1] = my_map[0][1]
    x_map[2] = my_map[0][3]
    x_map[3] = my_map[1][0]
    x_map[4] = my_map[1][1]
    x_map[5] = my_map[1][3]
    x_map[6] = np.ones((n_pix, n_pix)) * state.to_play
    return x_map


def v_conv(state, net):
    x = conv_features(state)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    value, probs = net(x)
    if state.done:
        value = state.victory
    return value, probs, x

def compare_models(mod1, mod2, n_exp=100, ref=False, n_sim=5, temp=0.5, replay_folder=None):
    if replay_folder is not None:
        save_game = True
    else:
        save_game = False
    elo1, model1, v1 = mod1
    elo2, model2, v2 = mod2
    vic = np.zeros(n_exp)
    K = 10
    eps = 0.1
    for i in range(n_exp):
        print('Playing game number: ', i + 1, ' of ', n_exp)
        if i < n_exp // 2:
            gm = game.Game(n_pix, v1, v2, randomize=True, rand_frac=0.2)
            gm.mcts_vs_mcts(model1, model2, eps=eps, n_sim=n_sim, save_game=save_game, replay_folder=replay_folder)
            vic[i] = (gm.state.victory + 1) * 0.5
        else:
            gm = game.Game(n_pix, v2, v1, randomize=True, rand_frac=0.2)
            gm.mcts_vs_mcts(model2, model1, eps=eps, n_sim=n_sim, save_game=save_game, replay_folder=replay_folder)
            vic[i] = 1 - (gm.state.victory + 1) * 0.5
        print('Mean score: ', np.mean(vic[:i+1]))
        exp1, exp2 = get_exp(elo1, elo2)
        
        elo1 = elo1 + K * (vic[i] - exp1)
        if not ref:
            elo2 = elo2 + K * ((1 - vic[i]) - exp2)
    print('Mean score: ', np.mean(vic))
    return elo1, elo2, np.mean(vic)


def get_exp(e1, e2):
    q1 = 10 ** (e1 / 400)
    q2 = 10 ** (e2 / 400)
    qsum = q1 + q2
    e1 = q1 / qsum
    e2 = q2 / qsum
    return e1, e2 

n_pix = 4
def load_ref(name='ref'):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

# # Check to see if we have a GPU to use for training
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print('A {} device was detected.'.format(device))
device = 'cpu'
# Print the name of the cuda device, if detected
if device=='cuda':
  print (torch.cuda.get_device_name(device=device))

mod = load_ref('ref_conv_15_alldata_lr02_512_100epoch_085')

mod = [mod[0], mod[1], v_conv]

def load_ref(name='ref'):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

# mod2 = load_ref('ref_freak')
# mod2 = [mod2[0], mod2[1], v3]
mod2 = load_ref('ref_conv_15_alldata_lr015_512_100epoch')
mod2 = [mod2[0], mod2[1], v_conv]
elo_new, _, mean_score = compare_models(mod, mod2, n_exp=200, ref=True, n_sim=5, replay_folder='comparisons')
