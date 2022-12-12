import torch
# from torch import nn
import numpy as np
import copy
import os
import pickle
from time import sleep
import game.state as state
import game.game as game
import mcts as my_mcts
from scipy.special import softmax
import model
import tools


n_pix = 4

n_feat = n_feat = 2 * n_pix ** 2 + 9
# elo, my_model = load_ref(name='ref_conv_2_alldata_lr015_100epoch')
# mod = [elo, my_model, v_conv]
# view_replay(220, mod, folder='comparisons')

my_game = game.Game(n_pix, tools.v_conv, tools.v_conv, randomize=False, rand_frac=0.5)

# my_model = model.GameModel(n_feat, n_action=4 * state.n_pix ** 2)
# elo, my_model = load_ref(name='ref_replays_all_clean')
elo, my_model = tools.load_ref(name='ref_conv_best')
mod = [elo, my_model, tools.v_conv]
# my_model = model.ResModel(state.n_pix, n_action=4 * state.n_pix ** 2)


# tools.view_replay(250, mod, folder='conv')

my_game.vs_mcts(num_sim=150, net=my_model)
