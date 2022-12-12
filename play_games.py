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


n_pix = 4
eps = 0.0

model_name = 'ref_conv_best'
replay_folder = 'conv4'
_, my_model = tools.load_ref(name=model_name)

n_games = 10000
for i in range(n_games):
    my_game = game.Game(n_pix, v_conv, v_conv, randomize=True, rand_frac=0.8, random_start=True)
    try:
        my_game.mcts_vs_mcts(my_model, my_model, eps=eps, n_sim=150, replay_folder=replay_folder)
    except RecursionError:
        print('Encountered a recursion error')
        continue
    replay = my_game.replay
    victory = my_game.state.victory

    n_replay = len(replay) - 1

    print('length of game: ', n_replay + 1, ' score: ', victory)