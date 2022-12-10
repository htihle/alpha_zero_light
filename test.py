import torch
from torch import nn
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

def q(state, action):
    newstate = copy.deepcopy(state)
    newstate.step(action)
    return v(newstate)

def v(state, net):
    x = get_feat_nn(state)
    # print(x.shape)
    if state.to_play == 0:
        value = (x[0] + x[2] + x[4]/12 + x[6] * 5 - x[7] * 5 - x[1] - x[3] - x[5]/12) / 30
        if state.done:
            if state.victory == 0.0:  # draw
                return 0.0, x
            else:
                return state.victory, x
    if state.to_play == 1:
        value = -(x[0] + x[2] + x[4]/12 + x[6] * 5 - x[7] * 5 - x[1] - x[3] - x[5]/12) / 30
        if state.done:
            if state.victory == 0.0:  # draw
                return 0.5, x
            else:
                return -state.victory, x  
    return value, x

def v2(state, net):
    x = get_feat_nn(state)
    # print(x.shape)

    value = (x[0] + x[2] + x[4]/12 + x[6] * 5 - x[7] * 5 - x[1] - x[3] - x[5]/12) / 30
    if state.done:
        if state.victory == 0.0:  # draw
            return 0.0, x
        else:
            return state.victory, x

    return value, x

def v3(state, net):
    x = get_feat_nn(state)
    x = torch.tensor(x, dtype=torch.float32)
    value, probs = net(x)
    if state.done:
        value = state.victory
    return value, probs, x

def load_ref(name='ref'):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

n_pix = 4
n_aux = 1
my_game = game.Game(n_pix, q, v3, randomize=False, rand_frac=0.5)

# my_model = model.GameModel(n_feat, n_action=4 * n_pix ** 2)
elo, my_model = load_ref(name='ref_replays_all_clean')

res_model = model.ResModel(n_pix, n_aux, n_action=4 * n_pix ** 2)

print(len(my_model.parameters))
print(len(res_model.parameters))

# elo, my_model = load_ref(name='ref_replays_all_clean')
# my_game.vs_mcts(num_sim=150, net=my_model)

# my_game.state.step([0, 0, 0, 1])

# my_game.state.step([1, 0, 0, 0])
# my_game.state.step([0, 1, 0, 1])
# my_game.state.step([1, 1, 0, 2])
# my_game.state.step([0, 1, 0, 1])
# my_game.state.step([1, 1, 0, 3])
# my_game.state.step([0, 1, 0, 0])
# my_game.state.step([1, 0, 0, 1])
# my_game.state.step([0, 1, 0, 0])
# my_game.state.step([1, 0, 0, 0])
# my_game.state.step([0, 0, 0, 0])
# my_game.state.visualize()

# mcts = my_mcts.MCTS(valuefunc=v3)
# probs, best_val = mcts.get_mcts_policy(my_game.state, my_model, n_sim=150)
# for i in range(4):
#     print(probs[:, :, i])
####

# action = [0, 0, 0, 0] 
# action2 = [1, 1, 0, 3] 
# my_game.state.step(action)
# my_game.state.step(action2)
# my_game.state.step(action)

# my_game.state.step(action2)

# action2 = [1, 1, 0, 2] 
# my_game.state.step(action)
# my_game.state.step(action2)
# my_game.state.step(action)

#####

# action = [0, 1, 0, 1] 
# action2 = [1, 0, 0, 0] 
# my_game.state.step(action)
# my_game.state.step(action2)
# my_game.state.step(action)

# my_game.state.step(action2)

# action = [0, 1, 0, 0] 
# my_game.state.step(action)
# my_game.state.step(action2)
# my_game.state.step(action)
# my_game.state.step(action2)
# my_game.state.step(action)

# my_game.state.visualize()
# maps = my_game.state.return_maps()

# print(maps)
# my_game.vs_mcts(my_model, num_sim=150)
# x = torch.tensor(get_feat_nn(my_game.state), dtype=torch.float32)
# mask = my_game.state.invalid_action_mask()
# value, priors = my_model(x)
# priors = priors.detach().numpy()
# priors = softmax(priors)
# priors = priors.reshape(n_pix, n_pix, 4)
# value = value.detach().numpy()
# masked = priors * mask
# print(value)
# for i in range(4):
#     print(masked[:, :, i])
# for i in range(4):
#     print(priors[:, :, i])
# mcts = my_mcts.MCTS(valuefunc=v3)
# probs, best_val = mcts.get_mcts_policy(my_game.state, my_model, n_sim=15, temp=1.0)
# print(best_val)
# for i in range(4):
#     print(np.array(probs)[:, :, i])
# net = 1
# mcts = my_mcts.MCTS(valuefunc=v)
# print(my_game.state.to_play)
# probs = mcts.get_mcts_policy(my_game.state, net, n_sim=300)
# print(probs[:, :, 2])


# action = [0, 1, 0, 1] 
# action2 = [1, 0, 0, 0] 
# my_game.state.step(action)
# my_game.state.step(action2)
# my_game.state.step(action)

# my_game.state.step(action2)

# action = [0, 1, 0, 0] 
# my_game.state.step(action)
# my_game.state.step(action2)
# # my_game.state.step(action)
# # my_game.state.step(action2)
# # my_game.state.step(action)
# my_game.state.visualize()
# net = 1
# mcts = my_mcts.MCTS(valuefunc=v)
# print(my_game.state.to_play)
# probs = mcts.get_mcts_policy(my_game.state, net, n_sim=300)
# print(probs[:, :, 0])

# elo, my_model = load_ref(name='ref_best')

# print(v(my_game.state, net))



# def v(state):
#     x = get_feat_nn(state)
#     # print(x.shape)
#     if state.to_play == 0:
#         value = (x[0] + x[2] + x[4]/12 + x[6] * 5 - x[7] * 5 - x[1] - x[3] - x[5]/12) / 30
#         if state.done:
#             if state.victory == 0.0:  # draw
#                 return 0.0, x
#             else:
#                 return state.victory, x
#     if state.to_play == 1:
#         value = -(x[0] + x[2] + x[4]/12 + x[6] * 5 - x[7] * 5 - x[1] - x[3] - x[5]/12) / 30
#         if state.done:
#             if state.victory == 0.0:  # draw
#                 return 0.5, x
#             else:
#                 return -state.victory, x  
#     return value, x

# n_pix = 4

# my_game = game.Game(n_pix, q, v, randomize=True, rand_frac=0.9)

# my_game.vs_mcts(num_sim=1500)