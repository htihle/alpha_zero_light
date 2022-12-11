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


def load_game(gameid):
    filename = 'replays/game_{0:d}'.format(gameid)
    with open(filename + '.pkl', 'rb') as f:
        return pickle.load(f)


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
    x = torch.tensor(x, dtype=torch.float32)
    value = net(x)
    if state.done:
        if state.victory == 0.0:
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

def view_replay(gameid, net):
    replay, victory = load_game(gameid)
    n_turns = len(replay)
    print(n_turns)
    print(replay[0])
    print(len(replay))
    p = 0
    for state, action in replay:
        state.visualize()
        print(action)
        x = get_feat_nn(state, action)
        
        print(net.predict(x[None, :]))
        print(state.done)
        p += 1
        print(p)
        sleep(1.5)
    print(victory)


def train_nn(net, optim, loss_fn, elo=0.0):
    for i in range(10):
        print(i)
        eps = 0.0
        n_games = 500
        lens = np.zeros(n_games)
        accuracy = []
        victories = []
        x = []
        y = []
        p = []
        for gameid in range(n_games):
            print(gameid)
            my_game = game.Game(n_pix, q, v3, randomize=True, rand_frac=0.8, random_start=True)
            try:
                my_game.mcts_vs_mcts(net, net, eps=eps, n_sim=50)
            except RecursionError:
                print('Encountered a recursion error')
                continue
            # replay, victory, _ = load_game(gameid)
            replay = my_game.replay
            victory = my_game.state.victory

            n_replay = len(replay) - 1
            lens[gameid] = n_replay

            victories.append(victory)
            print('length of game: ', n_replay + 1, ' score: ', victory)
            for i in range(n_replay):
                
                state, action, probs = replay[i]

                x1 = get_feat_nn(state)
                y.append(victory)
                x.append(x1[:])
                p.append(probs.flatten())

        
        y = torch.tensor(y, dtype=torch.float32)
        x = torch.tensor(x, dtype=torch.float32)
        p = torch.tensor(p, dtype=torch.float32)
        n_epochs = 1

        for i in range(n_epochs):

            preds, priors = net(x)
            # print(preds)
            loss = loss_fn(preds, y[:, None]) + 10*loss_p_fn(priors, p)
            
            optim.zero_grad()
            loss.backward()
            optim.step()

        print('Average result', np.array(victories).mean())
        print('Median game length: ', np.median(lens))
        filename = 'nn_3_0'
        with open(filename + '.pkl', 'wb') as f:
            pickle.dump(net, f, pickle.HIGHEST_PROTOCOL)
    mod = [elo, net]
    save_ref(mod, 'ref_current')

    mod = [elo, net]
    mod2 = load_ref('ref_best')
    elo_new, _, mean_score = compare_models(mod, mod2, n_exp=500, ref=True)
    mod = [elo_new, net]

    
    if (elo_new > elo + 15) and (mean_score > 0.53):
        print("New elo: ", elo_new)
        save_ref(mod, 'ref_' + str(int(elo_new)))
        save_ref(mod, 'ref_best')
        return [elo_new, net]
    else:
        return load_ref('ref_best')

def compare_models(mod1, mod2, n_exp=100, ref=False, n_sim=5, temp=0.5):

    elo1, model1 = mod1
    elo2, model2 = mod2
    vic = np.zeros(n_exp)
    K = 10
    eps = 0.1
    for i in range(n_exp):
        print('Playing game number: ', i + 1, ' of ', n_exp)
        if i < n_exp // 2:
            gm = game.Game(n_pix, q, v3, randomize=True, rand_frac=0.2)
            gm.mcts_vs_mcts(model1, model2, eps=eps, n_sim=n_sim, save_game=False)
            vic[i] = (gm.state.victory + 1) * 0.5
        else:
            gm = game.Game(n_pix, q, v3, randomize=True, rand_frac=0.2)
            gm.mcts_vs_mcts(model2, model1, eps=eps, n_sim=n_sim, save_game=False)
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

def load_nn():
    filename = 'nn_3_0'
    with open(filename + '.pkl', 'rb') as f:
        return pickle.load(f)


def save_ref(mod, name='ref'):
    with open(name + '.pkl', 'wb') as f:
            pickle.dump(mod, f, pickle.HIGHEST_PROTOCOL)


def load_ref(name='ref'):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


n_pix = 4
n_feat = 2 * n_pix ** 2 + 9
learning_rate = 0.0002 #0.0001

# elo = 100
# my_model = model.GameModel(n_feat, n_action=4 * n_pix ** 2)
# elo, my_model = load_ref(name='ref_best')

# loss_fn = torch.nn.MSELoss()
# loss_p_fn = torch.nn.CrossEntropyLoss()

# optim = torch.optim.Adam(my_model.parameters(), lr=learning_rate, weight_decay=1e-05)


# train_nn(my_model, optim, loss_fn, elo)

for i in range(10):

    elo, my_model = load_ref(name='ref_best')
    # mod = [100, my_model]
    # save_ref(mod, 'ref_best')

    loss_fn = torch.nn.MSELoss()
    loss_p_fn = torch.nn.CrossEntropyLoss()

    optim = torch.optim.Adam(my_model.parameters(), lr=learning_rate, weight_decay=1e-05)


    train_nn(my_model, optim, loss_fn, elo)
