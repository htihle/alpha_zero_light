import numpy as np
import copy
import os
import pickle
import game.state as st
from time import sleep
import mcts as my_mcts


class Game():
    def __init__(self, n_pix, q, v, state=None, randomize=False, rand_frac=0.9, max_turns=50, random_start=False):
        self.n_pix = n_pix
        self.q = q
        self.v = v
        self.max_turns = max_turns
        if state is None:
            if random_start:
                to_play = np.random.randint(0, 2)
                self.state = st.State(self.n_pix, randomize=randomize, rand_frac=rand_frac, to_play=to_play)
            else:
                self.state = st.State(self.n_pix, randomize=randomize, rand_frac=rand_frac)
        else:
            self.state = state
        self.turn = 1
        
        self.replay = []

    def vs_mcts(self, net, num_sim=100, sample=False):
        mcts = my_mcts.MCTS(valuefunc=self.v)
        walking = {
            "s": 0,
            "d": 1,
            "w": 2,
            "a": 3,
        }
        while (not self.state.done):
            self.state.visualize()
            action = [0]
            action.append(int(input('0 for cities, 1 for units: ')))
            if (action[1] == 0):
                print('You have %s resources' % self.state.players[0][0][0].res)
                action.append(0)
                action.append(int(input('0 for growth, 1 for unit: ')))
            elif (action[1] == 1):
                for i, unit in enumerate(self.state.players[action[0]][1]):
                    print('You have unit at (%i, %i) with size %i, this is unit %i in %i' % (
                        unit.position[0], unit.position[1], unit.size, i + 1, len(self.state.players[action[0]][1])))
                    my_input = input('wasd to move or f to skip ')
                    if (my_input == 'f'):
                        pass
                    else:
                        action.append(i)
                        action.append(walking[my_input])
                        break
            self.replay.append([copy.deepcopy(self.state), action])
            self.state.step(action)

            mcts.clear()

            probs, best_val = mcts.get_mcts_policy(self.state, net, num_sim)
            # for i in range(4): 
            #     print(probs[:, :, i])
            if sample:
                idx = np.unravel_index(np.argmax(np.random.multinomial(1, probs.flatten()), axis=None), probs.shape)
            else:
                idx = np.unravel_index(np.argmax(probs, axis=None), probs.shape)
            my_map = self.state.return_maps()
            if ((idx[0], idx[1]) == (0, 0) or (idx[0], idx[1]) == (self.state.n_pix - 1, self.state.n_pix -1)):
                i = 0
                unit_index = 0
            else:
                i = 1
                unit_index = int(my_map[self.state.to_play, 2, idx[0], idx[1]])
                assert unit_index >= 0, "no unit in position chosen"
            action = [self.state.to_play, i, unit_index, idx[2]]
            print('Action: ', action)
            print('Model judgement: ', float(best_val))
            self.replay.append([copy.deepcopy(self.state), action])  # fix this with probs (if needed)
            self.state.step(action)
            # self.v(self.state)
            self.turn += 1
            if self.turn > 150:
                self.state.done = True
                self.state.victory = 0.0
        if self.state.done:
            # self.save_game()
            print(self.state.victory)

    def mcts_vs_mcts(self, net, net2, eps=0.0, sample=True, n_sim=150, save_game=True, temp=1.0, replay_folder='replays'):
        mcts = my_mcts.MCTS(valuefunc=self.v)
        while (not self.state.done):
            #print(self.turn)
            #### first player
            mcts.clear()

            probs, best_val = mcts.get_mcts_policy(self.state, net, n_sim=n_sim, temp=temp)
            if np.random.rand() < eps:
                i = np.random.randint(2)
                if len(self.state.players[self.state.to_play][i]) > 0:
                    j = np.random.randint(len(self.state.players[self.state.to_play][i]))
                else:
                    i = 1 - i
                    j = np.random.randint(len(self.state.players[self.state.to_play][i]))
                a = np.random.randint(self.state.players[self.state.to_play][i][j].n_actions)
                action = [self.state.to_play, i, j, a]
            else:
                
                if sample:
                    idx = np.unravel_index(np.argmax(np.random.multinomial(1, probs.flatten()), axis=None), probs.shape)
                else:
                    idx = np.unravel_index(np.argmax(probs, axis=None), probs.shape)
                # idx = np.unravel_index(np.argmax(probs, axis=None), probs.shape)
                my_map = self.state.return_maps()
                if ((idx[0], idx[1]) == (0, 0) or (idx[0], idx[1]) == (self.state.n_pix - 1, self.state.n_pix -1)):
                    i = 0
                    unit_index = 0
                else:
                    i = 1
                    unit_index = int(my_map[self.state.to_play, 2, idx[0], idx[1]])
                    assert unit_index >= 0, "no unit in position chosen"
                action = [self.state.to_play, i, unit_index, idx[2]]
                # print(q_max)
            self.replay.append([copy.deepcopy(self.state), action, probs])
            self.state.step(action)
            
            self.turn += 1
            if self.turn > self.max_turns:
                self.state.done = True
                self.state.victory = 0.0
                
            if self.state.done:
                break
            
            #### second player
            mcts.clear()

            probs, best_val = mcts.get_mcts_policy(self.state, net2, n_sim=n_sim, temp=temp)  
            if np.random.rand() < eps:
                i = np.random.randint(2)
                if len(self.state.players[self.state.to_play][i]) > 0:
                    j = np.random.randint(len(self.state.players[self.state.to_play][i]))
                else:
                    i = 1 - i
                    j = np.random.randint(len(self.state.players[self.state.to_play][i]))
                a = np.random.randint(self.state.players[self.state.to_play][i][j].n_actions)
                action = [self.state.to_play, i, j, a]
            else:
                if sample:
                    idx = np.unravel_index(np.argmax(np.random.multinomial(1, probs.flatten()), axis=None), probs.shape)
                else:
                    idx = np.unravel_index(np.argmax(probs, axis=None), probs.shape)
                # idx = np.unravel_index(np.argmax(probs, axis=None), probs.shape)
                my_map = self.state.return_maps()
                if ((idx[0], idx[1]) == (0, 0) or (idx[0], idx[1]) == (self.state.n_pix - 1, self.state.n_pix -1)):
                    i = 0
                    unit_index = 0
                else:
                    i = 1
                    unit_index = int(my_map[self.state.to_play, 2, idx[0], idx[1]])
                    assert unit_index >= 0, "no unit in position chosen"
                action = [self.state.to_play, i, unit_index, idx[2]]
            self.replay.append([copy.deepcopy(self.state), action, probs])
            self.state.step(action)
            self.turn += 1
            if self.turn > self.max_turns:
                self.state.done = True
                self.state.victory = 0.0
        if self.state.done:
            #self.replay.append([copy.deepcopy(self.state), action, probs])
            if save_game:
                self.save_game(replay_folder)



    def save_game(self, replay_folder='replays'):
        all_info = [self.replay, self.state.victory]
        gameid = 0
        while os.path.isfile(os.path.join(
                replay_folder + '/game_{0:d}.pkl'.format(gameid))):
            gameid += 1
        print('Current game_id: %i' % gameid)
        filename = replay_folder + '/game_{0:d}'.format(gameid)
        # filename = 'replays/game_1'
        with open(filename + '.pkl', 'wb') as f:
            pickle.dump(all_info, f, pickle.HIGHEST_PROTOCOL)
