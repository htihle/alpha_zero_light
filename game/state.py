import numpy as np
import copy
import os
import pickle
import game.units as units
from time import sleep


class State():
    def __init__(self, n_pix, players=None, randomize=False, rand_frac=0.9, to_play=0):
        self.to_play = to_play
        self.n_pix = n_pix
        self.turn = 0
        self.done = False
        if players is None:
            self.players = [
                [[units.Center((0, 0))], [units.Unit((0, 1))]],
                [[units.Center((self.n_pix - 1, self.n_pix - 1))],
                 [units.Unit((self.n_pix - 1, self.n_pix - 2))]]
            ]
            if randomize and np.random.rand() < rand_frac:
                for player in self.players:
                    for t in player:
                        for u in t:
                            # u.size += np.random.randint(0, 10)
                            # u.size += np.random.randint(0, 5)
                            u.size += np.random.randint(0, 1)
                self.players[0][1][0].position += np.random.randint(0, 2, 2)
                self.players[1][1][0].position -= np.random.randint(0, 2, 2)
                # self.players[0][1][0].position += np.random.randint(0, 3, 2)
                # self.players[1][1][0].position -= np.random.randint(0, 3, 2)
                self.players[0][0][0].res = np.random.randint(5, 10)
                self.players[1][0][0].res = np.random.randint(5, 10)
            # else:
            #     print('starting standard game')
        else:
            self.players = players

    def return_maps(self):
        my_map = np.zeros((len(self.players), len(self.players[0]) + 2, self.n_pix, self.n_pix))
        my_map[:, len(self.players[0])] = -1
        for i, player in enumerate(self.players):
            for j, unit_class in enumerate(player):
                for k, unit in enumerate(unit_class):
                    my_map[i, j, unit.position[0], unit.position[1]] = unit.size
                    my_map[i, len(self.players[0]), unit.position[0], unit.position[1]] = k  # indices
                    if j == 0:
                        my_map[i, len(self.players[0]) + 1, unit.position[0], unit.position[1]] = unit.res
        return my_map
    
    def invalid_action_mask(self):
        my_map = self.return_maps()
        summap = my_map[0][1] - my_map[1][1] + my_map[0][0] - my_map[1][0]
        mask = np.zeros((self.n_pix, self.n_pix, 4))
        one_mask = np.zeros((self.n_pix, self.n_pix))
        if self.to_play == 0:
            one_mask[(summap > 0)] = 1   # where player 0 has units
            mask[:, :, :] = one_mask[:, :, None] 
            mask[0, 0, 2:] = 0  # center only has two actions
            mask[0, :, 2] = 0 # can't go out of the top
            mask[:, 0, 3] = 0 # can't go out to the left
            mask[-1, :, 0] = 0 # can't go out the bottom
            mask[:, -1, 1] = 0 # can't go out to the right
        if self.to_play == 1:
            one_mask[(summap < 0)] = 1  # where player 1 has units
            mask[:, :, :] = one_mask[:, :, None]
            mask[self.n_pix-1, self.n_pix-1, 2:] = 0  # center only has two actions
            mask[0, :, 2] = 0 # can't go out of the top
            mask[:, 0, 3] = 0 # can't go out to the left
            mask[-1, :-1, 0] = 0 # can't go out the bottom
            mask[:-1, -1, 1] = 0 # can't go out to the right
        return mask

    def visualize(self):
        my_map = self.return_maps()
        print('Money %i and %i' % (np.sum(my_map[0][3].flatten()), np.sum(my_map[1][3].flatten())))
        summap = my_map[0][1] - my_map[1][1] + my_map[0][0] - my_map[1][0]
        print(summap)
    
    def state2string(self):
        my_map = self.return_maps()
        summap = my_map[0][1] - my_map[1][1] + my_map[0][0] - my_map[1][0]
        st = ''
        state_string = str(self.turn) + ' '+ str(self.to_play) + ' ' + str(np.sum(my_map[0][3].flatten())) + ' ' + str(np.sum(my_map[1][3].flatten())) + ' ' + st.join([str(el) for el in summap])
        return state_string

    def step(self, action):
        assert action[0] == self.to_play, "inconsistent actor " + str(action)
        self.players[self.to_play][action[1]][action[2]].act(self, action)  
        
        if self.to_play == 1:
            for player in self.players:
                for centre in player[0]:
                    centre.work()
                    centre.work()
        
        self.to_play = 1 - self.to_play
        self.turn += 1
        