import numpy as np
import copy
import os
import pickle


class Unit():
    def __init__(self, position):
        self.position = np.array(position)
        self.size = 1
        self.steps = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        self.n_actions = len(self.steps)

    def combine(self, unit2):
        self.position = unit2.position
        self.size = self.size + unit2.size

    def act(self, state, action):
        my_map = state.return_maps()
        newpos = self.position + self.steps[action[3]]
        if (np.any(newpos < 0)) or (np.any(newpos >= state.n_pix)):
            return 0
        elif (my_map[action[0], action[1], newpos[0], newpos[1]] > 0):
            other_index = int(my_map[action[0], 2, newpos[0], newpos[1]])
            other_unit = state.players[action[0]][action[1]][other_index]
            self.combine(other_unit)
            state.players[action[0]][action[1]].pop(other_index)
            return 0
        elif (my_map[action[0], 0, newpos[0], newpos[1]] > 0):
            other_index = int(my_map[action[0], 2, newpos[0], newpos[1]])
            other_unit = state.players[action[0]][0][other_index]
            other_unit.size += state.players[action[0]][action[1]][action[2]].size
            state.players[action[0]][action[1]].pop(action[2])
            return 0
        elif (my_map[1 - action[0], action[1], newpos[0], newpos[1]] > 0):
            other_index = int(my_map[1 - action[0], 2, newpos[0], newpos[1]])
            other_unit = state.players[1 - action[0]][action[1]][other_index]
            s = other_unit.size
            other_unit.size -= self.size // 2 + self.size % 2
            self.size -= s // 3 + 1
            if (self.size <= 0) and (other_unit.size <= 0):
                state.players[1 - action[0]][action[1]].pop(other_index)
                state.players[action[0]][action[1]].pop(action[2])
                return 0
            elif (self.size <= 0) and (other_unit.size > 0):
                state.players[action[0]][action[1]].pop(action[2])
                return 0
            elif (self.size > 0) and (other_unit.size <= 0):
                self.position = newpos
                state.players[1 - action[0]][action[1]].pop(other_index)
                return 0
            else:
                return 0
        elif (my_map[1 - action[0], 0, newpos[0], newpos[1]] > 0):
            self.position = newpos
            state.players[1 - action[0]][0][0].size = 0
            state.players[1 - action[0]][0][0].res = 0
            if state.done:
                pass
            else:
                state.done = True
                state.victory = (action[0] == 0) * 1.0 - (action[0] == 1) * 1.0 
            # print('Player %i' % (action[0] + 1))
            return 0
        else:
            self.position = newpos
            return 0


class Center():
    def __init__(self, position, unit=None):
        self.position = np.array(position)
        self.n_actions = 2
        if unit is None:
            self.size = 1
        else:
            self.size = unit.size
        self.res = 10

    def work(self):
        self.res += self.size * 0.5

    def act(self, state, action):
        if action[3] == 0:
            if self.res >= 5: 
                self.size += 1
                self.res -= 5
                return 0
            else:
                return 0
        elif action[3] == 1:
            if self.res >= 10:
                my_map = state.return_maps()
                offset = (action[0] == 0) * 1 + (action[0] == 1) * -1
                newpos = self.position + (offset, 0)
                # print('newpos', newpos)
                if (my_map[action[0], 1, newpos[0], newpos[1]] > 0):
                    other_index = int(my_map[action[0], 2, newpos[0], newpos[1]])
                    # print(state.players[action[0]][1][other_index].size)
                    state.players[action[0]][1][other_index].size += 1
                    # print(state.players[action[0]][1][other_index].size)
                    self.res -= 10
                    return 0
                elif (my_map[1 - action[0], 1, newpos[0], newpos[1]] > 0):
                    return 0
                else:
                    state.players[action[0]][1].append(Unit(newpos))
                    self.res -= 10
                    return 0
            else:
                return 0