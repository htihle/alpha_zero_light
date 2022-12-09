import numpy as np
import copy
from scipy.special import softmax

import torch

n_pix = 4

priors_uni = np.zeros((n_pix, n_pix,  4)) + 1.0
# priors[1, 0, 2] = 100.0

c_base = 1.0
pb_c_init = 1.0
num_simulations = 150

class MCTS():
    def __init__(self, valuefunc):
        self.Qsa = {}  # value of action a in state s
        self.Nsa = {}  # number of times the action a is taken in state s
        self.Ns = {}  # number of times state s is visited
        self.Ps = {}  # policy in state s (also called priors)
        self.v = valuefunc

    def clear(self):
        self.Qsa = {}  # value of action a in state s
        self.Nsa = {}  # number of times the action a is taken in state s
        self.Ns = {}  # number of times state s is visited
        self.Ps = {}  # policy in state s (also called priors)

    def get_mcts_policy(self, state, net, n_sim=num_simulations, temp=1.0):
        state_str = state.state2string()

        for i in range(n_sim):
            # print(' ')
            self.find_leaf_node(state, net, root=state_str)

        

        counts = np.zeros((state.n_pix, state.n_pix, 4))
        best_value = -1.0
        # loop over all possible actions
        for i, unittype in enumerate(state.players[state.to_play]):
            for j, unit in enumerate(unittype):
                for a in range(unit.n_actions):
                    action_str = ' '.join((str(i), str(j), str(a)))
                    sa = state_str + ' ' + action_str
                    if sa in self.Nsa:
                        counts[unit.position[0], unit.position[1], a] = self.Nsa[sa]
                        value = self.Qsa[sa]
                        if value > best_value:
                            best_value = value

        if state.to_play == 1:
            best_value = - best_value

        return softmax_probs(counts, temp), best_value
    
    def find_leaf_node(self, state, net, root):
        
        state_str = state.state2string()
        # print(state_str)
        if state.done:
            # print('to play when done', state.to_play)
            if state.to_play == 1:
                return state.victory
            else:
                return -state.victory

        if state_str not in self.Ps:
            # this means that we have reached a leaf node
            # here we call the policy and value functions
            # get policy of state

            value, priors, _ = self.v(state, net)
            mask = state.invalid_action_mask()
            priors = priors.detach().numpy()
            priors = softmax(priors)
            if state_str == root: 
                priors = 0.75 * priors + 0.25 * np.random.dirichlet(0.5 * np.ones_like(priors))
            priors = priors.reshape(n_pix, n_pix, 4)
            value = value.detach().numpy()
            ##### remove this
            # priors = priors_uni
            #####
            policy = priors * mask
            polsum = np.sum(policy.flatten())
            if polsum > 0:
                policy = policy / polsum # normalize policy
            else:
                print(mask, priors)
                sys.exit()
            
            # print('value', value)
            self.Ps[state_str] = policy 
            self.Ns[state_str] = 0
            if state.to_play == 1:
                value = - value
            return -value

        best_u = -np.inf

        # loop over all possible actions
        for i, unittype in enumerate(state.players[state.to_play]):
            for j, unit in enumerate(unittype):
                for a in range(unit.n_actions):
                    action_str = ' '.join((str(i), str(j), str(a)))
                    sa = state_str + ' ' + action_str
                    if sa in self.Qsa:
                        u = self.Qsa[sa] + c_base * self.Ps[state_str][unit.position[0], unit.position[1], a] * np.sqrt(self.Ns[state_str]) / (1.0 + self.Nsa[sa])
                    else:
                        u = c_base * self.Ps[state_str][unit.position[0], unit.position[1], a] * np.sqrt(self.Ns[state_str] + 1e-5)
                    if u > best_u:
                        best_u = u
                        best_action = (i, j, a)
                        best_action_str = action_str

        new_s = copy.deepcopy(state)
        new_s.step([new_s.to_play, best_action[0], best_action[1], best_action[2]])
        val = self.find_leaf_node(new_s, net, root=root)
        sa = state_str + ' ' + best_action_str
        if sa in self.Qsa:
            # print(sa, val, self.Qsa[sa])
            self.Qsa[sa] = (self.Nsa[sa] * self.Qsa[sa] + val) / (self.Nsa[sa] + 1)
            self.Nsa[sa] += 1
        else:
            self.Qsa[sa] = val
            self.Nsa[sa] = 1
            # print(sa, val)

        self.Ns[state_str] += 1
        return -val

def softmax_probs(counts, temp):
    if temp == 0.0:
        probs = np.zeros_like(counts)
        idx = np.unravel_index(np.argmax(counts, axis=None), counts.shape)
        probs[idx] = 1.0
        return probs  
    exps = np.array(counts) ** (1.0 / temp)
    probs = exps / np.sum(exps.flatten())
    return probs