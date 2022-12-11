import torch
from torch import nn
from torch.utils.data import DataLoader,random_split, Dataset
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import pickle
from time import sleep
import game.state as state
import game.game as game
import model
import glob 

import sys
sys.setrecursionlimit(10000)


def load_ref(name='ref'):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def conv_features(state):
    my_map = state.return_maps()
    n_pix = len(my_map[0][0][0])
    x_map = np.zeros((7, n_pix, n_pix))
    x_map[0] = my_map[0][0]
    x_map[1] = my_map[0][1]
    x_map[2] = my_map[0][3]
    x_map[3] = my_map[1][0]
    x_map[4] = my_map[1][1]
    x_map[5] = my_map[1][3]
    x_map[6] = np.ones((n_pix, n_pix)) * state.to_play
    return x_map

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

class Replays(Dataset):
    def __init__(self, x_func, use_data=(2)):
        x = []
        y = []
        p = []
        if 0 in use_data:
            replay_folder = 'replays_old/'
            replay_files = glob.glob(replay_folder + "game_*.pkl", recursive=True) #[:2000]
            replays = []
            for filename in replay_files:
                try:
                    with open(filename, 'rb') as f:
                        rep, vic = pickle.load(f)
                    assert len(rep[0]) == 3, 'data missing in replay ' + filename
                    replays.append([rep, vic])
                except AssertionError:
                    pass
            print('number of games in dataset ', len(replays))

            for replay in replays: 
                rep, vic = replay
                for i, r  in enumerate(rep):
                    my_state, action, probs = r
                    
                    if i>0:  # the probs from previous turn was saved, so we did this to fix
                        p.append(probs.flatten())
                    if i < len(rep)-1:
                        y.append(vic)
                        x.append(x_func(my_state))
        
        if 1 in use_data:
            replay_folder = 'replays_uni/'
            replay_files_uni = glob.glob(replay_folder + "game_*.pkl", recursive=True) #[:2000]

            replay_folder = '../alpha_zero/replays_uni/'
            replay_files_uni1 = glob.glob(replay_folder + "game_*.pkl", recursive=True) #[:2000]
            replay_folder = '../alpha_zero/replays_uni2/'
            replay_files_uni2 = glob.glob(replay_folder + "game_*.pkl", recursive=True) #[:2000]
            replay_folder = '../alpha_zero/replays_ref_best/'
            replay_files_best1 = glob.glob(replay_folder + "game_*.pkl", recursive=True) #[:2000]
            replay_folder = '../alpha_zero/replays_ref_best/'
            replay_files_best2 = glob.glob(replay_folder + "game_*.pkl", recursive=True) #[:2000]
            replay_files = replay_files_uni + replay_files_uni1 + replay_files_uni2 + replay_files_best1 + replay_files_best2
            replays = []
            for filename in replay_files:
                try:
                    with open(filename, 'rb') as f:
                        rep, vic = pickle.load(f)
                    assert len(rep[0]) == 3, 'data missing in replay ' + filename
                    replays.append([rep, vic])
                except AssertionError:
                    pass
            print('number of games in dataset ', len(replays))
            for replay in replays: 
                rep, vic = replay
                for i, r  in enumerate(rep):
                    my_state, action, probs = r
                        
                    if i < len(rep)-1:
                        p.append(probs.flatten())
                        y.append(vic)
                        x.append(x_func(my_state))
        
        if 2 in use_data:
            replay_folder = 'replays_clean_mac/'
            replay_files_uni = glob.glob(replay_folder + "game_*.pkl", recursive=True) #[:2000]
            replay_folder = 'replays_clean/'
            replay_files_uni1 = glob.glob(replay_folder + "game_*.pkl", recursive=True) #[:2000]
            replay_folder = 'replays_clean2/'
            replay_files_uni2 = glob.glob(replay_folder + "game_*.pkl", recursive=True) #[:2000]
            replay_folder = 'replays_freak/'
            replay_files_best1 = glob.glob(replay_folder + "game_*.pkl", recursive=True) #[:2000]
            replay_folder = 'replays_freak2/'
            replay_files_best2 = glob.glob(replay_folder + "game_*.pkl", recursive=True) #[:2000]
            replay_files = replay_files_uni + replay_files_uni1 + replay_files_uni2 + replay_files_best1 + replay_files_best2
            replays = []
            for filename in replay_files:
                try:
                    with open(filename, 'rb') as f:
                        rep, vic = pickle.load(f)
                    assert len(rep[0]) == 3, 'data missing in replay ' + filename
                    replays.append([rep, vic])
                except AssertionError:
                    pass
            print('number of games in dataset ', len(replays))
            for replay in replays: 
                rep, vic = replay
                for i, r  in enumerate(rep):
                    my_state, action, probs = r
                        
                    if i < len(rep)-1:
                        p.append(probs.flatten())
                        y.append(vic)
                        x.append(x_func(my_state))        
        
        assert len(x) == len(y), 'inconsistent length of data'
        self.n_samples = len(x)
        print('total number of positions in data ', self.n_samples)
        self.x_data = torch.tensor(x, dtype=torch.float32)
        self.y_data = torch.tensor(y, dtype=torch.float32)
        self.p_data = torch.tensor(p, dtype=torch.float32)

    
    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        x, y, p = self.x_data[index], self.y_data[index], self.p_data[index]
        return x, y, p

    def __len__(self):
        return self.n_samples



# Check to see if we have a GPU to use for training
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('A {} device was detected.'.format(device))

# Print the name of the cuda device, if detected
if device=='cuda':
  print (torch.cuda.get_device_name(device=device))


batch_size = 512

dataset = Replays(conv_features, use_data=(2,))

n_samp = len(dataset)
n_train = int(n_samp * 0.85)
n_val = int(n_samp * 0.1)
n_test = n_samp - n_train - n_val
train_data, val_data, test_data = random_split(dataset, lengths=[n_train, n_val, n_test])

# The dataloaders handle shuffling, batching, etc...
train_loader = DataLoader(train_data, batch_size=batch_size)
valid_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size,shuffle=True)

n_pix = 4

n_feat = n_feat = 2 * n_pix ** 2 + 9

# _, my_model = load_ref(name='ref_best')
# _, my_model = load_ref(name='ref_replays_all_clean')
_, my_model = load_ref(name='ref_conv_2_rat')
# my_model = model.GameModel(n_feat, n_action=4 * n_pix ** 2)
# my_model = model.ResModel(n_pix, n_action=4 * n_pix ** 2)

lr = 0.001 #0.0005
num_epochs = 100
prior_weight = 1.5

loss_fn = torch.nn.MSELoss()
loss_p_fn = torch.nn.CrossEntropyLoss()

optim = torch.optim.Adam(my_model.parameters(), lr=lr, weight_decay=1e-05)
# optim = torch.optim.SGD(my_model.parameters(), lr=lr, weight_decay=1e-05)

my_model.to(device)

### Training function
def train_epoch(my_model, device, dataloader, loss_fn, optimizer, prior_weight=10):
    # Set train mode for both the encoder and the decoder
    my_model.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for x_batch, y_batch, p_batch in dataloader: # with "_" we just ignore the labels (the second element of the dataloader tuple)
        # Move tensor to the proper device
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        p_batch = p_batch.to(device)
        # Encode data
        prediction, probs = my_model(x_batch)
        prediction = prediction[:, 0]
        # print(prediction.shape)
        # print(y_batch.shape)
        # sys.exit()
        # Evaluate loss
        loss = loss_fn(prediction, y_batch) + prior_weight * loss_p_fn(probs, p_batch)
        # loss = loss_p_fn(probs, p_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        #print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())


    return np.mean(train_loss)

### Testing function
def test_epoch(my_model, device, dataloader, loss_fn, make_plot=False, prior_weight=prior_weight):
    # Set evaluation mode for encoder and decoder
    my_model.eval()
    with torch.no_grad(): # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_y = []
        conc_p = []
        conc_p_out = []

        # for name, param in my_model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)



        for x_batch, y_batch, p_batch in dataloader:
            # Move tensor to the proper device
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            p_batch = p_batch.to(device)
            # predict data
            prediction, probs = my_model(x_batch)
            prediction = prediction[:, 0]
            # Append the network output and the original image to the lists
            conc_out.append(prediction.cpu())
            conc_y.append(y_batch.cpu())
            conc_p.append(p_batch.cpu())
            conc_p_out.append(probs.cpu())

            # conc_mask.append(mask_batch.cpu())
        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out) #[:, 0]
        conc_y = torch.cat(conc_y)
        conc_p = torch.cat(conc_p)
        conc_p_out = torch.cat(conc_p_out)
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_y) + prior_weight * loss_p_fn(conc_p_out, conc_p)
        # val_loss = loss_p_fn(conc_p_out, conc_p)
        idx = np.random.randint(0, len(conc_y.detach().cpu().numpy()), 10)
        print(conc_y.detach().cpu().numpy()[idx])
        print(conc_out.detach().cpu().numpy()[idx])

        # plt.figure()
        if make_plot:
            plt.scatter(conc_y.detach().cpu().numpy(), conc_out.detach().cpu().numpy(), s=0.5)
            # xs = np.linspace(0, 1, 2)
            # plt.plot(xs, xs)
            # plt.show()
        # # Find loss of each individual sample and plot them
        # individual_loss = loss_fn_ind(conc_out, conc_label).data
        # print(individual_loss)
        # print(individual_loss.shape)
        # plt.plot(individual_loss.numpy().sum((1, 2, 3)))
        # plt.show()
    return val_loss.data






history={'train_loss':[],'val_loss':[]}
for epoch in range(num_epochs):

    train_loss = train_epoch(my_model,device,train_loader,loss_fn,optim, prior_weight=prior_weight)
    val_loss = test_epoch(my_model,device,valid_loader,loss_fn, prior_weight=prior_weight)
    print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, num_epochs,train_loss,val_loss))
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)


def save_ref(mod, name='ref'):
    with open(name + '.pkl', 'wb') as f:
            pickle.dump(mod, f, pickle.HIGHEST_PROTOCOL)



test_epoch(my_model,device,test_loader,loss_fn).item()

test_epoch(my_model,device,test_loader,loss_fn, True).item() # Plot losses
plt.figure(figsize=(10,8))
plt.semilogy(history['train_loss'], label='Train')
plt.semilogy(history['val_loss'], label='Valid')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.grid()
plt.legend()
plt.plot()