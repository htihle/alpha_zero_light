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
import tools


import sys
sys.setrecursionlimit(10000)


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

        if 3 in use_data:
            replay_folder = 'replays_conv_mac/'
            replay_files_uni = glob.glob(replay_folder + "game_*.pkl", recursive=True) #[:2000]
            replay_folder = 'conv/'
            replay_files_uni1 = glob.glob(replay_folder + "game_*.pkl", recursive=True) #[:2000]
            replay_folder = 'conv2/'
            replay_files_uni2 = glob.glob(replay_folder + "game_*.pkl", recursive=True) #[:2000]
            replay_folder = 'conv3/'
            replay_files_best1 = glob.glob(replay_folder + "game_*.pkl", recursive=True) #[:2000]
            replay_folder = 'conv4/'
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

dataset = Replays(tools.conv_features, use_data=(3,))  # choose whic data to include 0 to 3 with 3 being the newest

n_samp = len(dataset)
n_train = int(n_samp * 0.85)
n_val = int(n_samp * 0.1)
n_test = n_samp - n_train - n_val
train_data, val_data, test_data = random_split(dataset, lengths=[n_train, n_val, n_test])

train_loader = DataLoader(train_data, batch_size=batch_size)
valid_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size,shuffle=True)

n_pix = 4

n_feat = n_feat = 2 * n_pix ** 2 + 9  # number of features for fully connected model

# _, my_model = load_ref(name='ref_best')
# _, my_model = load_ref(name='ref_replays_all_clean')
_, my_model = tools.load_ref(name='ref_conv_best')
# my_model = model.GameModel(n_feat, n_action=4 * n_pix ** 2)
# my_model = model.ResModel(n_pix, n_action=4 * n_pix ** 2)

lr = 0.0005 #0.0005
num_epochs = 100
prior_weight = 5.0

loss_fn = torch.nn.MSELoss()
loss_p_fn = torch.nn.CrossEntropyLoss()

optim = torch.optim.Adam(my_model.parameters(), lr=lr, weight_decay=1e-05)
# optim = torch.optim.SGD(my_model.parameters(), lr=lr, weight_decay=1e-05)

my_model.to(device)

### Training function
def train_epoch(my_model, device, dataloader, loss_fn, optimizer, prior_weight=10):
    # Set train mode
    my_model.train()
    train_loss = []
    
    for x_batch, y_batch, p_batch in dataloader: 
        # Move tensor to device
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        p_batch = p_batch.to(device)
        # inference
        value, probs = my_model(x_batch)
        value = value[:, 0]

        # Evaluate loss
        loss = loss_fn(value, y_batch) + prior_weight * loss_p_fn(probs, p_batch)
        # loss = loss_p_fn(probs, p_batch)
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # batch loss
        #print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())


    return np.mean(train_loss)

## Testing function
def test_epoch(my_model, device, dataloader, loss_fn, make_plot=False, prior_weight=prior_weight):
    my_model.eval()
    with torch.no_grad():
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_y = []
        conc_p = []
        conc_p_out = []

        # for name, param in my_model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)



        for x_batch, y_batch, p_batch in dataloader:
            # Move tensors to device
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            p_batch = p_batch.to(device)
            # inference
            value, probs = my_model(x_batch)
            value = value[:, 0]
            # Append the network output and the targets
            conc_out.append(value.cpu())
            conc_y.append(y_batch.cpu())
            conc_p.append(p_batch.cpu())
            conc_p_out.append(probs.cpu())

        conc_out = torch.cat(conc_out)
        conc_y = torch.cat(conc_y)
        conc_p = torch.cat(conc_p)
        conc_p_out = torch.cat(conc_p_out)
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_y) + prior_weight * loss_p_fn(conc_p_out, conc_p)
        # val_loss = loss_p_fn(conc_p_out, conc_p)
        idx = np.random.randint(0, len(conc_y.detach().cpu().numpy()), 4)
        print(conc_y.detach().cpu().numpy()[idx])
        print(conc_out.detach().cpu().numpy()[idx])

        # plt.figure()
        if make_plot:
            plt.scatter(conc_y.detach().cpu().numpy(), conc_out.detach().cpu().numpy(), s=0.5)

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