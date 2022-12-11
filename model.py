import torch
from torch import nn
from torch.nn import Conv2d 
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import Tanh
# from torch.nn import Softmax
from torchvision import transforms
import numpy as np


class GameModel(nn.Module):
    def __init__(self, n_in, n_action):
        super(GameModel, self).__init__()
        self.n_in = n_in
        self.n_action = n_action
        self.n_fc1 = 256
        self.n_fc2 = 128
        self.n_fc3 = 128
        self.fc1 = nn.Linear(n_in, self.n_fc1)
        self.relu1 = ReLU()
        self.fc2 = nn.Linear(self.n_fc1, self.n_fc2)
        self.relu2 = ReLU()
        self.fc3 = nn.Linear(self.n_fc2, self.n_fc3)
        self.relu3 = ReLU()
        self.pred = nn.Linear(self.n_fc3, 1)
        self.tanh = Tanh()
        self.p_head = nn.Linear(self.n_fc3, self.n_action)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)

        p = self.p_head(x)
        x = self.pred(x)
        x = self.tanh(x)
        return x, p


class ResModel(nn.Module):
    def __init__(self, n_pix, n_action):
        super(ResModel, self).__init__()
        self.n_pix = n_pix
        self.in_channels = 7
        self.n_action = n_action
        self.n_filters=32
        self.conv_size = self.n_filters * self.n_pix ** 2
        self.conv_in = nn.Sequential(
            nn.Conv2d(self.in_channels, self.n_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.n_filters),
            nn.LeakyReLU()
        )

        # residual layers
        self.conv_1 = nn.Sequential(
            nn.Conv2d(self.n_filters, self.n_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.n_filters),
            nn.LeakyReLU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(self.n_filters, self.n_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.n_filters),
            nn.LeakyReLU()
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(self.n_filters, self.n_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.n_filters),
            nn.LeakyReLU()
        )

        self.flatten = nn.Flatten(start_dim=1)

        self.policy_head = nn.Sequential(
            nn.Conv2d(self.n_filters, 4, kernel_size=1),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(),
            nn.Flatten(start_dim=1), 
            nn.Linear(self.n_action, self.n_action)
        )

        self.pred = nn.Linear(self.conv_size, 1)
        self.tanh = Tanh()


    def forward(self, x):
        x = self.conv_in(x)
        x = x + self.conv_1(x)
        x = x + self.conv_2(x)
        x = x + self.conv_3(x)
        
        p = self.policy_head(x)
        x = self.flatten(x)
        x = self.pred(x)
        x = self.tanh(x)
        return x, p