

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

class Policy(nn.Module):
    def __init__(self, action_space, obs_space, num_layers=2, num_hidden=32):
        super(Policy, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.action_size = action_space
        self.obs_size = obs_space

        # specify network architecture
        self.layer_list = nn.ModuleList([nn.Linear(obs_space, num_hidden), nn.ReLU()])
        for i in range(num_layers):
            self.layer_list.append(nn.Linear(num_hidden, num_hidden))
            self.layer_list.append(nn.ReLU())
        self.layer_list.append(nn.Linear(num_hidden, self.action_size))
        self.layer_list.append(nn.Softmax(dim=-1))

        self._layers = nn.Sequential(*self.layer_list)

        self.optimizer = torch.optim.Adam(self._layers.parameters(), lr=3e-4)


    def forward(self, x):
        dist = self._layers(x)
        return Categorical(dist)
