

import torch
import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module):
    def __init__(self, action_space, obs_space, num_layers=2, num_hidden=32):
        super(Policy, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.action_size = action_space
        self.obs_size = obs_space

        # specify network architecture
        layer_list = [nn.Linear(self.obs_size, num_hidden), nn.ReLU()]
        for i in range(num_layers):
            layer_list.append(nn.Linear(num_hidden, num_hidden))
            layer_list.append(nn.ReLU())
        layer_list.append(nn.Linear(num_hidden, self.action_size))
        layer_list.append(nn.Softmax(dim=0))

        self._layers = nn.Sequential(*layer_list)

        

    def forward(self, x):
        return self._layers(x)
