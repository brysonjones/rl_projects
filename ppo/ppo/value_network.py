
import torch
import torch.nn as nn
import torch.nn.functional as F
from ppo.helper import layer_init

class ValueNet(nn.Module):
    def __init__(self, obs_space, num_layers=2, num_hidden=32):
        super(ValueNet, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # specify network architecture

        self.layer_list = nn.ModuleList([layer_init(nn.Linear(obs_space, num_hidden)), 
                                         nn.LeakyReLU()])
        for i in range(num_layers):
            self.layer_list.append(layer_init(nn.Linear(num_hidden, num_hidden)))
            self.layer_list.append(nn.LeakyReLU())
        self.layer_list.append(layer_init(nn.Linear(num_hidden, 1), std=1.0))

        self._layers = nn.Sequential(*self.layer_list)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=2.5e-4, eps=1e-5)


    def forward(self, x):
        return self._layers(x)

