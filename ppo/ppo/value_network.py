
import torch
import torch.nn as nn
import torch.nn.functional as F

class ValueNet(nn.Module):
    def __init__(self, obs_space, num_layers=2, num_hidden=32):
        super(ValueNet, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # specify network architecture

        self.layer_list = nn.Sequential([nn.Linear(obs_space, num_hidden), 
                                         nn.Tanh(),
                                         nn.Linear(num_hidden, num_hidden),
                                         nn.Tanh(), 
                                         nn.Linear(num_hidden, 1)])
        # self.layer_list = nn.ModuleList([nn.Linear(obs_space, num_hidden), nn.Tanh()])
        # for i in range(num_layers):
        #     self.layer_list.append(nn.Linear(num_hidden, num_hidden))
        #     self.layer_list.append(nn.Tanh())
        # self.layer_list.append(nn.Linear(num_hidden, 1))

        # self._layers = nn.Sequential(*self.layer_list)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)


    def forward(self, x):
        return self.layer_list(x)

