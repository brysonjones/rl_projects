
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, action_space, obs_space):
        super(DQN, self).__init__()
        self.action_size = action_space
        self.obs_size = obs_space

        # specify network architecture

    def forward(self, x):
        pass
