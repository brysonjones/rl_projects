
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, action_space, obs_space, espilion=0.3):
        super(DQN, self).__init__()
        self.action_size = action_space
        self.obs_size = obs_space

    def forward(x):
        pass

    def get_action(current_obs):
        pass
