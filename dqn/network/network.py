
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, action_space, obs_space):
        super(DQN, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.action_size = action_space
        self.obs_size = obs_space

        # specify network architecture
        # TODO: make this more modular
        self._layers = nn.Sequential(
            nn.Conv2d(4, 16, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2592, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_size)
        )

    def forward(self, x):
        x = torch.from_numpy(x).type(torch.FloatTensor).to(self.device)
        return self._layers(x)
