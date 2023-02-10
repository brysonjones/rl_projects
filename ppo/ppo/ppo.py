
import torch
import torch.nn as nn
import random

class PPO():
    def __init__(self, network, optimizer, espilion=0.3, discount_gamma=0.99):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._network = network.to(self.device)
        self._optimizer = optimizer
        self._espilion = espilion
        self._discount_gamma = discount_gamma

    def get_action(self, current_obs):
        if (random.random() < self._espilion):
            # explore
            return random.randint(0, self._network.action_size)
        else:
            # exploit
            q_values = self._network(current_obs)
            return q_values.cpu().data.numpy().argmax()