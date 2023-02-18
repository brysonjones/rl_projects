
import torch
import torch.nn as nn
import random
import numpy as np

class PPO():
    def __init__(self, policy, value_fcn, optimizer, espilion=0.3, discount_gamma=0.99):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._policy = policy.to(self.device)
        self._value_fcn = value_fcn.to(self.device)
        self._espilion = espilion
        self._discount_gamma = discount_gamma
        self._optimizer = optimizer

    def get_action(self, current_obs):
        if (random.random() < self._espilion):
            # explore
            return random.randint(0, self._network.action_size)
        else:
            # exploit
            q_values = self._policy(current_obs)
            return q_values.cpu().data.numpy().argmax()
    
    def get_value(self, state):
        return self._value_fcn(state)


    # start with very simple advantage function calculation and make it more complex later (GAE, etc)
    def _calculate_advantage(self, value_current_state_list, value_next_state_list, reward_list):
        advantage = np.zeros_like(reward_list)
        for i in range(len(reward_list)):
            advantage[i] = reward_list[i] + (self._discount_gamma * value_next_state_list[i]) - value_current_state_list[t]

    def _calculate_loss():
        pass

    def learn():
        pass