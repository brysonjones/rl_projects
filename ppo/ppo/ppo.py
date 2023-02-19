
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import wandb

class PPO():
    def __init__(self, policy, value_fcn, optimizer, action_space_size, 
                 epsilon=0.2, discount_gamma=0.99, num_epochs=3):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._policy = policy.to(self.device)
        self._value_fcn = value_fcn.to(self.device)
        self._optimizer = optimizer
        self._action_space_size = action_space_size
        self._epsilon = epsilon
        self._discount_gamma = discount_gamma
        self._num_epochs = num_epochs
        self._training_data = []

    def get_action(self, current_obs):
        action_prob_dist = self._policy(current_obs)
        action = action_prob_dist.sample()
        
        return action, action_prob_dist
    
    def get_value(self, state):
        return self._value_fcn(state)

    # start with very simple advantage function calculation and make it more complex later (GAE, etc)
    def store_rollout(self, rollout_data_list, value_target_t):
        # loop in reverse to prevent re-computing values
        for t in range(len(rollout_data_list)-1, -1, -1):
            value_target_t = rollout_data_list[t][2] + self._discount_gamma * value_target_t
            advantage_t = value_target_t - self._value_fcn(rollout_data_list[t][0])
            rollout_data_list[t] = rollout_data_list[t] + (advantage_t, 
                                                           value_target_t)
        self._training_data += rollout_data_list

    def _calculate_loss(self, sample):
        # extract data
        state_t, action_t, reward_t, next_state_t, \
            old_action_prob_t, advantage_t, value_target_t = sample
        _, action_prob_dist_t = self.get_action(state_t)
        prob_ratio = action_prob_dist_t.log_prob(action_t)/ old_action_prob_t
        
        # calculate losses
        loss_clip = torch.min(torch.tensor([prob_ratio*advantage_t, 
                                torch.clip(prob_ratio, 1-self._epsilon, 1+self._epsilon)]))
        loss_value = F.mse_loss(self._value_fcn(state_t).squeeze(), value_target_t)
        loss_total = -loss_clip + 0.5 * loss_value
        wandb.log({'loss_total': loss_total})

        # backprop and update weights
        self._optimizer.zero_grad()
        loss_total.backward()
        self._optimizer.step()

    def learn(self):
        # update policy
        for e in range(self._num_epochs):
            random.shuffle(self._training_data)
            # TODO: loop through individual samples for now, but switch to batches
            for sample in self._training_data:
                self._calculate_loss(sample)

        # clear all previous training data
        self._training_data.clear()