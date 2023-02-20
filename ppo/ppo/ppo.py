
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import wandb

class PPO():
    def __init__(self, policy, value_fcn, obs_space_size, action_space_size, 
                 epsilon=0.2, discount_gamma=0.99, num_epochs=1,
                 batch_size=32):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._policy = policy.to(self.device)
        self._value_fcn = value_fcn.to(self.device)
        self._obs_space_size = obs_space_size
        self._action_space_size = action_space_size
        self._epsilon = epsilon
        self._discount_gamma = discount_gamma
        self._num_epochs = num_epochs
        self._batch_size = batch_size
        self._training_data_raw = []
        self._training_data_batched = []

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
            rollout_data_list[t] = rollout_data_list[t] + (advantage_t.reshape((1, 1)), 
                                                           value_target_t.reshape((1, 1)))
        self._training_data_raw += rollout_data_list
        random.shuffle(self._training_data_raw)

    def create_single_batch(self):
        state_tensor = torch.empty((0, self._obs_space_size)).to(self.device)
        action_tensor = torch.empty((0, 1)).to(self.device)
        old_action_prob_tensor = torch.empty((0, 1)).to(self.device)
        advantage_tensor = torch.empty((0, 1)).to(self.device)
        value_target_tensor = torch.empty((0, 1)).to(self.device)
        current_size = 0
        for i in range(self._batch_size):
            if not self._training_data_raw:
                if current_size > 0:
                    self._training_data_batched.append((state_tensor, action_tensor, advantage_tensor, \
                                                        old_action_prob_tensor, value_target_tensor))
                return False  # ran out of samples
            state_t, action_t, _, _, \
                old_action_prob_t, advantage_t, value_target_t = self._training_data_raw.pop(0)
            state_tensor = torch.cat((state_tensor, state_t), dim=0)
            action_tensor = torch.cat((action_tensor, action_t), dim=0)
            old_action_prob_tensor = torch.cat((old_action_prob_tensor, old_action_prob_t), dim=0)
            advantage_tensor = torch.cat((advantage_tensor, advantage_t), dim=0)
            value_target_tensor = torch.cat((value_target_tensor, value_target_t), dim=0)
            current_size =+ 1

        self._training_data_batched.append((state_tensor, action_tensor, advantage_tensor, \
                                            old_action_prob_tensor, value_target_tensor))
        return True

    def batch_data(self):
        while(self.create_single_batch()):
            return


    def _calculate_loss(self, batch):
        # extract data
        state_tensor, action_tensor, old_action_prob_tensor, \
            advantage_tensor, value_target_tensor = batch
        _, action_prob_dist_t = self.get_action(state_tensor)
        prob_ratio = action_prob_dist_t.log_prob(action_tensor).exp() / old_action_prob_tensor.exp()
        
        # calculate losses
        loss_clip = torch.min(prob_ratio*advantage_tensor, 
                              torch.clip(prob_ratio, 1-self._epsilon, 1+self._epsilon)*advantage_tensor).mean()
        loss_value = F.mse_loss(self._value_fcn(state_tensor), value_target_tensor)
        loss_entropy = action_prob_dist_t.entropy().mean()
        loss_total = -loss_clip + 0.9 * loss_value - 0.15 * loss_entropy
        # wandb.log({'loss_total': loss_total})

        # backprop and update weights
        self._value_fcn.optimizer.zero_grad()
        self._policy.optimizer.zero_grad()
        loss_total.backward()
        self._value_fcn.optimizer.step()
        self._policy.optimizer.step()

    def learn(self):
        # update policy
        self.batch_data()
        for e in range(self._num_epochs):
            # TODO: loop through individual samples for now, but switch to batches
            for batch in self._training_data_batched:
                self._calculate_loss(batch)

        # clear all previous training data
        self._training_data_batched.clear()
