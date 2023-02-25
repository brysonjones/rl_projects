
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import random
import numpy as np
import wandb

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPO_Agent():
    def __init__(self, obs_space_size, action_space_size, 
                 epsilon=0.2, discount_gamma=0.99, gae_lambda=0.95, num_epochs=4,
                 batch_size=64):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self._obs_space_size = obs_space_size
        self._action_space_size = action_space_size
        self._epsilon = epsilon
        self._discount_gamma = discount_gamma
        self._gae_lambda = gae_lambda
        self._num_epochs = num_epochs
        self._batch_size = batch_size

        self.state_data = []
        self.action_data = []
        self.logprobs_data = []
        self.rewards_data = []
        self.dones_data = []
        self.value_ests_data = []

        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_space_size).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_space_size), std=0.01),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_space_size).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4, eps=1e-5)

    def get_action(self, current_state):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.get_value(current_state)
    
    def get_value(self, state):
        return self.critic(state)

    # start with very simple advantage function calculation and make it more complex later (GAE, etc)
    def store_rollout(self, state, action, log_probs, rewards, done, value_ests):
        self.state_data.append(state)
        self.action_data.append(action)
        self.logprobs_data.append(log_probs)
        self.rewards_data.append(rewards)
        self.dones_data.append(done)
        self.value_ests_data.append(value_ests)

    def calc_advantage(self, next_state, next_done, num_steps):
         with torch.no_grad():
            next_value = self.get_value(next_state).reshape(1, -1)
            advantages = torch.zeros_like(self.rewards_data).to(self.device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones_data[t + 1]
                    nextvalues = self.value_ests_data[t + 1]
                delta = self.rewards_data[t] + self.discount_gamma * nextvalues * nextnonterminal - self.value_ests_data[t]
                advantages[t] = lastgaelam = delta + self.discount_gamma * self.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + self.value_ests_data

            return returns, advantages

    def batch_data(self):
        pass

    def _calculate_loss(self, batch):        



        wandb.log({'loss_clip': loss_clip})
        wandb.log({'loss_value': loss_value})
        wandb.log({'loss_entropy': loss_entropy})
        wandb.log({'loss_total': loss_total})

        # backprop and update weights
        self.optimizer.zero_grad()
        loss_total.backward()
        self.optimizer.step()

    def learn(self):        
        b_inds = np.arange(self._batch_size)
        clipfracs = []
        
        for e in range(self._num_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self._batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

        # clear all previous training data
        self._training_data_batched.clear()
