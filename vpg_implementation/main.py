'''
@brief implementation of Vanilla Polict Gradient algorithm to learn more about
       the field of deep RL
'''

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
import numpy as np
import gym
from collections import deque


if __name__ == "__main__":
    # define policy network
    class PolicyNetwork(nn.Module):
        def __init__(self, state_size, num_layers, hidden_size, action_size):
            super(PolicyNetwork, self).__init__()
            self.optimizer = None

            self.layers = torch.nn.ModuleList()

            self.layers.append(nn.Linear(state_size, hidden_size))
            self.layers.append(nn.ReLU(hidden_size))

            for i in range(1, num_layers-1):
                self.layers.append(nn.Linear(hidden_size, hidden_size))
                self.layers.append(nn.ReLU(hidden_size))

            self.layers.append(nn.Linear(hidden_size, action_size))

        # define forward pass with one hidden layer with ReLU activation and sofmax after output layer
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            x = F.softmax(x, dim=1)
            return x

        def update(self, reward_list, log_prob_list):
            disc_rewards_list = []
            gamma = 0.99  # TODO: make input
            for i in range(len(reward_list)):
                discounted_r = 0
                discount_counter = 0
                for j in range(i, len(reward_list)):
                    discounted_r += (gamma**discount_counter) * reward_list[j]
                    discount_counter += 1
                disc_rewards_list.append(discounted_r)

            disc_rewards_tensor = torch.tensor(disc_rewards_list)
            disc_rewards_tensor_norm = (disc_rewards_tensor - disc_rewards_tensor.mean()) / (disc_rewards_tensor.std() + 1e-9)

            # calculae the polict gradient
            policy_gradient = []
            for log_prob, Gt in zip(log_prob_list, disc_rewards_tensor_norm):
                policy_gradient.append(-log_prob * Gt)
            policy_gradient_tensor = torch.stack(policy_gradient).sum()

            # back propagate
            self.optimizer.zero_grad()
            policy_gradient_tensor.backward()
            self.optimizer.step()

    # create environment
    env = gym.make("CartPole-v1")
    # instantiate the policy
    policy = PolicyNetwork(state_size=env.observation_space.shape[0],
                           num_layers=3,
                           hidden_size=40,
                           action_size=env.action_space.n)
    # create an optimizer
    _lr = 3e-3
    policy.optimizer = torch.optim.Adam(policy.parameters(), lr=_lr)
    # initialize gamma and stats
    gamma=0.99
    render_rate = 100 # render every render_rate episodes

    current_episode = 0
    while True:
        state = env.reset()
        log_prob_list = []
        reward_list = []
        steps = 0
        current_episode += 1
        while True:
            steps += 1
            if current_episode % render_rate == 0:
                env.render()
            # get action with highest prob
            action_probs = policy.forward(Variable(torch.from_numpy(state).float().unsqueeze(0)))
            highest_prob_action = np.random.choice(env.action_space.n, p=np.squeeze(action_probs.detach().numpy()))
            log_prob = torch.log(action_probs.squeeze(0)[highest_prob_action])
            # take step
            state, reward, done, info = env.step(highest_prob_action)

            log_prob_list.append(log_prob)
            reward_list.append(reward)
            if done:
                policy.update(reward_list, log_prob_list)
                if current_episode % 1 == 0:
                    sys.stdout.write("episode: {}, total reward: {}, length: {}\n".format(current_episode,
                                                                                          np.round(np.sum(reward_list), decimals=3),
                                                                                          steps))
                break


