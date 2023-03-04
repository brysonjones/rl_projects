
import sys
import gym
import torch
import numpy as np

if __name__ == "__main__":

    # init environment
    env = gym.make("") # TODO: pick a good, simple continuous action space
    action_space_size = env.action_space.n
    obs_space_size = env.observation_space.shape[0]

    # initialize networks
    # DQN
    # policy

    # init replay buffer

    # create target networks
    # TODO: is that just copying?

    # loop()
        # select action, and add zero mean gaussian noise to selected actions
        # execute action
        # get next_state, rewards, done signal
        # store all data in replay buffer
        