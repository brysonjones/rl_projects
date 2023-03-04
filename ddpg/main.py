
import sys
import gym
import torch
import numpy as np

if __name__ == "__main__":

    # init environment
    env = gym.make("") # TODO: pick a good, simple continuous action space
    action_space_size = env.action_space.n
    obs_space_size = env.observation_space.shape[0]
