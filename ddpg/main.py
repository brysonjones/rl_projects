
import gymnasium as gym
import sys
import torch
import numpy as np
import yaml
from absl import app
from absl import flags

# import custom modules
import policy
import dqn
import memory

FLAGS = flags.FLAGS
flags.DEFINE_string("config_file_path", None, 
                    "A configuration file for the environment")

# Required flag.
flags.mark_flag_as_required("config_file_path")

def main(argv):
    config_file = open(FLAGS.config_file_path, "r")
    config = yaml.load(config_file, Loader=yaml.FullLoader)

    # init environment
    env = gym.make(config["env"])
    random_seed = config["random_seed"]
    action_space_size = env.action_space.shape[0]
    action_space_range = (env.action_space.low[0], env.action_space.high[0])
    obs_space_size = env.observation_space.shape[0]
    
    # initialize networks
    # q_network = dqn.DQN(obs_space_size, action_space_size,
    #                     config['network']['num_hidden'],
    #                     config['network']['num_layers'],
    #                     config['network']['activation'])
    policy_network = policy.PolicyNetwork(obs_space_size, action_space_size,
                        config['network']['num_hidden'],
                        config['network']['num_layers'],
                        config['network']['activation'])
    # policy

    # init replay buffer
    replay_buffer = memory.ReplayMemory(config['replay_buffer']['max_size'])

    # create target networks
    # TODO: is that just copying?

    # loop()
        # select action, and add zero mean gaussian noise to selected actions
        # execute action
        # get next_state, rewards, done signal
        # store all data in replay buffer

    # clean up resources
    config_file.close()
        

if __name__ == "__main__":
  app.run(main)