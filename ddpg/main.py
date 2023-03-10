
import gymnasium as gym
import sys
import torch
import numpy as np
import yaml
from absl import app
from absl import flags

# import custom modules
import ddpg

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
    state, _ = env.reset(seed=random_seed)
    action_space_size = env.action_space.shape[0]
    action_space_range = (env.action_space.low[0], env.action_space.high[0])
    obs_space_size = env.observation_space.shape[0]
    
    # initialize DDPG class
    ddpg_system = ddpg.DDPG(obs_space_size, action_space_size, 
                            action_space_range, config)

    # init environmen
    # loop()
    counter = 0
    while(True):
        # select action, and add zero mean gaussian noise to selected actions
        action = ddpg_system.select_action(state, True)
        # execute action - get next_state, rewards, done signal
        next_state, reward, done, _, _ = env.step(action)
        # store all data in replay buffer
        ddpg_system.store_memory(state, action, next_state, reward, done)
        # if (time to update):
        if (counter % config["update_period"] == 0):
           ddpg_system.update()


    # clean up resources
    config_file.close()
        

if __name__ == "__main__":
  app.run(main)