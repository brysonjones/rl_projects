
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

def run_episode(env, policy):
    rewards = 0
    steps = 0
    state, _ = env.reset()
    while(True):
        # select action, and add zero mean gaussian noise to selected actions
        action = policy.select_action(state, False)
        # execute action - get next_state, rewards, done signal
        state, reward, done, _, _ = env.step(action)
        rewards += reward
        if done:
            break

    sys.stdout.write("Full Episode Rewards: {}\n".format(rewards))

def main(argv):
    config_file = open(FLAGS.config_file_path, "r")
    config = yaml.load(config_file, Loader=yaml.FullLoader)

    # init environment
    env = gym.make(config["env"], render_mode=config["render_mode"])
    random_seed = config["random_seed"]
    state, _ = env.reset(seed=random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True

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

        counter += 1

        if (done):
            state, _ = env.reset()
        # if (time to update):
        if (counter % config["update_period"] == 0 and counter > config["learning_wait_period"]):
            for i in range(config["num_updates"]):
                ddpg_system.update()
        
        if (counter % config["render_period"] == 0):
            sys.stdout.write("Render Number: {}\n".format(counter / config["render_period"]))
            run_episode(env, ddpg_system)
        

    # clean up resources
    config_file.close()
        

if __name__ == "__main__":
  app.run(main)