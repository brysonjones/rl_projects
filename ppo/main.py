
import sys
import gym
import torch
from ppo.ppo import PPO
import ppo.policy
import ppo.value_network
import numpy as np
import wandb

if __name__ == "__main__":
    wandb.init(project="rl-ppo-project")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # init environment
    env = gym.make("CartPole-v0")
    action_space_size = env.action_space.n
    obs_space_size = env.observation_space.shape[0]

    # init models
    # wandb.init(config={"policy - num_layers": 2, 
    #                    "value_net - num_layers": 2,
    #                    "policy - num_hidden": 64, 
    #                    "value_net - num_hidden": 64, 
    #                    "optimizer - learning_rate": 1e-4})
    policy = ppo.policy.Policy(action_space_size, obs_space_size, num_layers=1, num_hidden=256)
    value_fcn = ppo.value_network.ValueNet(obs_space_size, num_layers=1, num_hidden=256)

    model = PPO(policy, value_fcn, obs_space_size, action_space_size)

    render_rate = 100
    num_episodes = 5000
    num_epochs = 3  # TODO: tune this
    for episode in range(num_episodes):
        state = env.reset()
        rollout_data_list = []
        reward_list = []
        steps = 0
        value_target_t = 0
        while True:
            steps += 1
            if episode % render_rate == 0:
                env.render()
            # get action and probability distribution
            state = torch.from_numpy(state).type(torch.FloatTensor).unsqueeze(0).to(device)
            action_t, action_prob_dist, value = model.get_action(state)
            # take step
            state_new, reward, done, info = env.step(action_t.item())

            # TODO: is a tuple really the best DS here? Maybe a dict or named-tuple is better?
            rollout_tuple_t = (state, 
                               action_t.reshape((1, 1)), 
                               torch.tensor(reward, dtype=torch.float).reshape((1, 1)).to(device), 
                               value, 
                               action_prob_dist.log_prob(action_t).reshape((1, 1)),
                               done)
            state = state_new
            rollout_data_list.append(rollout_tuple_t)
            reward_list.append(reward)

            if done:
                # TODO try to make all datasets the same length by resetting to a random state and continuing
                #      this will help with training efficiency by making data length the same for batching           
                value_target_t = 0  # TODO: revisit page 15 of the Bick paper to look at this estimation when the trajectory ends non-terminally
                if episode % 25 == 0:
                    sys.stdout.write("episode: {}, total reward: {}, length: {}\n".format(episode,
                                                                                          np.round(np.sum(reward_list), decimals=3),
                                                                                          steps))
                    wandb.log({'episode': episode, 
                               'total reward': np.round(np.sum(reward_list), decimals=3)})
                if len(model.training_data_raw) < 500:   
                    model.store_rollout(rollout_data_list, value_target_t)                     
                    state = env.reset()
                    rollout_data_list = []
                    reward_list = []
                    steps = 0
                    value_target_t = 0
                    continue
                break
        model.learn()
