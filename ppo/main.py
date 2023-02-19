
import sys
import gym
import torch
from ppo.ppo import PPO
import ppo.policy
import ppo.value_network
import numpy as np

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # init environment
    env = gym.make("CartPole-v1")
    action_space_size = env.action_space.n
    obs_space_size = env.observation_space.shape[0]

    # init models
    policy = ppo.policy.Policy(action_space_size, obs_space_size)
    value_fcn = ppo.value_network.ValueNet(obs_space_size)
    # TODO: check if this implementation for optimizing both sets of weights is valid
    optimizer = torch.optim.SGD(list(policy.parameters()) + list(value_fcn.parameters()), 
                                lr=1e-3, 
                                momentum=0.9)
    # loss_fn = torch.nn.MSELoss()
    model = PPO(policy, value_fcn, optimizer, action_space_size)

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
            action_t, action_prob_dist = model.get_action(torch.from_numpy(state).type(torch.FloatTensor).to(device))
            # take step
            state_new, reward, done, info = env.step(action_t)

            # TODO: is a tuple really the best DS here? Maybe a dict or named-tuple is better?
            rollout_tuple_t = (torch.from_numpy(state).type(torch.FloatTensor).to(device), 
                               torch.tensor(action_t, dtype=torch.int8).to(device),  
                               torch.tensor(reward, dtype=torch.float).to(device), 
                               torch.from_numpy(state_new).type(torch.FloatTensor).to(device), 
                               action_prob_dist[action_t])
            state = state_new
            rollout_data_list.append(rollout_tuple_t)
            reward_list.append(reward)

            if done:
                # TODO try to make all datasets the same length by resetting to a random state and continuing
                #      this will help with training efficiency by making data length the same for batching           
                value_target_t = 0  # TODO: revisit page 15 of the Bick paper to look at this estimation when the trajectory ends non-terminally
                if episode % 1 == 0:
                    sys.stdout.write("episode: {}, total reward: {}, length: {}\n".format(episode,
                                                                                          np.round(np.sum(reward_list), decimals=3),
                                                                                          steps))
                break
        model.store_rollout(rollout_data_list, value_target_t)
        model.learn()
