
import sys
import gym
import torch
from ppo.ppo import PPO_Agent
import ppo.policy
import ppo.value_network
import numpy as np
import wandb

if __name__ == "__main__":
    wandb.init(project="rl-ppo-project")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # init environment
    env = gym.make("CartPole-v1")
    action_space_size = env.action_space.n
    obs_space_size = env.observation_space.shape[0]
    num_steps = 500  # TODO: change this for not cartpoleV0

    # init models
    # wandb.init(config={"policy - num_layers": 2, 
    #                    "value_net - num_layers": 2,
    #                    "policy - num_hidden": 64, 
    #                    "value_net - num_hidden": 64, 
    #                    "optimizer - learning_rate": 1e-4})
    # policy = ppo.policy.Policy(action_space_size, obs_space_size, num_layers=1, num_hidden=256)
    # value_fcn = ppo.value_network.ValueNet(obs_space_size, num_layers=1, num_hidden=256)

    model = PPO_Agent(obs_space_size, action_space_size, num_steps=num_steps)
    model.optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
    model.to(device)

    render_rate = 100
    num_episodes = 5000
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        reward_list = []
        steps = 0
        for step in range(num_steps):
            steps += 1
            if episode % render_rate == 0:
                env.render()
            # get action and probability distribution
            state = torch.from_numpy(state).type(torch.FloatTensor).unsqueeze(0).to(device)
            with torch.no_grad():
                action_t, log_probs_t, _, value_est_t = model.get_action(state)
            # take step
            state_new, reward, done, info = env.step(action_t.item())
            model.store_rollout(step, state, action_t, log_probs_t, reward, done, value_est_t)      
            reward_list.append(reward)
            state = state_new

            if done:
                # TODO try to make all datasets the same length by resetting to a random state and continuing
                #      this will help with training efficiency by making data length the same for batching           
                if episode % 25 == 0:
                    sys.stdout.write("episode: {}, total reward: {}, length: {}\n".format(episode,
                                                                                          np.round(np.sum(reward_list), decimals=3),
                                                                                          steps))
                    wandb.log({'episode': episode, 
                               'total reward': np.round(np.sum(reward_list), decimals=3)})
                state = env.reset()
                reward_list = []
                steps = 0
                continue
        returns, advantages = model.calc_advantage(state_new, done, num_steps)
        model.learn(num_steps, returns, advantages)
