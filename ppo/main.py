
import sys
import gym
import torch
from ppo.ppo import PPO_Agent
import plotting
import numpy as np
import wandb

if __name__ == "__main__":
    wandb.init(project="rl-ppo-project")
    config = wandb.config
    config.learning_rate = 4e-4
    config.num_hidden_neurons = 256
    config.num_hidden_layers = 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # init environment
    env = gym.make("CartPole-v1")
    action_space_size = env.action_space.n
    obs_space_size = env.observation_space.shape[0]
    num_steps = 500  # TODO: change this for not cartpoleV0

    model = PPO_Agent(obs_space_size, action_space_size, num_steps=num_steps)
    model.to(device)

    render_rate = 100
    num_episodes = 5000
    cum_rewards = 0
    total_finished_episodes = 0
    rolling_rewards_list = []
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
                total_finished_episodes += 1           
                cum_rewards = (cum_rewards * (total_finished_episodes-1) + np.sum(reward_list)) / (total_finished_episodes)
                if episode % 25 == 0:
                    sys.stdout.write("episode: {}, rolling rewards: {}\n".format(episode, cum_rewards))
                state = env.reset()
                reward_list = []
                steps = 0
                continue
        wandb.log({'episode': episode, 
                   'rolling rewards': cum_rewards})
        rolling_rewards_list.append(cum_rewards)
        returns, advantages = model.calc_advantage(state_new, done, num_steps)
        model.learn(num_steps, returns, advantages)

    plotting.plot_rolling_rewards(rolling_rewards_list)
