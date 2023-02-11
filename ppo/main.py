
import sys
import gym
import torch
import ppo.ppo
import ppo.policy
import wandb
import numpy as np

if __name__ == "__main__":
    # init wandb logger
    # wandb.init(project="dqn-dev")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # init environment
    env = gym.make("CartPole-v1")
    action_space_size = env.action_space.n
    obs_space_size = env.observation_space

    # init model
    policy = ppo.policy.Policy(action_space_size, obs_space_size)
    optimizer = torch.optim.SGD(network.parameters(), lr=1e-3, momentum=0.9)
    loss_fn = torch.nn.MSELoss()
    model = ppo.ppo.PPO(policy, optimizer)

    render_rate = 100

    num_episodes = 5000
    for e in range(num_episodes):
        state = env.reset()
        log_prob_list = []
        reward_list = []
        steps = 0
        while True:
            steps += 1
            if e % render_rate == 0:
                env.render()
            # get action with highest prob
            action_probs = # TODO
            highest_prob_action = np.random.choice(env.action_space.n, p=np.squeeze(action_probs.detach().numpy()))
            log_prob = torch.log(action_probs.squeeze(0)[highest_prob_action])
            # take step
            state, reward, done, info = env.step(highest_prob_action)

            log_prob_list.append(log_prob)
            reward_list.append(reward)

            if done:
                # update policy
                if e % 1 == 0:
                    sys.stdout.write("episode: {}, total reward: {}, length: {}\n".format(e,
                                                                                          np.round(np.sum(reward_list), decimals=3),
                                                                                          steps))
                break