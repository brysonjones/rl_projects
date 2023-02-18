
import sys
import gym
import torch
from ppo.ppo import PPO
import ppo.policy
import ppo.value_network
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

    # init models
    policy = ppo.policy.Policy(action_space_size, obs_space_size)
    value_fcn = ppo.value_network.ValueNet(action_space_size, obs_space_size)
    # TODO: check if this implementation for optimizing both sets of weights is valid
    optimizer = torch.optim.SGD(list(policy.parameters()) + list(value_fcn.parameters()), 
                                lr=1e-3, 
                                momentum=0.9)
    loss_fn = torch.nn.MSELoss()
    model = PPO(policy, optimizer)

    render_rate = 100
    num_episodes = 5000
    for e in range(num_episodes):
        state = env.reset()
        rollout_data_list = []
        steps = 0
        value_target_t = 0
        while True:
            steps += 1
            if e % render_rate == 0:
                env.render()
            # get action with highest prob
            action_probs = model(state)
            highest_prob_action = np.random.choice(env.action_space.n, p=np.squeeze(action_probs.detach().numpy()))
            # log_prob = torch.log(action_probs.squeeze(0)[highest_prob_action]) TODO: determine if we need this?
            # take step
            state_new, reward, done, info = env.step(highest_prob_action)

            rollout_tuple_t = (state, highest_prob_action, reward, action_probs)
            state = state_new
            rollout_data_list.append(reward)

            if done:
                # TODO try to make all datasets the same length by resetting to a random state and continuing
                #      this will help with training efficiency by making data length the same for batching           
                value_target_t = model.get_value(rollout_data_list[-1][0])
                if e % 1 == 0:
                    sys.stdout.write("episode: {}, total reward: {}, length: {}\n".format(e,
                                                                                          np.round(np.sum(reward_list), decimals=3),
                                                                                          steps))
                break
        for t in range(len(rollout_data_list)-1, -1, -1):
            r_t = rollout_data_list[t]
            value_target_t = 0
            for t_discount in range(len(rollout_data_list)):
                r_t = rollout_data_list[t_discount]