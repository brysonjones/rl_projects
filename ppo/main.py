
import gym
import torch
import ppo.ppo
import ppo.network
import wandb

if __name__ == "__main__":
    # init wandb logger
    # wandb.init(project="dqn-dev")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # init environment
    env = gym.make("CartPole-v1")
    action_space_size = env.action_space.n
    obs_space_size = env.observation_space

    # init model
    policy = ppo.network.Policy(action_space_size, obs_space_size)
    optimizer = torch.optim.SGD(network.parameters(), lr=1e-3, momentum=0.9)
    loss_fn = torch.nn.MSELoss()
    model = ppo.network.PPO(policy, optimizer)

    num_episodes = 1000
    for e in range(num_episodes):
        obs = env.reset()
        obs_frame = env.render(mode = "rgb_array")
        state = model.update_state(obs_frame)
        # TODO: init seq
        # TODO: init preprocessed seq
        while True:
            action = model.get_action(state)
            obs, reward, done, _ = env.step(action)
            obs_frame = env.render(mode = "rgb_array")
            state = model.update_state(obs_frame, state, action, reward, done)
            # TODO: sample random mini batch of transitions from memory
            # TODO: set y_j = reward estimate
            # TODO: perform gradient descent step
            model.learn()