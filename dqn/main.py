
import gym
import torch
from network.network import DQN
from model import MainModel
import wandb

if __name__ == "__main__":
    # init wandb logger
    # wandb.init(project="dqn-dev")

    # init environment
    env = gym.make("CartPole-v0")
    action_space_size = env.action_space.n
    obs_space_size = env.observation_space

    # init model
    # network = DQN(action_space_size, obs_space_size)
    # optimizer = torch.optim.SGD(network.parameters(), lr=1e-3, momentum=0.9)
    # loss_fn = torch.nn.MSELoss()
    # model = MainModel(network)
    obs = env.reset()
    frane = env.render(mode = "rgb_array")
    import time
    time.sleep(20)

    num_episodes = 1000
    for e in range(num_episodes):
        obs = env.reset()
        # TODO: init seq
        # TODO: init preprocessed seq
        while True:
            action = model.get_action()
            obs, reward, done, _ = env.step(action)
            # TODO: update state
            # TODO: preprocess state
            # TODO: store transition in replay memory
            # TODO: sample random mini batch of transitions from memory
            # TODO: set y_j = reward estimate
            # TODO: perform gradient descent step


