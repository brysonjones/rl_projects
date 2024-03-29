
import sys
import torch
import dqn
import policy
import memory
from memory import Transition
import copy
import numpy as np

class DDPG():
    def __init__(self, obs_space_size, action_space_size, 
                 action_space_range, config) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.action_space_size = action_space_size
        self.action_space_range = action_space_range
        self.action_scale = (action_space_range[1] - action_space_range[0]) / 2.0
        self.noise_scale = self.action_scale * self.config["noise_scale"]
        self.batch_size = config["batch_size"]

        self.q_network = dqn.DQN(obs_space_size, action_space_size,
                                 config['network']['num_hidden'],
                                 config['network']['num_layers'],
                                 config['network']['activation'])
        self.policy_network = policy.PolicyNetwork(obs_space_size, action_space_size, self.action_scale,
                                                   config['network']['num_hidden'],
                                                   config['network']['num_layers'],
                                                   config['network']['activation'])

        self.replay_buffer = memory.ReplayMemory(config['replay_buffer']['max_size'], config["random_seed"])

        # copy networks to target networks
        self.target_q_network = copy.deepcopy(self.q_network)
        self.target_policy_network = copy.deepcopy(self.policy_network)

        # add everything to GPU if available
        self.q_network.to(self.device)
        self.policy_network.to(self.device)
        self.target_q_network.to(self.device)
        self.target_policy_network.to(self.device)

        self.q_optimizer = torch.optim.Adam(list(self.q_network.parameters()), self.config["optimizer"]["lr"])
        self.policy_optimizer = torch.optim.Adam(list(self.policy_network.parameters()), self.config["optimizer"]["lr"])

    def select_action(self, state, add_noise=False):
        state = torch.from_numpy(state).squeeze().float().to(self.device)
        action = self.target_policy_network(state).cpu().detach().numpy()
        if add_noise == True:
            action = action + np.random.normal(loc=0, scale=self.noise_scale, size=(self.action_space_size))
        action = np.clip(action, 
                         self.action_space_range[0], 
                         self.action_space_range[1])
        return action

    def store_memory(self, *args):
        self.replay_buffer.push(*args)

    def update_target_network_weights(self):  
        rho = self.config["target_network"]["rho"]

        # update the target network
        for param, target_param in zip(self.policy_network.parameters(), self.target_policy_network.parameters()):
            target_param.data.copy_((1-rho) * param.data + rho * target_param.data)
        for param, target_param in zip(self.q_network.parameters(), self.target_q_network.parameters()):
            target_param.data.copy_((1-rho) * param.data + rho * target_param.data)

    def get_batch(self):
        if len(self.replay_buffer) < self.batch_size:
            return None
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.tensor(np.array(batch.state)).float().to(self.device)
        action_batch = torch.tensor(np.array(batch.action)).float().to(self.device)
        reward_batch = torch.tensor(np.array(batch.reward)).float().to(self.device)
        next_state_batch = torch.tensor(np.array(batch.next_state)).float().to(self.device)
        done_batch = torch.tensor(batch.done).int().to(self.device)

        return (state_batch, action_batch, reward_batch, \
            done_batch, next_state_batch)

    def update(self):
        # sample batch from memory
        sample = self.get_batch()
        if not sample:
            return 
        state_batch, action_batch, reward_batch, \
                done_batch, next_state_batch = sample

        with torch.no_grad():
            next_action_batch = self.target_policy_network(next_state_batch)
            # compute targets with target networks
            target_state_action_batch = torch.hstack((next_state_batch, next_action_batch))
            y_target = reward_batch + self.config["discount_gamma"] * (1 - done_batch) * self.target_q_network(target_state_action_batch)
        
        # update q function
        state_action_batch = torch.hstack((state_batch, action_batch))
        q_pred = self.q_network(state_action_batch)
        q_loss = torch.nn.functional.mse_loss(q_pred, y_target)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # update policy
        state_policy_output_batch = torch.hstack((state_batch, self.policy_network(state_batch)))
        policy_loss = -self.q_network(state_policy_output_batch).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # update target networks with polyak-ing
        self.update_target_network_weights()

        # print loss
        # sys.stdout.write("policy_loss: {}\n".format(policy_loss))
        # sys.stdout.write("q_loss: {}\n".format(q_loss))

