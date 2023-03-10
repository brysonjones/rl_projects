
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
        self.batch_size = config["batch_size"]

        self.q_network = dqn.DQN(obs_space_size, action_space_size,
                                 config['network']['num_hidden'],
                                 config['network']['num_layers'],
                                 config['network']['activation'])
        self.policy_network = policy.PolicyNetwork(obs_space_size, action_space_size,
                                                   config['network']['num_hidden'],
                                                   config['network']['num_layers'],
                                                   config['network']['activation'])
        self.action_space_range = action_space_range

        self.replay_buffer = memory.ReplayMemory(config['replay_buffer']['max_size'])

        # copy networks to target networks
        self.target_q_network = copy.deepcopy(self.q_network)
        self.target_policy_network = copy.deepcopy(self.policy_network)

        # add everything to GPU if available
        self.q_network.to(self.device)
        self.policy_network.to(self.device)
        self.target_q_network.to(self.device)
        self.target_policy_network.to(self.device)

        self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), self.config["optimizer"]["lr"])
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), self.config["optimizer"]["lr"])

    def select_action(self, state, add_noise=False):
        state = torch.from_numpy(state).squeeze().float()
        action = self.target_policy_network(state).detach().numpy()
        if add_noise == True:
            action = action + np.random.normal(size=(self.action_space_size))
        action = np.clip(action, 
                         self.action_space_range[0], 
                         self.action_space_range[1])
        return action

    def store_memory(self, *args):
        self.replay_buffer.push(*args)

    def update_target_network_weights(self):  
        rho = self.config["target_network"]["rho"]

        # update target dqn
        dqn_params = self.q_network.named_parameters()
        target_dqn_params = self.target_q_network.named_parameters()

        dict_target_dqn_params = dict(target_dqn_params)

        for name1, param1 in dqn_params:
            if name1 in dict_target_dqn_params:
                dict_target_dqn_params[name1].data.copy_(rho*dict_target_dqn_params[name1].data + 
                                                         (1-rho)*param1.data)

        self.target_q_network.load_state_dict(dict_target_dqn_params)

        # update target policy
        policy_params = self.policy_network.named_parameters()
        target_policy_params = self.target_policy_network.named_parameters()

        dict_target_policy_params = dict(target_policy_params)

        for name1, param1 in policy_params:
            if name1 in dict_target_policy_params:
                dict_target_policy_params[name1].data.copy_(rho*dict_target_policy_params[name1].data + 
                                                            (1-rho)*param1.data)

        self.target_policy_network.load_state_dict(dict_target_policy_params)

    def get_batch(self):
        if len(self.replay_buffer) < self.batch_size:
            return None
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.tensor(batch.state).float()
        action_batch = torch.tensor(batch.action).float()
        reward_batch = torch.tensor(batch.reward).float()
        next_state_batch = torch.tensor(batch.next_state).float()
        done_batch = torch.tensor(batch.done).int()

        return (state_batch, action_batch, reward_batch, \
            done_batch, next_state_batch)

    def update(self):
        # sample batch from memory
        sample = self.get_batch()
        if not sample:
            return 
        state_batch, action_batch, reward_batch, \
                done_batch, next_state_batch = sample
        next__action_batch = self.target_policy_network(next_state_batch)
        
        # compute targets with target networks
        target_state_action_batch = torch.hstack((next_state_batch, next__action_batch))
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
