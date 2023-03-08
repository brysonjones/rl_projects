
import torch
import dqn
import policy
import memory
from memory import Transition
import copy
import numpy as np

class DDPG():
    def __init__(self, obs_space_size, action_space_size, 
                 action_space_range, config, mode="train") -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.action_space_size = action_space_size
        self.mode = mode
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


    def select_action(self, state):
        state = torch.from_numpy(state).squeeze()
        action = torch.clamp(self.target_policy_network(state), 
                             self.action_space_range[0], 
                             self.action_space_range[1]).numpy()
        if self.mode == 'train':
            action = action + np.random.normal(size=(self.action_space_size))
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
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is False,
                                            batch.done)), device=self.device, 
                                            dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.done
                                                    if s is False])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

    def update(self):
        # sample batch from memory
        # compute targets with target networks
        batch.
        # update q function
        # update policy
        # update target networks with polyak-ing
