
import dqn
import policy
import memory

class DDPG():
    def __init__(self, obs_space_size, action_space_size, 
                 action_space_range, config) -> None:
        
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
        