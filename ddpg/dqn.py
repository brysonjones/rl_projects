
import torch
import torch.nn as nn
import layers

class DQN(nn.Module):
    def __init__(self, obs_space_size, action_space_size,
                 num_hidden, num_layers, activation='ReLU'):
        super().__init__()


        layer_list = []
        for i_layer in range(num_layers):
            if len(layer_list) == 0:
                layer_list.append(nn.Linear(obs_space_size, num_hidden))
                layer_list.append(layers.activation_dict[activation]())
            else:
                layer_list.append(nn.Linear(num_hidden, num_hidden))
                layer_list.append(layers.activation_dict[activation]())

        layer_list.append((nn.Linear(num_hidden, action_space_size)))

        self.linear_activation_stack = nn.Sequential(nn.Flatten(), 
                                                     *layer_list)
                                                
    def forward(self, x):
        logits = self.linear_activation_stack(x)
        return logits