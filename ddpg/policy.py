
import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, obs_space_size, action_space_size,
                 num_hidden, num_layers, activation='ReLU'):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_activation_stack = nn.Sequential(

        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_activation_stack(x)
        return logits