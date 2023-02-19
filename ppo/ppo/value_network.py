
import torch
import torch.nn as nn
import torch.nn.functional as F

class ValueNet(nn.Module):
    def __init__(self, obs_space, num_layers=2, num_hidden=32):
        super(ValueNet, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # specify network architecture
        layer_list = [nn.Linear(obs_space, num_hidden), nn.ReLU()]
        for i in range(num_layers):
            layer_list.append(nn.Linear(num_hidden, num_hidden))
            layer_list.append(nn.ReLU())
        layer_list.append(nn.Linear(num_hidden, 1))

        self._layers = nn.Sequential(*layer_list)


    def forward(self, x):
        x = torch.from_numpy(x).type(torch.FloatTensor).to(self.device)
        return self._layers(x)

    def learn():
        pass
