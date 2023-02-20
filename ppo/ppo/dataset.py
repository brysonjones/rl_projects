
import torch
from torch.utils.data import Dataset, DataLoader

class PolicyDataset(Dataset):
    """PPO dataset"""

    def __init__(self, data):
        self.rollout_data = data

    def __len__(self):
        return len(self.rollout_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

       

        return sample