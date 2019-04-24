import numpy as np
import torch
from torch.utils.data.dataset import Dataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FrameDataset(Dataset):
    def __init__(self, data, labels=None):
        self.data = data
        self.labels = labels
        self.test = labels is None
        self.size = len(data)
        
        print('Dataset Device:', DEVICE)

    def __getitem__(self, index):
        instance = torch.from_numpy(self.data[index])
        if self.test:
            return instance.float().to(DEVICE)
        label_torch = torch.from_numpy(self.labels[index])
        return instance.float().to(DEVICE), label_torch.to(DEVICE)

    # Override to give PyTorch size of dataset
    def __len__(self):
        return self.size