# rallyrobopilot/ml/dataset.py
import torch
from torch.utils.data import Dataset

class RallyDataset(Dataset):
    """Custom Dataset for Rally Robopilot recordings."""
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]