import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class RoboDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features  
        self.labels = labels      

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        img = self.features[idx]          
        label = self.labels[idx]           

        img = np.transpose(img, (2, 0, 1))

        img_tensor = torch.tensor(img, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return img_tensor, label_tensor