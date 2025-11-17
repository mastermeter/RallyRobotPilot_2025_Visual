import torch
from torch.utils.data import Dataset
import numpy as np

from .process import normalize_image_ndarray 


class RoboDataset(Dataset):
    def __init__(self, all_imgs, all_ctrls, seq_len=8, delta_frames=1):
        self.all_imgs = all_imgs
        self.all_ctrls = all_ctrls
        self.seq_len = int(seq_len)
        self.delta = max(1, int(delta_frames))

        # index global: liste de (file_idx, t_end)
        self.index = []
        for f_idx, imgs in enumerate(self.all_imgs):
            N = len(imgs)
            if N <= 0:
                continue
            t_min = (self.seq_len - 1) * self.delta
            for t in range(t_min, N):
                self.index.append((f_idx, t))

        print(f"RoboDataset: {len(self.index)} sequence samples.")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        f_idx, t_end = self.index[idx]
        imgs = self.all_imgs[f_idx]
        ctrls = self.all_ctrls[f_idx]

        # indices de la séquence (en remontant de delta_frames)
        idxs = [t_end - i * self.delta for i in range(self.seq_len)][::-1]

        seq = []
        for i in idxs:
            img = imgs[i]                       # (H, W, 3) uint8
            img = normalize_image_ndarray(img)  # float32 [0,1]
            seq.append(img)

        seq = np.stack(seq, axis=0)            # (T, H, W, 3)
        seq = np.transpose(seq, (0, 3, 1, 2))  # (T, C, H, W)

        f, b, l, r = ctrls[t_end].astype(np.float32)
        lab = np.array([f, l, r], dtype=np.float32)

        seq_tensor = torch.tensor(seq, dtype=torch.float32)
        lab_tensor = torch.tensor(lab, dtype=torch.float32)
        return seq_tensor, lab_tensor
