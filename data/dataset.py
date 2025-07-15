import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class GoogleEmbedDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        base = self.file_list[idx]
        img = np.load(base + '_img.npy')  # Expected shape: [C, H, W]
        lbl = np.load(base + '_lbl.npy')  # Expected shape: [H, W]

        if img.shape[1:] != lbl.shape:
            raise ValueError(f"Shape mismatch at {base}: img shape {img.shape}, label shape {lbl.shape}")

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        if self.transform:
            img = self.transform(img)

        return img, lbl


def get_file_list(processed_dir):
    files = [os.path.join(processed_dir, f[:-8]) for f in os.listdir(processed_dir) if f.endswith('_img.npy')]
    return sorted(files)
