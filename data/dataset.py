import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class GoogleEmbedDataset(Dataset):
    def __init__(self, file_list, transform=None, check_files=True):
        self.transform = transform
        self.file_list = []

        for base in file_list:
            img_path = base + '_img.npy'
            lbl_path = base + '_lbl.npy'

            if check_files and (not os.path.exists(img_path) or not os.path.exists(lbl_path)):
                continue  # skip missing files

            self.file_list.append(base)

        if len(self.file_list) == 0:
            raise RuntimeError("GoogleEmbedDataset: no valid .npy file pairs found!")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        base = self.file_list[idx]
        img = np.load(base + '_img.npy')  # [C, H, W]
        lbl = np.load(base + '_lbl.npy')  # [H, W]

        if img.shape[1:] != lbl.shape:
            raise ValueError(f"Shape mismatch at {base}: image {img.shape}, label {lbl.shape}")

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        if self.transform:
            img = self.transform(img)

        return img, lbl


def get_file_list(processed_dir):
    """Return sorted list of base filenames (without _img/_lbl suffix)"""
    files = [
        os.path.join(processed_dir, f[:-8])
        for f in os.listdir(processed_dir)
        if f.endswith('_img.npy') and os.path.exists(os.path.join(processed_dir, f[:-8] + '_lbl.npy'))
    ]
    return sorted(files)
