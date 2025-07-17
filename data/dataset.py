# File: data/dataset.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from data.transform import basic_augmentation  # make sure this is defined

class GoogleEmbedDataset(Dataset):
    """
    Dataset for loading Google Satellite Embedding .npy files.

    Each sample consists of a multiband input image tensor and a single-channel label map.
    """
    def __init__(self, file_list, transform=None, check_files=True):
        """
        Args:
            file_list (List[str]): Base paths (excluding _img.npy/_lbl.npy) to samples
            transform (callable): Optional transform applied to input image (not labels)
            check_files (bool): If True, skip any base without both _img.npy and _lbl.npy
        """
        self.transform = transform or basic_augmentation()
        self.file_list = []

        for base in file_list:
            img_path = base + '_img.npy'
            lbl_path = base + '_lbl.npy'

            if check_files and (not os.path.exists(img_path) or not os.path.exists(lbl_path)):
                continue

            self.file_list.append(base)

        if not self.file_list:
            raise RuntimeError("GoogleEmbedDataset: No valid .npy file pairs found.")

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
            img, lbl = self.transform(img, lbl)

        return img, lbl


def get_file_list(processed_dir):
    """
    List all valid image/label base filenames from a processed directory.

    Args:
        processed_dir (str): Directory containing *_img.npy and *_lbl.npy pairs

    Returns:
        List[str]: Base paths for all valid image/label pairs
    """
    files = [
        os.path.join(processed_dir, f[:-8])
        for f in os.listdir(processed_dir)
        if f.endswith('_img.npy') and os.path.exists(os.path.join(processed_dir, f[:-8] + '_lbl.npy'))
    ]
    return sorted(files)
