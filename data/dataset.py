import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from data.transform import RandomAugment


class GoogleEmbedDataset(Dataset):
    """
    Dataset for loading Google Satellite Embedding .npy files.
    Each sample consists of a multiband input image tensor and a single-channel label map.
    """
    def __init__(self, file_list, transform=None, check_files=False, num_classes=None):
        """
        Args:
            file_list (List[str]): Base paths (excluding _img.npy/_lbl.npy) to samples
            transform (callable): Transform that takes (img, lbl) and returns (img, lbl)
            check_files (bool): If True, skip bad files or out-of-range labels
            num_classes (int): Used to filter invalid label values
        """
        self.transform = transform or RandomAugment()
        self.num_classes = num_classes or 13

        if check_files:
            print("[INFO] Checking for invalid label values...")
            clean_list = []
            for path in tqdm(file_list, desc="Validating labels"):
                lbl_path = path + "_lbl.npy"
                if not os.path.exists(path + "_img.npy") or not os.path.exists(lbl_path):
                    continue
                lbl = np.load(lbl_path)
                if np.any((lbl < 0) | (lbl >= self.num_classes)):
                    print(f"[WARN] Skipping {path} — label out of range")
                    continue
                clean_list.append(path)
            self.file_list = clean_list
            print(f"[INFO] {len(self.file_list)} valid tiles retained.")
        else:
            self.file_list = [
                base for base in file_list
                if os.path.exists(base + "_img.npy") and os.path.exists(base + "_lbl.npy")
            ]

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

        if self.num_classes is not None and np.any((lbl < 0) | (lbl >= self.num_classes)):
            raise ValueError(f"[ERROR] Invalid label values in {base}")

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        if self.transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl
