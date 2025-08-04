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
            transform (callable): Optional transform applied to input image and label
            check_files (bool): If True, performs label range validation
            num_classes (int): Total number of semantic classes (excluding 255)
        """
        self.transform = transform or RandomAugment()
        self.num_classes = num_classes or 13
        self.file_list = []

        if check_files:
            print("[INFO] Checking for invalid label values...")
            for path in tqdm(file_list, desc="Validating labels"):
                lbl_path = path + "_lbl.npy"
                try:
                    lbl = np.load(lbl_path)
                    invalid_mask = (lbl != 255) & ((lbl < 0) | (lbl >= self.num_classes))
                    if np.any(invalid_mask):
                        bad_vals = np.unique(lbl[invalid_mask])
                        print(f"[DEBUG] {lbl_path} → invalid labels: {bad_vals}")
                        print(f"[WARN] Skipping {path} — label out of range (excluding 255)")
                        continue
                except Exception as e:
                    print(f"[ERROR] Could not load {lbl_path}: {e}")
                    continue

                self.file_list.append(path)
            print(f"[INFO] {len(self.file_list)} valid tiles retained.")
        else:
            # Only check existence
            for base in file_list:
                img_path = base + "_img.npy"
                lbl_path = base + "_lbl.npy"
                if os.path.exists(img_path) and os.path.exists(lbl_path):
                    self.file_list.append(base)

        if not self.file_list:
            raise RuntimeError("GoogleEmbedDataset: No valid .npy file pairs found.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        base = self.file_list[idx]
        img = np.load(base + "_img.npy")  # shape: [C, H, W]
        lbl = np.load(base + "_lbl.npy")  # shape: [H, W]

        if img.shape[1:] != lbl.shape:
            raise ValueError(f"[ERROR] Shape mismatch at {base}: image {img.shape}, label {lbl.shape}")

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
