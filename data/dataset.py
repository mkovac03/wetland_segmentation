import os
import numpy as np
import torch
from torch.utils.data import Dataset
from data.transform import RandomAugment


class GoogleEmbedDataset(Dataset):
    """
    Dataset for loading multiband satellite embedding tiles (.npy format).
    Each sample consists of an image tensor [C, H, W] and a label map [H, W].
    """

    def __init__(self, file_list, transform=None, num_classes=13, expected_channels=None):
        """
        Args:
            file_list (List[str]): List of base paths (without _img.npy/_lbl.npy suffix)
            transform (callable): Optional transform to apply to (image, label)
            num_classes (int): Number of valid label classes (excludes ignore_index)
            expected_channels (int): Expected number of image input channels (e.g., 63); required
        """
        if expected_channels is None:
            raise ValueError("expected_channels must be explicitly provided (e.g. from config['input_channels'])")

        self.file_list = [f.replace("_img.npy", "").replace("_lbl.npy", "") for f in file_list]
        self.transform = transform or RandomAugment()
        self.num_classes = num_classes
        self.expected_channels = expected_channels

        if not self.file_list:
            raise RuntimeError("GoogleEmbedDataset: No .npy file pairs found.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        base = self.file_list[idx]
        img_path = base + "_img.npy"
        lbl_path = base + "_lbl.npy"

        try:
            img = np.load(img_path)
            lbl = np.load(lbl_path)

            # Validate shapes
            if img.ndim != 3 or lbl.ndim != 2:
                raise ValueError(f"Invalid dimensions: img {img.shape}, lbl {lbl.shape}")
            if img.shape[1:] != lbl.shape:
                raise ValueError(f"Spatial shape mismatch: img {img.shape}, lbl {lbl.shape}")
            if img.shape[0] != self.expected_channels:
                raise ValueError(f"Channel count mismatch: expected {self.expected_channels}, got {img.shape[0]}")

            # Validate labels
            max_lbl = lbl.max()
            if max_lbl >= self.num_classes and max_lbl != 255:
                raise ValueError(f"Label out of bounds: max = {max_lbl}")

            img = torch.from_numpy(img).float()
            lbl = torch.from_numpy(lbl).long()

            if self.transform:
                img, lbl = self.transform(img, lbl)

            return img, lbl

        except Exception as e:
            print(f"[SKIPPED] {base}: {e}")
            return None  # let collate_fn skip invalid samples


def get_file_list(processed_dir):
    """
    Scans directory for *_img.npy and *_lbl.npy pairs and returns base paths.
    """
    files = []
    for f in os.listdir(processed_dir):
        if f.endswith("_img.npy"):
            base = f.replace("_img.npy", "")
            img_path = os.path.join(processed_dir, base + "_img.npy")
            lbl_path = os.path.join(processed_dir, base + "_lbl.npy")
            if os.path.exists(lbl_path):
                files.append(os.path.join(processed_dir, base))
    return sorted(files)
