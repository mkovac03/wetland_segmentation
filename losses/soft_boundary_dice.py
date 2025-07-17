# File: losses/soft_boundary_dice.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import numpy as np

class SoftBoundaryDiceLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        num_classes = probs.shape[1]
        loss = 0.0
        count = 0

        for c in range(num_classes):
            if c == self.ignore_index:
                continue

            pred = probs[:, c, :, :]              # shape: [B, H, W]
            gt = (targets == c).float()           # shape: [B, H, W]

            # Compute signed distance transform for each sample in batch
            dt_batch = []
            for b in range(gt.shape[0]):
                gt_np = gt[b].cpu().numpy()
                if gt_np.max() == 0:
                    dt = np.zeros_like(gt_np)
                else:
                    pos_dist = distance_transform_edt(gt_np)
                    neg_dist = distance_transform_edt(1 - gt_np)
                    dt = neg_dist - pos_dist
                dt_batch.append(torch.from_numpy(dt).to(pred.device))

            dt_tensor = torch.stack(dt_batch)     # shape: [B, H, W]

            # Normalize per sample
            dt_min = dt_tensor.view(gt.shape[0], -1).min(dim=1)[0].view(-1, 1, 1)
            dt_max = dt_tensor.view(gt.shape[0], -1).max(dim=1)[0].view(-1, 1, 1)
            dt_tensor = (dt_tensor - dt_min) / (dt_max - dt_min + 1e-8)

            intersect = (pred * dt_tensor).sum(dim=(1, 2))
            denom = pred.pow(2).sum(dim=(1, 2)) + dt_tensor.pow(2).sum(dim=(1, 2)) + 1e-6
            dice = 2.0 * intersect / denom
            class_loss = (1 - dice).mean()

            loss += class_loss
            count += 1

        return loss / count if count > 0 else torch.tensor(0.0, device=logits.device)
