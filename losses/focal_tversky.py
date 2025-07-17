import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import binary_dilation
from .soft_boundary_dice import SoftBoundaryDiceLoss

class CombinedFocalTverskyLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        focal_alpha=0.25,
        focal_gamma=1.0,
        tversky_alpha=0.5,
        tversky_beta=0.7,
        use_boundary=True,
        boundary_weight=0.2,
        ignore_index=255
    ):
        super().__init__()
        self.num_classes = num_classes
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.boundary_weight = boundary_weight
        self.ignore_index = ignore_index
        self.use_boundary = use_boundary

        if use_boundary:
            self.boundary_loss_fn = SoftBoundaryDiceLoss(ignore_index=ignore_index)

    def forward(self, inputs, targets):
        device = inputs.device
        valid_mask = (targets != self.ignore_index).float().to(device)

        # --- Focal Cross Entropy ---
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss

        # Boundary weighting
        if self.use_boundary:
            boundary_mask = self._compute_boundary_mask(targets).to(device)
            weight_map = torch.where(boundary_mask.bool(), self.boundary_weight, 1.0)
        else:
            weight_map = torch.ones_like(valid_mask)

        focal = focal * valid_mask * weight_map
        focal_loss = focal.sum() / (valid_mask * weight_map).sum().clamp(min=1e-8)

        # --- Tversky Loss ---
        probs = F.softmax(inputs, dim=1)
        one_hot = F.one_hot(
            targets.clamp(0, self.num_classes - 1),
            num_classes=self.num_classes
        ).permute(0, 3, 1, 2).float().to(device)

        dims = (0, 2, 3)
        TP = (probs * one_hot).sum(dims)
        FP = (probs * (1 - one_hot)).sum(dims)
        FN = ((1 - probs) * one_hot).sum(dims)

        tversky = (TP + 1e-6) / (TP + self.tversky_alpha * FP + self.tversky_beta * FN + 1e-6)
        tversky_loss = 1 - tversky.mean()

        # --- Combine ---
        total_loss = 0.5 * focal_loss + 0.5 * tversky_loss

        # --- Optional Boundary Loss ---
        if self.use_boundary:
            boundary_loss = self.boundary_loss_fn(inputs, targets)
            total_loss += self.boundary_weight * boundary_loss

        return total_loss

    def _compute_boundary_mask(self, targets):
        """
        Dilate the valid (non-ignore) mask to form a soft boundary around valid areas.
        """
        b, h, w = targets.shape
        boundary = torch.zeros((b, h, w), dtype=torch.float32)

        for i in range(b):
            t = targets[i].detach().cpu().numpy()
            t_mask = (t != self.ignore_index)
            t_dilated = binary_dilation(t_mask, iterations=1)
            edges = t_dilated & ~t_mask
            boundary[i] = torch.from_numpy(edges.astype(np.float32))

        return boundary
