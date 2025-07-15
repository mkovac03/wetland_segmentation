# File: train/metrics.py
import torch
import numpy as np
from sklearn.metrics import f1_score

def compute_miou(preds, labels, num_classes):
    ious = []
    for cls in range(num_classes):
        intersection = sum(((p == cls) & (l == cls)).sum().item() for p, l in zip(preds, labels))
        union = sum(((p == cls) | (l == cls)).sum().item() for p, l in zip(preds, labels))
        if union == 0:
            continue
        ious.append(intersection / union)
    return np.mean(ious) if ious else 0.0

def compute_f1(preds, labels, num_classes):
    preds_flat = torch.cat([p.flatten() for p in preds]).numpy()
    labels_flat = torch.cat([l.flatten() for l in labels]).numpy()
    return f1_score(labels_flat, preds_flat, average='macro')
