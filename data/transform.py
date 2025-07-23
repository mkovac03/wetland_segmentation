# File: data/transform.py
import torch
import random
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import numpy as np

class RandomAugment:
    def __init__(self, p=0.5, noise_std=0.01, channel_dropout_p=0.1, blur_p=0.2):
        self.p = p
        self.noise_std = noise_std
        self.channel_dropout_p = channel_dropout_p
        self.blur_p = blur_p

    def __call__(self, img, lbl):
        # img: [C, H, W], lbl: [H, W]

        # ----- Geometric augmentations -----
        if random.random() < self.p:
            img = TF.hflip(img)
            lbl = TF.hflip(lbl)

        if random.random() < self.p:
            img = TF.vflip(img)
            lbl = TF.vflip(lbl)

        if random.random() < self.p:
            angle = random.choice([90, 180, 270])
            img = TF.rotate(img, angle)

            lbl = lbl.unsqueeze(0).float()
            lbl = TF.rotate(lbl, angle, interpolation=TF.InterpolationMode.NEAREST)
            lbl = lbl.squeeze(0).long()

        # ----- Channel dropout -----
        if random.random() < self.channel_dropout_p:
            num_channels = img.shape[0]
            n_drop = random.randint(1, min(3, num_channels // 4))
            drop_indices = random.sample(range(num_channels), n_drop)
            for i in drop_indices:
                img[i] = 0

        # ----- Additive Gaussian noise -----
        if random.random() < self.p:
            noise = torch.randn_like(img) * self.noise_std
            img = img + noise

        # ----- Brightness/contrast jitter -----
        if random.random() < self.p:
            img = TF.adjust_brightness(img, brightness_factor=random.uniform(0.9, 1.1))
            img = TF.adjust_contrast(img, contrast_factor=random.uniform(0.9, 1.1))

        # ----- Gaussian blur -----
        if random.random() < self.blur_p:
            blur = T.GaussianBlur(kernel_size=3)
            img = blur(img)

        return img, lbl
