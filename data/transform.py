# File: data/transform.py
import torch
import random
import torchvision.transforms.functional as TF

class RandomFlipRotate:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, lbl):
        # img: [C, H, W], lbl: [H, W]
        if random.random() < self.p:
            img = TF.hflip(img)
            lbl = TF.hflip(lbl)

        if random.random() < self.p:
            img = TF.vflip(img)
            lbl = TF.vflip(lbl)

        if random.random() < self.p:
            angle = random.choice([90, 180, 270])
            img = TF.rotate(img, angle)

            # Convert lbl to fake "channel" format before rotation
            lbl = lbl.unsqueeze(0).float()
            lbl = TF.rotate(lbl, angle, interpolation=TF.InterpolationMode.NEAREST)
            lbl = lbl.squeeze(0).long()

        return img, lbl

