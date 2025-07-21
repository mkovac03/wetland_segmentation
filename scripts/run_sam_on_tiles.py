import numpy as np
import torch
import os
import cv2
from glob import glob
from tqdm import tqdm
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor

DEVICE = "cuda"
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = "models/sam_hq_vit_h.pth"
RGB_BANDS = [100, 60, 50]
IMAGE_SIZE = 512

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)
predictor = SamPredictor(sam)

with open("data/pseudo/seed_tiles.txt") as f:
    tile_paths = [l.strip() for l in f]

os.makedirs("outputs/sam_masks", exist_ok=True)

for img_path in tqdm(tile_paths):
    base = os.path.basename(img_path).replace("_img.npy", "")
    lbl_path = img_path.replace("_img.npy", "_lbl.npy")

    img = np.load(img_path)[RGB_BANDS]
    label = np.load(lbl_path)
    img = np.transpose(img, (1, 2, 0)).astype(np.uint8)

    slic = cv2.ximgproc.createSuperpixelSLIC(img, region_size=20, ruler=10.0)
    slic.iterate(10)
    labels = slic.getLabels()

    prompts = []
    for seg_val in np.unique(labels):
        ys, xs = np.where(labels == seg_val)
        if len(xs) == 0:
            continue
        center = [int(xs.mean()), int(ys.mean())]
        prompts.append(center)

    input_points = np.array(prompts)
    input_labels = np.ones(len(prompts))

    predictor.set_image(img)
    masks, scores, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=False
    )

    final_mask = np.full((IMAGE_SIZE, IMAGE_SIZE), 255, dtype=np.uint8)
    for mask in masks:
        masked = label[mask]
        if len(masked) == 0:
            continue
        values, counts = np.unique(masked, return_counts=True)
        cls = values[np.argmax(counts)]
        final_mask[mask] = cls

    out_path = f"data/pseudo/{base}_lbl.npy"
    np.save(out_path, final_mask)

    overlay = img.copy()
    overlay[final_mask != 255] = 0.6 * img[final_mask != 255] + 0.4 * np.array([255, 0, 0])
    Image.fromarray(overlay.astype(np.uint8)).save(f"outputs/sam_masks/{base}_overlay.png")
