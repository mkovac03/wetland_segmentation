# File: data/generate_pseudo_labels.py

import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import UMAP
from tqdm import tqdm
import argparse
import yaml


def extract_features(tile_tensor):
    # Use spectral features directly (C, H, W) → (H, W, C)
    return tile_tensor.transpose(1, 2, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output_dir", default="data/pseudo_labels")
    parser.add_argument("--num_clusters", type=int, default=10)
    parser.add_argument("--tile_limit", type=int, default=50)
    parser.add_argument("--reduce_dim", action="store_true")
    parser.add_argument("--umap_dim", type=int, default=10)
    parser.add_argument("--record_tiles", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    processed_dir = config["processed_dir"].format(now="").rstrip("/")
    file_list = [f[:-8] for f in os.listdir(processed_dir) if f.endswith("_img.npy")]
    file_list = sorted(file_list)[:args.tile_limit]

    all_pixels = []
    pixel_counts = []
    tile_shapes = []
    recorded_basenames = []

    for base in tqdm(file_list, desc="Extracting features"):
        path = os.path.join(processed_dir, base)
        img = np.load(path + "_img.npy")  # shape [C, H, W]
        feat_map = extract_features(img)   # → [H, W, C]

        H, W, C = feat_map.shape
        flat_feats = feat_map.reshape(-1, C)

        all_pixels.append(flat_feats)
        pixel_counts.append(len(flat_feats))
        tile_shapes.append((base, H, W))
        recorded_basenames.append(base)

    stacked = np.vstack(all_pixels)

    if args.reduce_dim:
        print(f"[INFO] Reducing dimensionality with UMAP to {args.umap_dim}D...")
        reducer = UMAP(n_components=args.umap_dim, random_state=42)
        stacked = reducer.fit_transform(stacked)

    print(f"Clustering {stacked.shape[0]} pixels into {args.num_clusters} clusters...")
    kmeans = KMeans(n_clusters=args.num_clusters, random_state=42).fit(stacked)

    os.makedirs(args.output_dir, exist_ok=True)
    start = 0
    for (base, H, W), n_pixels in zip(tile_shapes, pixel_counts):
        labels = kmeans.labels_[start:start+n_pixels].reshape(H, W).astype(np.uint8)
        pseudo_path = os.path.join(args.output_dir, base + "_pseudo_lbl.npy")
        np.save(pseudo_path, labels)
        start += n_pixels

    if args.record_tiles:
        registry_path = os.path.join(args.output_dir, "pseudo_tile_list.txt")
        with open(registry_path, "w") as f:
            for name in recorded_basenames:
                f.write(name + "\n")
        print(f"[INFO] Saved list of pseudo-labeled tiles to {registry_path}")

    print("[INFO] Pseudo-label generation complete.")


if __name__ == "__main__":
    main()
