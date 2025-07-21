import numpy as np
import joblib
from glob import glob
from tqdm import tqdm
import umap
import hdbscan
import os
import matplotlib.pyplot as plt

os.makedirs("data/pseudo", exist_ok=True)

img_paths = sorted(glob("data/processed/*_img.npy"))
vecs = np.stack([np.load(p).mean(axis=(1, 2)) for p in tqdm(img_paths)], dtype=np.float32)

umap_model = umap.UMAP(n_neighbors=40, min_dist=0.1, metric='cosine').fit(vecs)
emb = umap_model.embedding_

labels = hdbscan.HDBSCAN(min_cluster_size=50).fit_predict(emb)

plt.scatter(emb[:,0], emb[:,1], c=labels, cmap="tab20", s=5)
plt.savefig("outputs/umap_clusters.png", dpi=300)

subset_paths = []
for c in np.unique(labels[labels >= 0]):
    idxs = np.where(labels == c)[0]
    sub_vecs = vecs[idxs]
    dists = ((sub_vecs[:, None, :] - sub_vecs[None, :, :]) ** 2).sum(-1)
    medoid = idxs[np.argmin(dists.sum(1))]
    subset_paths.append(img_paths[medoid])

with open("data/pseudo/seed_tiles.txt", "w") as f:
    for p in subset_paths:
        f.write(p + "\n")

print(f"[INFO] Selected {len(subset_paths)} tiles for SAM pseudo-labeling.")
