import numpy as np
import matplotlib.pyplot as plt
import os

# Path to the .npy file
npy_path = "/media/lkm413/storage11/wetland_segmentation/data/processed/20250806_153626/tile_32630_36603_img.npy"

# Load data
data = np.load(npy_path)
nodata_val = -32768

# Mask nodata for all bands
masked = np.ma.masked_where(data == nodata_val, data)

# Count valid pixels per band
valid_pixel_counts = np.sum(masked.mask == False, axis=(1, 2))
total_pixels = data.shape[1] * data.shape[2]

print(f"Total pixels per band: {total_pixels}")
print("Valid pixels per band:")
for i, count in enumerate(valid_pixel_counts):
    print(f"  Band {i:02d}: {count:,} ({count / total_pixels:.2%} valid)")

# Plot all bands in a grid
cols = 8
rows = int(np.ceil(data.shape[0] / cols))
fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

for i in range(data.shape[0]):
    r, c = divmod(i, cols)
    ax = axes[r, c]
    im = ax.imshow(masked[i], cmap="viridis")
    ax.set_title(f"Band {i}", fontsize=8)
    ax.axis("off")

# Hide empty subplots
for j in range(data.shape[0], rows * cols):
    r, c = divmod(j, cols)
    axes[r, c].axis("off")

plt.tight_layout()
plt.show()
