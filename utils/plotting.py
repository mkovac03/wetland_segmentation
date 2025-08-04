# File: utils/plotting.py

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

def add_prediction_legend(ax, num_classes, label_names=None, cmap="tab20"):
    """
    Add a class legend to a matplotlib axis for semantic segmentation output.

    Parameters:
    - ax: matplotlib axis to add the legend to
    - num_classes: total number of classes
    - label_names: optional dict mapping class index to label
    - cmap: name of matplotlib colormap to use
    """
    class_colors = plt.get_cmap(cmap)(np.linspace(0, 1, num_classes))
    class_labels = label_names or {i: f"Class {i}" for i in range(num_classes)}

    patches = [
        mpatches.Patch(color=class_colors[i], label=class_labels.get(i, f"Class {i}"))
        for i in range(num_classes)
    ]
    ax.legend(handles=patches, loc="center left", bbox_to_anchor=(1.0, 0.5),
              fontsize="small", frameon=False)
