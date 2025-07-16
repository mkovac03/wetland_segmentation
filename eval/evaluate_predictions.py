import os
import glob
import numpy as np
import rasterio
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import csv

# ========== Dynamic path discovery ==========
BASE_DIR = "/media/lkm413/storage1/wetland_segmentation/outputs/"
INPUT_DIR = "/media/lkm413/storage1/gee_embedding_download/images/Denmark/2018/"

pattern = os.path.join(BASE_DIR, "predictions_Denmark2018_*")
pred_dirs = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)

if not pred_dirs:
    raise RuntimeError("No prediction output folders found matching pattern.")

PRED_DIR = pred_dirs[0]
timestamp = os.path.basename(PRED_DIR).split("_", maxsplit=2)[-1]
print(f"[INFO] Using latest predictions from: {PRED_DIR}")

# ========== Parameters ==========
NUM_CLASSES = 20
NODATA_VALUE = 255

def read_label_from_first_band(path):
    with rasterio.open(path) as src:
        return src.read(1)

def read_prediction(path):
    with rasterio.open(path) as src:
        return src.read(1)

def main():
    input_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.tif")))
    pred_files = sorted(glob.glob(os.path.join(PRED_DIR, "*.tif")))

    assert len(input_files) == len(pred_files), "Mismatch in number of input and prediction files"

    all_gt = []
    all_pred = []

    for input_path, pred_path in tqdm(zip(input_files, pred_files), total=len(input_files), desc="Evaluating images"):
        assert os.path.basename(input_path) == os.path.basename(pred_path), f"Filename mismatch: {input_path} vs {pred_path}"

        gt = read_label_from_first_band(input_path).flatten()
        pred = read_prediction(pred_path).flatten()

        mask = (gt != NODATA_VALUE)
        gt_masked = gt[mask]
        pred_masked = pred[mask]

        all_gt.append(gt_masked)
        all_pred.append(pred_masked)

    all_gt = np.concatenate(all_gt)
    all_pred = np.concatenate(all_pred)

    acc = accuracy_score(all_gt, all_pred)
    precision, recall, f1, support = precision_recall_fscore_support(all_gt, all_pred, labels=range(NUM_CLASSES), zero_division=0)
    conf_mat = confusion_matrix(all_gt, all_pred, labels=range(NUM_CLASSES))

    # Normalize confusion matrix by true label counts (row-wise)
    conf_mat_norm = conf_mat.astype(np.float32)
    row_sums = conf_mat_norm.sum(axis=1, keepdims=True)
    conf_mat_norm = np.divide(conf_mat_norm, row_sums, where=row_sums != 0)

    # Plot normalized confusion matrix
    cm_filename = f"confusion_matrix_normalized_{timestamp}.png"
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_mat_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=[str(i) for i in range(NUM_CLASSES)],
                yticklabels=[str(i) for i in range(NUM_CLASSES)])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Normalized Confusion Matrix (Recall per class)")
    plt.tight_layout()
    plt.savefig(cm_filename)
    plt.close()
    print(f"Normalized confusion matrix saved as {cm_filename}")

    # Save metrics to CSV
    csv_filename = f"metrics_summary_{timestamp}.csv"
    with open(csv_filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Class", "Precision", "Recall", "F1-Score", "Support"])
        for c in range(NUM_CLASSES):
            writer.writerow([c, f"{precision[c]:.4f}", f"{recall[c]:.4f}", f"{f1[c]:.4f}", support[c]])
        writer.writerow([])
        writer.writerow(["Overall Accuracy", f"{acc:.4f}"])
    print(f"Metrics summary saved as {csv_filename}")

if __name__ == "__main__":
    main()
