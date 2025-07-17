# Wetland Segmentation with ResUNet-ViT

This project performs high-resolution semantic segmentation of wetlands using a hybrid CNN-ViT architecture trained on 10-meter satellite imagery. It uses weak supervision from noisy 100-meter wetland type labels and powerful satellite-based feature embeddings.

---

## 📦 Project Structure

```bash
├── configs/
│   └── config.yaml              # Main configuration file (with {now} placeholders)
├── data/
│   ├── preprocess.py            # Reprojects and remaps raster inputs
│   ├── split_data.py            # Generates train/val/test splits
│   └── dataset.py               # PyTorch dataset class
├── models/
│   └── resunet_vit.py           # Hybrid ResNet+ViT architecture
├── train/
│   ├── train.py                 # Main training script
│   ├── metrics.py               # Computes mIoU, F1, etc.
│   └── losses/
│       ├── focal_tversky.py     # Custom Focal + Tversky loss with boundary masking
│       └── soft_boundary_dice.py# Optional soft boundary-aware Dice loss
├── predict/
│   ├── inference.py             # Patch-based inference with VRT support
│   └── evaluate_predictions.py  # Confusion matrix, metrics CSV
├── scripts/
│   ├── run_train.sh             # Bash script to run preprocessing + training
│   └── visualize_predictions.py # Plots random samples of GT vs. prediction
├── outputs/
│   └── ...                      # Saved models, logs, predictions
```

---

## 🚀 Quick Start

1. **Clone the repo**

```bash
git clone https://github.com/mkovac03/wetland-segmentation.git
cd wetland-segmentation
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run training**

```bash
bash run_train.sh
```

4. **Run inference**

```bash
python -m predict.inference --timestamp <TIMESTAMP>
```

---

## 🧠 Model Details

* The core segmentation model is a **ResNet-UNet-ViT hybrid**, designed for high-resolution land cover classification from satellite-derived features. This architecture combines the **local feature extraction strength of CNNs** with the **global attention modeling capabilities of Vision Transformers (ViT)**.

  ### 🔧 Architecture Overview

  * **Encoder**: A ResNet34 backbone pretrained on ImageNet, modified to accept 29 input channels. The encoder captures rich spatial features at progressively lower resolutions through convolutional layers and skip connections.
  * **Transformer Bottleneck**: The deepest features from the encoder are passed to a lightweight Vision Transformer. The ViT component:

    * Models long-range spatial dependencies across the entire patch,
    * Aggregates contextual information to assist in distinguishing subtle wetland types,
    * Operates at a reduced spatial resolution (e.g., 16x16 patches) for efficiency,
    * Parameters such as `vit_embed_dim`, `vit_depth`, `vit_heads`, `vit_patch_size`, and `vit_img_size` are configurable via `config.yaml`.

  * **Decoder**: A UNet-style upsampling path:

    * Uses `F.interpolate` for upsampling (rather than transposed convs) for smoother outputs with fewer artifacts,
    * Includes skip connections with encoder layers,
    * Adds optional dropout for regularization.

  This design allows the model to **combine texture- and boundary-level cues (via CNN)** with **global spatial patterns (via ViT)**—ideal for mapping heterogeneous and often ambiguous wetland landscapes.

---

## 🛠️ Loss Function

* **Focal Loss** with:

  * Tunable α-γ parameters to downweight easy negatives
  * Boundary masking to emphasize edge refinement

* **Tversky Loss**:

  * Generalized Dice-like formulation to handle extreme class imbalance
  * High recall sensitivity useful for detecting small/missing wetlands

* **Combined Loss** = Focal + Tversky (+ optional boundary-aware Dice)

---

## 📊 Evaluation

* **Metrics**:

  * Pixel Accuracy, mIoU, macro/micro F1
  * Per-class confusion matrix

* **Tools**:

  * `evaluate_predictions.py`: Aggregate statistics and error matrix
  * `visualize_predictions.py`: Patch-level GT vs. prediction plots

---

## 📍 Citation

If you use this in your research, please cite:

> Kovács et al. (2025). *High-resolution mapping of wetland types across Europe using CNN-ViT segmentation of satellite embeddings.*

---

## 📬 Contact

For questions or collaboration:

* 🧑‍💻 [Gyula Máté Kovács](https://github.com/mkovac03)
* 🌍 University of Copenhagen · Global Wetland Center
# wetland_segmentation
