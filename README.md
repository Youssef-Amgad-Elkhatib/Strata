# 🛰️ Strata — From-Scratch Vision Transformer vs. Two-Stage ResNet-50 on EuroSAT RGB

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-EuroSAT%20RGB%20(27k)-0EA5E9)](https://github.com/phelber/EuroSAT)
[![ViT Test Acc](https://img.shields.io/badge/ViT%20(Scratch)%20Test%20Acc-95.59%25-10B981)](#4-held-out-test-set-benchmark-2700-images)
[![ResNet-50 Test Acc](https://img.shields.io/badge/ResNet--50%20(Fine--Tuned)%20Test%20Acc-98.48%25-2563EB)](#4-held-out-test-set-benchmark-2700-images)

**Strata** is an end-to-end computer vision engineering and interpretability benchmark built in pure PyTorch. It implements a **Vision Transformer (ViT) completely from scratch**—without prebuilt Transformer backbones or ImageNet pretraining—and evaluates it head-to-head against an ImageNet-pretrained **ResNet-50** trained via a two-stage **Linear Probe → Differential-LR Fine-Tuning (LP-FT)** pipeline on the **EuroSAT RGB** Sentinel-2 satellite dataset.

Beyond raw accuracy, the project opens the Vision Transformer's internal representations using **Multi-Block Attention Rollout**, tracing how spatial patch information propagates across every encoder layer into the `[CLS]` token across all 10 land-use classes.

---

## ✨ 1. Project Highlights

- **100% From-Scratch Vision Transformer (`2.88M` params):** Custom PyTorch implementation of patch projection, learnable `[CLS]` and positional embeddings, multi-head self-attention, pre-norm Transformer encoder blocks, and an MLP classification head—reaching **95.59% test accuracy** on only 21,600 training images with **8.16x fewer parameters** than ResNet-50.
- **Controlled Two-Stage Transfer Learning (`LP-FT`):** Addresses the ground-to-satellite domain shift by first aligning a new classification head on frozen ImageNet features (with locked `BatchNorm` running statistics) before unfreezing the full ResNet-50 backbone with differential learning rates (`1e-4` backbone / `5e-4` head), climbing to **98.48% test accuracy**.
- **Strict Generalization Guardrail (`Gap <= 2.0%`):** Both models enforce identical checkpoint-saving criteria (`train_acc - val_acc <= 2.0%`) alongside `LabelSmoothing=0.1`, `AdamW`, and `CosineAnnealingLR` to prevent overfitting.
- **High-Throughput GPU Pipeline:** Preloads the dataset into RAM and executes mixed-precision training (`torch.amp`) and vectorized GPU channel re-normalization (`to_imagenet_norm`), cutting epoch runtime from ~30s to ~1.5s.
- **Full-Network Attention Rollout:** Uses forward pre/post hooks across all encoder blocks (accounting for residual skip connections) to visualize spatial focus across all 10 satellite land-use categories.

---

## 🌍 2. Dataset & Experimental Setup

- **Source:** EuroSAT RGB — 27,000 geo-referenced optical patches captured by the ESA **Sentinel-2** satellite at 10m/pixel spatial resolution (`64 × 64` pixels, covering `640m × 640m` per tile).
- **Classes (10):** `AnnualCrop`, `Forest`, `HerbaceousVegetation`, `Highway`, `Industrial`, `Pasture`, `PermanentCrop`, `Residential`, `River`, `SeaLake`.
- **Reproducible Stratified Split (`SEED = 42`):**
  - **Train Set:** `21,600` images (80%)
  - **Validation Set:** `2,700` images (10%)
  - **Unseen Test Set:** `2,700` images (10%)

---

## 🏗️ 3. Model Architectures & Training Methodology

### 🧩 Custom Vision Transformer (From Scratch)
Because EuroSAT tiles are `64 × 64`, standard ImageNet patch sizes (`16 × 16`) yield only 16 coarse tokens. Using `patch_size = 4` preserves fine linear structures such as highways, rivers, and crop rows:

| Hyperparameter / Component | Configuration |
| :--- | :--- |
| **Input Resolution & Patch Size** | `64 × 64 × 3` image → `4 × 4` patches (`16 × 16 = 256` patches) |
| **Sequence Length (N)** | `257` tokens (`256` spatial patch tokens + `1` learnable `[CLS]` token) |
| **Embedding Dimension (D)** | `192` |
| **Encoder Depth & MLP Width** | `6` Transformer Encoder Blocks, `768` hidden MLP nodes (`4 × D`) |
| **Total Parameters** | `2,884,426` (`~2.88M`) |
| **Normalization & Regularization** | Channel-wise EuroSAT standardization, `LabelSmoothing=0.1`, `AdamW`, `CosineAnnealingLR` |
| **Checkpoint Rule** | Save best `val_acc` only when `train_acc - val_acc <= 2.0%` |

### 🏛️ ResNet-50 Two-Stage Transfer Learning (`LP-FT`)
To benchmark pure self-attention against convolutional inductive bias while respecting the domain shift between ground-level ImageNet photos and top-down orbital views:

1. **On-the-Fly ImageNet Calibration (`to_imagenet_norm`):** Converts EuroSAT-standardized tensors back to `[0, 1]` and re-standardizes with ImageNet channel statistics on the GPU so pretrained filters receive activations in their expected numerical range.
2. **Stage 1 — Linear Probing (`15` Epochs):** Freezes all `23.5M` backbone weights, locks backbone `BatchNorm` layers in `.eval()`, and trains only the new `2048 -> 10` linear head (`lr = 1e-3`).
3. **Stage 2 — Full Fine-Tuning (`20` Epochs):** Loads the best Stage 1 checkpoint, unfreezes 100% of parameters, enables `.train()` across the network so `BatchNorm` running statistics adapt to satellite imagery, and optimizes with **differential learning rates** (`1e-4` for pretrained backbone filters, `5e-4` for the classification head).

---

## 🎯 4. Held-Out Test Set Benchmark (`2,700` Images)

### Overall Comparison

| Model | Pretraining | Parameters | Test Accuracy | Macro Precision | Macro Recall | Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Custom ViT (From Scratch)** | None (Random Init) | `2,884,426` (`2.88M`) | **95.59%** | `0.9544` | `0.9539` | `0.9541` |
| **ResNet-50 (Stage 2 Fine-Tuned)** | ImageNet-1K (`LP-FT`) | `23,528,522` (`23.53M`) | **98.48%** | `0.9851` | `0.9834` | `0.9842` |

### Per-Class F1-Score Breakdown

| Land-Use Class | Test Support | ViT Precision | ViT Recall | **ViT F1** | ResNet-50 Precision | ResNet-50 Recall | **ResNet-50 F1** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `AnnualCrop` | 300 | 0.9408 | 0.9533 | **0.9470** | 0.9702 | 0.9767 | **0.9734** |
| `Forest` | 300 | 0.9768 | 0.9833 | **0.9801** | 0.9967 | 0.9967 | **0.9967** |
| `HerbaceousVegetation` | 300 | 0.9567 | 0.9567 | **0.9567** | 0.9675 | 0.9933 | **0.9803** |
| `Highway` | 250 | 0.9388 | 0.9200 | **0.9293** | 0.9880 | 0.9840 | **0.9860** |
| `Industrial` | 250 | 0.9640 | 0.9640 | **0.9640** | 0.9841 | 0.9920 | **0.9880** |
| `Pasture` | 200 | 0.9406 | 0.9500 | **0.9453** | 0.9895 | 0.9450 | **0.9668** |
| `PermanentCrop` | 250 | 0.9292 | 0.8920 | **0.9102** | 0.9760 | 0.9760 | **0.9760** |
| `Residential` | 300 | 0.9673 | 0.9867 | **0.9769** | 0.9933 | 0.9867 | **0.9900** |
| `River` | 250 | 0.9363 | 0.9400 | **0.9381** | 0.9919 | 0.9840 | **0.9880** |
| `SeaLake` | 300 | 0.9933 | 0.9933 | **0.9933** | 0.9934 | 1.0000 | **0.9967** |
| **Macro Average** | **2,700** | **0.9544** | **0.9539** | **0.9541** | **0.9851** | **0.9834** | **0.9842** |

---

## 🔥 5. Visual Interpretability: Multi-Block Attention Rollout

Single-layer self-attention maps from the final Transformer block are often noisy because deep layers aggregate global context into summary tokens. Following Abnar & Zuidema (2020), **Attention Rollout** accounts for residual skip connections (`0.5 * A_l + 0.5 * I`) and multiplies row-normalized attention matrices recursively from Block 1 through Block 6 before bicubic upsampling the `16 × 16` patch weights back to `64 × 64`:

```text
Rollout(L) = ∏_{l=1..L} RowNorm(0.5 · A_l + 0.5 · I)
```

![Vision Transformer Attention Rollout Across All 10 Classes](vit_attention_rollout_all_classes.png)

- **Left Column:** Original Sentinel-2 RGB satellite patch (`64 × 64`).
- **Middle Column:** Full-network Attention Rollout intensity (`inferno` colormap).
- **Right Column:** Spatial overlay (`jet` colormap at `alpha=0.45`) confirming alignment with geometric land structures such as highways, rivers, building blocks, and field boundaries.

---

## 📈 6. Training Dynamics & Error Analysis

### Key Engineering Takeaways
1. **Self-Attention Without Pretraining Scales Remarkably Well:** Despite having zero built-in 2D translational equivariance and 8x fewer parameters than ResNet-50, the scratch ViT achieved **95.59% test accuracy** and **>97.6% F1** on `SeaLake`, `Forest`, and `Residential`.
2. **Where Convolutional Inductive Bias Wins:** The `2.89%` accuracy gap between fine-tuned ResNet-50 (`98.48%`) and the from-scratch ViT (`95.59%`) stems primarily from fine-grained agricultural texture classes—most notably `PermanentCrop` (`0.9102` vs. `0.9760` F1) and `AnnualCrop` (`0.9470` vs. `0.9734` F1)—where local multi-scale edge filters pretrained on ImageNet resolve subtle orchard/row spacing.
3. **Why Two-Stage `LP-FT` Matters:** Training the linear probe first prevents large random gradients from corrupting pretrained convolutional filters during early epochs, while differential learning rates in Stage 2 allow `BatchNorm` layers to smoothly adapt to satellite pixel statistics within the `2.0%` train–validation gap threshold.

| Vision Transformer (From Scratch) | ResNet-50 (Stage 2 Fine-Tuned) |
| :---: | :---: |
| ![ViT Training Curves](vit_training_curves.png) | ![ResNet-50 Stage 2 Training Curves](resnet50_stage2_curves.png) |
| ![ViT Confusion Matrix](vit_confusion_matrix.png) | ![ResNet-50 Stage 2 Confusion Matrix](resnet50_stage2_cm.png) |
