# Multi-Modal UNet with Learnable Band-Pass Fusion for Semantic Segmentation

A semantic segmentation framework built on U-Net with VGG16 backbone, extended with multi-modal input support (RGB + edge/detail maps + region maps) and a Learnable Band-Pass Filter (LBF) module for adaptive feature fusion.

---

## Overview

This repository implements a dual-branch UNet variant that fuses standard RGB imagery with auxiliary hand-crafted feature maps (e.g., DEX edge maps and LAB-based region maps) via a trainable band-pass filtering mechanism. The model is designed for binary segmentation tasks where fine boundary detail and region-level context are both critical.

---

## Environment

| Item | Version |
|---|---|
| OS | Linux |
| GPU | NVIDIA GeForce RTX 4090 (24 GB VRAM) |
| Python | 3.8 |
| PyTorch | 2.4.0 |
| CUDA | 11.8 |



---

## Dataset Preparation

The codebase follows a VOC-style directory layout. Prepare your dataset under a root directory (e.g., `VOCdevkit_dilate/`) with the following structure:

```
VOCdevkit_dilate/
└── VOC2007/
    ├── JPEGImages_Aug/            # Augmented RGB images (generated)
    ├── SegmentationClass_Aug/     # Augmented masks (generated)
    ├── Dexined_Aug/               # Augmented DEX maps (generated)
    ├── LAB_Aug/                   # Augmented LAB maps (generated)
    └── ImageSets/
        └── Segmentation/
            ├── train.txt
            └── val.txt
```

> **Important:** Segmentation mask pixel values must be `0` (background) and `1` (foreground). Values of `255` for foreground are a common pitfall — see [segmentation-format-fix](https://github.com/bubbliiiing/segmentation-format-fix) if your dataset needs conversion.

### Data Augmentation

Run the augmentation pipeline to generate the `*_Aug` directories before training:

```bash
python utils/augmentation_mixed.py
```

This applies spatial transforms (flip, rotate, elastic, affine) and photometric augmentations (brightness, contrast, gamma, color jitter) using [Albumentations](https://albumentations.ai/), with synchronized transformation across RGB, segmentation mask, DEX, and LAB maps. By default, each original image is augmented 10×.

---

## Training

Update the paths in `train.py` as needed, then run:

```bash
CUDA_VISIBLE_DEVICES=2 python train.py
```

### Training Configuration

| Parameter | Value |
|---|---|
| `num_classes` | 2 (background + foreground) |
| `backbone` | `vgg` |
| `input_shape` | 512 × 512 |
| `optimizer` | Adam (β₁=0.9, β₂=0.999, weight_decay=0) |
| `Init_lr` | 1e-4 |
| `Min_lr` | 1e-6 |
| `lr_decay_type` | cosine annealing |
| `Freeze_Epoch` | 50 |
| `UnFreeze_Epoch` | 300 |
| `batch_size` | 2 (both stages) |
| `dice_loss` | True |
| `focal_loss` | True (γ = 2) |
| `eval_period` | every 5 epochs |
| `num_workers` | 4 |
| `seed` | 11 |

### Two-Stage Training Strategy

**Stage 1 — Frozen backbone (epochs 0–50):**  
The VGG16 backbone is frozen. Only the decoder and the newly introduced modules (LBF, auxiliary branches) are optimized. This stabilizes early training.

**Stage 2 — Full fine-tuning (epochs 51–300):**  
All parameters are unfrozen and jointly optimized end-to-end.

### Initialization

- VGG16 backbone: ImageNet pretrained weights
- LBF parameters (τ, τ_l, τ_u): random initialization; default τ=0.05, τ_l=0.1, τ_u=0.4
- Fusion weights α and β: equal initialization (α = β = 0.5)

Checkpoints are saved to `train_weight_output/<timestamp>/`. The best model (lowest validation loss) is saved as `best_epoch_weights.pth`.

---

## Inference

Use `unet_double.py` for multi-modal inference (RGB + DEX + LAB):

```python
from unet_double import Unet
from PIL import Image

model = Unet()

rgb   = Image.open("path/to/image.jpg")
dex   = Image.open("path/to/dex.png").convert("L")
lab   = Image.open("path/to/lab.png").convert("L")

seg_result, contour_result = model.detect_image(rgb, dex, lab)
seg_result.save("segmentation.png")
contour_result.save("contour_overlay.png")
```

Use `unet.py` for the single-stream RGB-only baseline:

```python
from unet import Unet
from PIL import Image

model = Unet()
result = model.detect_image(Image.open("image.jpg"))
result.save("result.png")
```

**`mix_type` visualization modes:**

| Value | Output |
|---|---|
| `0` | Segmentation mask blended with original image (α=0.7) |
| `1` | Segmentation mask only |
| `2` | Original image with background removed |

---

## Evaluation Metrics

Evaluation runs automatically every `eval_period` epochs during training via `EvalCallback`. Metrics are logged to `epoch_miou.txt` and `metrics.csv` under the run's log directory.

Computed metrics include:

- **Per-class and mean IoU (mIoU)**
- **Dice coefficient**
- **Precision, Recall, F1-score**
- **Specificity**
- **Overall pixel accuracy and Kappa coefficient**
- **Frequency-weighted IoU (fwIoU)**
- **G-Mean** and **Balanced Accuracy** (for class-imbalanced datasets)
- **Foreground-weighted per-image metrics**

---


## Reproducibility

All configurations in `train.py` are set for direct reproduction of reported results. The random seed is fixed at `seed = 11` across Python, NumPy, and PyTorch. Training can be resumed from any checkpoint by setting `model_path` to the desired `.pth` file and adjusting `Init_Epoch` accordingly.

---

## Citation

If you use this code in your research, please cite this repository.

---

## License

This project is released for academic and research use. Please refer to `LICENSE` for details.
