# Meteorite Image Classification

Hybrid CNN-Transformer model (ResNet101 + ViT-B/16) for binary meteorite image classification.  
Course: 数据科学实践 | SUSTech 2026

## Results

| Model | Val Accuracy | vs. Sample (194 test) |
|-------|-------------|----------------------|
| ResNet101 + ViT-B/16 (50/50 fusion) | ~91% | **191/194 (98.5%)** |

---

## Dataset & Model Weights

All data and model weights are hosted on Hugging Face:

**[https://huggingface.co/datasets/YihangHu/meteorite-identification](https://huggingface.co/datasets/YihangHu/meteorite-identification)**

| Path | Description |
|------|-------------|
| `data/train/` | Training images (~5200 images, PNG) |
| `data/test_images/` | Test images (194 images, JPG) |
| `data/train_label.csv` | Training labels (`image_id`, `label`) |
| `data/sample_submission.csv` | Submission template |
| `last_model.pth` | Trained model weights (ResNet101 + ViT-B/16, 491 MB) |

---

## Environment

```bash
pip install torch torchvision tqdm pandas scikit-learn scipy pillow swanlab
```

Tested with: Python 3.9+, PyTorch 2.x, CUDA 11.8+

---

## Reproduce Test Results

### 1. Download data and weights

```bash
# Install git-lfs first
git lfs install

# Clone the dataset repo (includes data + weights)
git clone https://huggingface.co/datasets/YihangHu/meteorite-identification
```

Or download individual files from the Hugging Face web interface.

### 2. Directory structure

```
project/
├── hybrid_cnn_transformer.py   # Training script
├── infer.py                    # Inference-only script
├── dataset.py                  # Dataset utilities
├── data/
│   ├── train/                  # Training images
│   ├── test_images/            # Test images
│   ├── train_label.csv
│   └── sample_submission.csv
└── last_model.pth              # Pre-trained weights
```

### 3. Inference (reproduce submission)

Run inference with the provided weights to reproduce the submitted result:

```bash
python infer.py \
  --weights ./last_model.pth \
  --test-csv ./data/sample_submission.csv \
  --test-img-dir ./data/test_images \
  --output submission.csv \
  --threshold 0.52
```

Output: `submission.csv` with columns `id` and `label` (0 = non-meteorite, 1 = meteorite).

### 4. Training from scratch

```bash
python hybrid_cnn_transformer.py \
  --train-csv ./data/train_label.csv \
  --train-img-dir ./data/train \
  --test-csv ./data/sample_submission.csv \
  --test-img-dir ./data/test_images \
  --epochs 20 \
  --batch-size 32 \
  --mode single
```

Checkpoints and submission files are saved to `logs/run_YYYYMMDD_HHMMSS/`.

---

## Model Architecture

```
Input Image (224×224)
       ├── ResNet101 branch  →  p_cnn  (softmax)
       └── ViT-B/16  branch  →  p_vit  (softmax)
                  ↓
    p_fused = 0.5 × p_cnn + 0.5 × p_vit
                  ↓
    label = 1  if  p_fused[1] ≥ 0.52  else  0
```

**Training loss:** `0.755 × CE(CNN) + 0.245 × CE(ViT)`  
**Inference fusion:** equal weights (0.5 / 0.5)  
**Decision threshold:** 0.52

### Key Training Settings

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Batch size | 32 |
| Epochs | 20 |
| Scheduler | CosineAnnealingLR |
| Image size | 224 × 224 |
| Augmentation | RandomResizedCrop, Flip, ColorJitter, Rotation, MaskCorners |

---

## File Description

| File | Description |
|------|-------------|
| `hybrid_cnn_transformer.py` | Full training pipeline (train + inference + BCTS calibration) |
| `infer.py` | Lightweight inference-only script |
| `dataset.py` | Dataset class |
| `yoloe_preprocess.py` | Optional YOLOE preprocessing cache |
| `submission.csv` | Best submission result |
| `stage2_analysis.csv` | Per-image probability analysis (CNN/ViT branches) |
