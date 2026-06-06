# Meteorite Image Classification

Hybrid CNN-Transformer model (ResNet101 + ViT-B/16) for binary meteorite image classification.  
Course: 数据科学实践 | SUSTech 2026

## Results

| Model | Kaggle F1 Score |
|-------|----------------|
| ResNet101 + ViT-B/16 (50/50 fusion) | **0.83516** |

---

## Dataset & Model Weights

Data and model weights are hosted on Hugging Face (code only in this repo):

**[https://huggingface.co/datasets/YihangHu/meteorite-identification](https://huggingface.co/datasets/YihangHu/meteorite-identification)**

| Path on HF Hub | Description |
|----------------|-------------|
| `data/train/` | Training images (~5200 images, PNG) |
| `data/test_images/` | Test images (194 images, JPG) |
| `data/train_label.csv` | Training labels (`image_id`, `label`) |
| `data/sample_submission.csv` | Submission template |
| `last_model.pth` | Trained model weights (491 MB) |

---

## Environment

```bash
pip install torch torchvision tqdm pandas scikit-learn scipy pillow swanlab
```

Tested with: Python 3.9+, PyTorch 2.x, CUDA 11.8+

---

## Reproduce Test Results

### 1. Clone this repo and download data

```bash
# Clone code
git clone https://github.com/YihangHu2004/DeepLearning-meteorite-identification.git
cd DeepLearning-meteorite-identification

# Download data + weights from HF Hub
git lfs install
git clone https://huggingface.co/datasets/YihangHu/meteorite-identification hf_data
```

### 2. Directory structure

```
DeepLearning-meteorite-identification/
├── hybrid_cnn_transformer.py   # Training script
├── infer.py                    # Inference-only script
├── dataset.py                  # Dataset utilities
├── yoloe_preprocess.py         # Optional YOLOE preprocessing
└── hf_data/                    # Downloaded from HF Hub
    ├── last_model.pth
    └── data/
        ├── train/
        ├── test_images/
        ├── train_label.csv
        └── sample_submission.csv
```

### 3. Inference (reproduce submission)

```bash
python infer.py \
  --weights ./hf_data/last_model.pth \
  --test-csv ./hf_data/data/sample_submission.csv \
  --test-img-dir ./hf_data/data/test_images \
  --output submission.csv \
  --threshold 0.52
```

Output: `submission.csv` with columns `id` and `label` (0 = non-meteorite, 1 = meteorite).

### 4. Training from scratch

```bash
python hybrid_cnn_transformer.py \
  --train-csv ./hf_data/data/train_label.csv \
  --train-img-dir ./hf_data/data/train \
  --test-csv ./hf_data/data/sample_submission.csv \
  --test-img-dir ./hf_data/data/test_images \
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
