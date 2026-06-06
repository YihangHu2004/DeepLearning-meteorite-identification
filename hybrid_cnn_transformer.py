"""
Hybrid CNN-Transformer Architecture for Meteorite Image Classification
======================================================================

架构说明:
  - CNN 部分   : ResNet101 (负责提取局部空间特征)
  - ViT 部分   : ViT-B/16  (负责建模全局长距离依赖)
  - 融合模块   : P_i = λ1 * softmax(CNN_feat/T) + λ2 * softmax(ViT_feat/T)
  - 优化器     : AdamW, lr=1e-4, wd=1e-4
  - 第二阶段新增: 温度缩放 + EM先验估计 + 贝叶斯先验校正 + CNN↔ViT分支分歧策略

K折全折阈值策略:
  每折独立推理，只有所有折的 meteorite 概率都 >= kfold_thresh 才判为1。

运行模式:
  --mode single          : 单模型训练 + 推理
  --mode kfold           : K折训练 + 集成推理
  --tta N                : 启用TTA, N次增强推理取平均
  --use-yoloe-cache      : 使用 YOLOE 预处理缓存
  --kfold-thresh FLOAT   : K折全折阈值（默认 0.7）
  --disagree-tau FLOAT   : 分支分歧阈值（默认 0.3）
"""

import os
import time
import json
import random
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from tqdm import tqdm
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, top_k_accuracy_score,
)
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import (
    ResNet101_Weights,
    EfficientNet_B3_Weights,
    ConvNeXt_Small_Weights,
    ViT_B_16_Weights,
)
from PIL import Image
import swanlab


# ──────────────────────────────────────────────
# 数据遮挡工具类
# ──────────────────────────────────────────────
class MaskCorners:
    def __init__(self, mask_ratio=0.15, mask_top=True):
        self.mask_ratio = mask_ratio
        self.mask_top = mask_top

    def __call__(self, img):
        arr = np.array(img)
        h, w = arr.shape[:2]
        m = int(min(h, w) * self.mask_ratio)
        arr[:m, :m] = 128
        arr[:m, -m:] = 128
        arr[-m:, :m] = 128
        arr[-m:, -m:] = 128
        if self.mask_top:
            arr[:int(h * 0.08), :] = 128
        return Image.fromarray(arr)


# ──────────────────────────────────────────────
# 0. 可复现种子
# ──────────────────────────────────────────────
SEED = 42

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()

# ──────────────────────────────────────────────
# 1. 路径配置
# ──────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

def _first_existing_path(*candidates):
    for c in candidates:
        if Path(c).exists():
            return Path(c)
    return Path(candidates[0])

DEFAULT_TRAIN_CSV     = _first_existing_path(PROJECT_DIR / "/root/ds/assignment/project/陨石2/new_data/mix.csv", SCRIPT_DIR / "mix.csv")
DEFAULT_TRAIN_IMG_DIR = _first_existing_path(PROJECT_DIR / "/root/ds/assignment/project/陨石2/new_data/mix", SCRIPT_DIR / "mix")
DEFAULT_TEST_CSV      = _first_existing_path(PROJECT_DIR / "/root/ds/assignment/project/陨石2/submission.csv", SCRIPT_DIR / "submission.csv")
DEFAULT_TEST_IMG_DIR  = _first_existing_path(PROJECT_DIR / "/root/ds/assignment/project/陨石2/test_images", SCRIPT_DIR / "test_images")
OUTPUT_DIR = PROJECT_DIR / "logs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# 2. 超参数
# ──────────────────────────────────────────────
NUM_CLASSES            = 2
IMAGE_SIZE             = 224
BATCH_SIZE             = 32
EPOCHS                 = 20
MAX_LR                 = 1e-4
WEIGHT_DECAY           = 1e-4
NUM_WORKERS            = 4
LABEL_SMOOTHING        = 0.0
GRAD_CLIP_NORM         = 0
EARLY_STOPPING_PATIENCE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LAMBDA_CNN       = 0.755     # 训练损失权重
LAMBDA_VIT       = 0.245
LAMBDA_CNN_INFER = 0.5     # 推理融合权重（等权，后验分析最优）
LAMBDA_VIT_INFER = 0.5
DECISION_THRESHOLD = 0.52  # 推理决策阈值，由后验分析得出
DROPOUT_RATE_CNN  = 0.0    # CNN 分支：不压制，保持锐利
DROPOUT_RATE_VIT  = 0.0    # ViT 分支：高 dropout，主动压制至欠训练状态
CNN_BACKBONE  = "resnet101"

# 训练集先验（非陨石, 陨石）
TRAIN_PRIOR = [1 - 0.527, 0.527]

CNN_PRETRAINED_WEIGHTS = {
    "resnet101":      ResNet101_Weights.IMAGENET1K_V2,
    "efficientnet_b3": EfficientNet_B3_Weights.IMAGENET1K_V1,
    "convnext_small": ConvNeXt_Small_Weights.IMAGENET1K_V1,
}
print(f"[Config] Device: {DEVICE}")

# ──────────────────────────────────────────────
# 3. 数据增强
# ──────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomRotation(degrees=15),
    MaskCorners(mask_ratio=0.15, mask_top=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    MaskCorners(mask_ratio=0.15, mask_top=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

tta_transforms = [
    val_transform,
    transforms.Compose([transforms.Resize(256), transforms.CenterCrop(IMAGE_SIZE), transforms.RandomHorizontalFlip(p=1.0), MaskCorners(0.15, True), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
    transforms.Compose([transforms.Resize(240), transforms.CenterCrop(IMAGE_SIZE), MaskCorners(0.15, True), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
    transforms.Compose([transforms.Resize(256), transforms.CenterCrop(IMAGE_SIZE), transforms.RandomVerticalFlip(p=1.0), MaskCorners(0.15, True), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
    transforms.Compose([transforms.Resize(288), transforms.CenterCrop(IMAGE_SIZE), MaskCorners(0.15, True), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
]


# ──────────────────────────────────────────────
# 4. 数据集
# ──────────────────────────────────────────────
class MeteoriteDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, use_yoloe_cache=False, id_to_cache=None):
        self.transform = transform
        self.use_yoloe_cache = use_yoloe_cache
        self.id_to_cache = id_to_cache or {}
        self.df = df.reset_index(drop=True).copy()
        if "id" in self.df.columns and "image_id" not in self.df.columns:
            self.df = self.df.rename(columns={"id": "image_id"})
        self.image_dir = Path(image_dir)

    def __len__(self): return len(self.df)

    def _load_image(self, image_id):
        if self.use_yoloe_cache and image_id in self.id_to_cache:
            p = self.id_to_cache[image_id]
            if p.exists():
                try:
                    img = Image.open(p); img.load(); return img.convert("RGB")
                except: p.unlink(missing_ok=True)
        path = self.image_dir / image_id
        if not path.exists():
            cands = list(self.image_dir.glob(f"{Path(image_id).stem}.*"))
            if cands: path = cands[0]
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self._load_image(str(row["image_id"]))
        if self.transform: img = self.transform(img)
        return img, int(row["label"])


class MeteoriteTestDataset(Dataset):
    def __init__(self, test_img_dir, ids, transform=None, use_yoloe_cache=False, id_to_cache=None):
        self.transform = transform
        self.use_yoloe_cache = use_yoloe_cache
        self.id_to_cache = id_to_cache or {}
        self.test_img_dir = Path(test_img_dir)
        self.ids = list(ids)

    def __len__(self): return len(self.ids)
    def set_transform(self, t): self.transform = t

    def _load_image(self, image_id):
        if self.use_yoloe_cache and image_id in self.id_to_cache:
            p = self.id_to_cache[image_id]
            if p.exists():
                try:
                    img = Image.open(p); img.load(); return img.convert("RGB")
                except: p.unlink(missing_ok=True)
        path = self.test_img_dir / image_id
        if not path.exists():
            cands = list(self.test_img_dir.glob(f"{Path(image_id).stem}.*"))
            if cands: path = cands[0]
            else: raise FileNotFoundError(f"找不到: {image_id}")
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        img = self._load_image(image_id)
        if self.transform: img = self.transform(img)
        return img, image_id


# ──────────────────────────────────────────────
# 5. 模型（支持温度缩放 + 分支输出）
# ──────────────────────────────────────────────
class HybridCNNTransformer(nn.Module):
    def __init__(self, num_classes, lambda_cnn, lambda_vit,
                 lambda_cnn_infer=LAMBDA_CNN_INFER, lambda_vit_infer=LAMBDA_VIT_INFER,
                 backbone=CNN_BACKBONE):
        super().__init__()
        self.lambda_cnn       = lambda_cnn        # 训练损失权重
        self.lambda_vit       = lambda_vit
        self.lambda_cnn_infer = lambda_cnn_infer  # 推理融合权重
        self.lambda_vit_infer = lambda_vit_infer

        backbone = backbone.lower().strip()
        if backbone == "resnet101":
            cnn = models.resnet101(weights=CNN_PRETRAINED_WEIGHTS[backbone])
            in_feat = cnn.fc.in_features
            cnn.fc = nn.Sequential(nn.Dropout(p=DROPOUT_RATE_CNN), nn.Linear(in_feat, num_classes))
        elif backbone == "efficientnet_b3":
            cnn = models.efficientnet_b3(weights=CNN_PRETRAINED_WEIGHTS[backbone])
            in_feat = cnn.classifier[1].in_features
            cnn.classifier = nn.Sequential(nn.Dropout(p=DROPOUT_RATE_CNN), nn.Linear(in_feat, num_classes))
        elif backbone == "convnext_small":
            cnn = models.convnext_small(weights=CNN_PRETRAINED_WEIGHTS[backbone])
            in_feat = cnn.classifier[2].in_features
            cnn.classifier[2] = nn.Sequential(nn.Dropout(p=DROPOUT_RATE_CNN), nn.Linear(in_feat, num_classes))
        else:
            raise ValueError(f"未知骨干: {backbone}")
        self.cnn_branch = cnn
        print(f"  CNN 骨干: {backbone}  (分类头维度={in_feat})")

        vit = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        in_feat_vit = vit.heads.head.in_features
        vit.heads.head = nn.Sequential(nn.Dropout(p=DROPOUT_RATE_VIT), nn.Linear(in_feat_vit, num_classes))
        self.vit_branch = vit
        print(f"  ViT 骨干: vit_b_16  (分类头维度={in_feat_vit})")
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, fusion_mode=False, return_branches=False, temperature=1.0):
        cnn_logits = self.cnn_branch(x)
        vit_logits = self.vit_branch(x)
        if fusion_mode:
            # 温度缩放后再 softmax，推理时使用等权融合
            p_cnn = self.softmax(cnn_logits / temperature)
            p_vit = self.softmax(vit_logits / temperature)
            fused = LAMBDA_CNN_INFER * p_cnn + LAMBDA_VIT_INFER * p_vit
            if return_branches:
                return fused, p_cnn, p_vit
            return fused
        return cnn_logits, vit_logits


# ──────────────────────────────────────────────
# 6. 评估指标
# ──────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob=None, num_classes=2):
    top1 = np.mean(np.array(y_true) == np.array(y_pred))
    top5 = top1 if num_classes <= 2 else (
        top_k_accuracy_score(y_true, y_prob, k=min(5, num_classes)) if y_prob is not None else top1)
    return {
        "Top-1 Accuracy": top1, "Top-5 Accuracy": top5,
        "Mean Recall":    recall_score(y_true, y_pred, average="macro", zero_division=0),
        "Mean Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "F1-Score":       f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Confusion Matrix": confusion_matrix(y_true, y_pred),
    }

def _running_acc_f1(y_true, y_pred):
    if not y_true: return 0.0, 0.0
    return (float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
            float(f1_score(y_true, y_pred, average="macro", zero_division=0)))


# ──────────────────────────────────────────────
# 7. 训练
# ──────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, device, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    all_labels, all_preds = [], []
    progress = tqdm(loader, desc="  Train", leave=False, dynamic_ncols=True)
    for imgs, labels in progress:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        cnn_logits, vit_logits = model(imgs, fusion_mode=False)
        loss = model.lambda_cnn * criterion(cnn_logits, labels) + model.lambda_vit * criterion(vit_logits, labels)
        loss.backward()
        if GRAD_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        with torch.no_grad():
            fused = LAMBDA_CNN_INFER * torch.softmax(cnn_logits.detach(), 1) + LAMBDA_VIT_INFER * torch.softmax(vit_logits.detach(), 1)
            preds = fused.argmax(1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
        acc, f1 = _running_acc_f1(all_labels, all_preds)
        progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.4f}", f1=f"{f1:.4f}")
    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, device, num_classes, criterion):
    model.eval()
    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []
    # 同时收集 logits 用于温度缩放
    all_cnn_logits, all_vit_logits = [], []

    progress = tqdm(loader, desc="  Valid", leave=False, dynamic_ncols=True)
    for imgs, labels in progress:
        imgs, labels = imgs.to(device), labels.to(device)
        cnn_logits, vit_logits = model(imgs, fusion_mode=False)
        loss = model.lambda_cnn * criterion(cnn_logits, labels) + model.lambda_vit * criterion(vit_logits, labels)
        total_loss += loss.item() * imgs.size(0)

        p_cnn = torch.softmax(cnn_logits, 1)
        p_vit = torch.softmax(vit_logits, 1)
        fused = LAMBDA_CNN_INFER * p_cnn + LAMBDA_VIT_INFER * p_vit
        preds = fused.argmax(1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(fused.cpu().numpy())
        all_cnn_logits.extend(cnn_logits.cpu().numpy())
        all_vit_logits.extend(vit_logits.cpu().numpy())

        acc, f1 = _running_acc_f1(all_labels, all_preds)
        progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.4f}", f1=f"{f1:.4f}")

    metrics = compute_metrics(all_labels, all_preds, np.array(all_probs), num_classes)
    val_data = {
        "labels":      np.array(all_labels),
        "cnn_logits":  np.array(all_cnn_logits),
        "vit_logits":  np.array(all_vit_logits),
        "fused_probs": np.array(all_probs),
    }
    return total_loss / len(all_labels), metrics, val_data


def train_model(model, train_loader, val_loader, device, epochs, save_dir, fold_tag="",
                finetune_loaders=None, finetune_epochs=0, finetune_lr=1e-5):
    optimizer = optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999), eps=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    train_criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    eval_criterion  = nn.CrossEntropyLoss()

    swanlab.init(
        project="meteorite-classification",
        workspace="12311730",
        experiment_name=(f"hybrid_{fold_tag.replace(' ','_')}_{time.strftime('%H%M%S')}" if fold_tag else f"hybrid_single_{time.strftime('%H%M%S')}"),
        config={"architecture": "ResNet101+ViT-B/16", "lambda_cnn": model.lambda_cnn, "lambda_vit": model.lambda_vit,
                "learning_rate": MAX_LR, "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE, "epochs": epochs,
                "dropout_cnn": DROPOUT_RATE_CNN, "dropout_vit": DROPOUT_RATE_VIT, "label_smoothing": LABEL_SMOOTHING, "image_size": IMAGE_SIZE,
                "grad_clip": GRAD_CLIP_NORM, "optimizer": "AdamW", "scheduler": "CosineAnnealingLR", "eta_min": 1e-6},
    )

    best_acc = 0.0
    save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    last_path = save_dir / "last_model.pth"
    no_improve = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_top1": []}
    best_val_data = None   # ← 保存最佳 epoch 的验证集数据（用于温度缩放）

    tag = f" [{fold_tag}]" if fold_tag else ""
    print(f"\n{'='*60}\n  开始训练{tag}  epochs={epochs}  device={device}\n{'='*60}\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, train_criterion)
        val_loss, val_metrics, val_data = validate(model, val_loader, device, NUM_CLASSES, eval_criterion)
        scheduler.step()

        top1 = val_metrics["Top-1 Accuracy"]
        f1   = val_metrics["F1-Score"]
        lr   = scheduler.get_last_lr()[0]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_top1"].append(top1)

        if top1 > best_acc:
            best_acc = top1; no_improve = 0
            # 记录最佳 epoch 的验证数据，但不额外保存为文件（只保留内存中的 best_val_data）
            best_val_data = val_data   # ← 记录最佳时的验证数据
            flag = " <- best"
        else:
            no_improve += 1; flag = ""

        # 始终保存当前 epoch 的最后 checkpoint（覆盖），不再单独保存最优 checkpoint
        torch.save(model.state_dict(), str(last_path))

        elapsed = time.time() - t0
        print(f"Epoch [{epoch:3d}/{epochs}]{tag} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"| val_loss={val_loss:.4f} acc={top1:.4f} f1={f1:.4f} | lr={lr:.6f} time={elapsed:.1f}s{flag}")

        swanlab.log({"train/loss": train_loss, "train/acc": train_acc,
                     "val/loss": val_loss, "val/acc": top1, "val/f1": f1, "learning_rate": lr}, step=epoch)

        if no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    # ── 退火微调阶段（按序依次训练各 loader）──────────────
    if finetune_loaders and finetune_epochs > 0:
        global_ft_step = 0
        for phase_idx, (ft_loader, phase_name) in enumerate(finetune_loaders):
            print(f"\n{'='*60}")
            print(f"  退火微调{tag} [{phase_idx+1}/{len(finetune_loaders)}] {phase_name}: "
                  f"{finetune_epochs} epoch  lr={finetune_lr:.2e}  样本数={len(ft_loader.dataset)}")
            print(f"{'='*60}\n")
            for pg in optimizer.param_groups:
                pg["lr"] = finetune_lr
            ft_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=finetune_epochs, eta_min=1e-7)
            for ft_epoch in range(1, finetune_epochs + 1):
                t0 = time.time()
                ft_loss, ft_acc = train_one_epoch(model, ft_loader, optimizer, device, train_criterion)
                ft_scheduler.step()
                global_ft_step += 1
                lr = ft_scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                print(f"  [{phase_name}] Finetune [{ft_epoch:3d}/{finetune_epochs}]{tag} "
                      f"| ft_loss={ft_loss:.4f} ft_acc={ft_acc:.4f} | lr={lr:.2e} time={elapsed:.1f}s")
                swanlab.log({f"finetune_{phase_name}/loss": ft_loss,
                             f"finetune_{phase_name}/acc":  ft_acc,
                             "learning_rate": lr}, step=epochs + global_ft_step)
        # 微调全部结束后做一次验证，更新 best_val_data 用于 Stage2
        _, val_metrics_ft, val_data_ft = validate(model, val_loader, device, NUM_CLASSES, eval_criterion)
        top1_ft = val_metrics_ft["Top-1 Accuracy"]
        print(f"\n  微调后验证集 acc={top1_ft:.4f}  f1={val_metrics_ft['F1-Score']:.4f}")
        if top1_ft > best_acc:
            best_acc = top1_ft
            best_val_data = val_data_ft
            torch.save(model.state_dict(), str(last_path))
            print(f"  微调后 checkpoint 已更新 <- best")
        else:
            torch.save(model.state_dict(), str(last_path))  # 无论如何保存最终权重

    print(f"\n{tag} 训练完成! 最优 Top-1 Accuracy: {best_acc:.4f}")
    save_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(save_dir / "training_history.csv", index=False)
    swanlab.finish()

    return last_path, best_acc, history, best_val_data


# ──────────────────────────────────────────────
# 8. 推理（支持分支输出）
# ──────────────────────────────────────────────
@torch.no_grad()
def predict_probs(model, loader, device, temperature=1.0, return_branches=False):
    """返回 {image_id: prob_array}，可选同时返回各分支概率"""
    model.eval()
    id_to_prob, id_to_cnn, id_to_vit = {}, {}, {}
    for images, image_ids in tqdm(loader, desc="  Inference", leave=False):
        images = images.to(device, non_blocking=True)
        if return_branches:
            fused, p_cnn, p_vit = model(images, fusion_mode=True, return_branches=True, temperature=temperature)
            for prob, pc, pv, iid in zip(fused.cpu().numpy(), p_cnn.cpu().numpy(), p_vit.cpu().numpy(), image_ids):
                id_to_prob[iid] = prob
                id_to_cnn[iid]  = pc
                id_to_vit[iid]  = pv
        else:
            probs = model(images, fusion_mode=True, temperature=temperature)
            for prob, iid in zip(probs.cpu().numpy(), image_ids):
                id_to_prob[iid] = prob
    if return_branches:
        return id_to_prob, id_to_cnn, id_to_vit
    return id_to_prob


def predict_with_tta(model, test_dataset, device, tta_count=0, batch_size=BATCH_SIZE,
                     temperature=1.0, return_branches=False):
    if tta_count <= 0:
        test_dataset.set_transform(val_transform)
        loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        if return_branches:
            id_to_prob, id_to_cnn, id_to_vit = predict_probs(model, loader, device, temperature, True)
        else:
            id_to_prob = predict_probs(model, loader, device, temperature, False)
    else:
        n_augs = min(tta_count, len(tta_transforms))
        all_p, all_cnn, all_vit = {}, {}, {}
        for i in range(n_augs):
            test_dataset.set_transform(tta_transforms[i])
            loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
            if return_branches:
                p, pc, pv = predict_probs(model, loader, device, temperature, True)
                for iid in p:
                    all_p.setdefault(iid, []).append(p[iid])
                    all_cnn.setdefault(iid, []).append(pc[iid])
                    all_vit.setdefault(iid, []).append(pv[iid])
            else:
                p = predict_probs(model, loader, device, temperature, False)
                for iid in p: all_p.setdefault(iid, []).append(p[iid])

        id_to_prob = {iid: np.mean(v, 0) for iid, v in all_p.items()}
        if return_branches:
            id_to_cnn = {iid: np.mean(v, 0) for iid, v in all_cnn.items()}
            id_to_vit = {iid: np.mean(v, 0) for iid, v in all_vit.items()}

    id_to_pred = {iid: int(float(p[1]) >= DECISION_THRESHOLD) for iid, p in id_to_prob.items()}
    if return_branches:
        return id_to_pred, id_to_prob, id_to_cnn, id_to_vit
    return id_to_pred, id_to_prob


# ──────────────────────────────────────────────
# 8.5 第二阶段：温度缩放 + EM先验估计 + 先验校正 + 分支分歧
# ──────────────────────────────────────────────

def fit_temperature_scaling(val_data):
    """
    普通温度缩放（TS）：拟合单一温度参数 T。
    仅作对比基线使用，推荐用 fit_bcts 替代。
    """
    cnn_logits = torch.tensor(val_data["cnn_logits"], dtype=torch.float32)
    vit_logits = torch.tensor(val_data["vit_logits"], dtype=torch.float32)
    labels     = torch.tensor(val_data["labels"], dtype=torch.long)

    def nll(log_t):
        T     = float(np.exp(log_t))
        p_cnn = torch.softmax(cnn_logits / T, 1)
        p_vit = torch.softmax(vit_logits / T, 1)
        fused = LAMBDA_CNN_INFER * p_cnn + LAMBDA_VIT_INFER * p_vit
        return nn.CrossEntropyLoss()(torch.log(fused + 1e-9), labels).item()

    result = minimize_scalar(nll, bounds=(-2, 3), method="bounded")
    T = float(np.exp(result.x))
    print(f"[TS]   温度 T = {T:.4f}  (NLL = {result.fun:.4f})")
    return T


def fit_bcts(val_data):
    """
    偏差修正温度缩放（BCTS）：Alexandari, Kundaje & Shrikumar, ICML 2020。
    在普通 TS 基础上为每个类别增加独立偏置参数 b[k]，
    消除模型对某一类别的系统性高估/低估，使 EM 先验估计更可靠。

    校准公式：logit_calibrated[k] = logit[k] / T + b[k]
    若 b[1] < 0，说明模型对陨石类存在系统性高估，BCTS 自动修正。

    返回: (T: float, bias: np.array[2])
    """
    cnn_logits = torch.tensor(val_data["cnn_logits"], dtype=torch.float32)
    vit_logits = torch.tensor(val_data["vit_logits"], dtype=torch.float32)
    labels     = torch.tensor(val_data["labels"],     dtype=torch.long)

    log_T = nn.Parameter(torch.zeros(1))
    bias  = nn.Parameter(torch.zeros(NUM_CLASSES))
    optimizer = torch.optim.LBFGS([log_T, bias], lr=0.01, max_iter=1000,
                                   tolerance_grad=1e-7, tolerance_change=1e-9)

    def closure():
        optimizer.zero_grad()
        T     = torch.exp(log_T)
        p_cnn = torch.softmax(cnn_logits / T + bias, 1)
        p_vit = torch.softmax(vit_logits / T + bias, 1)
        fused = LAMBDA_CNN_INFER * p_cnn + LAMBDA_VIT_INFER * p_vit
        loss  = nn.functional.nll_loss(torch.log(fused + 1e-9), labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    T = float(torch.exp(log_T).detach().item())
    b = bias.detach().numpy().copy()

    print(f"[BCTS] 温度 T = {T:.4f}")
    print(f"[BCTS] 偏置   b = [非陨石: {b[0]:+.4f}, 陨石: {b[1]:+.4f}]")
    if b[1] < -0.05:
        print(f"[BCTS] ✓ 检测到陨石类系统性高估 (b[1]={b[1]:.4f})，已修正")
    elif b[1] > 0.05:
        print(f"[BCTS] ✓ 检测到陨石类系统性低估 (b[1]={b[1]:.4f})，已修正")
    else:
        print(f"[BCTS] 无显著系统性偏置，结果与普通 TS 接近")
    return T, b


def estimate_prior_em(calibrated_probs, train_prior, max_iter=200, tol=1e-7):
    """
    SLD/EM 算法：从无标签测试集估计真实先验分布。
    calibrated_probs : np.array [N, 2]，温度缩放后的测试概率
    train_prior      : list [p_non_meteorite, p_meteorite]
    返回: 估计的测试先验 np.array [2]
    """
    p_train = np.array(train_prior, dtype=np.float64)
    p_test  = p_train.copy()   # 初始化为训练先验

    for i in range(max_iter):
        p_old = p_test.copy()
        # E 步：用当前先验比率重新加权
        r        = p_test / (p_train + 1e-12)
        weighted = calibrated_probs * r[None, :]
        corrected = weighted / (weighted.sum(axis=1, keepdims=True) + 1e-12)
        # M 步：新先验 = 平均重加权后验
        p_test = corrected.mean(axis=0)
        p_test = p_test / p_test.sum()   # 归一化
        if np.abs(p_test - p_old).max() < tol:
            print(f"[EM先验] 收敛于第 {i+1} 次迭代")
            break

    print(f"[EM先验] 估计测试先验 → 非陨石: {p_test[0]:.4f}  陨石: {p_test[1]:.4f}")
    print(f"[EM先验] 对比训练先验 → 非陨石: {p_train[0]:.4f}  陨石: {p_train[1]:.4f}")
    return p_test


def apply_prior_correction(probs, p_train, p_test):
    """
    贝叶斯先验校正（Saerens 2002）。
    probs  : np.array [N, 2]
    返回   : 校正后的陨石概率 np.array [N]，决策阈值可用 0.5
    """
    p_train = np.array(p_train, dtype=np.float64)
    p_test  = np.array(p_test,  dtype=np.float64)
    r = p_test / (p_train + 1e-12)
    weighted  = probs * r[None, :]
    corrected = weighted / (weighted.sum(axis=1, keepdims=True) + 1e-12)
    return corrected[:, 1]   # 陨石的校正后概率


def apply_branch_disagreement(id_to_prob_cnn, id_to_prob_vit, id_to_corrected_prob,
                               test_ids, tau=0.3):
    """
    CNN↔ViT 分支分歧策略：
    当 |p_cnn_meteorite - p_vit_meteorite| > tau 时，
    不信任模型，保守预测非陨石（利用测试集 80% 非陨石先验）。
    否则，使用先验校正后的概率做决策。

    返回: {image_id: label}, disagreement_stats
    """
    preds = {}
    n_disagree = 0

    for img_id in test_ids:
        p_cnn = float(id_to_prob_cnn[img_id][1])   # CNN 陨石概率
        p_vit = float(id_to_prob_vit[img_id][1])   # ViT 陨石概率
        d     = abs(p_cnn - p_vit)
        p_corr = float(id_to_corrected_prob[img_id])

        if d > tau:
            preds[img_id] = 0   # 分歧 → 保守预测非陨石
            n_disagree += 1
        else:
            preds[img_id] = int(p_corr > 0.5)

    total = len(test_ids)
    n_meteor = sum(preds.values())
    stats = {
        "tau":          tau,
        "n_disagree":   n_disagree,
        "disagree_pct": n_disagree / total,
        "n_meteorite":  n_meteor,
        "n_non_meteor": total - n_meteor,
    }
    print(f"[分支分歧] tau={tau}  分歧样本: {n_disagree}/{total} ({n_disagree/total:.1%})")
    print(f"[分支分歧] 预测陨石(1): {n_meteor}  非陨石(0): {total - n_meteor}")
    return preds, stats


def run_stage2_pipeline(model, best_val_data, id_to_prob, id_to_cnn, id_to_vit,
                        test_ids, test_csv, run_dir, disagree_tau=0.3):
    """
    第二阶段完整流水线：
      BCTS校准 → EM先验估计 → 贝叶斯先验校正 → 分支分歧策略

    生成四份 submission：
      submission_empirical_corrected.csv : BCTS + 经验先验(86/194)校正
      submission_em_corrected.csv        : BCTS + EM先验估计 + 校正（ICML 2020最优方案）
      submission_disagreement.csv        : BCTS + EM先验 + 校正 + 分支分歧
      submission_top86.csv               : Top-K=86 强制提交（最直接利用先验知识）
    """
    print(f"\n{'='*60}")
    print(f"  第二阶段：BCTS校准 + EM先验估计 + 先验校正 + 分支分歧")
    print(f"{'='*60}")

    # Step 1: BCTS 校准（Alexandari et al. ICML 2020，优于普通 TS）
    # 普通 TS 无法消除每类的系统性偏置，导致 EM 估计不可靠
    # BCTS 增加每类独立偏置参数，修正后 EM 估计才有意义
    T, bcts_bias = fit_bcts(best_val_data)
    fit_temperature_scaling(best_val_data)   # 同时跑普通 TS 作对比参考

    # 将测试集原始概率用 BCTS 偏置重新校准
    raw_probs  = np.array([id_to_prob[iid] for iid in test_ids])   # [N, 2]
    bias_t     = torch.tensor(bcts_bias, dtype=torch.float32)
    raw_logits = torch.tensor(np.log(np.clip(raw_probs, 1e-9, 1)), dtype=torch.float32)
    bcts_probs = torch.softmax(raw_logits / T + bias_t, dim=1).numpy()   # [N, 2]

    print(f"\n[BCTS校准效果]")
    print(f"  校准前陨石平均概率: {raw_probs[:,1].mean():.4f}  "
          f"→ 校准后: {bcts_probs[:,1].mean():.4f}")

    # Step 2a: 经验先验校正（基于经验估算 86/194，使用 BCTS 概率）
    empirical_ratio  = 86 / 194
    known_test_prior = [1 - empirical_ratio, empirical_ratio]
    p1_known  = apply_prior_correction(bcts_probs, TRAIN_PRIOR, known_test_prior)
    known_preds = {iid: int(p1_known[i] > 0.5) for i, iid in enumerate(test_ids)}
    make_submission(known_preds, str(test_csv),
                    str(run_dir / "submission_empirical_corrected.csv"), allow_partial=True)
    print(f"[经验先验校正] 先验=[{known_test_prior[0]:.3f}, {known_test_prior[1]:.3f}]  "
          f"陨石: {sum(known_preds.values())}  非陨石: {len(test_ids)-sum(known_preds.values())}")

    # Step 2b: EM 估计测试先验（使用 BCTS 校准概率，系统性偏置已消除）
    print(f"\n[EM先验] 使用 BCTS 校准概率（系统性偏置已消除，估计更可靠）")
    em_test_prior = estimate_prior_em(bcts_probs, TRAIN_PRIOR)

    # Step 3: EM先验校正
    p1_em    = apply_prior_correction(bcts_probs, TRAIN_PRIOR, em_test_prior)
    em_preds = {iid: int(p1_em[i] > 0.5) for i, iid in enumerate(test_ids)}
    make_submission(em_preds, str(test_csv),
                    str(run_dir / "submission_em_corrected.csv"), allow_partial=True)
    print(f"[EM先验校正] EM先验=[{em_test_prior[0]:.3f}, {em_test_prior[1]:.3f}]  "
          f"陨石: {sum(em_preds.values())}  非陨石: {len(test_ids)-sum(em_preds.values())}")

    # Step 4: 分支分歧策略（基于 EM 校正后的概率）
    id_to_em_corrected = {iid: p1_em[i] for i, iid in enumerate(test_ids)}
    disagree_preds, disagree_stats = apply_branch_disagreement(
        id_to_cnn, id_to_vit, id_to_em_corrected, test_ids, tau=disagree_tau)
    make_submission(disagree_preds, str(test_csv),
                    str(run_dir / "submission_disagreement.csv"), allow_partial=True)

    # Step 5: Top-K=86（直接利用经验先验知识，不依赖模型概率分布）
    make_submission_topk(id_to_prob, str(test_csv),
                         str(run_dir / "submission_top86.csv"), k=86)

    # 保存详细分析数据
    rows = []
    for i, iid in enumerate(test_ids):
        rows.append({
            "id":               iid,
            "p_fused_raw":      float(raw_probs[i, 1]),
            "p_fused_bcts":     float(bcts_probs[i, 1]),
            "p_cnn":            float(id_to_cnn[iid][1]),
            "p_vit":            float(id_to_vit[iid][1]),
            "branch_disagree":  abs(float(id_to_cnn[iid][1]) - float(id_to_vit[iid][1])),
            "p_empirical_corr": float(p1_known[i]),
            "p_em_corr":        float(p1_em[i]),
            "pred_original":    int(np.argmax(raw_probs[i])),
            "pred_empirical":   known_preds[iid],
            "pred_em_corr":     em_preds[iid],
            "pred_disagree":    disagree_preds[iid],
        })
    stage2_df = pd.DataFrame(rows)
    stage2_df.to_csv(run_dir / "stage2_analysis.csv", index=False)

    # 打印对比摘要
    print(f"\n{'─'*58}")
    print(f"  策略对比摘要（目标：接近 86 张陨石）")
    print(f"{'─'*58}")
    print(f"  原始预测                → 陨石: {sum(r['pred_original']  for r in rows)}")
    print(f"  BCTS + 经验先验校正     → 陨石: {sum(r['pred_empirical'] for r in rows)}")
    print(f"  BCTS + EM先验校正       → 陨石: {sum(r['pred_em_corr']   for r in rows)}")
    print(f"  BCTS + EM + 分支分歧    → 陨石: {sum(r['pred_disagree']  for r in rows)}")
    print(f"  Top-K=86（强制）        → 陨石: 86")
    print(f"{'─'*58}")
    print(f"  BCTS: T={T:.4f}  b=[{bcts_bias[0]:+.3f}, {bcts_bias[1]:+.3f}]")
    print(f"  EM先验: [{em_test_prior[0]:.3f}, {em_test_prior[1]:.3f}]  tau={disagree_tau}")
    print(f"  stage2_analysis.csv → {run_dir / 'stage2_analysis.csv'}")

    return T, bcts_bias, em_test_prior, disagree_stats


# ──────────────────────────────────────────────
# 9. 集成推理
# ──────────────────────────────────────────────
def ensemble_predict(models_list, test_dataset, device, tta_count=0, batch_size=BATCH_SIZE):
    n_models = len(models_list)
    all_model_probs = {}
    for model in models_list:
        _, id_to_prob = predict_with_tta(model, test_dataset, device, tta_count, batch_size)
        for iid, prob in id_to_prob.items():
            all_model_probs.setdefault(iid, []).append(prob)
    ensemble_probs = {iid: np.mean(v, 0) for iid, v in all_model_probs.items()}
    ensemble_preds = {iid: int(float(p[1]) >= DECISION_THRESHOLD) for iid, p in ensemble_probs.items()}
    return ensemble_preds, ensemble_probs


# ──────────────────────────────────────────────
# 10. submission 生成
# ──────────────────────────────────────────────
def make_submission(id_to_pred, template_csv_path, output_path, allow_partial=False):
    ids_df = pd.read_csv(template_csv_path)
    if allow_partial:
        ids_df = ids_df[ids_df["id"].astype(str).isin(id_to_pred.keys())].copy()
    ids_df["label"] = ids_df["id"].map(id_to_pred)
    if ids_df["label"].isna().any():
        missing = ids_df.loc[ids_df["label"].isna(), "id"].head(5).tolist()
        raise RuntimeError(f"Missing predictions: {missing}")
    ids_df["label"] = ids_df["label"].astype(int)
    ids_df.to_csv(output_path, index=False)
    print(f"  → {output_path}")


def make_submission_with_threshold(id_to_prob, template_csv_path, output_dir, thresholds=None, allow_partial=False):
    if thresholds is None: thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    meteor_probs = {iid: float(p[1]) if hasattr(p, '__len__') else float(p) for iid, p in id_to_prob.items()}
    pd.DataFrame(meteor_probs.items(), columns=["id", "prob_meteorite"]).to_csv(output_dir / "probs.csv", index=False)
    ids_df = pd.read_csv(template_csv_path)
    if allow_partial: ids_df = ids_df[ids_df["id"].astype(str).isin(meteor_probs.keys())].copy()
    total = len(ids_df)
    print(f"\n{'─'*52}"); print(f"  阈值  预测陨石  非陨石  占比"); print(f"{'─'*52}")
    for t in thresholds:
        ids_df["label"] = (ids_df["id"].map(meteor_probs) >= t).astype(int)
        n1 = int(ids_df["label"].sum())
        ids_df[["id","label"]].to_csv(output_dir / f"submission_t{int(t*100):02d}.csv", index=False)
        print(f"  {t:.2f}  {n1:8d}  {total-n1:7d}  {n1/total:6.1%}")
    print(f"{'─'*52}")


def make_submission_topk(id_to_prob, template_csv_path, output_path, k=101):
    meteor_probs = {iid: float(p[1]) if hasattr(p, '__len__') else float(p) for iid, p in id_to_prob.items()}
    ids_df = pd.read_csv(template_csv_path)
    ids_df["prob"] = ids_df["id"].map(meteor_probs)
    ids_sorted = ids_df.sort_values("prob", ascending=False).reset_index(drop=True)
    ids_sorted["label"] = 0
    ids_sorted.loc[:min(k, len(ids_sorted))-1, "label"] = 1
    result = ids_df[["id"]].merge(ids_sorted[["id","label"]], on="id", how="left")
    result["label"] = result["label"].fillna(0).astype(int)
    result.to_csv(output_path, index=False)
    n1 = int(result["label"].sum())
    print(f"[Top-K] k={k}  陨石={n1}  非陨石={len(result)-n1}")
    return result


# ──────────────────────────────────────────────
# 11. 主流程
# ──────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv",      type=str,   default=str(DEFAULT_TRAIN_CSV))
    parser.add_argument("--train-img-dir",  type=str,   default=str(DEFAULT_TRAIN_IMG_DIR))
    parser.add_argument("--test-csv",       type=str,   default=str(DEFAULT_TEST_CSV))
    parser.add_argument("--test-img-dir",   type=str,   default=str(DEFAULT_TEST_IMG_DIR))
    parser.add_argument("--output-dir",     type=str,   default=str(OUTPUT_DIR))
    parser.add_argument("--epochs",         type=int,   default=EPOCHS)
    parser.add_argument("--batch-size",     type=int,   default=BATCH_SIZE)
    parser.add_argument("--num-workers",    type=int,   default=NUM_WORKERS)
    parser.add_argument("--cnn-backbone",   type=str,   default=CNN_BACKBONE,
                        choices=["resnet101","efficientnet_b3","convnext_small"])
    parser.add_argument("--mode",           type=str,   default="single", choices=["single","kfold"])
    parser.add_argument("--n-folds",        type=int,   default=5)
    parser.add_argument("--tta",            type=int,   default=0)
    parser.add_argument("--smoke",          action="store_true")
    parser.add_argument("--use-yoloe-cache",action="store_true")
    parser.add_argument("--yoloe-cache-dir",type=str,   default=str(Path(__file__).resolve().parents[2] / "yoloe_cache"))
    parser.add_argument("--kfold-thresh",   type=float, default=0.7)
    parser.add_argument("--disagree-tau",   type=float, default=0.3,
                        help="CNN↔ViT 分支分歧阈值，超过此值保守预测非陨石")
    parser.add_argument("--finetune-epochs", type=int,   default=10,
                        help="训练结束后对末尾难例微调的 epoch 数（0=不启用）")
    parser.add_argument("--finetune-lr",    type=float, default=5e-5,
                        help="难例微调阶段学习率（默认 1e-5）")
    parser.add_argument("--hard-samples",   type=int,   default=100,
                        help="训练集末尾视为难例的样本数（默认 100）")
    return parser.parse_args()


def build_model(backbone):
    return HybridCNNTransformer(NUM_CLASSES, LAMBDA_CNN, LAMBDA_VIT, backbone).to(DEVICE)


def get_test_ids(test_csv, test_img_dir, smoke=False):
    raw_ids = pd.read_csv(test_csv)["id"].astype(str).tolist()
    valid_ids = [tid for tid in raw_ids if (Path(test_img_dir)/tid).exists()
                 or list(Path(test_img_dir).glob(f"{Path(tid).stem}.*"))]
    print(f"  测试集: CSV {len(raw_ids)} 条, 实际找到 {len(valid_ids)} 张")
    return valid_ids[:64] if smoke else valid_ids


def _load_yoloe_cache(cache_dir, all_ids, split):
    cache_dir = Path(cache_dir)
    id_to_cache = {iid: cache_dir / split / f"{Path(iid).stem}.png" for iid in all_ids}
    hit = sum(p.exists() for p in id_to_cache.values())
    print(f"[YOLOE缓存/{split}] 命中: {hit}/{len(all_ids)}")
    if hit == 0: print("[YOLOE缓存] ⚠️  缓存为空，请先运行: python yoloe_preprocess.py")
    return id_to_cache


def run_single(args):
    train_csv = Path(args.train_csv); train_img_dir = Path(args.train_img_dir)
    test_csv  = Path(args.test_csv);  test_img_dir  = Path(args.test_img_dir)
    use_yoloe = args.use_yoloe_cache

    run_id  = time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_csv)
    if "id" in train_df.columns and "image_id" not in train_df.columns:
        train_df = train_df.rename(columns={"id": "image_id"})
    if args.smoke: train_df = train_df.sample(128, random_state=SEED).reset_index(drop=True)

    all_train_ids = train_df["image_id"].astype(str).tolist()
    test_ids      = get_test_ids(test_csv, test_img_dir, args.smoke)

    train_cache = _load_yoloe_cache(args.yoloe_cache_dir, all_train_ids, "train") if use_yoloe else {}
    test_cache  = _load_yoloe_cache(args.yoloe_cache_dir, test_ids, "test")       if use_yoloe else {}

    # 使用 9:1 划分训练/验证集（以前为 8:2）
    n_val = max(1, int(0.1 * len(train_df))); n_train = len(train_df) - n_val
    # 不打乱：固定取前 10% 为验证集
    train_df_part = train_df.iloc[n_val:]   # 训练部分（末尾含难例）
    train_set = MeteoriteDataset(train_df_part,          train_img_dir, train_transform, use_yoloe, train_cache)
    val_set   = MeteoriteDataset(train_df.iloc[:n_val],  train_img_dir, val_transform,   use_yoloe, train_cache)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # 退火微调 loaders：先验证集，再末尾难例
    n_hard = min(args.hard_samples, len(train_df_part))
    hard_df   = train_df_part.iloc[-n_hard:]
    hard_ft_set = MeteoriteDataset(hard_df, train_img_dir, train_transform, use_yoloe, train_cache)
    hard_ft_loader = DataLoader(hard_ft_set, batch_size=min(args.batch_size, n_hard),
                                shuffle=True, num_workers=args.num_workers, pin_memory=True)
    finetune_loaders = [(hard_ft_loader, "难例")]
    print(f"  训练集: {n_train}  验证集: {n_val}  难例（末尾）: {n_hard}")

    model = build_model(args.cnn_backbone)
    epochs = 1 if args.smoke else args.epochs
    best_path, best_acc, history, best_val_data = train_model(
        model, train_loader, val_loader, DEVICE, epochs=epochs, save_dir=run_dir,
        finetune_loaders=finetune_loaders, finetune_epochs=args.finetune_epochs, finetune_lr=args.finetune_lr)
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))

    # 推理（含分支概率，用于分歧策略）
    test_dataset = MeteoriteTestDataset(test_img_dir, test_ids, use_yoloe_cache=use_yoloe, id_to_cache=test_cache)
    _, id_to_prob, id_to_cnn, id_to_vit = predict_with_tta(
        model, test_dataset, DEVICE, tta_count=args.tta,
        batch_size=args.batch_size, return_branches=True)

    # 原始 submission
    id_to_pred = {iid: int(float(p[1]) >= DECISION_THRESHOLD) for iid, p in id_to_prob.items()}
    make_submission(id_to_pred, str(test_csv), str(run_dir / "submission.csv"), allow_partial=True)
    make_submission_with_threshold(id_to_prob, str(test_csv), run_dir / "thresholds",
                                   thresholds=[0.3,0.4,0.5,0.6,0.7,0.8,0.9], allow_partial=True)
    for k in [80, 83, 86, 89, 92]:
        make_submission_topk(id_to_prob, str(test_csv), str(run_dir / f"submission_top{k}.csv"), k=k)

    # ── 第二阶段流水线 ──────────────────────────
    T, bcts_bias, em_prior, _ = run_stage2_pipeline(
        model, best_val_data, id_to_prob, id_to_cnn, id_to_vit,
        test_ids, test_csv, run_dir, disagree_tau=args.disagree_tau)

    result = {"run_id": run_id, "mode": "single", "tta": args.tta,
              "use_yoloe": use_yoloe, "best_acc": float(best_acc)}
    with open(run_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n全部完成! → {run_dir}")
    return run_dir


def run_kfold(args):
    train_csv = Path(args.train_csv); train_img_dir = Path(args.train_img_dir)
    test_csv  = Path(args.test_csv);  test_img_dir  = Path(args.test_img_dir)
    use_yoloe = args.use_yoloe_cache

    run_id  = time.strftime("kfold_%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_csv)
    if "id" in train_df.columns and "image_id" not in train_df.columns:
        train_df = train_df.rename(columns={"id": "image_id"})
    if args.smoke: train_df = train_df.sample(128, random_state=SEED).reset_index(drop=True)

    all_train_ids = train_df["image_id"].astype(str).tolist()
    test_ids      = get_test_ids(test_csv, test_img_dir, args.smoke)

    train_cache = _load_yoloe_cache(args.yoloe_cache_dir, all_train_ids, "train") if use_yoloe else {}
    test_cache  = _load_yoloe_cache(args.yoloe_cache_dir, test_ids, "test")       if use_yoloe else {}

    labels  = train_df["label"].values
    n_folds = args.n_folds
    skf     = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    fold_models, fold_accs = [], []
    oof_preds = np.full(len(train_df), -1, dtype=int)
    oof_probs = np.zeros((len(train_df), NUM_CLASSES))
    all_best_val_data = []   # 收集各折的验证数据用于温度缩放

    print(f"\n{'#'*60}\n  K-Fold 训练: {n_folds} 折\n{'#'*60}")

    for fold_i, (train_idx, val_idx) in enumerate(skf.split(train_df, labels)):
        fold_tag = f"Fold {fold_i+1}/{n_folds}"
        fold_dir = run_dir / f"fold_{fold_i+1}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'─'*60}\n  {fold_tag}: train={len(train_idx)} val={len(val_idx)}\n{'─'*60}")

        train_set = MeteoriteDataset(train_df.iloc[train_idx], train_img_dir, train_transform, use_yoloe, train_cache)
        val_set   = MeteoriteDataset(train_df.iloc[val_idx],   train_img_dir, val_transform,   use_yoloe, train_cache)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=True)
        val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        seed_everything(SEED + fold_i)
        model = build_model(args.cnn_backbone)
        epochs = 1 if args.smoke else args.epochs
        best_path, best_acc, _, best_val_data = train_model(
            model, train_loader, val_loader, DEVICE, epochs=epochs, save_dir=fold_dir, fold_tag=fold_tag)
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        fold_models.append(model); fold_accs.append(best_acc)
        all_best_val_data.append(best_val_data)

        # OOF
        val_ids_list = train_df.iloc[val_idx]["image_id"].astype(str).tolist()
        oof_ds = MeteoriteTestDataset(train_img_dir, val_ids_list, use_yoloe_cache=use_yoloe, id_to_cache=train_cache)
        oof_ds.set_transform(val_transform)
        oof_loader = DataLoader(oof_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
        oof_prob_dict = predict_probs(model, oof_loader, DEVICE)
        for gi in val_idx:
            iid = str(train_df.iloc[gi]["image_id"])
            if iid in oof_prob_dict:
                oof_probs[gi] = oof_prob_dict[iid]
                oof_preds[gi] = int(np.argmax(oof_prob_dict[iid]))
        print(f"  {fold_tag} 完成: val_acc={best_acc:.4f}")

    # OOF 评估
    valid_mask  = oof_preds >= 0
    oof_metrics = compute_metrics(labels[valid_mask], oof_preds[valid_mask], oof_probs[valid_mask], NUM_CLASSES)
    print(f"\n{'='*60}\n  K-Fold OOF 评估\n{'='*60}")
    print(f"  各折 val_acc: {[f'{a:.4f}' for a in fold_accs]}")
    print(f"  平均 val_acc: {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")
    for k, v in oof_metrics.items():
        if k == "Confusion Matrix": print(f"  {k}:\n{v}")
        else: print(f"  {k}: {v:.4f}")

    oof_df = train_df.copy()
    oof_df["oof_pred"] = oof_preds; oof_df["oof_prob_0"] = oof_probs[:,0]; oof_df["oof_prob_1"] = oof_probs[:,1]
    oof_df.to_csv(run_dir / "oof_predictions.csv", index=False)

    # 集成推理（含分支）
    print("\n[集成推理] 对测试集进行多模型集成推理...")
    test_dataset = MeteoriteTestDataset(test_img_dir, test_ids, use_yoloe_cache=use_yoloe, id_to_cache=test_cache)

    fold_meteor_probs = {iid: [] for iid in test_ids}
    fold_cnn_probs    = {iid: [] for iid in test_ids}
    fold_vit_probs    = {iid: [] for iid in test_ids}

    for fold_i, model in enumerate(fold_models):
        print(f"  Fold {fold_i+1}/{len(fold_models)} 推理中...")
        _, id_to_prob, id_to_cnn, id_to_vit = predict_with_tta(
            model, test_dataset, DEVICE, tta_count=args.tta,
            batch_size=args.batch_size, return_branches=True)

        fold_preds = {iid: int(np.argmax(p)) for iid, p in id_to_prob.items()}
        make_submission(fold_preds, str(test_csv), str(run_dir / f"submission_fold{fold_i+1}.csv"), allow_partial=True)

        for iid in test_ids:
            if iid in id_to_prob:
                fold_meteor_probs[iid].append(float(id_to_prob[iid][1]))
                fold_cnn_probs[iid].append(id_to_cnn[iid])
                fold_vit_probs[iid].append(id_to_vit[iid])

    # 全折阈值策略（主 submission）
    kfold_thresh = args.kfold_thresh
    conservative_preds = {}
    avg_probs_dict = {}
    avg_cnn_dict   = {}
    avg_vit_dict   = {}

    for iid in test_ids:
        probs_list = fold_meteor_probs[iid]
        all_above  = all(p >= kfold_thresh for p in probs_list)
        conservative_preds[iid] = 1 if all_above else 0
        avg_p1 = float(np.mean(probs_list))
        avg_probs_dict[iid] = np.array([1 - avg_p1, avg_p1])
        avg_cnn_dict[iid]   = np.mean(fold_cnn_probs[iid], axis=0)
        avg_vit_dict[iid]   = np.mean(fold_vit_probs[iid], axis=0)

    n1 = sum(conservative_preds.values())
    print(f"[全折阈值] threshold={kfold_thresh}  陨石: {n1}  非陨石: {len(test_ids)-n1}")

    make_submission(conservative_preds, str(test_csv), str(run_dir / "submission.csv"), allow_partial=True)

    avg_preds = {iid: int(float(p[1]) >= DECISION_THRESHOLD) for iid, p in avg_probs_dict.items()}
    make_submission(avg_preds, str(test_csv), str(run_dir / "submission_avg_prob.csv"), allow_partial=True)

    make_submission_with_threshold(avg_probs_dict, str(test_csv), run_dir/"thresholds",
                                   thresholds=[0.3,0.4,0.5,0.6,0.7,0.8,0.9], allow_partial=True)

    # 保存各折概率
    prob_rows = []
    for iid in test_ids:
        row = {"id": iid}
        for fi, p in enumerate(fold_meteor_probs[iid]): row[f"fold{fi+1}_prob"] = p
        row["all_above_thresh"] = int(conservative_preds[iid])
        row["avg_prob"] = float(np.mean(fold_meteor_probs[iid]))
        prob_rows.append(row)
    pd.DataFrame(prob_rows).to_csv(run_dir / "fold_probs.csv", index=False)

    for k in [30, 33, 36, 39, 42, 45]:
        make_submission_topk(avg_probs_dict, str(test_csv), str(run_dir / f"submission_top{k}.csv"), k=k)

    # ── 第二阶段流水线（用各折验证数据合并后的 val_data）──
    # 合并所有折的验证数据
    merged_val_data = {
        "cnn_logits": np.concatenate([d["cnn_logits"] for d in all_best_val_data]),
        "vit_logits": np.concatenate([d["vit_logits"] for d in all_best_val_data]),
        "labels":     np.concatenate([d["labels"]     for d in all_best_val_data]),
    }
    T, bcts_bias, em_prior, _ = run_stage2_pipeline(
        None, merged_val_data, avg_probs_dict, avg_cnn_dict, avg_vit_dict,
        test_ids, test_csv, run_dir, disagree_tau=args.disagree_tau)

    result = {
        "run_id": run_id, "mode": "kfold", "n_folds": n_folds,
        "tta": args.tta, "fold_accs": fold_accs,
        "mean_acc": float(np.mean(fold_accs)), "std_acc": float(np.std(fold_accs)),
    }
    with open(run_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n全部完成! → {run_dir}")
    return run_dir


def main():
    args = parse_args()
    for p, name in [(args.train_csv,"Train CSV"),(args.train_img_dir,"Train images"),
                    (args.test_csv,"Test CSV"),(args.test_img_dir,"Test images")]:
        if not Path(p).exists():
            raise FileNotFoundError(f"{name} not found: {p}")
    print(f"[Config] mode={args.mode}  tta={args.tta}  backbone={args.cnn_backbone}  disagree_tau={args.disagree_tau}")
    if args.mode == "kfold": print(f"[Config] n_folds={args.n_folds}  kfold_thresh={args.kfold_thresh}")
    if args.mode == "single": run_single(args)
    elif args.mode == "kfold": run_kfold(args)

if __name__ == "__main__":
    main()