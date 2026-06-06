"""
infer.py — 纯推理脚本
用法：python infer.py --weights /path/to/last_model.pth
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet101_Weights, ViT_B_16_Weights
from PIL import Image


SEED               = 42
NUM_CLASSES        = 2
IMAGE_SIZE         = 224
BATCH_SIZE         = 32
DEVICE             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAMBDA_CNN_INFER   = 0.5
LAMBDA_VIT_INFER   = 0.5
DECISION_THRESHOLD = 0.52
DROPOUT_RATE_CNN   = 0.0
DROPOUT_RATE_VIT   = 0.0
IMAGENET_MEAN      = [0.485, 0.456, 0.406]
IMAGENET_STD       = [0.229, 0.224, 0.225]

_SERVER_ROOT      = Path("/root/ds/assignment/project/陨石2")
_DEFAULT_TEST_CSV = _SERVER_ROOT / "submission.csv"
_DEFAULT_TEST_DIR = _SERVER_ROOT / "test_images"


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()


class MaskCorners:
    def __init__(self, mask_ratio=0.15, mask_top=True):
        self.mask_ratio = mask_ratio
        self.mask_top   = mask_top

    def __call__(self, img):
        arr = np.array(img)
        h, w = arr.shape[:2]
        m = int(min(h, w) * self.mask_ratio)
        arr[:m, :m] = arr[:m, -m:] = arr[-m:, :m] = arr[-m:, -m:] = 128
        if self.mask_top:
            arr[:int(h * 0.08), :] = 128
        return Image.fromarray(arr)


val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    MaskCorners(mask_ratio=0.15, mask_top=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class MeteoriteTestDataset(Dataset):
    def __init__(self, test_img_dir, ids, transform=None):
        self.test_img_dir = Path(test_img_dir)
        self.ids          = list(ids)
        self.transform    = transform

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        iid  = self.ids[idx]
        path = self.test_img_dir / iid
        if not path.exists():
            cands = list(self.test_img_dir.glob(f"{Path(iid).stem}.*"))
            if cands: path = cands[0]
            else: raise FileNotFoundError(f"找不到: {iid}")
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, iid


class HybridCNNTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        cnn = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
        in_feat = cnn.fc.in_features
        cnn.fc  = nn.Sequential(nn.Dropout(p=DROPOUT_RATE_CNN), nn.Linear(in_feat, NUM_CLASSES))
        self.cnn_branch = cnn

        vit = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        in_feat_vit = vit.heads.head.in_features
        vit.heads.head = nn.Sequential(nn.Dropout(p=DROPOUT_RATE_VIT), nn.Linear(in_feat_vit, NUM_CLASSES))
        self.vit_branch = vit
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        p_cnn = self.softmax(self.cnn_branch(x))
        p_vit = self.softmax(self.vit_branch(x))
        return LAMBDA_CNN_INFER * p_cnn + LAMBDA_VIT_INFER * p_vit


@torch.no_grad()
def run_inference(model, loader):
    model.eval()
    id_to_prob = {}
    for images, image_ids in tqdm(loader, desc="Inference"):
        images = images.to(DEVICE, non_blocking=True)
        probs  = model(images)
        for prob, iid in zip(probs.cpu().numpy(), image_ids):
            id_to_prob[iid] = prob
    return id_to_prob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",      type=str, required=True)
    parser.add_argument("--test-csv",     type=str, default=str(_DEFAULT_TEST_CSV))
    parser.add_argument("--test-img-dir", type=str, default=str(_DEFAULT_TEST_DIR))
    parser.add_argument("--output",       type=str, default="submission_infer.csv")
    parser.add_argument("--batch-size",   type=int, default=BATCH_SIZE)
    parser.add_argument("--threshold",    type=float, default=DECISION_THRESHOLD)
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Weights: {args.weights}")
    print(f"Threshold: {args.threshold}")

    model = HybridCNNTransformer().to(DEVICE)
    model.load_state_dict(torch.load(args.weights, map_location=DEVICE))

    test_csv     = Path(args.test_csv)
    test_img_dir = Path(args.test_img_dir)
    raw_ids      = pd.read_csv(test_csv)["id"].astype(str).tolist()
    test_ids     = [tid for tid in raw_ids
                    if (test_img_dir / tid).exists()
                    or list(test_img_dir.glob(f"{Path(tid).stem}.*"))]
    print(f"Test images: {len(test_ids)}/{len(raw_ids)}")

    dataset = MeteoriteTestDataset(test_img_dir, test_ids, val_transform)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    id_to_prob = run_inference(model, loader)
    id_to_pred = {iid: int(float(p[1]) >= args.threshold) for iid, p in id_to_prob.items()}

    ids_df = pd.read_csv(test_csv)
    ids_df = ids_df[ids_df["id"].astype(str).isin(id_to_pred)].copy()
    ids_df["label"] = ids_df["id"].map(id_to_pred).astype(int)
    ids_df.to_csv(args.output, index=False)

    n1 = int(ids_df["label"].sum())
    print(f"label=1: {n1},  label=0: {len(ids_df)-n1}")
    print(f"→ {args.output}")


if __name__ == "__main__":
    main()
