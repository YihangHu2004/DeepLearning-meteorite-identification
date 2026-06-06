"""
YOLOE 石头提取预处理脚本
========================
两级提取策略（已移除 SAM 全自动）：
  1. YOLOE 文本提示（带边缘过滤）
  2. OWL-ViT 定位 + SAM 框提示（OWL-ViT 检测到时不做边缘过滤）

提取不到的图片直接跳过（不保存），仅保留成功提取的图片。
输出：
  - 提取后的图片（白色背景）→ cache_dir/train/
  - 过滤后的标签文件       → cache_dir/train_labels.csv（仅包含提取成功的图片）

边缘过滤逻辑（YOLOE 共用，OWL-ViT+SAM 框提示不做边缘过滤）：
  - 扩展边缘带：距图片边缘 SAM_MARGIN(5%) 以内的像素视为边缘
  - 硬过滤：border_contact > SAM_BORDER_THRESH(20%) 视为背景
  - 兜底：全部被过滤时不过滤（石头布满边缘的情况）
"""

import argparse
import cv2
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

try:
    from ultralytics import YOLOE
    _YOLOE_AVAILABLE = True
except ImportError:
    _YOLOE_AVAILABLE = False

# ──────────────────────────────────────────────
# 默认路径配置
# ──────────────────────────────────────────────
BASE_DIR              = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_CSV     = Path("/root/ds/assignment/project/陨石2/train_labels.csv")
DEFAULT_TRAIN_IMG_DIR = BASE_DIR / "/root/ds/assignment/project/陨石2/train_images/train_images"
DEFAULT_TEST_CSV      = BASE_DIR / "/root/ds/assignment/project/陨石2/test_images/test_images"
DEFAULT_TEST_IMG_DIR  = BASE_DIR / "/root/ds/assignment/project/陨石2/sample_submission.csv"
DEFAULT_CACHE_DIR     = BASE_DIR / "/root/ds/assignment/project/陨石2/yolo_data"

# ──────────────────────────────────────────────
# 超参
# ──────────────────────────────────────────────
DEFAULT_MODEL       = "/root/ds/assignment/project/yoloe-26x-seg.pt"
DEFAULT_SAM_CKPT    = Path("/root/ds/assignment/project/sam_vit_b_01ec64.pth")
DEFAULT_OWLVIT_DIR  = Path("/root/ds/assignment/project/陨石/scripts/owlvit-base-patch32")
DEFAULT_DEVICE      = "cuda"
DEFAULT_CONF        = 0.00999
PROMPTS             = ["rock", "stone", "mineral", "pebble", "ore", "fossil",
                       "rock fragment", "geological specimen"]
OWLVIT_LABELS       = PROMPTS
OWLVIT_SCORE_THRESH = 0.05
SAM_BORDER_THRESH   = 0.20   # 边缘接触率阈值，超过视为背景
SAM_MARGIN          = 0.05   # 扩展边缘带宽度（占图片比例）
BG_COLOR            = (255, 255, 255)  # 白色背景
PAD                 = 20

# 全局模型实例（懒加载）
_model_prompt       = None
_owlvit_processor   = None
_owlvit_model       = None
_sam_predictor_inst = None


# ──────────────────────────────────────────────
# 通用边缘工具
# ──────────────────────────────────────────────
def _border_contact(mask: np.ndarray, h: int, w: int,
                    margin: float = SAM_MARGIN) -> float:
    """扩展边缘接触率：距图片边缘 margin 比例以内的像素都算边缘。"""
    margin_h = max(1, int(h * margin))
    margin_w = max(1, int(w * margin))
    border = np.zeros((h, w), dtype=bool)
    border[:margin_h, :]  = True
    border[-margin_h:, :] = True
    border[:, :margin_w]  = True
    border[:, -margin_w:] = True
    return float((mask & border).sum()) / max(mask.sum(), 1)


def _is_border_mask(mask: np.ndarray, h: int, w: int) -> bool:
    """扩展边缘接触率 > SAM_BORDER_THRESH 则视为背景边缘 mask"""
    return _border_contact(mask, h, w) > SAM_BORDER_THRESH


def _cleanup_mask(mask: np.ndarray) -> np.ndarray:
    """连通域清理：只保留最大连通区域"""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest)


def _apply_mask(pil_img: Image.Image, mask: np.ndarray) -> Image.Image:
    """白色背景填充 + 按掩码 bbox 裁剪"""
    h, w = mask.shape
    arr = np.array(pil_img).copy()
    arr[~mask] = BG_COLOR
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    arr = arr[max(0, r0-PAD):min(h, r1+PAD), max(0, c0-PAD):min(w, c1+PAD)]
    return Image.fromarray(arr)


# ──────────────────────────────────────────────
# YOLOE 推理
# ──────────────────────────────────────────────
def load_model(model_path: str, device: str) -> "YOLOE":
    global _model_prompt
    if _model_prompt is not None:
        return _model_prompt
    if not _YOLOE_AVAILABLE:
        raise ImportError("ultralytics 未安装: pip install ultralytics")
    print(f"[YOLOE] 加载: {model_path}")
    m = YOLOE(model_path)
    m.set_classes(PROMPTS)
    print(f"[YOLOE] 加载成功  提示词: {PROMPTS}")
    _model_prompt = m
    return m


def _run_inference(model, img_path: Path, device: str, conf_thresh: float):
    try:
        results = model(str(img_path), conf=max(conf_thresh, 0.001),
                        verbose=False, device=device)
        return results[0]
    except Exception as e:
        print(f"  ⚠️  推理失败 {img_path.name}: {e}")
        return None


def _pick_best_mask(result, h: int, w: int, conf_thresh: float):
    """
    YOLOE top-1 掩码选取（带边缘过滤）：
    1. 面积过滤（提前排除噪声 < 2.5%）
    2. 边缘硬过滤；全部被过滤时不过滤（兜底）
    3. 综合打分：conf² × area × (1-border)²
    4. 连通域清理
    """
    if result is None or result.masks is None or len(result.masks.data) == 0:
        return None

    items = []
    for i in range(len(result.masks.data)):
        mask = cv2.resize(
            result.masks.data[i].cpu().numpy().astype(np.uint8),
            (w, h), interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        conf = float(result.boxes.conf[i])
        if conf < conf_thresh:
            continue
        area_ratio = float(mask.sum()) / (h * w)
        if area_ratio < 0.025:
            continue
        bc = _border_contact(mask, h, w)
        items.append({"mask": mask, "conf": conf,
                      "area_ratio": area_ratio, "border_contact": bc})

    if not items:
        return None

    non_border = [it for it in items if not _is_border_mask(it["mask"], h, w)]
    pool = non_border if non_border else items

    best = max(pool, key=lambda x:
               x["conf"] ** 2 * x["area_ratio"] * (1 - x["border_contact"]) ** 2)
    return _cleanup_mask(best["mask"])


# ──────────────────────────────────────────────
# OWL-ViT + SAM 框提示（不做边缘过滤）
# ──────────────────────────────────────────────
def _load_owlvit(owlvit_dir: Path, device: str):
    global _owlvit_processor, _owlvit_model
    if _owlvit_model is not None:
        return _owlvit_processor, _owlvit_model
    try:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    except ImportError:
        raise ImportError("transformers 未安装: pip install transformers")
    if not owlvit_dir.exists():
        raise FileNotFoundError(f"OWL-ViT 模型目录不存在: {owlvit_dir}")
    print(f"[OWL-ViT] 加载: {owlvit_dir}")
    _owlvit_processor = AutoProcessor.from_pretrained(str(owlvit_dir), local_files_only=True)
    _owlvit_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        str(owlvit_dir), local_files_only=True).to(device)
    _owlvit_model.eval()
    print("[OWL-ViT] 加载成功")
    return _owlvit_processor, _owlvit_model


def _load_sam_predictor(sam_ckpt: Path, device: str):
    global _sam_predictor_inst
    if _sam_predictor_inst is not None:
        return _sam_predictor_inst
    from segment_anything import sam_model_registry, SamPredictor
    if not sam_ckpt.exists():
        raise FileNotFoundError(f"SAM 权重不存在: {sam_ckpt}")
    print(f"[SAM-box] 加载: {sam_ckpt}")
    sam = sam_model_registry["vit_b"](checkpoint=str(sam_ckpt))
    sam.to(device)
    _sam_predictor_inst = SamPredictor(sam)
    print("[SAM-box] 加载成功")
    return _sam_predictor_inst


def _owlvit_detect(pil_img: Image.Image, processor, model,
                   device: str, score_thresh: float = OWLVIT_SCORE_THRESH):
    import torch
    W, H = pil_img.size
    inputs = processor(text=[OWLVIT_LABELS], images=pil_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs, target_sizes=[(H, W)], threshold=score_thresh)[0]
    if len(results["scores"]) == 0:
        return None
    best_idx = int(results["scores"].argmax())
    return [int(x) for x in results["boxes"][best_idx].tolist()]


def _sam_box_extract(pil_img: Image.Image, box: list, sam_pred):
    """
    SAM 框提示分割。OWL-ViT 已定位石头位置，信任该框，不做边缘过滤。
    """
    arr = np.array(pil_img)
    h, w = arr.shape[:2]
    sam_pred.set_image(arr)
    masks, scores, _ = sam_pred.predict(
        box=np.array(box, dtype=float), multimask_output=True)
    valid = [(masks[i], scores[i]) for i in range(len(masks))
             if 0.01 * h * w < masks[i].sum() < 0.85 * h * w]
    if not valid:
        best_mask = masks[int(np.argmax([m.sum() for m in masks]))]
    else:
        best_mask = max(valid, key=lambda x: x[1])[0]
    return _cleanup_mask(best_mask)


def _owlvit_sam_extract(pil_img: Image.Image, img_path: Path,
                        sam_ckpt: Path, device: str) -> Image.Image | None:
    """
    OWL-ViT + SAM 框提示提取。
    - OWL-ViT 检测到 → SAM 框提示（不做边缘过滤）→ 返回提取结果
    - OWL-ViT 未检测到 → 返回 None（提取失败，跳过该图片）
    """
    try:
        processor, owlvit_model = _load_owlvit(DEFAULT_OWLVIT_DIR, device)
        box = _owlvit_detect(pil_img, processor, owlvit_model, device)
    except Exception as e:
        print(f"  ⚠️  OWL-ViT 失败 {img_path.name}: {e}")
        return None

    if box is None:
        # OWL-ViT 未检测到，提取失败
        return None

    # OWL-ViT 检测到 → SAM 框提示
    try:
        sam_pred = _load_sam_predictor(sam_ckpt, device)
        mask = _sam_box_extract(pil_img, box, sam_pred)
        if mask is not None:
            print(f"  [OWL-ViT+SAM-box] {img_path.name}  box={box}")
            return _apply_mask(pil_img, mask)
    except Exception as e:
        print(f"  ⚠️  SAM 框提示失败 {img_path.name}: {e}")

    return None


# ──────────────────────────────────────────────
# 主提取函数
# 返回 Image（成功）或 None（失败，跳过）
# ──────────────────────────────────────────────
def extract_rock(pil_img: Image.Image, img_path: Path,
                 model, device: str, conf_thresh: float,
                 sam_ckpt: Path = DEFAULT_SAM_CKPT) -> Image.Image | None:
    img_array = np.array(pil_img)
    h, w = img_array.shape[:2]

    # Step 1: YOLOE（带边缘过滤）
    result = _run_inference(model, img_path, device, conf_thresh)
    mask   = _pick_best_mask(result, h, w, conf_thresh)
    if mask is not None:
        return _apply_mask(pil_img, mask)

    # Step 2: OWL-ViT + SAM 框提示（检测不到返回 None）
    print(f"  [OWL-ViT+SAM] {img_path.name}  YOLOE 未检测到")
    return _owlvit_sam_extract(pil_img, img_path, sam_ckpt, device)


# ──────────────────────────────────────────────
# 批量处理
# 返回成功提取的 image_id 集合
# ──────────────────────────────────────────────
def process_images(image_ids: list, image_dir: Path, cache_dir: Path,
                   model, device: str, conf_thresh: float,
                   force: bool, sam_ckpt: Path = DEFAULT_SAM_CKPT) -> set:
    cache_dir.mkdir(parents=True, exist_ok=True)
    extracted_ids = set()
    skip = 0

    for img_id in tqdm(image_ids, desc=f"提取 → {cache_dir.name}"):
        stem = Path(img_id).stem
        cache_path = cache_dir / f"{stem}.png"

        if cache_path.exists() and not force:
            skip += 1
            extracted_ids.add(img_id)
            continue

        orig_path = image_dir / img_id
        if not orig_path.exists():
            candidates = list(image_dir.glob(f"{stem}.*"))
            if not candidates:
                print(f"  ⚠️  找不到原图: {img_id}")
                continue
            orig_path = candidates[0]

        try:
            pil = Image.open(orig_path).convert("RGB")
            result = extract_rock(pil, orig_path, model, device, conf_thresh, sam_ckpt)
            if result is not None:
                result.save(str(cache_path))
                extracted_ids.add(img_id)
            else:
                print(f"  ✗ 提取失败，跳过: {img_id}")
        except Exception as e:
            print(f"  ⚠️  处理异常 {img_id}: {e}")

    total = len(image_ids)
    print(f"[提取] 完成: {len(extracted_ids)}/{total} 张成功  (跳过已缓存: {skip}  失败丢弃: {total - len(extracted_ids)})")
    return extracted_ids


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="YOLOE + OWL-ViT + SAM 石头提取预处理（两级，失败丢弃）")
    p.add_argument("--train-csv",     default=str(DEFAULT_TRAIN_CSV),
                   help="原始训练标签（默认: 陨石2/train_labels.csv）")
    p.add_argument("--train-img-dir", default=str(DEFAULT_TRAIN_IMG_DIR))
    p.add_argument("--cache-dir",     default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--model",         default=DEFAULT_MODEL,
                   help="YOLOE 权重路径")
    p.add_argument("--sam-ckpt",      default=str(DEFAULT_SAM_CKPT),
                   help="SAM 权重路径")
    p.add_argument("--owlvit-dir",    default=str(DEFAULT_OWLVIT_DIR),
                   help="OWL-ViT 本地模型目录")
    p.add_argument("--device",        default=DEFAULT_DEVICE)
    p.add_argument("--conf-thresh",   type=float, default=DEFAULT_CONF)
    p.add_argument("--force",         action="store_true",
                   help="强制重新提取，忽略已有缓存")
    return p.parse_args()


def main():
    args = parse_args()

    global DEFAULT_OWLVIT_DIR
    DEFAULT_OWLVIT_DIR = Path(args.owlvit_dir)

    cache_dir = Path(args.cache_dir)

    print(f"[Config] YOLOE:      {args.model}")
    print(f"[Config] SAM:        {args.sam_ckpt}")
    print(f"[Config] OWL-ViT:    {args.owlvit_dir}")
    print(f"[Config] train_csv:  {args.train_csv}")
    print(f"[Config] device={args.device}  conf_thresh={args.conf_thresh}  force={args.force}")
    print(f"[Config] cache_dir:  {cache_dir}")

    model = load_model(args.model, args.device)

    # ── 训练集 ─────────────────────────────────
    train_df = pd.read_csv(args.train_csv)
    if "id" in train_df.columns and "image_id" not in train_df.columns:
        train_df = train_df.rename(columns={"id": "image_id"})
    train_ids = train_df["image_id"].astype(str).tolist()

    print(f"[训练集] 原始: {len(train_ids)} 张")
    extracted_train_ids = process_images(
        train_ids, Path(args.train_img_dir),
        cache_dir / "train",
        model, args.device, args.conf_thresh,
        args.force, Path(args.sam_ckpt),
    )

    # 过滤标签：只保留提取成功的图片
    filtered_df = train_df[train_df["image_id"].astype(str).isin(extracted_train_ids)].copy()
    filtered_df = filtered_df.reset_index(drop=True)

    # 写出过滤后的标签
    out_label_path = cache_dir / "train_labels.csv"
    filtered_df.to_csv(out_label_path, index=False)

    print(f"\n[标签] 原始: {len(train_df)} 条  →  提取后: {len(filtered_df)} 条")
    print(f"[标签] 保存至: {out_label_path}")
    if "label" in filtered_df.columns:
        vc = filtered_df["label"].value_counts()
        print(f"[标签] 类别分布: {vc.to_dict()}")

    print(f"\n✅ 全部完成！")
    print(f"   提取图片目录: {cache_dir}/train/")
    print(f"   过滤后标签:   {out_label_path}")
    print(f"\n   使用提取结果训练:")
    print(f"   python hybrid_cnn_transformer.py --no-sam \\")
    print(f"     --train-csv {out_label_path} \\")
    print(f"     --train-img-dir {cache_dir}/train")


if __name__ == "__main__":
    main()