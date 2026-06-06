"""
将 best_model.pth 中的 CNN/ViT 主干权重
覆盖 torchvision 缓存的 ImageNet 预训练文件。
代码无需任何改动，加载时自动使用这份权重。
"""
import torch
from pathlib import Path
import argparse

def extract_and_replace(best_model_path):
    ckpt = torch.load(best_model_path, map_location="cpu")

    # ── 1. 提取 CNN 主干（去掉 cnn_branch. 前缀，去掉自定义 fc）──
    cnn_state = {}
    for k, v in ckpt.items():
        if not k.startswith("cnn_branch."): continue
        new_k = k[len("cnn_branch."):]
        if new_k.startswith("fc."):         # 自定义分类头，跳过
            continue
        cnn_state[new_k] = v
    # torchvision 要求有 fc.weight/bias（1000类），用零填充骗过 strict=False
    # 实际加载后代码立即替换 fc，所以内容无所谓
    cnn_state["fc.weight"] = torch.zeros(1000, 2048)
    cnn_state["fc.bias"]   = torch.zeros(1000)

    # ── 2. 提取 ViT 主干（去掉 vit_branch. 前缀，去掉自定义 heads）──
    vit_state = {}
    for k, v in ckpt.items():
        if not k.startswith("vit_branch."): continue
        new_k = k[len("vit_branch."):]
        if new_k.startswith("heads."):      # 自定义分类头，跳过
            continue
        vit_state[new_k] = v
    vit_state["heads.head.weight"] = torch.zeros(1000, 768)
    vit_state["heads.head.bias"]   = torch.zeros(1000)

    # ── 3. 找 torchvision 缓存目录 ──
    import torch.hub
    cache_dir = Path(torch.hub.get_dir()) / "checkpoints"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 找现有缓存文件名（含 hash）
    resnet_files = list(cache_dir.glob("resnet101*.pth"))
    vit_files    = list(cache_dir.glob("vit_b_16*.pth"))

    if not resnet_files:
        print("[!] 未找到 resnet101 缓存，请先运行一次训练让 torchvision 下载")
    else:
        dst = resnet_files[0]
        torch.save(cnn_state, str(dst))
        print(f"[OK] ResNet101 缓存已替换: {dst}")

    if not vit_files:
        print("[!] 未找到 vit_b_16 缓存，请先运行一次训练让 torchvision 下载")
    else:
        dst = vit_files[0]
        torch.save(vit_state, str(dst))
        print(f"[OK] ViT-B/16 缓存已替换: {dst}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="best_model.pth 路径")
    args = parser.parse_args()
    extract_and_replace(args.weights)
