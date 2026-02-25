"""
Precompute L14 backbone embeddings for all images in the dataset.
Each image -> 768-dim vector saved as .pt file (~3KB each).

Usage:
  python precompute_embeddings.py
  python precompute_embeddings.py --checkpoint checkpoints/best_model.pth
  python precompute_embeddings.py --images_dir images --output_dir cached_embeddings
"""

import argparse
import glob
import os
import time

import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from model import MobileCLIPRanker, get_norm_stats
from utils import load_config

HF_CHECKPOINT = (
    "/root/.cache/huggingface/hub/models--Nightfury16--clipick/"
    "snapshots/3a4a7d5ac48bd8ab20b8763d135c32de49f712c8/best_model_2602.pth"
)


def find_checkpoint(path=None):
    if path and os.path.exists(path):
        return path
    for p in [HF_CHECKPOINT, "checkpoints/best_model.pth", "checkpoints/last.pth"]:
        if os.path.exists(p):
            return p
    hits = sorted(glob.glob("checkpoints/*.pth"), key=os.path.getmtime)
    return hits[-1] if hits else None


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--checkpoint", default=None)
    pa.add_argument("--images_dir", default="images")
    pa.add_argument("--output_dir", default="cached_embeddings")
    pa.add_argument("--batch_size", type=int, default=32)
    pa.add_argument("--csv", default=None, help="Only process images in this CSV (uses file_path column)")
    args = pa.parse_args()

    ckpt_path = find_checkpoint(args.checkpoint)
    if not ckpt_path:
        print("No checkpoint found. Use --checkpoint <path>")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Load L14 backbone only
    cfg = load_config("config.yml")
    cfg.model.name = "mobileclip2_l14"

    print(f"Loading L14 from {ckpt_path}")
    model = MobileCLIPRanker(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = model.backbone.to(device)

    norm_mean, norm_std = get_norm_stats("mobileclip2_l14")
    process = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])

    # Gather image paths
    if args.csv and os.path.exists(args.csv):
        df = pd.read_csv(args.csv)
        if "file_path" not in df.columns:
            df["file_path"] = df.index.map(lambda x: os.path.join(args.images_dir, f"{x}.jpg"))
        image_paths = df["file_path"].tolist()
    else:
        image_paths = sorted(
            glob.glob(os.path.join(args.images_dir, "*.jpg"))
            + glob.glob(os.path.join(args.images_dir, "*.png"))
            + glob.glob(os.path.join(args.images_dir, "*.jpeg"))
        )

    # Skip already cached
    total = len(image_paths)
    image_paths = [
        p for p in image_paths
        if not os.path.exists(os.path.join(
            args.output_dir,
            os.path.splitext(os.path.basename(p))[0] + ".pt"
        ))
    ]
    skipped = total - len(image_paths)
    print(f"Images: {total} total, {skipped} already cached, {len(image_paths)} to process")

    if not image_paths:
        print("Nothing to do.")
        return

    # Process in batches
    t0 = time.perf_counter()
    processed = 0
    failed = 0

    for i in tqdm(range(0, len(image_paths), args.batch_size), desc="Embedding"):
        batch_paths = image_paths[i : i + args.batch_size]
        tensors = []
        valid_paths = []

        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                tensors.append(process(img))
                valid_paths.append(p)
            except Exception:
                failed += 1

        if not tensors:
            continue

        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            embeddings = backbone(batch).cpu()

        for j, p in enumerate(valid_paths):
            name = os.path.splitext(os.path.basename(p))[0]
            torch.save(embeddings[j], os.path.join(args.output_dir, f"{name}.pt"))
            processed += 1

    elapsed = time.perf_counter() - t0
    print(f"\nDone: {processed} cached, {failed} failed, {elapsed:.1f}s")
    print(f"Speed: {processed / elapsed:.1f} images/sec")
    print(f"Cache dir: {args.output_dir}/")


if __name__ == "__main__":
    main()
