"""
Precompute backbone features for all images and cache to disk.
Since the backbone is frozen during training, this eliminates all image I/O
and backbone forward passes — typically 10-50x faster training.

Usage:
    python precompute_features.py
"""

import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from model import MobileCLIPRanker, get_norm_stats
from utils import load_config


def extract_features(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = getattr(cfg.data, "cached_features_dir", "cached_features")
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Device: {device}")
    print(f"Cache dir: {cache_dir}")

    # Load model (only need the backbone)
    model = MobileCLIPRanker(cfg).to(device)
    model.eval()

    norm_mean, norm_std = get_norm_stats(cfg.model.name)
    process = transforms.Compose([
        transforms.Resize(cfg.data.img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(cfg.data.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])

    df = pd.read_csv(cfg.data.csv_path)
    if "file_path" not in df.columns:
        df["file_path"] = df.index.map(lambda x: os.path.join("images", f"{x}.jpg"))

    # Collect unique image paths
    paths = df["file_path"].unique().tolist()
    paths = [p for p in paths if os.path.exists(p)]
    print(f"Extracting features for {len(paths)} images...")

    batch_size = 64
    use_amp = device.type == "cuda"

    with torch.no_grad():
        for start in tqdm(range(0, len(paths), batch_size)):
            batch_paths = paths[start : start + batch_size]
            tensors = []
            valid_paths = []

            for p in batch_paths:
                try:
                    with Image.open(p) as img:
                        tensors.append(process(img.convert("RGB")))
                    valid_paths.append(p)
                except Exception as e:
                    print(f"Skipping {p}: {e}")

            if not tensors:
                continue

            batch = torch.stack(tensors).to(device)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    features = model.backbone(batch)
            else:
                features = model.backbone(batch)

            features = features.float().cpu()

            for path, feat in zip(valid_paths, features):
                basename = os.path.splitext(os.path.basename(path))[0]
                out_path = os.path.join(cache_dir, f"{basename}.pt")
                torch.save(feat, out_path)

    print(f"Done. Cached {len(paths)} feature vectors to {cache_dir}/")


if __name__ == "__main__":
    cfg = load_config("config.yml")
    extract_features(cfg)
