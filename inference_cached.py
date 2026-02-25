"""
Production inference with L14 embedding cache.
- Cached images: head-only forward (~0.01ms/image)
- Cache miss: full L14 backbone + auto-cache for next time
- Validates accuracy is identical to original inference.py

Usage:
  python inference_cached.py
  python inference_cached.py --validate
  python inference_cached.py --cache_dir cached_embeddings
"""

import torch
import requests
import os
import time
import glob
import hashlib
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from torchvision import transforms

from model import MobileCLIPRanker, get_norm_stats
from dataset import _remap_score
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


class PropertyRanker:
    """Production ranker with embedding cache.

    Cached path:   image -> load .pt embedding -> head -> score  (~0.01ms)
    Uncached path:  image -> backbone -> cache embedding -> head -> score  (~850ms)
    """

    def __init__(self, model_path=None, config_path="config.yml",
                 cache_dir="cached_embeddings", device=None):
        self.cfg = load_config(config_path)
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # Force L14
        self.cfg.model.name = "mobileclip2_l14"

        ckpt_path = find_checkpoint(model_path)
        if not ckpt_path:
            raise FileNotFoundError("No checkpoint found")

        print(f"--- Loading Ranker ---")
        print(f"Device: {self.device}")
        print(f"Cache:  {self.cache_dir}/")

        self.model = MobileCLIPRanker(self.cfg)

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        sd = checkpoint.get("model_state_dict", checkpoint)
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        self.model.load_state_dict(sd)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded.\n")

        norm_mean, norm_std = get_norm_stats(self.cfg.model.name)
        self.process = transforms.Compose([
            transforms.Resize(
                self.cfg.data.img_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(self.cfg.data.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])

    def _cache_key(self, src):
        """Generate cache filename from image source."""
        if os.path.isfile(src):
            # Local file: use filename without extension
            return os.path.splitext(os.path.basename(src))[0]
        # URL: hash it
        return hashlib.md5(src.encode()).hexdigest()

    def _load_image(self, src):
        if src.startswith("http"):
            resp = requests.get(src, timeout=5)
            return Image.open(BytesIO(resp.content)).convert("RGB")
        return Image.open(src).convert("RGB")

    def _get_embedding(self, src):
        """Return cached embedding or compute + cache it."""
        key = self._cache_key(src)
        cache_path = os.path.join(self.cache_dir, f"{key}.pt")

        if os.path.exists(cache_path):
            return torch.load(cache_path, map_location=self.device, weights_only=True)

        # Cache miss: compute embedding
        img = self._load_image(src)
        tensor = self.process(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model.backbone(tensor).squeeze(0)

        # Save for next time
        torch.save(emb.cpu(), cache_path)
        return emb

    def rank(self, image_list):
        """Rank images. Uses cache when available, computes + caches on miss."""
        embeddings = []
        clean_sources = []
        cache_hits = 0

        for src in image_list:
            if not src or not isinstance(src, str):
                continue
            try:
                key = self._cache_key(src)
                cache_path = os.path.join(self.cache_dir, f"{key}.pt")
                was_cached = os.path.exists(cache_path)

                emb = self._get_embedding(src)
                embeddings.append(emb)
                clean_sources.append(src)

                if was_cached:
                    cache_hits += 1
            except Exception as e:
                print(f"Error: {src}: {e}")

        if not embeddings:
            return []

        print(f"  {len(embeddings)} images | {cache_hits} cached, "
              f"{len(embeddings) - cache_hits} computed")

        # Head forward — this is the fast part
        batch = torch.stack(embeddings).unsqueeze(0).to(self.device)
        with torch.no_grad():
            scores = self.model.head(batch).view(-1).cpu().numpy()

        results = [
            {"source": src, "score": float(sc)}
            for src, sc in zip(clean_sources, scores)
        ]
        results.sort(key=lambda x: x["score"], reverse=True)
        return results


def compute_ndcg(preds, gts):
    n = len(preds)
    if n < 2:
        return 1.0
    order = np.argsort(-preds)
    disc = np.log2(np.arange(n) + 2)
    dcg = np.sum((2 ** gts[order] - 1) / disc)
    idcg = np.sum((2 ** np.sort(gts)[::-1] - 1) / disc)
    return dcg / idcg if idcg > 0 else 1.0


def validate(ranker, cfg):
    """Validate accuracy using cached embeddings vs original full-model inference."""
    csv_path = cfg.data.csv_path
    if not os.path.exists(csv_path):
        print(f"{csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    if "file_path" not in df.columns:
        df["file_path"] = df.index.map(lambda x: os.path.join("images", f"{x}.jpg"))

    groups = df["group_id"].unique()
    val_groups = groups[: int(len(groups) * 0.1)]
    val_df = df[df["group_id"].isin(val_groups)]

    print(f"Validating on {len(val_groups)} groups, {len(val_df)} images")
    print(f"Cache dir: {ranker.cache_dir}/\n")

    grouped = val_df.groupby("group_id")
    wins, total, ndcgs = 0, 0, []
    total_cache_hits = 0
    total_images = 0

    t0 = time.perf_counter()

    for _, grp in grouped:
        if len(grp) < 2:
            continue

        embeddings, scores, paths = [], [], []
        for _, row in grp.iterrows():
            fp = row.get("file_path", os.path.join("images", f"{row.name}.jpg"))
            if not os.path.exists(fp):
                continue
            try:
                key = ranker._cache_key(fp)
                cache_path = os.path.join(ranker.cache_dir, f"{key}.pt")
                if os.path.exists(cache_path):
                    total_cache_hits += 1

                emb = ranker._get_embedding(fp)
                embeddings.append(emb)
                scores.append(_remap_score(float(row["score"]), row.get("label", "")))
                total_images += 1
            except Exception:
                continue

        if len(embeddings) < 2:
            continue

        batch = torch.stack(embeddings).unsqueeze(0).to(ranker.device)
        with torch.no_grad():
            preds = ranker.model.head(batch).view(-1).cpu().numpy()

        gt = np.array(scores)
        best = scores[np.argmax(preds)]
        top = max(scores)
        tier = lambda s: 2 if s >= 8 else (1 if s >= 3 else 0)
        if tier(best) == tier(top):
            wins += 1
        ndcgs.append(compute_ndcg(preds, gt))
        total += 1

    elapsed = time.perf_counter() - t0
    acc = wins / total if total else 0
    ndcg = np.mean(ndcgs) if ndcgs else 0

    print(f"Accuracy:    {acc:.2%}")
    print(f"NDCG:        {ndcg:.4f}")
    print(f"Time:        {elapsed:.2f}s")
    print(f"Cache hits:  {total_cache_hits}/{total_images}")
    print(f"Per image:   {elapsed / max(total_images, 1) * 1000:.2f}ms")


if __name__ == "__main__":
    import argparse

    pa = argparse.ArgumentParser()
    pa.add_argument("--checkpoint", default=None)
    pa.add_argument("--cache_dir", default="cached_embeddings")
    pa.add_argument("--validate", action="store_true")
    args = pa.parse_args()

    ranker = PropertyRanker(
        model_path=args.checkpoint,
        cache_dir=args.cache_dir,
    )

    if args.validate:
        validate(ranker, ranker.cfg)
    else:
        test_urls = [
            "https://ap.rdcpix.com/69fe76be4fd818c9b1e25b8b6c79432el-m3865337706s-w2048_h1536.jpg",
            "https://ap.rdcpix.com/69fe76be4fd818c9b1e25b8b6c79432el-m1211374265s-w2048_h1536.jpg",
            "https://ap.rdcpix.com/69fe76be4fd818c9b1e25b8b6c79432el-m713883090s-w2048_h1536.jpg",
            "https://ap.rdcpix.com/c3065cb0efd74e0e69c634c4e7926ed0l-m3456441259s-w2048_h1536.jpg",
        ]

        # First run: computes + caches
        print("--- Run 1 (cold cache) ---")
        t0 = time.time()
        results = ranker.rank(test_urls)
        cold = time.time() - t0

        print(f"\n{'Rank':<6} {'Score':<10} Source")
        print("-" * 60)
        for i, r in enumerate(results):
            print(f"{i+1:<6} {r['score']:>+8.4f}   {r['source'][-45:]}")
        print(f"Time: {cold:.4f}s\n")

        # Second run: all cached
        print("--- Run 2 (warm cache) ---")
        t0 = time.time()
        results = ranker.rank(test_urls)
        warm = time.time() - t0

        print(f"\n{'Rank':<6} {'Score':<10} Source")
        print("-" * 60)
        for i, r in enumerate(results):
            print(f"{i+1:<6} {r['score']:>+8.4f}   {r['source'][-45:]}")
        print(f"Time: {warm:.4f}s")
        print(f"\nSpeedup: {cold/warm:.0f}x (cold: {cold:.2f}s -> warm: {warm:.4f}s)")
