"""
PyTorch 2.x CPU optimization benchmark for ViT-L/14.

Tests:
  1. FP32 baseline
  2. FP16 (model.half())
  3. FP32 + channels_last memory format
  4. FP16 + channels_last
  5. torch.inference_mode vs torch.no_grad
  6. torch.compile (default backend)

Usage:
  python benchmark_cpu.py
  python benchmark_cpu.py --validate
  python benchmark_cpu.py --threads 4
"""

import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
import requests
import torch
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


def build_transform(img_size=224):
    mean, std = get_norm_stats("mobileclip2_l14")
    return transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def load_image(src):
    if src.startswith("http"):
        return Image.open(BytesIO(requests.get(src, timeout=10).content)).convert("RGB")
    return Image.open(src).convert("RGB")


def load_model(checkpoint, cfg):
    cfg.model.name = "mobileclip2_l14"
    model = MobileCLIPRanker(cfg)
    ckpt = torch.load(checkpoint, map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    return model


def find_checkpoint(path=None):
    if path and os.path.exists(path):
        return path
    for p in [HF_CHECKPOINT, "checkpoints/best_model.pth", "checkpoints/last.pth"]:
        if os.path.exists(p):
            return p
    hits = sorted(glob.glob("checkpoints/*.pth"), key=os.path.getmtime)
    return hits[-1] if hits else None


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
class FP32Backend:
    name = "FP32"

    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def __call__(self, batch_np):
        t = torch.from_numpy(batch_np)
        return self.model.head(self.model.backbone(t)).numpy().flatten()


class FP16Backend:
    name = "FP16 (half)"

    def __init__(self, model):
        import copy
        self.model = copy.deepcopy(model).half()

    @torch.no_grad()
    def __call__(self, batch_np):
        t = torch.from_numpy(batch_np).half()
        out = self.model.head(self.model.backbone(t))
        return out.float().numpy().flatten()


class ChannelsLastBackend:
    name = "FP32 + channels_last"

    def __init__(self, model):
        import copy
        self.model = copy.deepcopy(model)
        self.model = self.model.to(memory_format=torch.channels_last)

    @torch.no_grad()
    def __call__(self, batch_np):
        t = torch.from_numpy(batch_np).to(memory_format=torch.channels_last)
        return self.model.head(self.model.backbone(t)).numpy().flatten()


class FP16ChannelsLastBackend:
    name = "FP16 + channels_last"

    def __init__(self, model):
        import copy
        self.model = copy.deepcopy(model).half()
        self.model = self.model.to(memory_format=torch.channels_last)

    @torch.no_grad()
    def __call__(self, batch_np):
        t = torch.from_numpy(batch_np).half().to(memory_format=torch.channels_last)
        out = self.model.head(self.model.backbone(t))
        return out.float().numpy().flatten()


class InferenceModeBackend:
    name = "FP32 + inference_mode"

    def __init__(self, model):
        self.model = model

    def __call__(self, batch_np):
        with torch.inference_mode():
            t = torch.from_numpy(batch_np)
            return self.model.head(self.model.backbone(t)).numpy().flatten()


class TorchCompileBackend:
    name = "torch.compile"

    def __init__(self, model):
        import copy
        m = copy.deepcopy(model)
        self.backbone = torch.compile(m.backbone)
        self.head = torch.compile(m.head)

    @torch.no_grad()
    def __call__(self, batch_np):
        t = torch.from_numpy(batch_np)
        return self.head(self.backbone(t)).numpy().flatten()


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def benchmark(backend, batch_np, warmup=5, runs=20):
    n = batch_np.shape[0]

    # Warmup (torch.compile compiles here)
    for i in range(warmup):
        try:
            backend(batch_np)
        except Exception as e:
            if i == 0:
                return {"name": backend.name, "total_ms": -1, "std_ms": 0,
                        "per_img_ms": -1, "scores": None, "error": str(e)}

    times = []
    scores = None
    for _ in range(runs):
        t0 = time.perf_counter()
        scores = backend(batch_np)
        times.append(time.perf_counter() - t0)

    return {
        "name": backend.name,
        "total_ms": np.mean(times) * 1000,
        "std_ms": np.std(times) * 1000,
        "per_img_ms": np.mean(times) / n * 1000,
        "scores": scores,
    }


def compute_ndcg(preds, gts):
    n = len(preds)
    if n < 2:
        return 1.0
    order = np.argsort(-preds)
    disc = np.log2(np.arange(n) + 2)
    dcg = np.sum((2 ** gts[order] - 1) / disc)
    idcg = np.sum((2 ** np.sort(gts)[::-1] - 1) / disc)
    return dcg / idcg if idcg > 0 else 1.0


def validate(backend, val_df, tf, images_dir="images"):
    grouped = val_df.groupby("group_id")
    wins, total, ndcgs = 0, 0, []

    for _, grp in grouped:
        if len(grp) < 2:
            continue
        imgs, scores = [], []
        for _, row in grp.iterrows():
            fp = row.get("file_path", os.path.join(images_dir, f"{row.name}.jpg"))
            if not os.path.exists(fp):
                continue
            imgs.append(tf(Image.open(fp).convert("RGB")))
            scores.append(_remap_score(float(row["score"]), row.get("label", "")))
        if len(imgs) < 2:
            continue

        preds = backend(torch.stack(imgs).numpy())
        gt = np.array(scores)

        best = scores[np.argmax(preds)]
        top = max(scores)
        tier = lambda s: 2 if s >= 8 else (1 if s >= 3 else 0)
        if tier(best) == tier(top):
            wins += 1
        ndcgs.append(compute_ndcg(preds, gt))
        total += 1

    acc = wins / total if total else 0
    ndcg = np.mean(ndcgs) if ndcgs else 0
    return acc, ndcg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--checkpoint", default=None)
    pa.add_argument("--validate", action="store_true")
    pa.add_argument("--threads", type=int, default=None)
    pa.add_argument("--runs", type=int, default=20)
    args = pa.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)

    print(f"PyTorch {torch.__version__}")
    print(f"CPU threads: {torch.get_num_threads()}")

    cfg = load_config("config.yml")
    tf = build_transform(cfg.data.img_size)

    ckpt = find_checkpoint(args.checkpoint)
    if not ckpt:
        print("No checkpoint found. Use --checkpoint <path>")
        return

    print(f"Loading L14 from {ckpt}\n")
    model = load_model(ckpt, cfg)

    # Build backends
    print("Initializing backends...")
    backends = [
        FP32Backend(model),
        FP16Backend(model),
        ChannelsLastBackend(model),
        FP16ChannelsLastBackend(model),
        InferenceModeBackend(model),
    ]

    try:
        print("  torch.compile... (first run will be slow)")
        backends.append(TorchCompileBackend(model))
    except Exception as e:
        print(f"  torch.compile skipped: {e}")

    # Prepare test batch
    test_urls = [
        "https://ap.rdcpix.com/69fe76be4fd818c9b1e25b8b6c79432el-m3865337706s-w2048_h1536.jpg",
        "https://ap.rdcpix.com/69fe76be4fd818c9b1e25b8b6c79432el-m1211374265s-w2048_h1536.jpg",
        "https://ap.rdcpix.com/69fe76be4fd818c9b1e25b8b6c79432el-m713883090s-w2048_h1536.jpg",
        "https://ap.rdcpix.com/c3065cb0efd74e0e69c634c4e7926ed0l-m3456441259s-w2048_h1536.jpg",
    ]
    print(f"\nDownloading {len(test_urls)} test images...")
    tensors = [tf(load_image(u)) for u in test_urls]
    batch = torch.stack(tensors).numpy()
    print(f"Batch: {batch.shape}\n")

    # =====================================================================
    # TIMING
    # =====================================================================
    print("=" * 80)
    print(f"{'Backend':<25} {'Total (ms)':<16} {'Per Image (ms)':<18} {'vs FP32'}")
    print("-" * 80)

    results = []
    for b in backends:
        r = benchmark(b, batch, runs=args.runs)
        results.append(r)

    base = results[0]["total_ms"]
    for r in results:
        if r["total_ms"] < 0:
            print(f"{r['name']:<25} FAILED: {r.get('error', 'unknown')}")
        else:
            speedup = base / r["total_ms"]
            print(f"{r['name']:<25} {r['total_ms']:>7.1f} +/- {r['std_ms']:<5.1f}  "
                  f"{r['per_img_ms']:>10.1f}          {speedup:>5.2f}x")

    # =====================================================================
    # RANKING AGREEMENT
    # =====================================================================
    ref_scores = results[0]["scores"]
    if ref_scores is not None:
        ref_order = np.argsort(-ref_scores)
        print(f"\n{'Backend':<25} Rank Order   Match?  Max Score Diff")
        print("-" * 65)
        for r in results:
            if r["scores"] is None:
                print(f"{r['name']:<25} FAILED")
                continue
            order = np.argsort(-r["scores"])
            match = "YES" if np.array_equal(order, ref_order) else "NO"
            diff = np.abs(ref_scores - r["scores"]).max()
            print(f"{r['name']:<25} {list(order)}    {match}     {diff:.6f}")

    # =====================================================================
    # VALIDATION
    # =====================================================================
    if args.validate:
        csv_path = cfg.data.csv_path
        if not os.path.exists(csv_path):
            print(f"\n{csv_path} not found, can't validate.")
            return

        df = pd.read_csv(csv_path)
        if "file_path" not in df.columns:
            df["file_path"] = df.index.map(lambda x: os.path.join("images", f"{x}.jpg"))

        groups = df["group_id"].unique()
        val_groups = groups[:int(len(groups) * 0.1)]
        val_df = df[df["group_id"].isin(val_groups)]

        print(f"\n{'=' * 80}")
        print(f"VALIDATION  |  {len(val_groups)} groups, {len(val_df)} images")
        print(f"{'=' * 80}")
        print(f"{'Backend':<25} {'Accuracy':<12} {'NDCG':<10} {'Time (s)'}")
        print("-" * 60)

        for i, b in enumerate(backends):
            if results[i]["total_ms"] < 0:
                print(f"{b.name:<25} SKIPPED (failed benchmark)")
                continue
            t0 = time.perf_counter()
            acc, ndcg = validate(b, val_df, tf)
            elapsed = time.perf_counter() - t0
            print(f"{b.name:<25} {acc:>8.2%}    {ndcg:>6.4f}    {elapsed:>7.1f}")

    print()


if __name__ == "__main__":
    main()
