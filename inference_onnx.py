"""
Benchmark: PyTorch vs ONNX FP32 vs ONNX Optimized vs OpenVINO.
Reports per-image latency and (optionally) accuracy on the validation set.

Usage:
  python inference_onnx.py                                    # timing only
  python inference_onnx.py --validate                         # + accuracy
  python inference_onnx.py --checkpoint path.pth --threads 4  # custom
"""

import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
import requests
import torch
import onnxruntime as ort
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


# ---------------------------------------------------------------------------
# Preprocessing (matches training exactly)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
class PyTorchBackend:
    name = "PyTorch"

    def __init__(self, checkpoint, cfg):
        cfg.model.name = "mobileclip2_l14"
        self.model = MobileCLIPRanker(cfg)
        ckpt = torch.load(checkpoint, map_location="cpu")
        sd = ckpt.get("model_state_dict", ckpt)
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        self.model.load_state_dict(sd)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, batch_np):
        t = torch.from_numpy(batch_np)
        feat = self.model.backbone(t)
        return self.model.head(feat).numpy().flatten()


class ONNXBackend:
    def __init__(self, path, name, threads=None):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            so.intra_op_num_threads = threads
            so.inter_op_num_threads = threads
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.name = name

    def __call__(self, batch_np):
        return self.sess.run(None, {"images": batch_np})[0].flatten()


class OpenVINOBackend:
    def __init__(self, xml_path, name="OpenVINO", threads=None):
        import openvino as ov
        core = ov.Core()
        if threads:
            core.set_property("CPU", {"INFERENCE_NUM_THREADS": str(threads)})
        model = core.read_model(xml_path)
        self.compiled = core.compile_model(model, "CPU")
        self.output_key = self.compiled.output(0)
        self.name = name

    def __call__(self, batch_np):
        return self.compiled({0: batch_np})[self.output_key].flatten()


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def benchmark(backend, batch_np, warmup=5, runs=20):
    n = batch_np.shape[0]
    for _ in range(warmup):
        backend(batch_np)
    times = []
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


# ---------------------------------------------------------------------------
# Validation (mirrors train_ddp.py validate logic)
# ---------------------------------------------------------------------------
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
# Rank (production-style: URL list in, sorted results out)
# ---------------------------------------------------------------------------
def rank_images(backend, image_list, tf):
    tensors, clean = [], []
    for src in image_list:
        try:
            tensors.append(tf(load_image(src)))
            clean.append(src)
        except Exception as e:
            print(f"  Skip {src}: {e}")
    if not tensors:
        return []
    scores = backend(torch.stack(tensors).numpy())
    results = sorted(zip(clean, scores), key=lambda x: x[1], reverse=True)
    return [{"source": s, "score": float(sc)} for s, sc in results]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
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
    pa.add_argument("--onnx_dir", default="onnx_models")
    pa.add_argument("--validate", action="store_true")
    pa.add_argument("--threads", type=int, default=None)
    pa.add_argument("--runs", type=int, default=20)
    args = pa.parse_args()

    cfg = load_config("config.yml")
    tf = build_transform(cfg.data.img_size)

    # --- Load backends ---
    backends = []

    ckpt = find_checkpoint(args.checkpoint)
    if ckpt:
        print(f"Loading PyTorch L14 from {ckpt}")
        backends.append(PyTorchBackend(ckpt, cfg))
    else:
        print("No .pth found, skipping PyTorch backend")

    onnx_files = [
        ("ranker_fp32.onnx", "ONNX FP32"),
        ("ranker_optimized.onnx", "ONNX Optimized"),
    ]
    for fname, label in onnx_files:
        fpath = os.path.join(args.onnx_dir, fname)
        if os.path.exists(fpath):
            print(f"Loading {label} from {fpath}")
            backends.append(ONNXBackend(fpath, label, args.threads))
        else:
            print(f"  {fpath} not found, skipping")

    # OpenVINO
    ov_xml = os.path.join(args.onnx_dir, "openvino", "ranker.xml")
    if os.path.exists(ov_xml):
        try:
            print(f"Loading OpenVINO from {ov_xml}")
            backends.append(OpenVINOBackend(ov_xml, threads=args.threads))
        except ImportError:
            print("  openvino not installed, skipping")
    else:
        print(f"  {ov_xml} not found, skipping OpenVINO")

    if not backends:
        print("No backends available.")
        return

    # --- Download test images ---
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
    # TIMING BENCHMARK
    # =====================================================================
    print("=" * 75)
    print(f"{'Backend':<18} {'Total (ms)':<16} {'Per Image (ms)':<18} {'vs PyTorch'}")
    print("-" * 75)

    results = []
    for b in backends:
        r = benchmark(b, batch, runs=args.runs)
        results.append(r)

    base = results[0]["total_ms"]
    for r in results:
        speedup = base / r["total_ms"]
        print(f"{r['name']:<18} {r['total_ms']:>7.1f} +/- {r['std_ms']:<5.1f}  "
              f"{r['per_img_ms']:>10.1f}          {speedup:>5.2f}x")

    # =====================================================================
    # SCORE COMPARISON (same images, check ranking agreement)
    # =====================================================================
    print(f"\n{'Backend':<18} Scores (should match ranking order)")
    print("-" * 75)
    for r in results:
        sc = "  ".join(f"{s:+.4f}" for s in r["scores"])
        print(f"{r['name']:<18} {sc}")

    # Ranking agreement check
    ref_order = np.argsort(-results[0]["scores"])
    print(f"\n{'Backend':<18} Rank Order   Match?")
    print("-" * 50)
    for r in results:
        order = np.argsort(-r["scores"])
        match = "YES" if np.array_equal(order, ref_order) else "NO"
        print(f"{r['name']:<18} {list(order)}    {match}")

    # =====================================================================
    # PRODUCTION RANKING DEMO
    # =====================================================================
    fastest = min(results, key=lambda r: r["total_ms"])
    print(f"\nRanking with fastest backend ({fastest['name']}):")
    ranked = rank_images(backends[results.index(fastest)], test_urls, tf)
    for i, item in enumerate(ranked):
        print(f"  {i+1}. {item['score']:+.4f}  {item['source'][-40:]}")

    # =====================================================================
    # VALIDATION (optional)
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

        print(f"\n{'=' * 75}")
        print(f"VALIDATION  |  {len(val_groups)} groups, {len(val_df)} images")
        print(f"{'=' * 75}")
        print(f"{'Backend':<18} {'Accuracy':<12} {'NDCG':<10} {'Time (s)'}")
        print("-" * 55)

        for b in backends:
            t0 = time.perf_counter()
            acc, ndcg = validate(b, val_df, tf)
            elapsed = time.perf_counter() - t0
            print(f"{b.name:<18} {acc:>8.2%}    {ndcg:>6.4f}    {elapsed:>7.1f}")

    print()


if __name__ == "__main__":
    main()
