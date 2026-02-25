"""
Convert MobileCLIP2-L14 checkpoint -> ONNX variants (FP32, Optimized) + OpenVINO IR.

Usage:
  python convert_onnx.py --checkpoint checkpoints/best_model.pth
"""

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort

from model import MobileCLIPRanker
from utils import load_config


HF_CHECKPOINT = (
    "/root/.cache/huggingface/hub/models--Nightfury16--clipick/"
    "snapshots/3a4a7d5ac48bd8ab20b8763d135c32de49f712c8/best_model_2602.pth"
)


class FlatRanker(nn.Module):
    """Flat (B,3,H,W) -> (B,1) wrapper for ONNX export."""

    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        return self.head(self.backbone(x))


def find_checkpoint(path=None):
    if path and os.path.exists(path):
        return path
    for p in [HF_CHECKPOINT, "checkpoints/best_model.pth", "checkpoints/last.pth"]:
        if os.path.exists(p):
            return p
    hits = sorted(glob.glob("checkpoints/*.pth"), key=os.path.getmtime)
    return hits[-1] if hits else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--output_dir", default="onnx_models")
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()

    ckpt_path = find_checkpoint(args.checkpoint)
    if not ckpt_path:
        print("No checkpoint found. Use --checkpoint <path>")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Force L14
    cfg = load_config("config.yml")
    cfg.model.name = "mobileclip2_l14"

    print(f"Loading L14 from {ckpt_path}")
    model = MobileCLIPRanker(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()

    flat = FlatRanker(model.backbone, model.head)
    flat.eval()

    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        ref = flat(dummy).numpy()
    print(f"PyTorch ref: {ref.flatten()}")

    # --- 1. FP32 ---
    fp32 = os.path.join(args.output_dir, "ranker_fp32.onnx")
    print(f"\n[1/3] FP32 -> {fp32}")
    torch.onnx.export(
        flat, dummy, fp32,
        input_names=["images"], output_names=["scores"],
        dynamic_axes={"images": {0: "batch"}, "scores": {0: "batch"}},
        opset_version=args.opset, do_constant_folding=True,
    )
    out = ort.InferenceSession(fp32, providers=["CPUExecutionProvider"]).run(
        None, {"images": dummy.numpy()}
    )[0]
    print(f"  Verify: {out.flatten()}, diff={np.abs(ref - out).max():.6f}")

    # --- 2. Optimized (graph fusions: MatMul+Add, attention, etc.) ---
    opt = os.path.join(args.output_dir, "ranker_optimized.onnx")
    print(f"\n[2/3] Optimized -> {opt}")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.optimized_model_filepath = opt
    ort.InferenceSession(fp32, so, providers=["CPUExecutionProvider"])
    out = ort.InferenceSession(opt, providers=["CPUExecutionProvider"]).run(
        None, {"images": dummy.numpy()}
    )[0]
    print(f"  Verify: {out.flatten()}, diff={np.abs(ref - out).max():.6f}")

    # --- 3. OpenVINO IR (best for Intel Xeon CPUs) ---
    print(f"\n[3/3] OpenVINO IR")
    try:
        import openvino as ov
        core = ov.Core()
        ov_model = core.read_model(fp32)
        ov_dir = os.path.join(args.output_dir, "openvino")
        os.makedirs(ov_dir, exist_ok=True)
        ov.save_model(ov_model, os.path.join(ov_dir, "ranker.xml"))
        print(f"  Saved to {ov_dir}/ranker.xml + ranker.bin")
    except ImportError:
        print("  openvino not installed, skipping. pip install openvino")
    except Exception as e:
        print(f"  OpenVINO conversion failed: {e}")

    # --- Summary ---
    print(f"\n{'Model':<12} {'Size MB':<10}")
    print("-" * 22)
    for name, path in [("FP32", fp32), ("Optimized", opt)]:
        print(f"{name:<12} {os.path.getsize(path)/1024/1024:<10.1f}")
    ov_bin = os.path.join(args.output_dir, "openvino", "ranker.bin")
    if os.path.exists(ov_bin):
        ov_xml = os.path.join(args.output_dir, "openvino", "ranker.xml")
        total = os.path.getsize(ov_bin) + os.path.getsize(ov_xml)
        print(f"{'OpenVINO':<12} {total/1024/1024:<10.1f}")

    print("\nDone. Now run: python inference_onnx.py --validate")


if __name__ == "__main__":
    main()
