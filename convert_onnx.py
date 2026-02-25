"""
Convert MobileCLIP2-L14 checkpoint -> 3 ONNX variants (FP32, Optimized, INT8).

Usage:
  python convert_onnx.py --checkpoint checkpoints/best_model.pth
  python convert_onnx.py --checkpoint /kaggle/input/notebooks/mithunparab/preference-optimization/clip_rank/checkpoints/best_model.pth
"""

import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

from model import MobileCLIPRanker
from utils import load_config


class FlatRanker(nn.Module):
    """Unwraps the grouped forward into flat (B,3,H,W) -> (B,1) for ONNX."""

    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        return self.head(self.backbone(x))


def find_checkpoint(path=None):
    if path and os.path.exists(path):
        return path
    for p in ["checkpoints/best_model.pth", "checkpoints/last.pth"]:
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

    # --- 3. INT8 dynamic quantization ---
    # Preprocess: fix shape inference for ViT before quantization
    preprocessed = os.path.join(args.output_dir, "ranker_preprocessed.onnx")
    print(f"\n[3/3] INT8 -> preprocessing shape inference...")
    model_proto = onnx.load(fp32)
    try:
        model_proto = onnx.shape_inference.infer_shapes(model_proto, strict_mode=False)
    except Exception as e:
        print(f"  Shape inference partial (expected for ViT): {e}")
    onnx.save(model_proto, preprocessed)

    int8 = os.path.join(args.output_dir, "ranker_int8.onnx")
    print(f"  Quantizing -> {int8}")
    quantize_dynamic(preprocessed, int8, weight_type=QuantType.QUInt8)
    out = ort.InferenceSession(int8, providers=["CPUExecutionProvider"]).run(
        None, {"images": dummy.numpy()}
    )[0]
    print(f"  Verify: {out.flatten()}, diff={np.abs(ref - out).max():.6f}")

    # --- Summary ---
    print(f"\n{'Model':<12} {'Size MB':<10}")
    print("-" * 22)
    for name, path in [("FP32", fp32), ("Optimized", opt), ("INT8", int8)]:
        print(f"{name:<12} {os.path.getsize(path)/1024/1024:<10.1f}")

    print("\nDone. Now run: python inference_onnx.py --validate")


if __name__ == "__main__":
    main()
