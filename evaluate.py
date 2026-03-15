"""
Evaluate a trained ranker against real ground-truth data.

Usage:
    python evaluate.py --csv /path/to/ground_truth.csv \
                       --model /path/to/best_model.pth \
                       [--config config.yml] [--workers 8]

Parallelization:
    - Properties split across all available GPUs (torch.multiprocessing)
    - Image downloads parallelized with ThreadPoolExecutor per property

Expected CSV: property_id, image_url, is_ground_truth (TRUE/FALSE), ...
"""

import torch
import torch.multiprocessing as mp
import requests
import yaml
import argparse
import os
import glob
import csv
import time
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from io import BytesIO
from types import SimpleNamespace
from torchvision import transforms
from collections import defaultdict
from model import MobileCLIPRanker, get_norm_stats


def load_config(path="config.yml"):
    with open(path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    def recursive_namespace(d):
        if isinstance(d, dict):
            for k, v in d.items():
                d[k] = recursive_namespace(v)
            return SimpleNamespace(**d)
        return d
    return recursive_namespace(cfg_dict)


def _download_one(src, process_fn):
    """Download and preprocess a single image. Returns (url, tensor) or None."""
    if not src or not isinstance(src, str):
        return None
    try:
        if src.startswith("http"):
            resp = requests.get(src, timeout=10)
            img = Image.open(BytesIO(resp.content)).convert('RGB')
        else:
            img = Image.open(src).convert('RGB')
        return (src, process_fn(img))
    except Exception:
        return None


class PropertyRanker:
    def __init__(self, model_path, config_path='config.yml', device=None, download_workers=8):
        self.cfg = load_config(config_path)
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.download_workers = download_workers

        print(f"[{self.device}] Loading {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k.replace("module.", "")] = v

        # Auto-detect head type from checkpoint keys
        has_attn = any('head.attn.' in k for k in new_state_dict)
        self.cfg.model.use_attention = has_attn
        print(f"[{self.device}] Head: {'attention' if has_attn else 'independent'}")

        self.model = MobileCLIPRanker(self.cfg)
        self.model.load_state_dict(new_state_dict)
        self.model.to(self.device)
        self.model.eval()

        norm_mean, norm_std = get_norm_stats(self.cfg.model.name)
        self.process = transforms.Compose([
            transforms.Resize(self.cfg.data.img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.cfg.data.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std)
        ])

    def rank(self, image_list):
        # Parallel image download
        with ThreadPoolExecutor(max_workers=self.download_workers) as pool:
            futures = [pool.submit(_download_one, src, self.process) for src in image_list]
            results_raw = [f.result() for f in futures]

        valid = [(url, t) for r in results_raw if r is not None for url, t in [r]]
        if not valid:
            return []

        valid_urls = [v[0] for v in valid]
        valid_tensors = [v[1] for v in valid]

        with torch.no_grad():
            batch = torch.stack(valid_tensors).unsqueeze(0).to(self.device)
            valid_len = torch.tensor([len(valid_tensors)]).to(self.device)
            raw_scores = self.model(batch, valid_lens=valid_len).view(-1).cpu().numpy()

        results = [{'url': valid_urls[i], 'score': float(s)} for i, s in enumerate(raw_scores)]
        results.sort(key=lambda x: x['score'], reverse=True)
        return results


def load_csv(csv_path):
    """Load CSV and group images by property_id."""
    properties = defaultdict(lambda: {'images': [], 'ground_truth': None, 'total': 0})

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row['property_id']
            url = row['image_url']
            is_gt = row['is_ground_truth'].strip().upper() == 'TRUE'
            properties[pid]['images'].append(url)
            if 'total_scored_images' in row:
                properties[pid]['total'] = int(row['total_scored_images'])
            if is_gt:
                properties[pid]['ground_truth'] = url

    return dict(properties)


def evaluate_shard(gpu_id, prop_ids, properties, model_path, config_path, download_workers, return_dict):
    """Evaluate a shard of properties on a single GPU."""
    device = f"cuda:{gpu_id}"
    ranker = PropertyRanker(model_path, config_path, device=device, download_workers=download_workers)

    rank_distribution = defaultdict(int)
    total = 0
    failed = 0

    for i, pid in enumerate(prop_ids):
        prop = properties[pid]
        gt_url = prop['ground_truth']

        if not gt_url or len(prop['images']) < 2:
            continue

        total += 1

        if (i + 1) % 50 == 0:
            correct_so_far = rank_distribution.get(1, 0)
            print(f"  [GPU {gpu_id}] [{i+1}/{len(prop_ids)}] Accuracy: {correct_so_far}/{total} = {correct_so_far/total*100:.1f}%")

        ranked = ranker.rank(prop['images'])

        if not ranked:
            failed += 1
            continue

        gt_rank = None
        for r, item in enumerate(ranked):
            if item['url'] == gt_url:
                gt_rank = r + 1
                break

        if gt_rank is None:
            failed += 1
            continue

        rank_distribution[gt_rank] += 1

    return_dict[gpu_id] = {
        'rank_distribution': dict(rank_distribution),
        'total': total,
        'failed': failed,
    }


def print_report(rank_distribution, total, failed):
    matched = total - failed
    print("\n" + "=" * 60)
    print("  BEST IMAGE ACCURACY REPORT")
    print("=" * 60)
    print(f"  Total properties evaluated : {total}")
    print(f"  Failed to process          : {failed}")
    print(f"  Successfully ranked        : {matched}")
    print()

    if matched == 0:
        print("  No data to report.")
        return

    correct = rank_distribution.get(1, 0)
    top3 = sum(rank_distribution.get(r, 0) for r in range(1, 4))
    top5 = sum(rank_distribution.get(r, 0) for r in range(1, 6))

    print(f"  Top-1 Accuracy : {correct}/{matched} = {correct/matched*100:.1f}%")
    print(f"  Top-3 Accuracy : {top3}/{matched} = {top3/matched*100:.1f}%")
    print(f"  Top-5 Accuracy : {top5}/{matched} = {top5/matched*100:.1f}%")
    print()

    print("  Rank Distribution:")
    max_rank = max(rank_distribution.keys()) if rank_distribution else 0
    for r in range(1, min(max_rank + 1, 16)):
        count = rank_distribution.get(r, 0)
        pct = count / matched * 100
        bar = "#" * int(pct)
        label = " <-- model correct" if r == 1 else ""
        print(f"    Rank #{r:2d}: {count:4d} ({pct:5.1f}%) {bar}{label}")

    if max_rank > 15:
        remainder = sum(rank_distribution.get(r, 0) for r in range(16, max_rank + 1))
        pct = remainder / matched * 100
        print(f"    Rank 16+: {remainder:4d} ({pct:5.1f}%)")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate ranker on ground-truth CSV")
    parser.add_argument("--csv", required=True, help="Path to ground-truth CSV")
    parser.add_argument("--model", default=None, help="Path to model checkpoint")
    parser.add_argument("--config", default="config.yml")
    parser.add_argument("--workers", type=int, default=8, help="Download threads per GPU")
    args = parser.parse_args()

    # Auto-detect model
    model_path = args.model
    if not model_path:
        if os.path.exists("checkpoints/best_model.pth"):
            model_path = "checkpoints/best_model.pth"
        elif os.path.exists("checkpoints/last.pth"):
            model_path = "checkpoints/last.pth"
        else:
            checkpoints = sorted(glob.glob("checkpoints/*.pth"), key=os.path.getmtime)
            model_path = checkpoints[-1] if checkpoints else None

    if not model_path:
        print("No model found!")
        exit(1)

    print(f"CSV   : {args.csv}")
    print(f"Model : {model_path}")

    properties = load_csv(args.csv)

    # Filter to evaluable properties
    eval_pids = [pid for pid, p in properties.items() if p['ground_truth'] and len(p['images']) >= 2]
    print(f"Loaded {len(properties)} properties, {len(eval_pids)} evaluable\n")

    n_gpus = torch.cuda.device_count()
    if n_gpus < 2:
        # Single GPU / CPU fallback
        print(f"Using 1 GPU\n")
        ranker = PropertyRanker(model_path, args.config, download_workers=args.workers)
        start = time.time()

        rank_distribution = defaultdict(int)
        total = 0
        failed = 0

        for i, pid in enumerate(eval_pids):
            prop = properties[pid]
            total += 1

            if (i + 1) % 50 == 0:
                correct = rank_distribution.get(1, 0)
                print(f"  [{i+1}/{len(eval_pids)}] Accuracy: {correct}/{total} = {correct/total*100:.1f}%")

            ranked = ranker.rank(prop['images'])
            if not ranked:
                failed += 1
                continue

            gt_rank = None
            for r, item in enumerate(ranked):
                if item['url'] == prop['ground_truth']:
                    gt_rank = r + 1
                    break
            if gt_rank is None:
                failed += 1
                continue
            rank_distribution[gt_rank] += 1

        elapsed = time.time() - start
        print_report(dict(rank_distribution), total, failed)
        print(f"  Total eval time: {elapsed:.1f}s")
        return

    # Multi-GPU: split properties across GPUs
    print(f"Using {n_gpus} GPUs\n")

    shards = [[] for _ in range(n_gpus)]
    for i, pid in enumerate(eval_pids):
        shards[i % n_gpus].append(pid)

    for g in range(n_gpus):
        print(f"  GPU {g}: {len(shards[g])} properties")
    print()

    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    return_dict = manager.dict()

    start = time.time()
    processes = []
    for gpu_id in range(n_gpus):
        p = mp.Process(
            target=evaluate_shard,
            args=(gpu_id, shards[gpu_id], properties, model_path, args.config, args.workers, return_dict)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    elapsed = time.time() - start

    # Merge results from all GPUs
    merged_dist = defaultdict(int)
    merged_total = 0
    merged_failed = 0

    for gpu_id in range(n_gpus):
        result = return_dict[gpu_id]
        for rank, count in result['rank_distribution'].items():
            merged_dist[rank] += count
        merged_total += result['total']
        merged_failed += result['failed']

    print_report(dict(merged_dist), merged_total, merged_failed)
    print(f"  Total eval time: {elapsed:.1f}s ({n_gpus} GPUs)")


if __name__ == "__main__":
    main()
