"""
Evaluate a trained ranker against real ground-truth data.

Usage:
    python evaluate.py --csv /path/to/ground_truth.csv \
                       --model /path/to/best_model.pth \
                       [--config config.yml]

Expected CSV: property_id, image_url, is_ground_truth (TRUE/FALSE), ...
"""

import argparse
import os
import time
import csv
import glob
import numpy as np
from collections import defaultdict
from inference import PropertyRanker


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


def evaluate(ranker, properties):
    rank_distribution = defaultdict(int)
    total = 0
    failed = 0
    mrr_sum = 0.0
    results_log = []

    prop_ids = list(properties.keys())
    print(f"Evaluating {len(prop_ids)} properties...\n")

    for i, pid in enumerate(prop_ids):
        prop = properties[pid]
        gt_url = prop['ground_truth']

        if not gt_url or len(prop['images']) < 2:
            continue

        total += 1

        if (i + 1) % 50 == 0:
            correct_so_far = rank_distribution.get(1, 0)
            print(f"  [{i+1}/{len(prop_ids)}] Accuracy so far: {correct_so_far}/{total} = {correct_so_far/total*100:.1f}%")

        ranked = ranker.rank(prop['images'])

        if not ranked:
            failed += 1
            continue

        # Find ground truth rank in model's ranking
        gt_rank = None
        for r, item in enumerate(ranked):
            if item['source'] == gt_url:
                gt_rank = r + 1
                break

        if gt_rank is None:
            failed += 1
            continue

        rank_distribution[gt_rank] += 1
        mrr_sum += 1.0 / gt_rank

        results_log.append({
            'property_id': pid,
            'total_images': len(prop['images']),
            'gt_rank': gt_rank,
            'model_top_url': ranked[0]['source'],
            'model_top_score': ranked[0]['score'],
            'gt_url': gt_url,
            'gt_score': next(r['score'] for r in ranked if r['source'] == gt_url),
        })

    return rank_distribution, total, failed, mrr_sum, results_log


def print_report(rank_distribution, total, failed, mrr_sum):
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
    print(f"  MRR            : {mrr_sum/matched:.4f}")
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
    args = parser.parse_args()

    # Auto-detect model if not provided
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
    print(f"Loaded {len(properties)} properties\n")

    ranker = PropertyRanker(model_path=model_path, config_path=args.config)

    start = time.time()
    rank_dist, total, failed, mrr_sum, results_log = evaluate(ranker, properties)
    elapsed = time.time() - start

    print_report(rank_dist, total, failed, mrr_sum)
    print(f"  Total eval time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
