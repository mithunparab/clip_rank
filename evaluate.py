"""
Evaluate a trained ranker against real ground-truth data.

Usage:
    python evaluate.py --csv /path/to/ground_truth.csv \
                       --model /path/to/best_model.pth \
                       [--config config.yml]

Expected CSV columns (auto-detected):
    - group col:  property_id | group_id | listing_id
    - image col:  image_url | url | photo_url
    - truth col:  selected (0/1) | score (numeric) | position/rank (lower=better)
"""

import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
from inference import PropertyRanker


# ── column detection ────────────────────────────────────────────────

def detect_columns(df):
    cols = {c.lower(): c for c in df.columns}

    # group column
    for candidate in ["property_id", "group_id", "listing_id", "propid"]:
        if candidate in cols:
            group_col = cols[candidate]
            break
    else:
        raise ValueError(f"No group column found. Have: {list(df.columns)}")

    # image column
    for candidate in ["image_url", "url", "photo_url", "img_url", "image"]:
        if candidate in cols:
            image_col = cols[candidate]
            break
    else:
        raise ValueError(f"No image URL column found. Have: {list(df.columns)}")

    # truth column + type
    if "selected" in cols:
        truth_col, truth_type = cols["selected"], "binary"
    elif "score" in cols:
        truth_col, truth_type = cols["score"], "score"
    elif "position" in cols:
        truth_col, truth_type = cols["position"], "position"
    elif "rank" in cols:
        truth_col, truth_type = cols["rank"], "position"
    else:
        raise ValueError(f"No ground-truth column found. Have: {list(df.columns)}")

    return group_col, image_col, truth_col, truth_type


# ── metrics ─────────────────────────────────────────────────────────

def compute_ndcg(pred_scores, gt_relevance):
    n = len(pred_scores)
    if n < 2:
        return 1.0
    order = np.argsort(-pred_scores)
    ordered_gt = gt_relevance[order]
    discounts = np.log2(np.arange(n) + 2)
    dcg = np.sum((2 ** ordered_gt - 1) / discounts)
    ideal_gt = np.sort(gt_relevance)[::-1]
    idcg = np.sum((2 ** ideal_gt - 1) / discounts)
    return dcg / idcg if idcg > 0 else 1.0


def evaluate(ranker, df, group_col, image_col, truth_col, truth_type, top_k=1):
    """Run model on every group, compare to ground truth."""
    from scipy.stats import spearmanr

    groups = df.groupby(group_col)
    total = len(groups)

    # accumulators
    top1_correct = 0
    topk_correct = 0
    groups_evaluated = 0
    ndcg_scores = []
    spearman_scores = []
    mrr_sum = 0.0
    failures = []

    for gid, group in groups:
        urls = group[image_col].tolist()

        # ground truth relevance
        if truth_type == "binary":
            gt = group[truth_col].values.astype(float)
        elif truth_type == "score":
            gt = group[truth_col].values.astype(float)
        elif truth_type == "position":
            # lower position = better → invert to relevance
            pos = group[truth_col].values.astype(float)
            gt = pos.max() - pos + 1
        else:
            continue

        if len(urls) < 2:
            continue

        # skip groups with no positive signal
        if gt.max() == gt.min():
            continue

        # run model
        try:
            results = ranker.rank(urls)
        except Exception as e:
            failures.append((gid, str(e)))
            continue

        if len(results) < 2:
            failures.append((gid, "too few images loaded"))
            continue

        # map model scores back to original order
        url_to_model_score = {r["source"]: r["score"] for r in results}
        pred_scores = []
        gt_aligned = []
        for url, g in zip(urls, gt):
            if url in url_to_model_score:
                pred_scores.append(url_to_model_score[url])
                gt_aligned.append(g)

        if len(pred_scores) < 2:
            continue

        pred_scores = np.array(pred_scores)
        gt_aligned = np.array(gt_aligned)

        groups_evaluated += 1

        # ── top-1 accuracy ──
        model_pick = np.argmax(pred_scores)
        if truth_type == "binary":
            is_correct = gt_aligned[model_pick] == 1
        else:
            is_correct = gt_aligned[model_pick] == gt_aligned.max()

        if is_correct:
            top1_correct += 1

        # ── top-k accuracy ──
        topk_indices = np.argsort(-pred_scores)[:top_k]
        if truth_type == "binary":
            topk_hit = any(gt_aligned[i] == 1 for i in topk_indices)
        else:
            topk_hit = any(gt_aligned[i] == gt_aligned.max() for i in topk_indices)
        if topk_hit:
            topk_correct += 1

        # ── MRR ──
        ranked_indices = np.argsort(-pred_scores)
        for rank_pos, idx in enumerate(ranked_indices, 1):
            if truth_type == "binary":
                if gt_aligned[idx] == 1:
                    mrr_sum += 1.0 / rank_pos
                    break
            else:
                if gt_aligned[idx] == gt_aligned.max():
                    mrr_sum += 1.0 / rank_pos
                    break

        # ── NDCG ──
        ndcg_scores.append(compute_ndcg(pred_scores, gt_aligned))

        # ── Spearman ──
        if len(np.unique(gt_aligned)) > 1:
            rho, _ = spearmanr(pred_scores, gt_aligned)
            if not np.isnan(rho):
                spearman_scores.append(rho)

    return {
        "groups_evaluated": groups_evaluated,
        "groups_total": total,
        "failures": len(failures),
        "top1_accuracy": top1_correct / groups_evaluated if groups_evaluated else 0,
        "topk_accuracy": topk_correct / groups_evaluated if groups_evaluated else 0,
        "top_k": top_k,
        "mrr": mrr_sum / groups_evaluated if groups_evaluated else 0,
        "ndcg": np.mean(ndcg_scores) if ndcg_scores else 0,
        "spearman": np.mean(spearman_scores) if spearman_scores else 0,
        "failure_details": failures[:10],
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate ranker on ground-truth CSV")
    parser.add_argument("--csv", required=True, help="Path to ground-truth CSV")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--config", default="config.yml")
    parser.add_argument("--top-k", type=int, default=3, help="k for top-k accuracy")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")
    print(f"Columns: {list(df.columns)}")

    group_col, image_col, truth_col, truth_type = detect_columns(df)
    print(f"Detected: group={group_col}, image={image_col}, truth={truth_col} ({truth_type})")

    n_groups = df[group_col].nunique()
    print(f"Groups: {n_groups}\n")

    ranker = PropertyRanker(model_path=args.model, config_path=args.config)

    start = time.time()
    metrics = evaluate(ranker, df, group_col, image_col, truth_col, truth_type, top_k=args.top_k)
    elapsed = time.time() - start

    print("\n" + "=" * 55)
    print("  EVALUATION RESULTS")
    print("=" * 55)
    print(f"  Groups evaluated : {metrics['groups_evaluated']} / {metrics['groups_total']}")
    print(f"  Failures         : {metrics['failures']}")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  Top-1 Accuracy   : {metrics['top1_accuracy']:.2%}")
    print(f"  Top-{metrics['top_k']} Accuracy   : {metrics['topk_accuracy']:.2%}")
    print(f"  MRR              : {metrics['mrr']:.4f}")
    print(f"  NDCG             : {metrics['ndcg']:.4f}")
    print(f"  Spearman ρ       : {metrics['spearman']:.4f}")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  Time             : {elapsed:.1f}s")
    print("=" * 55)

    if metrics["failure_details"]:
        print(f"\nFirst {len(metrics['failure_details'])} failures:")
        for gid, err in metrics["failure_details"]:
            print(f"  {gid}: {err}")


if __name__ == "__main__":
    main()
