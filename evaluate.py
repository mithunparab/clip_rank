import torch
import requests
import yaml
import os
import glob
import csv
import time
from PIL import Image
from io import BytesIO
from types import SimpleNamespace
from torchvision import transforms
from collections import defaultdict
from model import MobileCLIPRanker, OrdinalRanker, LDLRanker, get_norm_stats


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


class PropertyRanker:
    def __init__(self, model_path, config_path='config.yml', device=None):
        self.cfg = load_config(config_path)
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        print(f"--- Loading Ranker ---")
        print(f"Device: {self.device}")
        print(f"Loading Weights: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k.replace("module.", "")] = v

        self.is_ordinal = any('head.biases' in k for k in new_state_dict)
        self.is_ldl = any('score_values' in k for k in new_state_dict)

        if self.is_ldl:
            print("Detected LDL model")
            self.model = LDLRanker(self.cfg)
        elif self.is_ordinal:
            print("Detected ordinal (CORAL) model")
            self.model = OrdinalRanker(self.cfg)
        else:
            self.model = MobileCLIPRanker(self.cfg)

        self.model.load_state_dict(new_state_dict)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully.\n")

        norm_mean, norm_std = get_norm_stats(self.cfg.model.name)
        self.process = transforms.Compose([
            transforms.Resize(self.cfg.data.img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.cfg.data.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std)
        ])

    def rank(self, image_list):
        valid_tensors = []
        valid_urls = []

        for src in image_list:
            if not src or not isinstance(src, str):
                continue
            try:
                if src.startswith("http"):
                    resp = requests.get(src, timeout=10)
                    img = Image.open(BytesIO(resp.content)).convert('RGB')
                else:
                    img = Image.open(src).convert('RGB')
                valid_tensors.append(self.process(img))
                valid_urls.append(src)
            except Exception as e:
                print(f"  Skip: {e}")

        if not valid_tensors:
            return []

        with torch.no_grad():
            if self.is_ldl or self.is_ordinal:
                batch = torch.stack(valid_tensors).to(self.device)
                raw_scores = self.model.score(batch).cpu().numpy()
            else:
                batch = torch.stack(valid_tensors).unsqueeze(0).to(self.device)
                valid_len = torch.tensor([len(valid_tensors)]).to(self.device)
                raw_scores = self.model(batch, valid_lens=valid_len).view(-1).cpu().numpy()

        results = []
        for i, score in enumerate(raw_scores):
            results.append({'url': valid_urls[i], 'score': float(score)})

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
            properties[pid]['total'] = int(row['total_scored_images'])
            if is_gt:
                properties[pid]['ground_truth'] = url

    return dict(properties)


def evaluate(ranker, properties):
    rank_distribution = defaultdict(int)
    total = 0
    failed = 0
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
            if item['url'] == gt_url:
                gt_rank = r + 1
                break

        if gt_rank is None:
            failed += 1
            continue

        rank_distribution[gt_rank] += 1
        results_log.append({
            'property_id': pid,
            'total_images': len(prop['images']),
            'gt_rank': gt_rank,
            'model_top_url': ranked[0]['url'],
            'model_top_score': ranked[0]['score'],
            'gt_url': gt_url,
            'gt_score': next(r['score'] for r in ranked if r['url'] == gt_url),
        })

    return rank_distribution, total, failed, results_log


def print_report(rank_distribution, total, failed):
    matched = total - failed
    print("\n" + "=" * 60)
    print("BEST IMAGE ACCURACY REPORT")
    print("=" * 60)
    print(f"Total properties evaluated: {total}")
    print(f"Failed to process: {failed}")
    print(f"Successfully ranked: {matched}")
    print()

    if matched == 0:
        print("No data to report.")
        return

    correct = rank_distribution.get(1, 0)
    top3 = sum(rank_distribution.get(r, 0) for r in range(1, 4))
    top5 = sum(rank_distribution.get(r, 0) for r in range(1, 6))

    print(f"Top-1 Accuracy (exact match): {correct}/{matched} = {correct/matched*100:.1f}%")
    print(f"Top-3 Accuracy: {top3}/{matched} = {top3/matched*100:.1f}%")
    print(f"Top-5 Accuracy: {top5}/{matched} = {top5/matched*100:.1f}%")
    print()

    print("Rank Distribution:")
    max_rank = max(rank_distribution.keys()) if rank_distribution else 0
    for r in range(1, min(max_rank + 1, 16)):
        count = rank_distribution.get(r, 0)
        pct = count / matched * 100
        bar = "#" * int(pct)
        label = " <-- model correct" if r == 1 else ""
        print(f"  Rank #{r:2d}: {count:4d} ({pct:5.1f}%) {bar}{label}")

    if max_rank > 15:
        remainder = sum(rank_distribution.get(r, 0) for r in range(16, max_rank + 1))
        pct = remainder / matched * 100
        print(f"  Rank 16+: {remainder:4d} ({pct:5.1f}%)")

    print("=" * 60)


if __name__ == "__main__":
    # Kaggle paths
    KAGGLE_DIR = "/kaggle/inputs/ranking"
    # Local fallbacks
    LOCAL_CSV = "ranking_model_vs_ground_truth_2days.csv"

    # CSV: prefer Kaggle, fallback local
    if os.path.exists(f"{KAGGLE_DIR}/ranking_model_vs_ground_truth_2days.csv"):
        CSV_PATH = f"{KAGGLE_DIR}/ranking_model_vs_ground_truth_2days.csv"
    elif os.path.exists(LOCAL_CSV):
        CSV_PATH = LOCAL_CSV
    else:
        print("No eval CSV found!")
        exit()

    # Model: prefer Kaggle, fallback local checkpoints
    if os.path.exists(f"{KAGGLE_DIR}/best_model.pth"):
        model_path = f"{KAGGLE_DIR}/best_model.pth"
        print("Using Kaggle model.")
    elif os.path.exists("checkpoints/best_model.pth"):
        model_path = "checkpoints/best_model.pth"
        print("Using Best Model.")
    elif os.path.exists("checkpoints/last.pth"):
        model_path = "checkpoints/last.pth"
        print("Using Last Epoch.")
    else:
        checkpoints = sorted(glob.glob("checkpoints/*.pth"), key=os.path.getmtime)
        model_path = checkpoints[-1] if checkpoints else None

    if not model_path:
        print("No model found!")
        exit()

    print(f"CSV: {CSV_PATH}")
    properties = load_csv(CSV_PATH)
    print(f"Loaded {len(properties)} properties from CSV\n")

    ranker = PropertyRanker(model_path=model_path)

    start = time.time()
    rank_dist, total, failed, results_log = evaluate(ranker, properties)
    elapsed = time.time() - start

    print_report(rank_dist, total, failed)
    print(f"Total eval time: {elapsed:.1f}s")
