"""Binary classification trainer for best-image selection.

Phase 1: Precompute backbone features (once, cached to disk).
Phase 2: Train a lightweight MLP head on cached features (~seconds/epoch).

Usage:
    python train_binary.py                  # runs both phases
    python train_binary.py --skip-cache     # skip phase 1 if already cached
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from model import _build_backbone, _find_unfrozen_modules, get_norm_stats
from utils import load_config

BINARY_CSV = "best_image_training_data.csv"


class BinaryClassifier(nn.Module):
    """Pointwise binary classifier: backbone -> MLP -> sigmoid."""

    def __init__(self, cfg):
        super().__init__()
        self.backbone, self.backbone_dim = _build_backbone(cfg)
        self._unfrozen_modules = _find_unfrozen_modules(self.backbone)

        hidden = getattr(cfg.model, 'head_hidden_dim', 256)
        dropout = getattr(cfg.model, 'head_dropout', 0.1)
        self.head = nn.Sequential(
            nn.LayerNorm(self.backbone_dim),
            nn.Linear(self.backbone_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        if mode:
            for module in self._unfrozen_modules:
                module.train()
                for sub in module.modules():
                    if isinstance(sub, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
                        sub.eval()
        return self

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features).squeeze(-1)


# ── Phase 1: Precompute features ──────────────────────────────────────

class ImageOnlyDataset(Dataset):
    """Returns (image_tensor, index) for feature extraction."""

    def __init__(self, file_paths, img_size=224, norm_stats=None):
        self.paths = file_paths
        mean, std = norm_stats or ((0.481, 0.457, 0.408), (0.268, 0.261, 0.275))
        self.transform = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            with Image.open(self.paths[idx]) as img:
                return self.transform(img.convert('RGB')), idx
        except Exception:
            return torch.zeros(3, 224, 224), idx


def _extract_on_device(backbone, loader, extract_paths, cache_dir, gpu_id):
    """Run feature extraction on a single GPU."""
    device = torch.device(f"cuda:{gpu_id}")
    backbone = backbone.to(device)
    count = 0
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for batch, batch_indices in loader:
            batch = batch.to(device)
            feats = backbone(batch).cpu()
            for j, feat in zip(batch_indices, feats):
                fname = os.path.basename(extract_paths[j.item()]).replace('.jpg', '.pt')
                torch.save(feat, os.path.join(cache_dir, fname))
                count += 1
    return count


def precompute_features(df, cfg, device, cache_dir):
    """Extract backbone features using all available GPUs in parallel."""
    os.makedirs(cache_dir, exist_ok=True)

    # Check which images still need extraction
    paths = df['file_path'].tolist()
    indices_to_extract = []
    for i, p in enumerate(paths):
        cache_path = os.path.join(cache_dir, f"{os.path.basename(p).replace('.jpg', '.pt')}")
        if not os.path.exists(cache_path) and os.path.exists(p):
            indices_to_extract.append(i)

    if not indices_to_extract:
        print(f"All {len(paths)} features already cached in {cache_dir}/")
        return

    norm_stats = get_norm_stats(cfg.model.name)
    extract_paths = [paths[i] for i in indices_to_extract]
    n_gpus = torch.cuda.device_count()

    print(f"Extracting features for {len(indices_to_extract)}/{len(paths)} images on {n_gpus} GPU(s)...")

    if n_gpus <= 1:
        # Single GPU path
        ds = ImageOnlyDataset(extract_paths, img_size=cfg.data.img_size, norm_stats=norm_stats)
        loader = DataLoader(ds, batch_size=32, num_workers=cfg.system.num_workers,
                            pin_memory=True, shuffle=False)
        backbone, _ = _build_backbone(cfg)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False
        _extract_on_device(backbone, tqdm(loader, desc="Extracting"), extract_paths, cache_dir, 0)
        del backbone
    else:
        # Multi-GPU: split images across GPUs, run in parallel threads
        import threading
        chunk_size = (len(extract_paths) + n_gpus - 1) // n_gpus
        threads = []
        progress_bars = []

        for gpu_id in range(n_gpus):
            start = gpu_id * chunk_size
            end = min(start + chunk_size, len(extract_paths))
            if start >= len(extract_paths):
                break
            gpu_paths = extract_paths[start:end]
            # Remap indices: ImageOnlyDataset uses local indices, but we need global indices for naming
            ds = ImageOnlyDataset(gpu_paths, img_size=cfg.data.img_size, norm_stats=norm_stats)
            loader = DataLoader(ds, batch_size=32, num_workers=2, pin_memory=True, shuffle=False)

            # Build a separate backbone copy per GPU
            backbone, _ = _build_backbone(cfg)
            backbone.eval()
            for p in backbone.parameters():
                p.requires_grad = False

            pbar = tqdm(loader, desc=f"GPU {gpu_id}", position=gpu_id)
            t = threading.Thread(target=_extract_on_device,
                                 args=(backbone, pbar, gpu_paths, cache_dir, gpu_id))
            threads.append(t)
            progress_bars.append(pbar)

        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for pb in progress_bars:
            pb.close()

    torch.cuda.empty_cache()
    print(f"Cached {len(indices_to_extract)} features to {cache_dir}/")


# ── Phase 2: Train head on cached features ────────────────────────────

class CachedBinaryDataset(Dataset):
    """Loads precomputed features + binary labels."""

    def __init__(self, df, cache_dir):
        self.cache_dir = cache_dir
        self.items = []
        for _, row in df.iterrows():
            fname = os.path.basename(row['file_path']).replace('.jpg', '.pt')
            cache_path = os.path.join(cache_dir, fname)
            if os.path.exists(cache_path):
                self.items.append((cache_path, int(row['selected'])))
        self.labels = np.array([item[1] for item in self.items])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        feat = torch.load(path, weights_only=True)
        return feat, torch.tensor(label, dtype=torch.float32)


class HeadOnly(nn.Module):
    """Standalone MLP head for training on cached features."""

    def __init__(self, in_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.head(x).squeeze(-1)


def validate_cached(model, val_loader, device):
    model.eval()
    correct = total = tp = fp = fn = tn = 0
    with torch.no_grad():
        for feats, labels in val_loader:
            feats, labels = feats.to(device), labels.to(device)
            preds = (model(feats) > 0).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()
    acc = correct / total if total else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return acc, prec, rec, f1


def validate_group_ranking_cached(model, df_val, cache_dir, device):
    """Gold accuracy on cached features: does the top-1 prediction pick selected=1?"""
    model.eval()
    top1_hits = total = 0
    with torch.no_grad():
        for _, group in df_val.groupby('group_id'):
            if len(group) < 2 or not (group['selected'] == 1).any():
                continue

            feats, gt_labels = [], []
            for _, row in group.iterrows():
                fname = os.path.basename(row['file_path']).replace('.jpg', '.pt')
                cache_path = os.path.join(cache_dir, fname)
                if not os.path.exists(cache_path):
                    continue
                feats.append(torch.load(cache_path, weights_only=True))
                gt_labels.append(int(row['selected']))

            if len(feats) < 2:
                continue

            batch = torch.stack(feats).to(device)
            logits = model(batch)
            top1_hits += gt_labels[logits.argmax().item()]
            total += 1

    return top1_hits / total if total else 0.0


def train_head(df, cfg, device, cache_dir):
    """Train the MLP head on precomputed features."""
    import random
    seed = getattr(cfg.train, 'seed', 42)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    def bcfg(key, default=None):
        return getattr(getattr(cfg, 'binary', None), key, None) or getattr(cfg.train, key, default)

    lr_head = bcfg('lr_head', 1e-3)
    accum_steps = bcfg('gradient_accumulation_steps', 4)

    # Train/val split by group
    rng = np.random.RandomState(seed)
    all_groups = df['group_id'].unique().tolist()
    rng.shuffle(all_groups)
    val_size = int(len(all_groups) * 0.1)
    val_groups = set(all_groups[:val_size])

    train_df = df[~df['group_id'].isin(val_groups)]
    val_df = df[df['group_id'].isin(val_groups)]

    train_ds = CachedBinaryDataset(train_df, cache_dir)
    val_ds = CachedBinaryDataset(val_df, cache_dir)

    n_pos = train_ds.labels.sum()
    n_neg = len(train_ds.labels) - n_pos

    # Weighted sampler for class balance
    class_counts = np.bincount(train_ds.labels)
    weights = 1.0 / class_counts[train_ds.labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=512, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    # Detect feature dim
    sample_feat, _ = train_ds[0]
    feat_dim = sample_feat.shape[-1]

    hidden = getattr(cfg.model, 'head_hidden_dim', 256)
    dropout = getattr(cfg.model, 'head_dropout', 0.1)
    model = HeadOnly(feat_dim, hidden, dropout).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr_head, weight_decay=cfg.train.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs, eta_min=1e-6)

    pos_weight = torch.tensor([n_neg / n_pos]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    patience = getattr(cfg.train, 'patience', 20)
    patience_counter = 0
    best_metric = 0.0

    print(f"Training head | {len(train_ds)} train, {len(val_ds)} val | "
          f"feat_dim={feat_dim} | lr={lr_head:.1e} | pos_weight={pos_weight.item():.2f}")

    os.makedirs(cfg.train.save_dir, exist_ok=True)

    for epoch in range(cfg.train.epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, (feats, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)):
            feats, labels = feats.to(device), labels.to(device)
            logits = model(feats)
            loss = criterion(logits, labels) / accum_steps
            loss.backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * accum_steps

        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        acc, prec, rec, f1 = validate_cached(model, val_loader, device)
        gold_acc = validate_group_ranking_cached(model, val_df, cache_dir, device)
        lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Acc: {acc:.2%} | "
              f"P: {prec:.2%} R: {rec:.2%} F1: {f1:.2%} | GoldAcc: {gold_acc:.2%} | LR: {lr:.2e}")

        if gold_acc > best_metric:
            best_metric = gold_acc
            patience_counter = 0
            torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                        'feat_dim': feat_dim}, f"{cfg.train.save_dir}/best_binary_head.pth")
            print(f"  -> New best GoldAcc: {gold_acc:.2%}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping. Best GoldAcc: {best_metric:.2%}")
            break

    # Save full model (backbone + head) for inference compatibility
    print("Assembling full model for inference...")
    full_model = BinaryClassifier(cfg)
    head_ckpt = torch.load(f"{cfg.train.save_dir}/best_binary_head.pth", weights_only=True)
    # Map HeadOnly state_dict to BinaryClassifier's head
    full_model.head.load_state_dict({k.replace('head.', ''): v
                                     for k, v in head_ckpt['model_state_dict'].items()})
    torch.save({'epoch': head_ckpt['epoch'],
                'model_state_dict': full_model.state_dict()},
               f"{cfg.train.save_dir}/best_model.pth")
    print(f"Saved full model to {cfg.train.save_dir}/best_model.pth")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-cache', action='store_true', help='Skip feature extraction (use existing cache)')
    args = parser.parse_args()

    cfg = load_config("config.yml")
    cache_dir = "cached_features_binary"
    device = torch.device(cfg.system.device if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(BINARY_CSV)
    df = df.rename(columns={'property_id': 'group_id', 'image_url': 'url'})
    df['file_path'] = df.index.map(lambda x: os.path.join('images', f"{x}.jpg"))

    print(f"Binary data: {len(df)} images, {df['group_id'].nunique()} groups")

    # Phase 1: Extract features
    if not args.skip_cache:
        precompute_features(df, cfg, device, cache_dir)

    # Phase 2: Train head
    train_head(df, cfg, device, cache_dir)


if __name__ == "__main__":
    main()
