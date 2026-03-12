"""Binary classification trainer for best-image selection.

Phase 1: Precompute backbone features (once, cached to disk).
Phase 2: Contrastive group-level training — for each group, softmax over
         image scores, CE loss on the selected image. Directly optimizes
         "pick the hero shot from the group."

Usage:
    python train_binary.py                  # runs both phases
    python train_binary.py --skip-cache     # skip phase 1 if already cached
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from model import _build_backbone, _find_unfrozen_modules, get_norm_stats
from utils import load_config

BINARY_CSV = "best_image_training_data.csv"


class BinaryClassifier(nn.Module):
    """Pointwise binary classifier: backbone -> MLP -> score. Used for inference."""

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
            ds = ImageOnlyDataset(gpu_paths, img_size=cfg.data.img_size, norm_stats=norm_stats)
            loader = DataLoader(ds, batch_size=32, num_workers=2, pin_memory=True, shuffle=False)

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


# ── Phase 2: Contrastive group-level training ─────────────────────────

class GroupDataset(Dataset):
    """Each sample is a group: returns (features[N, D], target_idx, valid_len).

    The model sees all images in a group and must pick the selected one.
    Groups are padded to max_per_group; target_idx is the position of the
    selected image within the group.
    """

    def __init__(self, df, cache_dir, max_per_group=15):
        self.cache_dir = cache_dir
        self.max_per_group = max_per_group
        self.groups = []

        for gid, group in df.groupby('group_id'):
            items = []
            target_idx = -1
            for _, row in group.iterrows():
                fname = os.path.basename(row['file_path']).replace('.jpg', '.pt')
                cache_path = os.path.join(cache_dir, fname)
                if not os.path.exists(cache_path):
                    continue
                if row['selected'] == 1 and target_idx == -1:
                    target_idx = len(items)
                items.append(cache_path)

            # Need at least 2 images and a selected one
            if len(items) >= 2 and target_idx >= 0:
                self.groups.append((items[:max_per_group],
                                    min(target_idx, max_per_group - 1)))

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        items, target_idx = self.groups[idx]
        feats = []
        for path in items:
            feats.append(torch.load(path, weights_only=True).float())

        valid_len = len(feats)

        # Pad to max_per_group
        if valid_len < self.max_per_group:
            pad_dim = feats[0].shape[-1]
            feats += [torch.zeros(pad_dim)] * (self.max_per_group - valid_len)

        return (torch.stack(feats),
                torch.tensor(target_idx, dtype=torch.long),
                torch.tensor(valid_len, dtype=torch.long))


class ContrastiveHead(nn.Module):
    """Scores each image in a group. Trained with CE: selected image should get highest score.

    Uses self-attention so each image's score is informed by what else is in the group.
    A mediocre photo looks worse when compared to a great one — attention captures this.
    """

    def __init__(self, in_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        n_heads = 8
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = nn.MultiheadAttention(in_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        # Zero-init attention output so it starts as identity (pure MLP)
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, x, valid_lens=None):
        """x: (B, G, D) -> (B, G) scores"""
        B, G, _ = x.shape

        key_padding_mask = None
        if valid_lens is not None:
            key_padding_mask = (
                torch.arange(G, device=x.device).unsqueeze(0) >= valid_lens.unsqueeze(1)
            )

        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, key_padding_mask=key_padding_mask)
        x = x + attn_out
        scores = self.mlp(self.norm2(x)).squeeze(-1)  # (B, G)

        # Mask padded positions to -inf so they never win softmax
        if valid_lens is not None:
            mask = torch.arange(G, device=x.device).unsqueeze(0) >= valid_lens.unsqueeze(1)
            scores = scores.masked_fill(mask, float('-inf'))

        return scores


def validate_contrastive(model, val_loader, device):
    """GoldAcc: how often does argmax(scores) land on the selected image?"""
    model.eval()
    hits = total = 0
    with torch.no_grad():
        for feats, targets, vlens in val_loader:
            feats, targets, vlens = feats.to(device), targets.to(device), vlens.to(device)
            scores = model(feats, vlens)
            preds = scores.argmax(dim=-1)
            hits += (preds == targets).sum().item()
            total += targets.size(0)
    return hits / total if total else 0.0


def train_head(df, cfg, device, cache_dir):
    """Train contrastive head on precomputed features."""
    import random
    seed = getattr(cfg.train, 'seed', 42)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    def bcfg(key, default=None):
        return getattr(getattr(cfg, 'binary', None), key, None) or getattr(cfg.train, key, default)

    lr_head = bcfg('lr_head', 1e-3)

    # Train/val split by group
    rng = np.random.RandomState(seed)
    all_groups = df['group_id'].unique().tolist()
    rng.shuffle(all_groups)
    val_size = int(len(all_groups) * 0.1)
    val_groups = set(all_groups[:val_size])

    train_df = df[~df['group_id'].isin(val_groups)]
    val_df = df[df['group_id'].isin(val_groups)]

    max_per_group = getattr(cfg.data, 'max_images_per_group', 15)
    train_ds = GroupDataset(train_df, cache_dir, max_per_group)
    val_ds = GroupDataset(val_df, cache_dir, max_per_group)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # Detect feature dim
    sample_feats, _, _ = train_ds[0]
    feat_dim = sample_feats.shape[-1]

    hidden = getattr(cfg.model, 'head_hidden_dim', 256)
    dropout = getattr(cfg.model, 'head_dropout', 0.1)
    model = ContrastiveHead(feat_dim, hidden, dropout).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr_head, weight_decay=cfg.train.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    patience = getattr(cfg.train, 'patience', 20)
    patience_counter = 0
    best_metric = 0.0

    print(f"Contrastive training | {len(train_ds)} train groups, {len(val_ds)} val groups | "
          f"feat_dim={feat_dim} | max_per_group={max_per_group} | lr={lr_head:.1e}")

    os.makedirs(cfg.train.save_dir, exist_ok=True)

    for epoch in range(cfg.train.epochs):
        model.train()
        total_loss = 0.0

        for feats, targets, vlens in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            feats, targets, vlens = feats.to(device), targets.to(device), vlens.to(device)
            scores = model(feats, vlens)
            loss = criterion(scores, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        gold_acc = validate_contrastive(model, val_loader, device)
        lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | GoldAcc: {gold_acc:.2%} | LR: {lr:.2e}")

        if gold_acc > best_metric:
            best_metric = gold_acc
            patience_counter = 0
            torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                        'feat_dim': feat_dim},
                       f"{cfg.train.save_dir}/best_binary_head.pth")
            print(f"  -> New best GoldAcc: {gold_acc:.2%}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping. Best GoldAcc: {best_metric:.2%}")
            break

    # Save full model for inference (backbone + head mapped to MobileCLIPRanker format)
    print("Assembling full model for inference...")
    from model import MobileCLIPRanker
    full_model = MobileCLIPRanker(cfg)
    head_ckpt = torch.load(f"{cfg.train.save_dir}/best_binary_head.pth", weights_only=True)
    # ContrastiveHead has the same structure as RankingHead — load directly
    full_model.head.load_state_dict(
        {k: v for k, v in head_ckpt['model_state_dict'].items()})
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

    # Phase 2: Train contrastive head
    train_head(df, cfg, device, cache_dir)


if __name__ == "__main__":
    main()
