"""Binary best-image trainer using group-level KL-divergence loss.

Phase 1: Precompute frozen backbone features (cached to disk).
Phase 2: Train attention head with KL-div soft targets + gold-set loss.
         Same loss recipe that achieved 81% on human-annotated data.

Usage:
    python train_binary.py                  # both phases
    python train_binary.py --skip-cache     # reuse cached features
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from model import _build_backbone, _find_unfrozen_modules, get_norm_stats, MobileCLIPRanker
from utils import load_config

BINARY_CSV = "best_image_training_data.csv"


# ── Phase 1: Precompute features ──────────────────────────────────────

class ImageOnlyDataset(Dataset):
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
    device = torch.device(f"cuda:{gpu_id}")
    backbone = backbone.to(device)
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for batch, batch_indices in loader:
            batch = batch.to(device)
            feats = backbone(batch).cpu()
            for j, feat in zip(batch_indices, feats):
                fname = os.path.basename(extract_paths[j.item()]).replace('.jpg', '.pt')
                torch.save(feat, os.path.join(cache_dir, fname))


def precompute_features(df, cfg, device, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    paths = df['file_path'].tolist()
    to_extract = [i for i, p in enumerate(paths)
                  if not os.path.exists(os.path.join(cache_dir, os.path.basename(p).replace('.jpg', '.pt')))
                  and os.path.exists(p)]
    if not to_extract:
        print(f"All {len(paths)} features already cached.")
        return

    norm_stats = get_norm_stats(cfg.model.name)
    extract_paths = [paths[i] for i in to_extract]
    n_gpus = torch.cuda.device_count()
    print(f"Extracting {len(to_extract)}/{len(paths)} features on {n_gpus} GPU(s)...")

    if n_gpus <= 1:
        ds = ImageOnlyDataset(extract_paths, img_size=cfg.data.img_size, norm_stats=norm_stats)
        loader = DataLoader(ds, batch_size=32, num_workers=cfg.system.num_workers, pin_memory=True)
        backbone, _ = _build_backbone(cfg)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False
        _extract_on_device(backbone, tqdm(loader, desc="Extracting"), extract_paths, cache_dir, 0)
        del backbone
    else:
        import threading
        chunk = (len(extract_paths) + n_gpus - 1) // n_gpus
        threads, pbars = [], []
        for gpu_id in range(n_gpus):
            s, e = gpu_id * chunk, min((gpu_id + 1) * chunk, len(extract_paths))
            if s >= len(extract_paths):
                break
            gpu_paths = extract_paths[s:e]
            ds = ImageOnlyDataset(gpu_paths, img_size=cfg.data.img_size, norm_stats=norm_stats)
            loader = DataLoader(ds, batch_size=32, num_workers=2, pin_memory=True)
            backbone, _ = _build_backbone(cfg)
            backbone.eval()
            for p in backbone.parameters():
                p.requires_grad = False
            pbar = tqdm(loader, desc=f"GPU {gpu_id}", position=gpu_id)
            t = threading.Thread(target=_extract_on_device, args=(backbone, pbar, gpu_paths, cache_dir, gpu_id))
            threads.append(t); pbars.append(pbar)
        for t in threads: t.start()
        for t in threads: t.join()
        for pb in pbars: pb.close()

    torch.cuda.empty_cache()
    print(f"Cached to {cache_dir}/")


# ── Phase 2: Group KL-div + GoldSet training ─────────────────────────

class GroupDataset(Dataset):
    """Each sample = one property group. Returns (feats, scores, valid_len).
    Scores: selected=10.0, not-selected=0.0. Padded with -100."""

    def __init__(self, df, cache_dir, max_per_group=15):
        self.max_per_group = max_per_group
        self.groups = []
        for _, group in df.groupby('group_id'):
            items = []
            has_pos = False
            for _, row in group.iterrows():
                fname = os.path.basename(row['file_path']).replace('.jpg', '.pt')
                cp = os.path.join(cache_dir, fname)
                if not os.path.exists(cp):
                    continue
                score = 10.0 if row['selected'] == 1 else 0.0
                if row['selected'] == 1:
                    has_pos = True
                items.append((cp, score))
            if len(items) >= 2 and has_pos:
                self.groups.append(items[:max_per_group])

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        items = self.groups[idx]
        feats, scores = [], []
        for path, score in items:
            feats.append(torch.load(path, weights_only=True).float())
            scores.append(score)
        vlen = len(feats)
        if vlen < self.max_per_group:
            feats += [torch.zeros_like(feats[0])] * (self.max_per_group - vlen)
            scores += [-100.0] * (self.max_per_group - vlen)
        return torch.stack(feats), torch.tensor(scores), torch.tensor(vlen, dtype=torch.long)


class RankingHead(nn.Module):
    """Self-attention + MLP head. Same structure as model.py RankingHead."""

    def __init__(self, in_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = nn.MultiheadAttention(in_dim, 8, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, x, valid_lens=None):
        """x: (B, G, D) -> (B, G) scores"""
        G = x.shape[1]
        key_padding_mask = None
        if valid_lens is not None:
            key_padding_mask = torch.arange(G, device=x.device).unsqueeze(0) >= valid_lens.unsqueeze(1)
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, key_padding_mask=key_padding_mask)
        x = x + attn_out
        return self.mlp(self.norm2(x)).squeeze(-1)


# ── Losses (from train_ddp.py — proven at 81%) ───────────────────────

GOLD_THRESHOLD = 7.0  # selected=10 is gold, not-selected=0 is not


def gold_set_loss(pred_scores, gt_scores, valid_len):
    """P(argmax ∈ gold) loss. -log Σ softmax(pred)[gold]."""
    loss = 0.0
    count = 0
    for b in range(pred_scores.shape[0]):
        n = int(valid_len[b].item())
        if n < 2:
            continue
        logits = pred_scores[b, :n]
        gts = gt_scores[b, :n]
        gold_mask = gts >= GOLD_THRESHOLD
        if gold_mask.any() and not gold_mask.all():
            loss += torch.logsumexp(logits, dim=0) - torch.logsumexp(logits[gold_mask], dim=0)
            count += 1
    return loss / count if count > 0 else pred_scores.sum() * 0.0


def kl_div_loss(pred_scores, gt_scores, valid_len, temperature=1.0):
    """KL(target_dist || pred_dist) over valid positions per group."""
    loss = 0.0
    count = 0
    for b in range(pred_scores.shape[0]):
        n = int(valid_len[b].item())
        if n < 2:
            continue
        preds = pred_scores[b, :n]
        gts = gt_scores[b, :n]

        # Soft target distribution from ground-truth scores
        target_dist = F.softmax(gts / temperature, dim=0)
        pred_log_dist = F.log_softmax(preds, dim=0)

        loss += F.kl_div(pred_log_dist, target_dist, reduction='sum')
        count += 1
    return loss / count if count > 0 else pred_scores.sum() * 0.0


def plackett_luce_loss(pred_scores, gt_scores, valid_len, temperature=1.0):
    """Plackett-Luce listwise ranking loss."""
    loss = 0.0
    count = 0
    for b in range(pred_scores.shape[0]):
        n = int(valid_len[b].item())
        if n < 2:
            continue
        preds = pred_scores[b, :n]
        gts = gt_scores[b, :n]
        gt_noisy = gts + torch.randn_like(gts) * 1e-4
        order = torch.argsort(gt_noisy, descending=True)
        sorted_preds = preds[order] / temperature
        log_cumsum = torch.flip(torch.logcumsumexp(torch.flip(sorted_preds, [0]), dim=0), [0])
        loss += (log_cumsum[:-1] - sorted_preds[:-1]).mean()
        count += 1
    return loss / count if count > 0 else pred_scores.sum() * 0.0


# ── Training loop ────────────────────────────────────────────────────

def validate(model, val_loader, device):
    """GoldAcc: does argmax(pred) land on a selected=1 image?"""
    model.eval()
    hits = total = 0
    with torch.no_grad():
        for feats, scores, vlens in val_loader:
            feats, scores, vlens = feats.to(device), scores.to(device), vlens.to(device)
            preds = model(feats, vlens)
            for b in range(preds.shape[0]):
                n = int(vlens[b].item())
                if n < 2:
                    continue
                gt = scores[b, :n]
                if (gt >= GOLD_THRESHOLD).any():
                    top1 = preds[b, :n].argmax().item()
                    hits += int(gt[top1] >= GOLD_THRESHOLD)
                    total += 1
    return hits / total if total else 0.0


def train_head(df, cfg, device, cache_dir):
    import random
    seed = getattr(cfg.train, 'seed', 42)
    random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)

    def bcfg(key, default=None):
        return getattr(getattr(cfg, 'binary', None), key, None) or getattr(cfg.train, key, default)

    lr_head = bcfg('lr_head', 1e-3)
    temperature = getattr(cfg.train, 'temperature', 0.3)
    margin_weight = getattr(cfg.train, 'margin_weight', 0.3)

    rng = np.random.RandomState(seed)
    all_groups = df['group_id'].unique().tolist()
    rng.shuffle(all_groups)
    val_groups = set(all_groups[:int(len(all_groups) * 0.1)])

    max_pg = getattr(cfg.data, 'max_images_per_group', 15)
    train_ds = GroupDataset(df[~df['group_id'].isin(val_groups)], cache_dir, max_pg)
    val_ds = GroupDataset(df[df['group_id'].isin(val_groups)], cache_dir, max_pg)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    feat_dim = train_ds[0][0].shape[-1]
    hidden = getattr(cfg.model, 'head_hidden_dim', 256)
    dropout = getattr(cfg.model, 'head_dropout', 0.1)
    model = RankingHead(feat_dim, hidden, dropout).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr_head, weight_decay=cfg.train.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs, eta_min=1e-6)

    patience = getattr(cfg.train, 'patience', 20)
    patience_counter = 0
    best_metric = 0.0
    os.makedirs(cfg.train.save_dir, exist_ok=True)

    print(f"KL-div + GoldSet + PL training | {len(train_ds)} train, {len(val_ds)} val | "
          f"feat_dim={feat_dim} | lr={lr_head:.1e} | T={temperature} | PL_w={margin_weight}")

    for epoch in range(cfg.train.epochs):
        model.train()
        total_loss = 0.0

        for feats, scores, vlens in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            feats, scores, vlens = feats.to(device), scores.to(device), vlens.to(device)
            preds = model(feats, vlens)

            gs = gold_set_loss(preds, scores, vlens)
            kl = kl_div_loss(preds, scores, vlens, temperature=temperature)
            pl = plackett_luce_loss(preds, scores, vlens, temperature=temperature)
            loss = gs + kl + margin_weight * pl

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        gold_acc = validate(model, val_loader, device)
        lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | GoldAcc: {gold_acc:.2%} | LR: {lr:.2e}")

        if gold_acc > best_metric:
            best_metric = gold_acc
            patience_counter = 0
            torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 'feat_dim': feat_dim},
                       f"{cfg.train.save_dir}/best_binary_head.pth")
            print(f"  -> New best GoldAcc: {gold_acc:.2%}")
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping. Best GoldAcc: {best_metric:.2%}")
            break

    # Assemble full model for inference
    print("Assembling full model...")
    full_model = MobileCLIPRanker(cfg)
    head_ckpt = torch.load(f"{cfg.train.save_dir}/best_binary_head.pth", weights_only=True)
    full_model.head.load_state_dict(head_ckpt['model_state_dict'])
    torch.save({'epoch': head_ckpt['epoch'], 'model_state_dict': full_model.state_dict()},
               f"{cfg.train.save_dir}/best_model.pth")
    print(f"Saved to {cfg.train.save_dir}/best_model.pth")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-cache', action='store_true')
    args = parser.parse_args()

    cfg = load_config("config.yml")
    cache_dir = "cached_features_binary"
    device = torch.device(cfg.system.device if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(BINARY_CSV)
    df = df.rename(columns={'property_id': 'group_id', 'image_url': 'url'})
    df['file_path'] = df.index.map(lambda x: os.path.join('images', f"{x}.jpg"))
    print(f"Binary data: {len(df)} images, {df['group_id'].nunique()} groups")

    if not args.skip_cache:
        precompute_features(df, cfg, device, cache_dir)
    train_head(df, cfg, device, cache_dir)


if __name__ == "__main__":
    main()
