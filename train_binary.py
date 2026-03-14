"""End-to-end binary best-image trainer with Focal BCE + Gold-Set loss.

Mirrors the proven train_ddp.py architecture (backbone fine-tuning + attention
head) but with losses adapted for binary labels (selected=1 vs 0).

Usage:
    python train_binary.py
"""
import os
import argparse
import random
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
from model import MobileCLIPRanker, get_norm_stats
from utils import load_config

BINARY_CSV = "best_image_training_data.csv"
GOLD_THRESHOLD = 7.0  # selected=10 is gold

# Inference compatibility: same architecture as MobileCLIPRanker
BinaryClassifier = MobileCLIPRanker


# ── Dataset ───────────────────────────────────────────────────────────

class BinaryGroupDataset(Dataset):
    """Groups images by property_id, maps selected=1→10.0, selected=0→0.0.
    Pads to max_per_group. Filters: ≥2 images AND at least 1 positive."""

    def __init__(self, df, images_dir="images", is_train=False, img_size=224,
                 max_per_group=15, norm_stats=None):
        self.img_size = img_size
        self.max_per_group = max_per_group
        mean, std = norm_stats or ((0.481, 0.457, 0.408), (0.268, 0.261, 0.275))

        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0),
                                             interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])

        # Build groups
        self.groups = []
        for _, group in df.groupby('group_id'):
            items = []
            has_pos = False
            for _, row in group.iterrows():
                fp = row['file_path']
                if not os.path.exists(fp):
                    continue
                score = 10.0 if row['selected'] == 1 else 0.0
                if row['selected'] == 1:
                    has_pos = True
                items.append((fp, score))
            if len(items) >= 2 and has_pos:
                self.groups.append(items[:max_per_group])

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        items = self.groups[idx]
        tensors, scores = [], []
        for path, score in items:
            tensors.append(self._load(path))
            scores.append(score)

        vlen = len(tensors)
        pad = self.max_per_group - vlen
        if pad > 0:
            tensors += [torch.zeros(3, self.img_size, self.img_size)] * pad
            scores += [-100.0] * pad

        return (torch.stack(tensors),
                torch.tensor(scores, dtype=torch.float32),
                torch.tensor(vlen, dtype=torch.long))

    def _load(self, path):
        try:
            with Image.open(path) as img:
                return self.transform(img.convert('RGB'))
        except Exception:
            return torch.zeros(3, self.img_size, self.img_size)


# ── Losses ────────────────────────────────────────────────────────────

def gold_set_loss(pred_scores, gt_scores, valid_len):
    """P(argmax ∈ gold) loss: logsumexp(all) - logsumexp(gold).
    Directly optimizes gold accuracy metric."""
    loss = 0.0
    count = 0
    for b in range(pred_scores.shape[0]):
        n = int(valid_len[b].item())
        if n < 2:
            continue
        logits = pred_scores[b, :n].view(-1)
        gts = gt_scores[b, :n]
        gold_mask = gts >= GOLD_THRESHOLD
        if gold_mask.any() and not gold_mask.all():
            loss += torch.logsumexp(logits, dim=0) - torch.logsumexp(logits[gold_mask], dim=0)
            count += 1
    return loss / count if count > 0 else pred_scores.sum() * 0.0


def focal_bce_loss(pred_scores, gt_scores, valid_len, alpha=0.75, gamma=2.0):
    """Per-image focal BCE. Handles extreme class imbalance (~90% neg, ~10% pos).
    Applied independently per image (flattened from groups)."""
    all_logits = []
    all_targets = []
    for b in range(pred_scores.shape[0]):
        n = int(valid_len[b].item())
        if n < 2:
            continue
        logits = pred_scores[b, :n].view(-1)
        gts = gt_scores[b, :n]
        # Binary target: gold (score >= GOLD_THRESHOLD) = 1, else 0
        targets = (gts >= GOLD_THRESHOLD).float()
        all_logits.append(logits)
        all_targets.append(targets)

    if not all_logits:
        return pred_scores.sum() * 0.0

    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)

    # Standard BCE
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

    # Focal modulation
    probs = torch.sigmoid(logits)
    p_t = targets * probs + (1 - targets) * (1 - probs)
    focal_weight = (1 - p_t) ** gamma

    # Alpha weighting: alpha for positives, (1-alpha) for negatives
    alpha_t = targets * alpha + (1 - targets) * (1 - alpha)

    return (alpha_t * focal_weight * bce).mean()


# ── Validation ────────────────────────────────────────────────────────

def validate(model, val_loader, device):
    """GoldAcc: does argmax(pred) land on a selected=1 image?"""
    model.eval()
    hits = total = 0
    with torch.no_grad():
        for images, scores, vlens in val_loader:
            images, scores, vlens = images.to(device), scores.to(device), vlens.to(device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                preds = model(images, valid_lens=vlens)
            for b in range(preds.shape[0]):
                n = int(vlens[b].item())
                if n < 2:
                    continue
                gt = scores[b, :n]
                if (gt >= GOLD_THRESHOLD).any():
                    top1 = preds[b, :n].view(-1).argmax().item()
                    hits += int(gt[top1] >= GOLD_THRESHOLD)
                    total += 1
    return hits / total if total else 0.0


# ── Training ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    cfg = load_config("config.yml")

    seed = getattr(cfg.train, 'seed', 42)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(cfg.system.device if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.train.save_dir, exist_ok=True)

    # Helper to read binary config with fallback to train config
    def bcfg(key, default=None):
        val = getattr(getattr(cfg, 'binary', None), key, None)
        if val is not None:
            return val
        return getattr(cfg.train, key, default)

    lr_head = bcfg('lr_head', 3e-4)
    lr_backbone = bcfg('lr_backbone', 5e-7)
    accum_steps = bcfg('gradient_accumulation_steps', 32)
    focal_alpha = bcfg('focal_alpha', 0.75)
    focal_gamma = bcfg('focal_gamma', 2.0)
    focal_weight = bcfg('focal_weight', 0.5)
    warmup_epochs = getattr(cfg.train, 'warmup_epochs', 1)

    # Load data
    df = pd.read_csv(BINARY_CSV)
    df = df.rename(columns={'property_id': 'group_id', 'image_url': 'url'})
    df['file_path'] = df.index.map(lambda x: os.path.join('images', f"{x}.jpg"))
    print(f"Binary data: {len(df)} images, {df['group_id'].nunique()} groups")

    # Train/val split by group
    rng = np.random.RandomState(seed)
    all_groups = df['group_id'].unique().tolist()
    rng.shuffle(all_groups)
    val_groups = set(all_groups[:int(len(all_groups) * 0.1)])

    norm_stats = get_norm_stats(cfg.model.name)
    max_pg = getattr(cfg.data, 'max_images_per_group', 15)

    train_ds = BinaryGroupDataset(
        df[~df['group_id'].isin(val_groups)], images_dir="images",
        is_train=True, img_size=cfg.data.img_size,
        max_per_group=max_pg, norm_stats=norm_stats)
    val_ds = BinaryGroupDataset(
        df[df['group_id'].isin(val_groups)], images_dir="images",
        is_train=False, img_size=cfg.data.img_size,
        max_per_group=max_pg, norm_stats=norm_stats)

    def worker_init_fn(worker_id):
        np.random.seed(seed + worker_id)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=cfg.system.num_workers, pin_memory=cfg.system.pin_memory,
        worker_init_fn=worker_init_fn)
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=cfg.system.num_workers, pin_memory=cfg.system.pin_memory)

    # Model: full MobileCLIPRanker (backbone + attention head)
    model = MobileCLIPRanker(cfg).to(device)

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        print(f"Resumed from {args.resume} (epoch {start_epoch})")

    # Differential LR: backbone vs head
    raw_model = model
    backbone_params = []
    head_params = []
    for name, param in raw_model.named_parameters():
        if not param.requires_grad:
            continue
        if "head" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    print(f"Trainable params: backbone={len(backbone_params)}, head={len(head_params)}")

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': lr_backbone},
        {'params': head_params, 'lr': lr_head},
    ], weight_decay=cfg.train.weight_decay)

    # Warmup → cosine annealing
    warmup_sched = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0 / max(warmup_epochs, 1), total_iters=warmup_epochs)
    cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.train.epochs - warmup_epochs, 1), eta_min=1e-7)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_sched, cosine_sched], milestones=[warmup_epochs])

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    patience = getattr(cfg.train, 'patience', 20)
    patience_counter = 0
    best_acc = 0.0

    print(f"Training: {len(train_ds)} train, {len(val_ds)} val | "
          f"GoldSet + FocalBCE(α={focal_alpha}, γ={focal_gamma}, w={focal_weight}) | "
          f"lr_head={lr_head:.1e}, lr_backbone={lr_backbone:.1e} | "
          f"accum={accum_steps} | AMP={use_amp}")

    for epoch in range(start_epoch, cfg.train.epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for step, (images, scores, vlens) in enumerate(iterator):
            images, scores, vlens = images.to(device), scores.to(device), vlens.to(device)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    preds = model(images, valid_lens=vlens)
                    gs = gold_set_loss(preds, scores, vlens)
                    fb = focal_bce_loss(preds, scores, vlens,
                                        alpha=focal_alpha, gamma=focal_gamma)
                    loss = gs + focal_weight * fb
                    loss = loss / accum_steps

                scaler.scale(loss).backward()

                if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                preds = model(images, valid_lens=vlens)
                gs = gold_set_loss(preds, scores, vlens)
                fb = focal_bce_loss(preds, scores, vlens,
                                    alpha=focal_alpha, gamma=focal_gamma)
                loss = gs + focal_weight * fb
                loss = loss / accum_steps

                loss.backward()

                if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * accum_steps

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        gold_acc = validate(model, val_loader, device)
        current_lr_bb = optimizer.param_groups[0]['lr']
        current_lr_hd = optimizer.param_groups[1]['lr']

        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | GoldAcc: {gold_acc:.2%} | "
              f"LR: bb={current_lr_bb:.2e} hd={current_lr_hd:.2e}")

        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }

        if gold_acc > best_acc:
            best_acc = gold_acc
            patience_counter = 0
            torch.save(checkpoint, f"{cfg.train.save_dir}/best_model.pth")
            print(f"  -> New best GoldAcc: {gold_acc:.2%}")
        else:
            patience_counter += 1

        torch.save(checkpoint, f"{cfg.train.save_dir}/last.pth")

        if patience_counter >= patience:
            print(f"Early stopping. Best GoldAcc: {best_acc:.2%}")
            break

    print(f"Done. Best GoldAcc: {best_acc:.2%}")


if __name__ == "__main__":
    main()
