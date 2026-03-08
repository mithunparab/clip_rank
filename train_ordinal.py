"""
Ordinal regression training (CORAL) for property image ranking.

Completely different from listwise/pairwise ranking:
- Pointwise: each image is an independent sample (15k samples vs 2k groups)
- Ordinal: learns 19 cumulative thresholds, preserving fine-grained ordering
- Gold-weighted: thresholds 7/8/9 get extra loss weight so hero-tier ordering is precise
- Score = sum(sigmoid(logits)) -> continuous, sortable ranking score
"""

import os
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr

from dataset_ordinal import OrdinalImageDataset, THRESHOLDS, NUM_THRESHOLDS
from model import OrdinalRanker, get_norm_stats
from utils import load_config


def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return rank, local_rank
    return 0, 0


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def build_threshold_weights(gold_weight=3.0):
    """Per-threshold loss weights. Gold-tier thresholds (7, 8, 9) get upweighted."""
    weights = torch.ones(NUM_THRESHOLDS)
    for i, t in enumerate(THRESHOLDS):
        if t >= 7:  # thresholds at 7, 8, 9
            weights[i] = gold_weight
    return weights


def coral_loss(logits, ordinal_labels, threshold_weights, device):
    """
    Weighted binary cross-entropy across all thresholds.
    logits: (B, K), ordinal_labels: (B, K), threshold_weights: (K,)
    """
    w = threshold_weights.to(device)
    loss = F.binary_cross_entropy_with_logits(logits, ordinal_labels, reduction='none')  # (B, K)
    loss = (loss * w).mean()
    return loss


def ordinal_score(logits):
    """Convert logits to continuous ranking score."""
    return torch.sigmoid(logits).sum(dim=-1)


def compute_ndcg(pred_scores, gt_scores):
    n = len(pred_scores)
    if n < 2:
        return 1.0
    pred_order = np.argsort(-pred_scores)
    ordered_gt = gt_scores[pred_order]
    discounts = np.log2(np.arange(n) + 2)
    dcg = np.sum((2 ** ordered_gt - 1) / discounts)
    ideal_gt = np.sort(gt_scores)[::-1]
    idcg = np.sum((2 ** ideal_gt - 1) / discounts)
    if idcg == 0:
        return 1.0
    return dcg / idcg


def validate(model, df_val, cfg, device, tta=True):
    """Group-level validation with optional TTA (5 crops + flips = 10 views)."""
    from torchvision import transforms
    from PIL import Image

    model.eval()
    norm_mean, norm_std = get_norm_stats(cfg.model.name)
    img_size = cfg.data.img_size

    # Standard center crop
    center_crop = transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    # TTA: 5 crops (center + 4 corners) x 2 (original + hflip) = 10 views
    def tta_transforms(pil_img):
        views = []
        w, h = pil_img.size
        crop_size = min(w, h)
        # Resize so short side = img_size * 1.14 (standard 5-crop ratio)
        resize_size = int(img_size * 1.143)
        resized = transforms.functional.resize(
            pil_img, resize_size,
            interpolation=transforms.InterpolationMode.BICUBIC
        )
        rw, rh = resized.size

        # 5 crop positions: center, top-left, top-right, bottom-left, bottom-right
        crops = [
            transforms.functional.center_crop(resized, img_size),
            transforms.functional.crop(resized, 0, 0, img_size, img_size),
            transforms.functional.crop(resized, 0, rw - img_size, img_size, img_size),
            transforms.functional.crop(resized, rh - img_size, 0, img_size, img_size),
            transforms.functional.crop(resized, rh - img_size, rw - img_size, img_size, img_size),
        ]

        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std)
        ])

        for crop in crops:
            views.append(to_tensor(crop))
            views.append(to_tensor(transforms.functional.hflip(crop)))
        return views  # 10 tensors

    if 'file_path' not in df_val.columns:
        df_val = df_val.copy()
        df_val['file_path'] = df_val.index.map(lambda x: f"images/{x}.jpg")

    grouped = df_val.groupby('group_id')
    top1_hits = 0
    total_groups = 0
    ndcg_scores = []
    spearman_scores = []

    with torch.no_grad():
        for _, group in grouped:
            if len(group) < 2:
                continue

            all_scores_list, gt_scores = [], []
            for _, row in group.iterrows():
                if not os.path.exists(row['file_path']):
                    continue
                try:
                    with Image.open(row['file_path']) as img:
                        pil = img.convert('RGB')
                        if tta:
                            views = tta_transforms(pil)
                        else:
                            views = [center_crop(pil)]
                        all_scores_list.append(views)
                    gt_scores.append(float(row['score']))
                except Exception:
                    continue

            if len(all_scores_list) < 2:
                continue

            # Score each image (average across TTA views)
            pred_scores_list = []
            for views in all_scores_list:
                batch = torch.stack(views).to(device)
                logits = model(batch)  # (num_views, K)
                # Average ordinal scores across views
                view_scores = ordinal_score(logits)  # (num_views,)
                pred_scores_list.append(view_scores.mean().item())

            pred_scores = np.array(pred_scores_list)
            gt_arr = np.array(gt_scores)

            # Gold accuracy
            if gt_arr.max() >= 7:
                top1_hits += int(gt_arr[np.argmax(pred_scores)] >= 7)
                total_groups += 1

            # Spearman
            if len(np.unique(gt_arr)) > 1:
                rho, _ = spearmanr(pred_scores, gt_arr)
                spearman_scores.append(float(rho) if not np.isnan(rho) else 0.0)

            ndcg_scores.append(compute_ndcg(pred_scores, gt_arr))

    gold_acc = top1_hits / total_groups if total_groups > 0 else 0.0
    spearman_val = np.mean(spearman_scores) if spearman_scores else 0.0
    ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
    return gold_acc, spearman_val, ndcg


def save_checkpoint(model, optimizer, epoch, path, is_best=False):
    raw_model = model.module if hasattr(model, "module") else model
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': raw_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_type': 'ordinal',
    }
    torch.save(checkpoint, path)
    if is_best:
        best_path = os.path.join(os.path.dirname(path), "best_model.pth")
        torch.save(checkpoint, best_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    rank, local_rank = setup_ddp()
    cfg = load_config("config.yml")

    seed = getattr(cfg.train, 'seed', 42)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs(cfg.train.save_dir, exist_ok=True)

    # Config
    gold_weight = getattr(cfg.train, 'gold_weight', 3.0)
    accum_steps = getattr(cfg.train, 'gradient_accumulation_steps', 4)
    warmup_epochs = getattr(cfg.train, 'warmup_epochs', 3)

    # Data
    df = pd.read_csv(cfg.data.csv_path)
    if 'file_path' not in df.columns:
        df['file_path'] = df.index.map(lambda x: os.path.join('images', f"{x}.jpg"))

    # Split by group_id so validation groups are unseen
    unique_groups = df['group_id'].unique()
    val_groups = set(unique_groups[:int(len(unique_groups) * 0.1)])
    train_df = df[~df['group_id'].isin(val_groups)].copy()
    val_df = df[df['group_id'].isin(val_groups)].copy()

    if rank == 0:
        print(f"Train: {len(train_df)} images | Val: {len(val_df)} images ({len(val_groups)} groups)")

    norm_mean, norm_std = get_norm_stats(cfg.model.name)
    train_ds = OrdinalImageDataset(
        train_df, images_dir="images", is_train=True,
        img_size=cfg.data.img_size, norm_mean=norm_mean, norm_std=norm_std
    )

    sampler = DistributedSampler(train_ds, shuffle=True, seed=seed) if dist.is_initialized() else None

    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, sampler=sampler,
        shuffle=(sampler is None),
        num_workers=cfg.system.num_workers,
        pin_memory=cfg.system.pin_memory,
        drop_last=True,
    )

    device = torch.device(f"cuda:{local_rank}")
    model = OrdinalRanker(cfg).to(device)

    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # Optimizer with differential LR
    raw_model = model.module if hasattr(model, "module") else model
    backbone_params = [p for n, p in raw_model.named_parameters() if p.requires_grad and "head" not in n]
    head_params = [p for n, p in raw_model.named_parameters() if p.requires_grad and "head" in n]

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': cfg.train.lr_backbone},
        {'params': head_params, 'lr': cfg.train.lr_head}
    ], weight_decay=cfg.train.weight_decay)

    warmup_sched = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0 / max(warmup_epochs, 1), total_iters=warmup_epochs
    )
    cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.train.epochs - warmup_epochs, 1), eta_min=1e-7
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_sched, cosine_sched], milestones=[warmup_epochs]
    )

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        raw_model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        print(f"Resumed from {args.resume} (epoch {ckpt.get('epoch', '?')})")

    threshold_weights = build_threshold_weights(gold_weight)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_gold_acc = 0.0
    patience = getattr(cfg.train, 'patience', 20)
    patience_counter = 0

    if rank == 0:
        print(f"CORAL ordinal training | {NUM_THRESHOLDS} thresholds | "
              f"gold_weight={gold_weight} | batch={cfg.train.batch_size} | "
              f"accum={accum_steps} | AMP={use_amp}")

    for epoch in range(cfg.train.epochs):
        model.train()
        total_loss = 0.0

        if sampler is not None:
            sampler.set_epoch(epoch)

        iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}") if rank == 0 else train_loader
        optimizer.zero_grad(set_to_none=True)

        for step, (images, ordinal_labels, _scores) in enumerate(iterator):
            images = images.to(device)
            ordinal_labels = ordinal_labels.to(device)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(images)  # (B, K)
                    loss = coral_loss(logits, ordinal_labels, threshold_weights, device)
                    loss = loss / accum_steps

                scaler.scale(loss).backward()

                if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                logits = model(images)
                loss = coral_loss(logits, ordinal_labels, threshold_weights, device)
                loss = loss / accum_steps

                loss.backward()

                if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * accum_steps

        scheduler.step()

        if rank == 0:
            avg_loss = total_loss / len(train_loader)
            raw_val = model.module if hasattr(model, 'module') else model
            gold_acc, spearman, ndcg = validate(raw_val, val_df, cfg, device)

            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | GoldAcc: {gold_acc:.2%} | "
                  f"Spearman: {spearman:.4f} | NDCG: {ndcg:.4f} | LR: {current_lr:.2e}")

            if gold_acc > best_gold_acc:
                best_gold_acc = gold_acc
                patience_counter = 0
                save_checkpoint(model, optimizer, epoch + 1,
                                f"{cfg.train.save_dir}/last.pth", is_best=True)
            else:
                patience_counter += 1
                save_checkpoint(model, optimizer, epoch + 1,
                                f"{cfg.train.save_dir}/last.pth", is_best=False)

            if patience_counter >= patience:
                print(f"Early stopping. Best GoldAcc: {best_gold_acc:.2%}")
                break

    cleanup_ddp()


if __name__ == "__main__":
    main()
