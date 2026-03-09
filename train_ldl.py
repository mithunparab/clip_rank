"""
Label Distribution Learning (LDL) training for property image ranking.

Each image predicts a probability distribution over 21 score values (-10 to 10).
Targets are Gaussian distributions centered at the true score, with tighter
sigma for gold-tier images to enforce precise hero ordering.
Loss = KL divergence between predicted and target distributions.
Ranking score = expected value = sum(score_k * P(score_k)).
"""

import os
import argparse
import random
import torch
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

from dataset_ldl import LDLImageDataset, SCORE_VALUES, NUM_BINS
from model import LDLRanker, get_norm_stats
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


def ldl_loss(logits, target_dist, scores=None, gold_weight=3.0):
    """KL divergence: D_KL(target || predicted), with per-sample gold weighting.

    target_dist: (B, 21) soft Gaussian targets (already normalized)
    logits: (B, 21) raw model output (pre-softmax)
    scores: (B,) raw scores for weighting — gold (>=7) gets upweighted
    """
    log_pred = F.log_softmax(logits, dim=-1)
    # Per-sample KL
    per_sample = F.kl_div(log_pred, target_dist, reduction='none').sum(dim=-1)  # (B,)

    if scores is not None:
        weights = torch.ones_like(per_sample)
        weights[scores >= 7] = gold_weight
        loss = (per_sample * weights).mean()
    else:
        loss = per_sample.mean()
    return loss


def expected_score(logits, score_values):
    """Expected value from predicted distribution."""
    probs = torch.softmax(logits, dim=-1)
    return (probs * score_values).sum(dim=-1)


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


def validate(model, df_val, cfg, device):
    """Group-level validation: score images individually, evaluate ranking per group."""
    from torchvision import transforms
    from PIL import Image

    model.eval()
    norm_mean, norm_std = get_norm_stats(cfg.model.name)
    process = transforms.Compose([
        transforms.Resize(cfg.data.img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(cfg.data.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

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

            images, scores = [], []
            for _, row in group.iterrows():
                if not os.path.exists(row['file_path']):
                    continue
                try:
                    with Image.open(row['file_path']) as img:
                        images.append(process(img.convert('RGB')))
                    scores.append(float(row['score']))
                except Exception:
                    continue

            if len(images) < 2:
                continue

            batch = torch.stack(images).to(device)
            pred_scores = model.score(batch).cpu().numpy()
            gt_arr = np.array(scores)

            if gt_arr.max() >= 7:
                top1_hits += int(gt_arr[np.argmax(pred_scores)] >= 7)
                total_groups += 1

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
        'model_type': 'ldl',
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

    accum_steps = getattr(cfg.train, 'gradient_accumulation_steps', 2)
    warmup_epochs = getattr(cfg.train, 'warmup_epochs', 3)
    sigma = getattr(cfg.train, 'ldl_sigma', 1.0)
    gold_sigma = getattr(cfg.train, 'ldl_gold_sigma', 0.5)

    # Data
    df = pd.read_csv(cfg.data.csv_path)
    if 'file_path' not in df.columns:
        df['file_path'] = df.index.map(lambda x: os.path.join('images', f"{x}.jpg"))

    # Shuffle groups before splitting so gold is evenly distributed
    unique_groups = df['group_id'].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_groups)
    val_groups = set(unique_groups[:int(len(unique_groups) * 0.1)])
    train_df = df[~df['group_id'].isin(val_groups)].copy()
    val_df = df[df['group_id'].isin(val_groups)].copy()

    if rank == 0:
        n_train_gold = (train_df['score'] >= 7).sum()
        n_val_gold_groups = val_df[val_df['score'] >= 7]['group_id'].nunique()
        print(f"Train: {len(train_df)} images ({n_train_gold} gold) | "
              f"Val: {len(val_df)} images ({len(val_groups)} groups, {n_val_gold_groups} with gold)")

    norm_mean, norm_std = get_norm_stats(cfg.model.name)
    train_ds = LDLImageDataset(
        train_df, images_dir="images", is_train=True,
        img_size=cfg.data.img_size, norm_mean=norm_mean, norm_std=norm_std,
        sigma=sigma, gold_sigma=gold_sigma,
    )

    # Weighted sampling: oversample gold images so model sees them more
    gold_weight_sample = getattr(cfg.train, 'gold_weight', 3.0)
    if dist.is_initialized():
        sampler = DistributedSampler(train_ds, shuffle=True, seed=seed)
    else:
        sample_weights = []
        for r in train_ds.records:
            sample_weights.append(gold_weight_sample if r['score'] >= 7 else 1.0)
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True
        )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, sampler=sampler,
        num_workers=cfg.system.num_workers,
        pin_memory=cfg.system.pin_memory,
        drop_last=True,
    )

    device = torch.device(f"cuda:{local_rank}")
    model = LDLRanker(cfg).to(device)

    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

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

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_gold_acc = 0.0
    patience = getattr(cfg.train, 'patience', 20)
    patience_counter = 0

    if rank == 0:
        print(f"LDL training | {NUM_BINS} bins | sigma={sigma} gold_sigma={gold_sigma} | "
              f"batch={cfg.train.batch_size} | accum={accum_steps} | AMP={use_amp}")

    for epoch in range(cfg.train.epochs):
        model.train()
        total_loss = 0.0

        if sampler is not None:
            sampler.set_epoch(epoch)

        iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}") if rank == 0 else train_loader
        optimizer.zero_grad(set_to_none=True)

        for step, (images, target_dist, scores_batch) in enumerate(iterator):
            images = images.to(device)
            target_dist = target_dist.to(device)
            scores_batch = scores_batch.to(device)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(images)
                    loss = ldl_loss(logits, target_dist, scores_batch, gold_weight_sample)
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
                loss = ldl_loss(logits, target_dist, scores_batch, gold_weight_sample)
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
