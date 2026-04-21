"""Regression training: per-image Huber on raw absolute scores.

Unlike train_ddp.py (listwise KL within groups), this produces a *global* score
scale — score 9 means the same thing across every property. Required for
"pick the best image to stage across the whole dataset" use case.

Oversampling: rare high-score groups are physically replicated based on decile
frequency, which balances training without needing a weighted sampler (works
transparently under DDP).

Primary validation metric: mean ground-truth score of the model's top-1%
predictions — a direct measure of "how good are the images we'd actually stage".
"""

import os
import argparse
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
from dataset import PropertyPreferenceDataset, CachedFeatureDataset, _remap_score
from model import MobileCLIPRanker
from utils import load_config
from train_ddp import EMA, save_checkpoint, setup_ddp, cleanup_ddp


PAD_SCORE = -100.0


def huber_loss(preds, targets, valid_lens, delta=1.0):
    """Per-image Huber regression on absolute scores. Padded positions are masked."""
    preds = preds.view(preds.shape[0], preds.shape[1])
    targets = targets.view(targets.shape[0], targets.shape[1])
    B, G = preds.shape

    mask = torch.arange(G, device=preds.device).unsqueeze(0) < valid_lens.to(preds.device).unsqueeze(1)
    diff = preds - targets
    abs_diff = diff.abs()
    quadratic = 0.5 * diff.pow(2)
    linear = delta * (abs_diff - 0.5 * delta)
    per_elem = torch.where(abs_diff <= delta, quadratic, linear)
    per_elem = per_elem * mask.float()

    denom = mask.float().sum().clamp(min=1.0)
    return per_elem.sum() / denom


def oversample_groups_by_decile(groups, n_bins=10, cap_ratio=8.0):
    """Replicate group references so rare-score bins appear more often.

    Bin each group by its max score (the thing we care about landing in top-1%),
    then replicate so every bin reaches ~equal total count. cap_ratio bounds
    per-group replication so the dataset can't balloon pathologically.
    """
    if not groups:
        return groups

    max_scores = np.array([max(float(r['score']) for r in g) for g in groups])
    # Quantile edges; bump the upper edge so the max value lands in the last bin.
    edges = np.quantile(max_scores, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-6
    bins = np.digitize(max_scores, edges[1:-1])  # 0..n_bins-1
    bin_counts = np.bincount(bins, minlength=n_bins)
    target = bin_counts.max()

    out = []
    for g, b in zip(groups, bins):
        copies = max(1, int(round(target / max(bin_counts[b], 1))))
        copies = min(copies, int(cap_ratio))
        out.extend([g] * copies)
    return out


def _collect_val_preds(model, df_val, cfg, device, use_cached, cache_dir):
    """Run model across all val images; return flat (preds, gts) arrays."""
    if use_cached:
        return _collect_val_preds_cached(model, df_val, device, cache_dir)

    ds = PropertyPreferenceDataset(
        pd.DataFrame({'group_id': [], 'score': [], 'label': []}),
        images_dir="images", is_train=False, img_size=cfg.data.img_size
    )
    if 'file_path' not in df_val.columns:
        df_val = df_val.copy()
        df_val['file_path'] = df_val.index.map(lambda x: f"images/{x}.jpg")

    preds_all, gts_all, group_ids = [], [], []
    grouped = df_val.groupby('group_id')

    with torch.no_grad():
        for gid, group in grouped:
            images, scores = [], []
            for _, row in group.iterrows():
                if not os.path.exists(row['file_path']):
                    continue
                images.append(ds._process(row['file_path']))
                scores.append(_remap_score(float(row['score']), row.get('label', '')))
            if len(images) < 2:
                continue
            batch = torch.stack(images).unsqueeze(0).to(device)
            vlen = torch.tensor([len(images)])
            preds = model(batch, valid_lens=vlen).view(-1).cpu().numpy()
            preds_all.extend(preds.tolist())
            gts_all.extend(scores)
            group_ids.extend([gid] * len(scores))

    return np.array(preds_all), np.array(gts_all), np.array(group_ids)


def _collect_val_preds_cached(model, df_val, device, cache_dir):
    if 'file_path' not in df_val.columns:
        df_val = df_val.copy()
        df_val['file_path'] = df_val.index.map(lambda x: f"images/{x}.jpg")

    preds_all, gts_all, group_ids = [], [], []
    grouped = df_val.groupby('group_id')

    with torch.no_grad():
        for gid, group in grouped:
            features, scores = [], []
            for _, row in group.iterrows():
                basename = os.path.splitext(os.path.basename(row['file_path']))[0]
                cache_path = os.path.join(cache_dir, f"{basename}.pt")
                if not os.path.exists(cache_path):
                    continue
                features.append(torch.load(cache_path, weights_only=True))
                scores.append(_remap_score(float(row['score']), row.get('label', '')))
            if len(features) < 2:
                continue
            batch = torch.stack(features).unsqueeze(0).to(device)
            preds = model.head(batch).view(-1).cpu().numpy()
            preds_all.extend(preds.tolist())
            gts_all.extend(scores)
            group_ids.extend([gid] * len(scores))

    return np.array(preds_all), np.array(gts_all), np.array(group_ids)


def validate_regression(model, df_val, cfg, device, use_cached=False, cache_dir="cached_features"):
    """Compute regression-oriented metrics across the full val set.

    Primary: top1_acc — per-group, did the model's highest-scoring image match
             the ground-truth highest-scoring image? Mirrors evaluate.py's metric.
             Ties at GT max all count as correct.
    Supporting: mean_gt_top1pct, precision@top-1%, Spearman, MAE, gold_acc.
    """
    from scipy.stats import spearmanr
    model.eval()

    preds, gts, group_ids = _collect_val_preds(model, df_val, cfg, device, use_cached, cache_dir)
    n = len(preds)
    if n == 0:
        return {'top1_acc': 0.0, 'mean_gt_top1pct': 0.0, 'precision_1pct': 0.0,
                'spearman': 0.0, 'mae': 0.0, 'gold_acc': 0.0}

    # Per-group top-1: did the model's argmax land on a GT-max image?
    top1_hits, top1_total = 0, 0
    gold_hits, gold_total = 0, 0
    for gid in np.unique(group_ids):
        m = group_ids == gid
        g_preds = preds[m]
        g_gts = gts[m]
        if len(g_gts) < 2:
            continue
        gt_max = g_gts.max()
        best_mask = g_gts == gt_max
        top1_hits += int(best_mask[int(np.argmax(g_preds))])
        top1_total += 1
        if gt_max >= 7:
            gold_hits += int(g_gts[int(np.argmax(g_preds))] >= 7)
            gold_total += 1

    top1_acc = top1_hits / top1_total if top1_total > 0 else 0.0
    gold_acc = gold_hits / gold_total if gold_total > 0 else 0.0

    # Global top-1% — kept for cross-property insight but no longer primary.
    k = max(1, n // 100)
    pred_top_idx = np.argsort(-preds)[:k]
    gt_top_set = set(np.argsort(-gts)[:k].tolist())
    mean_gt_top1pct = float(gts[pred_top_idx].mean())
    precision_1pct = len(set(pred_top_idx.tolist()) & gt_top_set) / k

    if len(np.unique(gts)) > 1:
        rho, _ = spearmanr(preds, gts)
        spearman = float(rho) if not np.isnan(rho) else 0.0
    else:
        spearman = 0.0
    mae = float(np.abs(preds - gts).mean())

    return {
        'top1_acc': top1_acc,
        'mean_gt_top1pct': mean_gt_top1pct,
        'precision_1pct': precision_1pct,
        'spearman': spearman,
        'mae': mae,
        'gold_acc': gold_acc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--cached', action='store_true',
                        help='Use precomputed features (run precompute_features.py first)')
    parser.add_argument('--no-oversample', action='store_true',
                        help='Disable decile oversampling (for ablation)')
    args = parser.parse_args()

    rank, local_rank = setup_ddp()
    cfg = load_config("config.yml")

    import random
    seed = getattr(cfg.train, 'seed', 42)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs(cfg.train.save_dir, exist_ok=True)

    accum_steps = getattr(cfg.train, 'gradient_accumulation_steps', 4)
    warmup_epochs = getattr(cfg.train, 'warmup_epochs', 1)
    use_cached = args.cached or getattr(cfg.train, 'use_cached_features', False)
    cache_dir = getattr(cfg.data, 'cached_features_dir', 'cached_features')
    huber_delta = getattr(cfg.train, 'huber_delta', 1.0)
    oversample_bins = getattr(cfg.train, 'oversample_bins', 10)
    oversample_cap = getattr(cfg.train, 'oversample_cap', 8.0)

    df = pd.read_csv(cfg.data.csv_path)
    if 'file_path' not in df.columns:
        df['file_path'] = df.index.map(lambda x: os.path.join('images', f"{x}.jpg"))

    unique_groups = df['group_id'].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_groups)
    val_groups = unique_groups[:int(len(unique_groups) * 0.1)]
    train_df = df[~df['group_id'].isin(val_groups)].copy()
    val_df = df[df['group_id'].isin(val_groups)].copy()

    if use_cached:
        if not os.path.isdir(cache_dir):
            print(f"ERROR: Cache dir '{cache_dir}' not found. Run precompute_features.py first.")
            return
        train_ds = CachedFeatureDataset(train_df, cache_dir=cache_dir)
        if rank == 0:
            print(f"Using cached features from {cache_dir}/")
    else:
        train_ds = PropertyPreferenceDataset(train_df, images_dir="images", is_train=True,
                                             img_size=cfg.data.img_size)

    n_raw_groups = len(train_ds.groups)
    if not args.no_oversample:
        train_ds.groups = oversample_groups_by_decile(
            train_ds.groups, n_bins=oversample_bins, cap_ratio=oversample_cap
        )
    n_eff_groups = len(train_ds.groups)

    sampler = DistributedSampler(train_ds, shuffle=True, seed=seed) if dist.is_initialized() else None

    def worker_init_fn(worker_id):
        np.random.seed(seed + worker_id)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, sampler=sampler,
        shuffle=(sampler is None),
        num_workers=cfg.system.num_workers if not use_cached else 2,
        pin_memory=cfg.system.pin_memory,
        worker_init_fn=worker_init_fn,
    )

    device = torch.device(f"cuda:{local_rank}")
    model = MobileCLIPRanker(cfg).to(device)

    start_epoch = 0
    if args.resume:
        if rank == 0:
            print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt.get('epoch', 0)

    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    raw_model = model.module if hasattr(model, "module") else model
    backbone_params, head_params = [], []
    for name, param in raw_model.named_parameters():
        if not param.requires_grad:
            continue
        (head_params if "head" in name else backbone_params).append(param)

    param_groups = [{'params': head_params, 'lr': cfg.train.lr_head}]
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': cfg.train.lr_backbone})
    optimizer = optim.AdamW(param_groups, weight_decay=cfg.train.weight_decay)

    eta_min = getattr(cfg.train, 'eta_min', 1e-7)
    warmup_sched = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0 / max(warmup_epochs, 1), total_iters=warmup_epochs
    )
    cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.train.epochs - warmup_epochs, 1), eta_min=eta_min
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_sched, cosine_sched], milestones=[warmup_epochs]
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    ema_decay = getattr(cfg.train, 'ema_decay', 0.998)
    ema = EMA(model, decay=ema_decay)

    best_metric = -float('inf')
    patience = getattr(cfg.train, 'patience', 15)
    patience_counter = 0

    if rank == 0:
        print(f"Training on {n_eff_groups} groups (raw={n_raw_groups}, "
              f"oversample={'off' if args.no_oversample else f'{oversample_bins}-bin cap={oversample_cap}x'}) | "
              f"Huber(δ={huber_delta}) | accum={accum_steps} | AMP={use_amp}")
        print(f"Primary metric: top1_acc (per-group: model's #1 pick == a GT-max image)")

    for epoch in range(start_epoch, cfg.train.epochs):
        model.train()
        total_loss = 0.0

        if sampler is not None:
            sampler.set_epoch(epoch)

        iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}") if rank == 0 else train_loader

        optimizer.zero_grad(set_to_none=True)

        for step, (data, scores, vlen) in enumerate(iterator):
            data, scores, vlen = data.to(device), scores.to(device), vlen.to(device)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    preds = model(data, vlen)
                    loss = huber_loss(preds, scores, vlen, delta=huber_delta) / accum_steps
                scaler.scale(loss).backward()

                if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    ema.update(model)
            else:
                preds = model(data, vlen)
                loss = huber_loss(preds, scores, vlen, delta=huber_delta) / accum_steps
                loss.backward()

                if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    ema.update(model)

            total_loss += loss.item() * accum_steps

        scheduler.step()

        if rank == 0:
            avg_loss = total_loss / max(len(train_loader), 1)

            ema.apply(model)
            raw_val = model.module if hasattr(model, 'module') else model
            m = validate_regression(raw_val, val_df, cfg, device,
                                    use_cached=use_cached, cache_dir=cache_dir)

            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | "
                f"Top1Acc: {m['top1_acc']:.2%} | "
                f"GoldAcc: {m['gold_acc']:.2%} | "
                f"MeanGT@top1%: {m['mean_gt_top1pct']:.3f} | "
                f"Spearman: {m['spearman']:.4f} | MAE: {m['mae']:.3f} | "
                f"LR: {current_lr:.2e}"
            )

            primary = m['top1_acc']
            if primary > best_metric:
                best_metric = primary
                patience_counter = 0
                save_checkpoint(model, optimizer, epoch + 1,
                                f"{cfg.train.save_dir}/last.pth", is_best=True)
            else:
                patience_counter += 1
                save_checkpoint(model, optimizer, epoch + 1,
                                f"{cfg.train.save_dir}/last.pth", is_best=False)

            ema.restore(model)

            if patience_counter >= patience:
                print(f"Early stopping. Best Top1Acc: {best_metric:.2%}")

        if dist.is_initialized():
            stop_flag = torch.tensor([1 if (rank == 0 and patience_counter >= patience) else 0],
                                     device=device)
            dist.broadcast(stop_flag, src=0)
            if stop_flag.item() == 1:
                break
        elif patience_counter >= patience:
            break

    cleanup_ddp()


if __name__ == "__main__":
    main()
