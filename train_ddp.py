import os
import argparse
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
from dataset import PropertyPreferenceDataset, CachedFeatureDataset, _remap_score
from model import MobileCLIPRanker
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


def listwise_kl_loss(pred_scores, gt_scores, valid_len, temperature=1.0, target_temperature=2.0):
    """
    KL divergence with score-proportional soft targets.
    Reached 97.27% — the terrain signal works. Needs stable LR to not crash.
    """
    loss = 0.0
    valid_batches = 0

    for b in range(pred_scores.shape[0]):
        n = int(valid_len[b].item())
        if n < 2:
            continue

        logits = pred_scores[b, :n].view(-1)
        gts = gt_scores[b, :n]

        # Score-proportional soft targets
        target_probs = F.softmax(gts / target_temperature, dim=0)

        log_probs = F.log_softmax(logits / temperature, dim=0)
        batch_loss = -torch.sum(target_probs * log_probs)

        loss += batch_loss
        valid_batches += 1

    if valid_batches > 0:
        return loss / valid_batches
    return pred_scores.sum() * 0.0


def pairwise_margin_loss(pred_scores, gt_scores, valid_len, margin=1.0):
    """
    Pairwise margin loss: higher-scored items should outscore lower-scored by margin.
    Complements KL by enforcing minimum score separation between tiers.
    """
    loss = 0.0
    count = 0

    for b in range(pred_scores.shape[0]):
        n = int(valid_len[b].item())
        if n < 2:
            continue

        logits = pred_scores[b, :n].view(-1)
        gts = gt_scores[b, :n]

        # Tier: 2=gold(>=8), 1=silver(>=3), 0=rest
        tiers = torch.zeros_like(gts, dtype=torch.float32)
        tiers[gts >= 3] = 1
        tiers[gts >= 8] = 2

        for i in range(n):
            for j in range(i + 1, n):
                if tiers[i] == tiers[j]:
                    continue
                if tiers[i] > tiers[j]:
                    hi, lo = i, j
                else:
                    hi, lo = j, i
                # hinge: want logits[hi] > logits[lo] + margin
                pair_loss = F.relu(margin - (logits[hi] - logits[lo]))
                loss += pair_loss
                count += 1

    if count > 0:
        return loss / count
    return pred_scores.sum() * 0.0


def compute_ndcg(pred_scores, gt_scores):
    """NDCG: measures full ranking quality, not just top-1 tier."""
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


def validate(model, df_val, cfg, device, use_cached=False, cache_dir="cached_features"):
    model.eval()

    if use_cached:
        return _validate_cached(model, df_val, device, cache_dir)

    ds = PropertyPreferenceDataset(
        pd.DataFrame({'group_id': [], 'score': [], 'label': []}),
        images_dir="images", is_train=False, img_size=cfg.data.img_size
    )
    if 'file_path' not in df_val.columns:
        df_val = df_val.copy()
        df_val['file_path'] = df_val.index.map(lambda x: f"images/{x}.jpg")

    grouped = df_val.groupby('group_id')
    strict_wins = 0
    total_groups = 0
    ndcg_scores = []

    with torch.no_grad():
        for _, group in grouped:
            if len(group) < 2:
                continue

            images, scores = [], []
            for _, row in group.iterrows():
                if not os.path.exists(row['file_path']):
                    continue
                images.append(ds._process(row['file_path']))
                scores.append(_remap_score(float(row['score']), row.get('label', '')))

            if len(images) < 2:
                continue
            batch = torch.stack(images).unsqueeze(0).to(device)
            valid_len = torch.tensor([len(images)])

            preds = model(batch, valid_lens=valid_len).view(-1).cpu().numpy()
            gt_arr = np.array(scores)

            # Strict tier accuracy
            best_idx = np.argmax(preds)
            best_score = scores[best_idx]
            max_possible = max(scores)

            picked_tier = 2 if best_score >= 8 else (1 if best_score >= 3 else 0)
            max_tier = 2 if max_possible >= 8 else (1 if max_possible >= 3 else 0)

            if picked_tier == max_tier:
                strict_wins += 1

            # NDCG — full ranking quality
            ndcg_scores.append(compute_ndcg(preds, gt_arr))

            total_groups += 1

    acc = strict_wins / total_groups if total_groups > 0 else 0.0
    ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
    return acc, ndcg


def _validate_cached(model, df_val, device, cache_dir):
    """Fast validation using precomputed features."""
    if 'file_path' not in df_val.columns:
        df_val = df_val.copy()
        df_val['file_path'] = df_val.index.map(lambda x: f"images/{x}.jpg")

    grouped = df_val.groupby('group_id')
    strict_wins = 0
    total_groups = 0
    ndcg_scores = []

    with torch.no_grad():
        for _, group in grouped:
            if len(group) < 2:
                continue

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
            valid_len = torch.tensor([len(features)])

            preds = model.head(batch).view(-1).cpu().numpy()
            gt_arr = np.array(scores)

            best_idx = np.argmax(preds)
            best_score = scores[best_idx]
            max_possible = max(scores)

            picked_tier = 2 if best_score >= 8 else (1 if best_score >= 3 else 0)
            max_tier = 2 if max_possible >= 8 else (1 if max_possible >= 3 else 0)

            if picked_tier == max_tier:
                strict_wins += 1

            ndcg_scores.append(compute_ndcg(preds, gt_arr))
            total_groups += 1

    acc = strict_wins / total_groups if total_groups > 0 else 0.0
    ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
    return acc, ndcg


def save_checkpoint(model, optimizer, epoch, path, is_best=False):
    raw_model = model.module if hasattr(model, "module") else model
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': raw_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, path)
    if is_best:
        best_path = os.path.join(os.path.dirname(path), "best_model.pth")
        torch.save(checkpoint, best_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--cached', action='store_true', help='Use precomputed features (run precompute_features.py first)')
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

    # Config with defaults for new params
    temperature = getattr(cfg.train, 'temperature', 0.3)
    target_temperature = getattr(cfg.train, 'target_temperature', 2.0)
    margin_weight = getattr(cfg.train, 'margin_weight', 0.5)
    accum_steps = getattr(cfg.train, 'gradient_accumulation_steps', 4)
    warmup_epochs = getattr(cfg.train, 'warmup_epochs', 1)
    use_cached = args.cached or getattr(cfg.train, 'use_cached_features', False)
    cache_dir = getattr(cfg.data, 'cached_features_dir', 'cached_features')

    df = pd.read_csv(cfg.data.csv_path)
    df['score'] = df['score'].clip(lower=0)

    unique_groups = df['group_id'].unique()
    val_groups = unique_groups[:int(len(unique_groups) * 0.1)]
    train_df = df[~df['group_id'].isin(val_groups)]
    val_df = df[df['group_id'].isin(val_groups)]

    if use_cached:
        if not os.path.isdir(cache_dir):
            print(f"ERROR: Cache dir '{cache_dir}' not found. Run precompute_features.py first.")
            return
        train_ds = CachedFeatureDataset(train_df, cache_dir=cache_dir)
        print(f"Using cached features from {cache_dir}/")
    else:
        train_ds = PropertyPreferenceDataset(train_df, images_dir="images", is_train=True, img_size=cfg.data.img_size)

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

    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    raw_model = model.module if hasattr(model, "module") else model
    backbone_params = []
    head_params = []
    for name, param in raw_model.named_parameters():
        if not param.requires_grad:
            continue
        if "head" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

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

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_acc = 0.0
    patience = getattr(cfg.train, 'patience', 15)
    patience_counter = 0

    if rank == 0:
        print(f"Training on {len(train_ds)} groups | KLD(T={temperature},T_tgt={target_temperature})+Margin(w={margin_weight}) | "
              f"accum_steps={accum_steps} | warmup={warmup_epochs} | AMP={use_amp}")

    for epoch in range(cfg.train.epochs):
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
                    kl = listwise_kl_loss(preds, scores, vlen, temperature=temperature, target_temperature=target_temperature)
                    margin = pairwise_margin_loss(preds, scores, vlen)
                    loss = kl + margin_weight * margin
                    loss = loss / accum_steps

                scaler.scale(loss).backward()

                if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                preds = model(data, vlen)
                kl = listwise_kl_loss(preds, scores, vlen, temperature=temperature, target_temperature=target_temperature)
                margin = pairwise_margin_loss(preds, scores, vlen)
                loss = kl + margin_weight * margin
                loss = loss / accum_steps

                loss.backward()

                if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * accum_steps  # undo scaling for logging

        scheduler.step()

        if rank == 0:
            avg_loss = total_loss / len(train_loader)
            raw_val = model.module if hasattr(model, 'module') else model
            acc, ndcg = validate(raw_val, val_df, cfg, device, use_cached=use_cached, cache_dir=cache_dir)

            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Acc: {acc:.2%} | NDCG: {ndcg:.4f} | LR: {current_lr:.2e}")

            if acc > best_acc:
                best_acc = acc
                patience_counter = 0
                save_checkpoint(model, optimizer, epoch + 1, f"{cfg.train.save_dir}/last.pth", is_best=True)
            else:
                patience_counter += 1
                save_checkpoint(model, optimizer, epoch + 1, f"{cfg.train.save_dir}/last.pth", is_best=False)

            if patience_counter >= patience:
                print(f"Early stopping. Best Acc: {best_acc:.2%}")
                break

    cleanup_ddp()


if __name__ == "__main__":
    main()
