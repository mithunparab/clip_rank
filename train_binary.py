"""End-to-end binary best-image trainer with cross-batch contrastive loss.

Instead of comparing images within a single group (1 pos vs ~12 neg),
pools all positives and negatives across the batch so each positive
competes against ALL negatives — the same trick that makes CLIP work.

Gold-set loss handles within-group ranking.
Cross-batch InfoNCE handles global quality discrimination.

Supports DDP multi-GPU:
    torchrun --nproc_per_node=2 train_binary.py
Single GPU:
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
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
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


def cross_batch_contrastive(pred_scores, gt_scores, valid_len, temperature=0.07,
                            queue=None):
    """InfoNCE pooled across all groups in the batch.

    Each positive competes against ALL negatives in the batch (not just its
    own group). With batch_size=4 and ~13 neg/group, each positive faces
    ~52 negatives instead of ~13. Optional score queue adds historical
    negatives for even larger pools.

    L_i = -log( exp(s_pos_i/τ) / (exp(s_pos_i/τ) + Σ_j exp(s_neg_j/τ)) )
        = CE(logits=[pos_i, neg_1, ..., neg_N], target=0)
    """
    all_pos = []
    all_neg = []
    for b in range(pred_scores.shape[0]):
        n = int(valid_len[b].item())
        if n < 2:
            continue
        scores = pred_scores[b, :n].view(-1)
        gts = gt_scores[b, :n]
        pos_mask = gts >= GOLD_THRESHOLD
        neg_mask = ~pos_mask
        if pos_mask.any():
            all_pos.append(scores[pos_mask])
        if neg_mask.any():
            all_neg.append(scores[neg_mask])

    if not all_pos or not all_neg:
        return pred_scores.sum() * 0.0

    pos = torch.cat(all_pos)  # (P,)
    neg = torch.cat(all_neg)  # (N,)

    # Append historical negatives from queue
    if queue is not None:
        queue_neg = queue.get()
        if queue_neg is not None:
            neg = torch.cat([neg, queue_neg.to(neg.device)])
        queue.push(neg)

    P, N = pos.shape[0], neg.shape[0]

    # InfoNCE: for each positive, classify it against all negatives
    # logits: (P, 1+N) — column 0 is the positive, rest are negatives
    logits = torch.cat([
        (pos / temperature).unsqueeze(1),                     # (P, 1)
        (neg / temperature).unsqueeze(0).expand(P, N),        # (P, N)
    ], dim=1)

    # Target: index 0 is always the positive
    targets = torch.zeros(P, dtype=torch.long, device=pos.device)
    return F.cross_entropy(logits, targets)


class ScoreQueue:
    """Stores recent negative scores for larger contrastive pools.
    Like MoCo's queue but for scalar scores — lightweight."""

    def __init__(self, max_size=2048):
        self.max_size = max_size
        self.buffer = []

    def push(self, neg_scores):
        self.buffer.append(neg_scores.detach())
        total = sum(s.shape[0] for s in self.buffer)
        while total > self.max_size and len(self.buffer) > 1:
            total -= self.buffer[0].shape[0]
            self.buffer.pop(0)

    def get(self):
        if not self.buffer:
            return None
        return torch.cat(self.buffer)


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

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.train.save_dir, exist_ok=True)

    # Helper to read binary config with fallback to train config
    def bcfg(key, default=None):
        val = getattr(getattr(cfg, 'binary', None), key, None)
        if val is not None:
            return val
        return getattr(cfg.train, key, default)

    lr_head = bcfg('lr_head', 1e-3)
    lr_backbone = bcfg('lr_backbone', 5e-6)
    accum_steps = bcfg('gradient_accumulation_steps', 4)
    batch_size = bcfg('batch_size', cfg.train.batch_size)
    contrast_temp = bcfg('contrastive_temperature', 0.07)
    contrast_weight = bcfg('contrastive_weight', 1.0)
    queue_size = bcfg('queue_size', 2048)
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

    sampler = DistributedSampler(train_ds, shuffle=True, seed=seed) if dist.is_initialized() else None

    def worker_init_fn(worker_id):
        np.random.seed(seed + worker_id)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        shuffle=(sampler is None),
        num_workers=cfg.system.num_workers, pin_memory=cfg.system.pin_memory,
        worker_init_fn=worker_init_fn)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=cfg.system.num_workers, pin_memory=cfg.system.pin_memory)

    # Model: full MobileCLIPRanker (backbone + attention head)
    model = MobileCLIPRanker(cfg).to(device)

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        if rank == 0:
            print(f"Resumed from {args.resume} (epoch {start_epoch})")

    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # Differential LR: backbone vs head
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

    if rank == 0:
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
    neg_queue = ScoreQueue(max_size=queue_size)

    if rank == 0:
        print(f"Training: {len(train_ds)} train, {len(val_ds)} val | "
              f"GoldSet + CrossBatch(τ={contrast_temp}, w={contrast_weight}, Q={queue_size}) | "
              f"lr_head={lr_head:.1e}, lr_backbone={lr_backbone:.1e} | "
              f"accum={accum_steps} | AMP={use_amp}")

    for epoch in range(start_epoch, cfg.train.epochs):
        model.train()
        total_loss = 0.0

        if sampler is not None:
            sampler.set_epoch(epoch)

        optimizer.zero_grad(set_to_none=True)

        iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}") if rank == 0 else train_loader
        for step, (images, scores, vlens) in enumerate(iterator):
            images, scores, vlens = images.to(device), scores.to(device), vlens.to(device)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    preds = model(images, valid_lens=vlens)
                    gs = gold_set_loss(preds, scores, vlens)
                    cb = cross_batch_contrastive(preds, scores, vlens,
                                                 temperature=contrast_temp,
                                                 queue=neg_queue)
                    loss = gs + contrast_weight * cb
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

        if rank == 0:
            avg_loss = total_loss / len(train_loader)
            raw_val = model.module if hasattr(model, 'module') else model
            gold_acc = validate(raw_val, val_loader, device)
            current_lr_bb = optimizer.param_groups[0]['lr']
            current_lr_hd = optimizer.param_groups[1]['lr']

            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | GoldAcc: {gold_acc:.2%} | "
                  f"LR: bb={current_lr_bb:.2e} hd={current_lr_hd:.2e}")

            if gold_acc > best_acc:
                best_acc = gold_acc
                patience_counter = 0
                save_checkpoint(model, optimizer, epoch + 1, f"{cfg.train.save_dir}/last.pth", is_best=True)
            else:
                patience_counter += 1
                save_checkpoint(model, optimizer, epoch + 1, f"{cfg.train.save_dir}/last.pth", is_best=False)

            if patience_counter >= patience:
                print(f"Early stopping. Best GoldAcc: {best_acc:.2%}")
                break

    cleanup_ddp()


if __name__ == "__main__":
    main()
