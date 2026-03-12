import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from model import _build_backbone, _find_unfrozen_modules, get_norm_stats
from utils import load_config

BINARY_CSV = "best_image_training_data.csv"


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


class BinaryImageDataset(Dataset):
    """Pointwise binary dataset: each sample is one image with label 0 or 1."""

    def __init__(self, df, images_dir="images", is_train=False, img_size=224, norm_stats=None):
        self.img_size = img_size
        self.df = df.copy()

        if 'file_path' not in self.df.columns:
            self.df['file_path'] = self.df.index.map(lambda x: os.path.join(images_dir, f"{x}.jpg"))
        self.df = self.df[self.df['file_path'].apply(os.path.exists)].reset_index(drop=True)

        # Use 'selected' column directly as label
        self.df['label'] = self.df['selected'].astype(int)

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

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            with Image.open(row['file_path']) as img:
                tensor = self.transform(img.convert('RGB'))
        except Exception:
            tensor = torch.zeros(3, self.img_size, self.img_size)

        return tensor, torch.tensor(row['label'], dtype=torch.float32)


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


def validate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    tp = fp = fn = tn = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = (logits > 0).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()

    acc = correct / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return acc, precision, recall, f1


def validate_group_ranking(model, df_val, cfg, device, norm_stats=None):
    """Gold accuracy: within each group, does the model's top-1 have selected=1?"""
    raw_model = model.module if hasattr(model, 'module') else model
    raw_model.eval()
    mean, std = norm_stats or ((0.481, 0.457, 0.408), (0.268, 0.261, 0.275))
    val_transform = transforms.Compose([
        transforms.Resize(cfg.data.img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(cfg.data.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    if 'file_path' not in df_val.columns:
        df_val = df_val.copy()
        df_val['file_path'] = df_val.index.map(lambda x: f"images/{x}.jpg")

    top1_hits = 0
    total = 0

    with torch.no_grad():
        for _, group in df_val.groupby('group_id'):
            if len(group) < 2:
                continue
            if not (group['selected'] == 1).any():
                continue

            images = []
            gt_labels = []
            for _, row in group.iterrows():
                if not os.path.exists(row['file_path']):
                    continue
                try:
                    with Image.open(row['file_path']) as img:
                        images.append(val_transform(img.convert('RGB')))
                    gt_labels.append(int(row['selected']))
                except Exception:
                    continue

            if len(images) < 2:
                continue

            batch = torch.stack(images).to(device)
            logits = raw_model(batch)
            top1_idx = logits.argmax().item()
            top1_hits += gt_labels[top1_idx]
            total += 1

    return top1_hits / total if total > 0 else 0.0


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
    warmup_epochs = getattr(cfg.train, 'warmup_epochs', 0)

    # Read best_image_training_data.csv directly
    df = pd.read_csv(BINARY_CSV)

    # Normalize columns: property_id -> group_id, image_url -> url
    df = df.rename(columns={'property_id': 'group_id', 'image_url': 'url'})
    df['file_path'] = df.index.map(lambda x: os.path.join('images', f"{x}.jpg"))

    n_pos = (df['selected'] == 1).sum()
    n_neg = (df['selected'] == 0).sum()

    # Train/val split by group
    rng = np.random.RandomState(seed)
    all_groups = df['group_id'].unique().tolist()
    rng.shuffle(all_groups)
    val_size = int(len(all_groups) * 0.1)
    val_groups = set(all_groups[:val_size])

    train_df = df[~df['group_id'].isin(val_groups)].copy()
    val_df = df[df['group_id'].isin(val_groups)].copy()

    norm_stats = get_norm_stats(cfg.model.name)
    train_ds = BinaryImageDataset(train_df, is_train=True, img_size=cfg.data.img_size, norm_stats=norm_stats)
    val_ds = BinaryImageDataset(val_df, is_train=False, img_size=cfg.data.img_size, norm_stats=norm_stats)

    # DDP: DistributedSampler, else WeightedRandomSampler for class balance
    if dist.is_initialized():
        sampler = DistributedSampler(train_ds, shuffle=True, seed=seed)
    else:
        labels = train_ds.df['label'].values
        class_counts = np.bincount(labels)
        weights = 1.0 / class_counts[labels]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    def worker_init_fn(worker_id):
        np.random.seed(seed + worker_id)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size * 8,
        sampler=sampler,
        num_workers=cfg.system.num_workers,
        pin_memory=cfg.system.pin_memory,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size * 8,
        shuffle=False,
        num_workers=cfg.system.num_workers,
        pin_memory=cfg.system.pin_memory,
    )

    device = torch.device(f"cuda:{local_rank}")
    model = BinaryClassifier(cfg).to(device)

    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # Differential LR
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
        {'params': head_params, 'lr': cfg.train.lr_head},
    ], weight_decay=cfg.train.weight_decay)

    warmup_sched = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0 / max(warmup_epochs, 1), total_iters=max(warmup_epochs, 1)
    )
    cosine_sched = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(cfg.train.epochs - warmup_epochs, 1), eta_min=1e-7
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_sched, cosine_sched], milestones=[max(warmup_epochs, 1)]
    )

    # BCE with pos_weight to handle class imbalance
    pos_weight = torch.tensor([n_neg / n_pos]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_metric = 0.0
    patience = getattr(cfg.train, 'patience', 20)
    patience_counter = 0

    if rank == 0:
        print(f"Binary data: {len(df)} images ({n_pos} selected, {n_neg} not selected) "
              f"from {df['group_id'].nunique()} groups")
        print(f"Training binary classifier | {len(train_ds)} train, {len(val_ds)} val | "
              f"pos_weight={pos_weight.item():.2f} | AMP={use_amp}")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        raw_model = model.module if hasattr(model, "module") else model
        raw_model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if rank == 0:
            print(f"Resumed from {args.resume}")

    for epoch in range(cfg.train.epochs):
        model.train()
        total_loss = 0.0

        if hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(epoch)

        optimizer.zero_grad(set_to_none=True)
        iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}") if rank == 0 else train_loader

        for step, (images, labels) in enumerate(iterator):
            images, labels = images.to(device), labels.to(device)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(images)
                    loss = criterion(logits, labels) / accum_steps

                scaler.scale(loss).backward()

                if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                logits = model(images)
                loss = criterion(logits, labels) / accum_steps
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
            acc, prec, rec, f1 = validate(raw_val, val_loader, device)
            gold_acc = validate_group_ranking(model, val_df, cfg, device, norm_stats=norm_stats)
            lr = optimizer.param_groups[0]['lr']

            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Acc: {acc:.2%} | "
                  f"P: {prec:.2%} R: {rec:.2%} F1: {f1:.2%} | GoldAcc: {gold_acc:.2%} | LR: {lr:.2e}")

            if gold_acc > best_metric:
                best_metric = gold_acc
                patience_counter = 0
                save_checkpoint(model, optimizer, epoch + 1, f"{cfg.train.save_dir}/last.pth", is_best=True)
                print(f"  -> New best GoldAcc: {gold_acc:.2%}")
            else:
                patience_counter += 1
                save_checkpoint(model, optimizer, epoch + 1, f"{cfg.train.save_dir}/last.pth", is_best=False)

            if patience_counter >= patience:
                print(f"Early stopping. Best GoldAcc: {best_metric:.2%}")
                break

    cleanup_ddp()


if __name__ == "__main__":
    main()
