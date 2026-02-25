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

from dataset import PropertyPreferenceDataset
from model import MobileCLIPRanker, get_norm_stats
from utils import load_config
from train_ddp import (
    setup_ddp, cleanup_ddp, listwise_kl_loss, pairwise_margin_loss,
    validate, save_checkpoint,
)


def load_teacher(cfg, distill_cfg, device):
    """Load frozen S4 teacher from checkpoint."""
    original_name = cfg.model.name
    cfg.model.name = distill_cfg.teacher_model_name
    teacher = MobileCLIPRanker(cfg)
    cfg.model.name = original_name

    ckpt = torch.load(distill_cfg.teacher_checkpoint, map_location=device, weights_only=True)
    state = ckpt["model_state_dict"]
    # Strip DDP 'module.' prefix if present
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    teacher.load_state_dict(state)

    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    print(f"Teacher loaded: {distill_cfg.teacher_model_name} "
          f"(backbone_dim={teacher.backbone_dim}, params frozen)")
    return teacher


def distillation_loss(teacher_scores, student_scores, valid_len, temperature):
    """Per-group KL divergence from teacher soft targets to student predictions."""
    loss = 0.0
    valid_batches = 0

    for b in range(teacher_scores.shape[0]):
        n = int(valid_len[b].item())
        if n < 2:
            continue

        t_logits = teacher_scores[b, :n].view(-1)
        s_logits = student_scores[b, :n].view(-1)

        teacher_probs = F.softmax(t_logits / temperature, dim=0)
        student_log_probs = F.log_softmax(s_logits / temperature, dim=0)

        # KL(teacher || student) scaled by T^2 (Hinton et al.)
        batch_loss = -torch.sum(teacher_probs * student_log_probs) * (temperature ** 2)
        loss += batch_loss
        valid_batches += 1

    if valid_batches > 0:
        return loss / valid_batches
    return student_scores.sum() * 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    rank, local_rank = setup_ddp()
    cfg = load_config("config.yml")
    dcfg = cfg.distill

    seed = getattr(cfg.train, "seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.makedirs(dcfg.save_dir, exist_ok=True)

    # Verify teacher and student use the same normalization
    teacher_norm = get_norm_stats(dcfg.teacher_model_name)
    student_norm = get_norm_stats(dcfg.student_model_name)
    assert teacher_norm == student_norm, (
        f"Normalization mismatch: teacher {teacher_norm} != student {student_norm}. "
        "Distillation requires matching input preprocessing."
    )
    norm_mean, norm_std = student_norm

    # Training config with defaults
    temperature = getattr(cfg.train, "temperature", 0.3)
    target_temperature = getattr(cfg.train, "target_temperature", 2.0)
    margin_weight = getattr(cfg.train, "margin_weight", 0.5)
    accum_steps = getattr(cfg.train, "gradient_accumulation_steps", 4)
    warmup_epochs = getattr(cfg.train, "warmup_epochs", 1)
    alpha = dcfg.alpha
    distill_temp = dcfg.distill_temperature

    # Data
    df = pd.read_csv(cfg.data.csv_path)
    unique_groups = df["group_id"].unique()
    val_groups = unique_groups[: int(len(unique_groups) * 0.1)]
    train_df = df[~df["group_id"].isin(val_groups)]
    val_df = df[df["group_id"].isin(val_groups)]

    train_ds = PropertyPreferenceDataset(
        train_df, images_dir="images", is_train=True, img_size=cfg.data.img_size,
        norm_mean=norm_mean, norm_std=norm_std,
    )

    sampler = DistributedSampler(train_ds, shuffle=True, seed=seed) if dist.is_initialized() else None

    def worker_init_fn(worker_id):
        np.random.seed(seed + worker_id)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, sampler=sampler,
        shuffle=(sampler is None),
        num_workers=cfg.system.num_workers,
        pin_memory=cfg.system.pin_memory,
        worker_init_fn=worker_init_fn,
    )

    device = torch.device(f"cuda:{local_rank}")

    # Load frozen teacher (no DDP — no gradients)
    teacher = load_teacher(cfg, dcfg, device)

    # Build trainable student
    cfg.model.name = dcfg.student_model_name
    student = MobileCLIPRanker(cfg).to(device)
    print(f"Student: {dcfg.student_model_name} (backbone_dim={student.backbone_dim})")

    if dist.is_initialized():
        student = DDP(student, device_ids=[local_rank], find_unused_parameters=True)

    raw_student = student.module if hasattr(student, "module") else student
    backbone_params = []
    head_params = []
    for name, param in raw_student.named_parameters():
        if not param.requires_grad:
            continue
        if "head" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": cfg.train.lr_backbone},
        {"params": head_params, "lr": cfg.train.lr_head},
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
    patience = getattr(cfg.train, "patience", 15)
    patience_counter = 0

    if rank == 0:
        print(
            f"Distilling {dcfg.teacher_model_name} -> {dcfg.student_model_name} | "
            f"alpha={alpha} T={distill_temp} | "
            f"hard: KLD(T={temperature})+Margin(w={margin_weight}) | "
            f"groups={len(train_ds)} | accum={accum_steps} | AMP={use_amp}"
        )

    for epoch in range(cfg.train.epochs):
        student.train()

        if sampler is not None:
            sampler.set_epoch(epoch)

        total_loss = 0.0
        iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}") if rank == 0 else train_loader
        optimizer.zero_grad(set_to_none=True)

        for step, (data, scores, vlen) in enumerate(iterator):
            data, scores, vlen = data.to(device), scores.to(device), vlen.to(device)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    with torch.no_grad():
                        teacher_preds = teacher(data, valid_lens=vlen)
                    student_preds = student(data, vlen)

                    # Distillation loss (soft targets from teacher)
                    d_loss = distillation_loss(teacher_preds, student_preds, vlen, distill_temp)

                    # Hard-label losses (ground-truth)
                    kl_hard = listwise_kl_loss(student_preds, scores, vlen,
                                               temperature=temperature,
                                               target_temperature=target_temperature)
                    margin_hard = pairwise_margin_loss(student_preds, scores, vlen)

                    loss = alpha * d_loss + (1 - alpha) * (kl_hard + margin_weight * margin_hard)
                    loss = loss / accum_steps

                scaler.scale(loss).backward()

                if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.train.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                with torch.no_grad():
                    teacher_preds = teacher(data, valid_lens=vlen)
                student_preds = student(data, vlen)

                d_loss = distillation_loss(teacher_preds, student_preds, vlen, distill_temp)
                kl_hard = listwise_kl_loss(student_preds, scores, vlen,
                                           temperature=temperature,
                                           target_temperature=target_temperature)
                margin_hard = pairwise_margin_loss(student_preds, scores, vlen)

                loss = alpha * d_loss + (1 - alpha) * (kl_hard + margin_weight * margin_hard)
                loss = loss / accum_steps

                loss.backward()

                if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.train.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * accum_steps

        scheduler.step()

        if rank == 0:
            avg_loss = total_loss / len(train_loader)
            raw_val = student.module if hasattr(student, "module") else student
            acc, ndcg = validate(raw_val, val_df, cfg, device,
                                 norm_mean=norm_mean, norm_std=norm_std)

            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Acc: {acc:.2%} | "
                  f"NDCG: {ndcg:.4f} | LR: {current_lr:.2e}")

            if acc > best_acc:
                best_acc = acc
                patience_counter = 0
                save_checkpoint(student, optimizer, epoch + 1,
                                f"{dcfg.save_dir}/last.pth", is_best=True)
            else:
                patience_counter += 1
                save_checkpoint(student, optimizer, epoch + 1,
                                f"{dcfg.save_dir}/last.pth", is_best=False)

            if patience_counter >= patience:
                print(f"Early stopping. Best Acc: {best_acc:.2%}")
                break

    cleanup_ddp()


if __name__ == "__main__":
    main()
