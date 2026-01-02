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
from dataset import PropertyPreferenceDataset
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
    if dist.is_initialized(): dist.destroy_process_group()

def listwise_kl_loss(pred_scores, gt_scores, valid_len):
    """
    Treats ranking as a probability distribution alignment.
    
    1. GT Distribution: 
       - Gold (Tier 2): High probability (shared if multiple).
       - Silver (Tier 1): Low probability.
       - Trash (Tier 0): Zero probability.
       
    2. Pred Distribution: Softmax(pred_scores).
    
    3. Loss: KL Divergence(GT || Pred).
    
    This forces the model to push Gold scores UP relative to everything else.
    It correlates directly with Accuracy.
    """
    loss = 0.0
    valid_batches = 0
    
    
    for b in range(pred_scores.shape[0]):
        n = int(valid_len[b].item())
        if n < 2: continue
        
        logits = pred_scores[b, :n].view(-1)
        gts = gt_scores[b, :n]
        
        target_probs = torch.zeros_like(logits)
        
        is_gold = gts >= 8
        is_silver = (gts >= 3) & (gts < 8)
        
        if is_gold.sum() > 0:
            target_probs[is_gold] = 1.0 / is_gold.sum()
        elif is_silver.sum() > 0:
            target_probs[is_silver] = 1.0 / is_silver.sum()
        else:
            target_probs.fill_(1.0 / n)
            
        log_probs = F.log_softmax(logits, dim=0)
        
        batch_loss = -torch.sum(target_probs * log_probs)
        
        loss += batch_loss
        valid_batches += 1
            
    if valid_batches > 0:
        return loss / valid_batches
    return pred_scores.sum() * 0.0

def validate(model, df_val, cfg, device):
    model.eval()
    ds = PropertyPreferenceDataset(
        pd.DataFrame({'group_id':[], 'score':[], 'label':[]}), 
        images_dir="images", is_train=False, img_size=cfg.data.img_size
    )
    df_val = df_val.copy()
    if 'file_path' not in df_val.columns: 
        df_val['file_path'] = df_val.index.map(lambda x: f"images/{x}.jpg")
    
    grouped = df_val.groupby('group_id')
    strict_wins = 0
    total_groups = 0
    
    with torch.no_grad():
        for _, group in grouped:
            if len(group) < 2: continue
            
            images, scores = [], []
            for _, row in group.iterrows():
                if not os.path.exists(row['file_path']): continue
                images.append(ds._process(row['file_path']))
                
                raw = float(row['score'])
                lbl = str(row.get('label', '')).lower()
                if raw >= 8 and lbl in ['outdoor','bathroom','other','balcony']: raw = 0.0
                if raw >= 8 and lbl == 'bedroom': raw = 3.0
                scores.append(raw)
            
            if len(images) < 2: continue
            batch = torch.stack(images).unsqueeze(0).to(device)
            valid_len = torch.tensor([len(images)])
            
            preds = model(batch, valid_lens=valid_len).view(-1).cpu().numpy()
            
            best_idx = np.argmax(preds)
            best_score = scores[best_idx]
            max_possible = max(scores)
            
            picked_tier = 2 if best_score >=8 else (1 if best_score >=3 else 0)
            max_tier = 2 if max_possible >=8 else (1 if max_possible >=3 else 0)
            
            if picked_tier == max_tier:
                strict_wins += 1
                
            total_groups += 1
            
    return strict_wins / total_groups if total_groups > 0 else 0.0

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
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    os.makedirs(cfg.train.save_dir, exist_ok=True)
    
    df = pd.read_csv(cfg.data.csv_path)
    
    unique_groups = df['group_id'].unique()
    val_groups = unique_groups[:int(len(unique_groups) * 0.1)]
    train_df = df[~df['group_id'].isin(val_groups)]
    val_df = df[df['group_id'].isin(val_groups)]
    
    train_ds = PropertyPreferenceDataset(train_df, images_dir="images", is_train=True, img_size=cfg.data.img_size)
    sampler = DistributedSampler(train_ds, shuffle=True) if dist.is_initialized() else None
    
    train_loader = DataLoader(
        train_ds, batch_size=1, sampler=sampler, 
        num_workers=cfg.system.num_workers, pin_memory=cfg.system.pin_memory
    )
    
    device = torch.device(f"cuda:{local_rank}")
    model = MobileCLIPRanker(cfg).to(device)
    
    if dist.is_initialized(): model = DDP(model, device_ids=[local_rank])
    
    optimizer = optim.AdamW(model.parameters(), lr=cfg.train.lr_head, weight_decay=cfg.train.weight_decay)
    
    best_acc = 0.0
    
    if rank == 0:
        print(f"Training on {len(train_ds)} groups using Listwise KL Loss.")

    for epoch in range(cfg.train.epochs):
        model.train()
        total_loss = 0.0
        
        iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}") if rank == 0 else train_loader
        
        for imgs, scores, vlen in iterator:
            imgs, scores, vlen = imgs.to(device), scores.to(device), vlen.to(device)
            
            optimizer.zero_grad()
            preds = model(imgs, vlen)
            
            loss = listwise_kl_loss(preds, scores, vlen)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            total_loss += loss.item()
            
        if rank == 0:
            avg_loss = total_loss / len(train_loader)
            raw_model = model.module if hasattr(model, 'module') else model
            acc = validate(raw_model, val_df, cfg, device)
            
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Strict Accuracy: {acc:.2%}")
            
            is_best = acc > best_acc
            if is_best: best_acc = acc
            
            save_checkpoint(model, optimizer, epoch + 1, f"{cfg.train.save_dir}/last.pth", is_best=is_best)

    cleanup_ddp()

if __name__ == "__main__":
    main()