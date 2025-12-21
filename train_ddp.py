import os
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
from sklearn.model_selection import GroupKFold
from dataset import PropertyPreferenceDataset
from model import MobileCLIPRanker
from utils import load_config

def setup_ddp():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    dist.destroy_process_group()

def weighted_preference_loss(s_win, s_lose, score_diffs):
    pred_diff = s_win - s_lose
    loss = F.softplus(-pred_diff)
    

    weights = torch.log1p(score_diffs).view(-1, 1)
    
    return (loss * weights).mean()

def train_epoch(model, loader, optimizer, device):
    model.train() 
    total_loss = torch.zeros(1).to(device)
    
    for win_img, lose_img, gt_diff in loader:
        win_img, lose_img = win_img.to(device), lose_img.to(device)
        gt_diff = gt_diff.to(device)
        
        optimizer.zero_grad()
        
        batch = torch.cat([win_img, lose_img], dim=0)
        all_scores = model(batch)
        s_win, s_lose = torch.split(all_scores, win_img.size(0), dim=0)
        
        loss = weighted_preference_loss(s_win, s_lose, gt_diff)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.detach()
        
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    return total_loss.item() / (len(loader) * dist.get_world_size())

def validate(model, df_val, cfg, device):
    model.eval()
    correct_strict = 0
    correct_relaxed = 0
    total = 0
    
    ds_helper = PropertyPreferenceDataset(pd.DataFrame(), 
                                          model_name=cfg.model.name, 
                                          img_size=cfg.data.img_size,
                                          is_train=False)
    
    df_val = df_val.copy()
    df_val['file_path'] = df_val.index.map(lambda x: f"{cfg.data.images_dir}/{x}.jpg")
    grouped = df_val.groupby(['group_id', 'label'])
    
    with torch.no_grad():
        for _, group in grouped:
            if len(group) < 2: continue
            recs = [r for r in group.to_dict('records') if os.path.exists(r['file_path'])]
            if len(recs) < 2: continue
            
            batch_tensors = []
            gt_scores = []
            
            for r in recs:
                tensor = ds_helper._load_local(r['file_path'])
                batch_tensors.append(tensor)
                gt_scores.append(r['score'])
            
            batch = torch.stack(batch_tensors).to(device)
            pred_scores = model(batch).squeeze().cpu().numpy()
            
            pred_winner_idx = np.argmax(pred_scores)
            max_gt_score = max(gt_scores)
            score_of_chosen = gt_scores[pred_winner_idx]
            
            if score_of_chosen == max_gt_score:
                correct_strict += 1
            if score_of_chosen >= (max_gt_score - 1.0):
                correct_relaxed += 1
                
            total += 1
            
    if total == 0: return 0, 0
    return (correct_strict / total), (correct_relaxed / total)

def main():
    setup_ddp()
    cfg = load_config()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)

    df = pd.read_csv(cfg.data.csv_path)
    gkf = GroupKFold(n_splits=5)
    groups = df['group_id'].values
    folds = list(gkf.split(df, groups=groups))
    train_idx, val_idx = folds[cfg.data.val_fold]
    
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    
    train_ds = PropertyPreferenceDataset(train_df, model_name=cfg.model.name, img_size=cfg.data.img_size, is_train=True)
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, sampler=train_sampler, num_workers=cfg.system.num_workers, pin_memory=cfg.system.pin_memory)
    
    model = MobileCLIPRanker(cfg).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    
    param_groups = [
        {'params': model.module.backbone.parameters(), 'lr': cfg.train.lr_backbone},
        {'params': model.module.score_head.parameters(), 'lr': cfg.train.lr_head}
    ]
    
    optimizer = optim.AdamW(param_groups, weight_decay=cfg.train.weight_decay)

    if local_rank == 0:
        print(f"--- Training {cfg.model.name} (Corrected Augmentation) ---")

    for epoch in range(cfg.train.epochs):
        train_sampler.set_epoch(epoch)
        loss = train_epoch(model, train_loader, optimizer, device)
        
        if local_rank == 0:
            strict, relaxed = validate(model, val_df, cfg, device)
            print(f"Epoch {epoch+1} | Loss: {loss:.4f} | Strict: {strict:.4f} | Relaxed: {relaxed:.4f}")
            
            if strict > 0.45:
                torch.save(model.module.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")

    cleanup_ddp()

if __name__ == "__main__":
    main()