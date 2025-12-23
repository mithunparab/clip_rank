import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import GroupKFold

from dataset import PropertyPreferenceDataset
from model import MobileCLIPRanker

def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return rank, local_rank
    else:
        return 0, 0

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def load_config(path='config.yml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def validate(model, df_val, device, img_size):
    """
    Computes Strict and Relaxed Accuracy.
    Strategy: Inference on all images in a group, then sort.
    """
    model.eval()
    
    grouped = df_val.groupby('group_id')
    
    strict_wins = 0
    relaxed_wins = 0
    total_groups = 0
    
    ds_helper = PropertyPreferenceDataset(pd.DataFrame(), is_train=False, img_size=img_size)
    
    with torch.no_grad():
        for _, group in grouped:
            if len(group) < 2: continue
            
            imgs = []
            scores_gt = []
            
            for _, row in group.iterrows():
                t_img = ds_helper._letterbox_image(row['file_path'])
                t_img = transforms.functional.to_tensor(t_img)
                t_img = ds_helper.normalize(t_img)
                imgs.append(t_img)
                scores_gt.append(row['score'])
            
            batch = torch.stack(imgs).to(device)
            
            pred_scores = model(batch).squeeze(-1).cpu().numpy()
            

            best_pred_idx = np.argmax(pred_scores)
            best_gt_score = max(scores_gt)
            
            score_of_selected = scores_gt[best_pred_idx]
            
            if score_of_selected == best_gt_score:
                strict_wins += 1
            
            if score_of_selected >= (best_gt_score - 1.0):
                relaxed_wins += 1
                
            total_groups += 1
    
    if total_groups == 0: return 0.0, 0.0
    return strict_wins / total_groups, relaxed_wins / total_groups

def main():
    rank, local_rank = setup_ddp()
    cfg = load_config()
    
    df = pd.read_csv(cfg['data']['csv_path'])
    
    gkf = GroupKFold(n_splits=5)
    groups = df['group_id'].values
    train_idx, val_idx = next(gkf.split(df, df['score'], groups))
    
    df_train = df.iloc[train_idx]
    df_val = df.iloc[val_idx]
    
    train_ds = PropertyPreferenceDataset(df_train, is_train=True, img_size=cfg['data']['img_size'])
    train_sampler = DistributedSampler(train_ds, shuffle=True) if dist.is_initialized() else None
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg['train']['batch_size'],
        sampler=train_sampler,
        num_workers=cfg['system']['num_workers'],
        pin_memory=cfg['system']['pin_memory'],
        shuffle=(train_sampler is None)
    )
    
    model = MobileCLIPRanker(
        head_hidden_dim=cfg['model']['head_hidden_dim'],
        dropout=cfg['model']['dropout']
    ).to(local_rank)
    
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    
    param_groups = [
        {'params': model.module.backbone.parameters(), 'lr': cfg['train']['lr_backbone']},
        {'params': model.module.head.parameters(), 'lr': cfg['train']['lr_head']}
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=cfg['train']['weight_decay'])
    

    criterion = nn.BCEWithLogitsLoss()
    target_ones = torch.ones(cfg['train']['batch_size'], 1).to(local_rank)

    os.makedirs(cfg['train']['save_dir'], exist_ok=True)
    
    for epoch in range(cfg['train']['epochs']):
        if dist.is_initialized():
            train_sampler.set_epoch(epoch)
        
        model.train()
        epoch_loss = 0.0
        
        iterator = train_loader
        if rank == 0:
            iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            
        for win_img, lose_img in iterator:
            win_img = win_img.to(local_rank)
            lose_img = lose_img.to(local_rank)
            
            optimizer.zero_grad()
            

            batch = torch.cat([win_img, lose_img], dim=0)
            scores = model(batch) 
            
            s_win, s_lose = torch.split(scores, win_img.size(0))
            

            diff = s_win - s_lose
            
            curr_batch_size = diff.size(0)
            loss = criterion(diff, torch.ones(curr_batch_size, 1).to(local_rank))
            
            loss.backward()
            
            if cfg['train']['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train']['grad_clip'])
                
            optimizer.step()
            epoch_loss += loss.item()
            
        if rank == 0:
            avg_loss = epoch_loss / len(train_loader)
            strict_acc, relaxed_acc = validate(model.module, df_val, local_rank, cfg['data']['img_size'])
            
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Strict Acc: {strict_acc:.4f} | Relaxed Acc: {relaxed_acc:.4f}")
            
            torch.save(model.module.state_dict(), f"{cfg['train']['save_dir']}/epoch_{epoch+1}.pt")
            
            if relaxed_acc > 0.85: 
                print("High relaxed accuracy reached. Snapshotting best model.")
                torch.save(model.module.state_dict(), f"{cfg['train']['save_dir']}/best_relaxed.pt")

    cleanup_ddp()

if __name__ == "__main__":
    main()