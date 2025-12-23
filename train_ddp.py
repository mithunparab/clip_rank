import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
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
    if dist.is_initialized():
        dist.destroy_process_group()

def validate(model, df_val, cfg, device):
    """
    Validates by ranking images within groups.
    Calculates Strict Accuracy and Relaxed Accuracy.
    """
    model.eval()
    
    ds_helper = PropertyPreferenceDataset(df_val, images_dir="images", is_train=False, img_size=cfg.data.img_size)
    
    if 'file_path' not in df_val.columns:
         df_val = df_val.copy()
         df_val['file_path'] = df_val.index.map(lambda x: f"images/{x}.jpg")
    
    grouped = df_val.groupby('group_id')
    strict_wins = 0
    relaxed_wins = 0
    total_groups = 0
    
    with torch.no_grad():
        for _, group in grouped:
            if len(group) < 2: continue
            
            images = []
            gt_scores = []
            
            for _, row in group.iterrows():
                path = row['file_path']
                if not os.path.exists(path): continue
                
                img = ds_helper._letterbox_image(path)
                t_img = ds_helper.normalize(transforms.functional.to_tensor(img))
                
                images.append(t_img)
                gt_scores.append(row['score'])
                
            if len(images) < 2: continue
            
            batch = torch.stack(images).to(device)
            pred_scores = model(batch).view(-1).cpu().numpy()
            
            best_pred_idx = np.argmax(pred_scores)
            score_of_model_choice = gt_scores[best_pred_idx]
            max_gt_score = max(gt_scores)
            
            if score_of_model_choice == max_gt_score:
                strict_wins += 1
            
            if score_of_model_choice >= (max_gt_score - 1.0):
                relaxed_wins += 1
                
            total_groups += 1
            
    if total_groups == 0: return 0.0, 0.0
    return strict_wins / total_groups, relaxed_wins / total_groups

def main():
    rank, local_rank = setup_ddp()
    cfg = load_config("config.yml")
    
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    
    df = pd.read_csv(cfg.data.csv_path)
    
    
    gkf = GroupKFold(n_splits=5)
    train_idx, val_idx = next(gkf.split(df, groups=df['group_id']))
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    
    train_ds = PropertyPreferenceDataset(
        train_df, 
        images_dir="images", 
        is_train=True, 
        img_size=cfg.data.img_size
    )
    
    sampler = DistributedSampler(train_ds, shuffle=True) if dist.is_initialized() else None
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg.train.batch_size, 
        sampler=sampler, 
        num_workers=cfg.system.num_workers, 
        pin_memory=cfg.system.pin_memory,
        shuffle=(sampler is None),
        drop_last=True 
    )
    
    device = torch.device(f"cuda:{local_rank}")
    model = MobileCLIPRanker(cfg).to(device)
    
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    
    criterion = nn.BCEWithLogitsLoss()
    
    param_groups = [
        {'params': model.module.backbone.parameters(), 'lr': cfg.train.lr_backbone},
        {'params': model.module.head.parameters(), 'lr': cfg.train.lr_head}
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=cfg.train.weight_decay)
    
    if rank == 0:
        print(f"Training on {len(train_ds)} pairs | Validation on {len(val_df)} images")

    for epoch in range(cfg.train.epochs):
        if dist.is_initialized():
            sampler.set_epoch(epoch)
            
        model.train()
        total_loss = 0.0
        
        iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}") if rank == 0 else train_loader
        
        for win_imgs, lose_imgs, targets in iterator:
            win_imgs, lose_imgs = win_imgs.to(device), lose_imgs.to(device)
            targets = targets.to(device).view(-1, 1)
            
            optimizer.zero_grad()
            
            combined = torch.cat([win_imgs, lose_imgs], dim=0)
            all_scores = model(combined)
            s_win, s_lose = torch.split(all_scores, win_imgs.size(0))
            
            diff = s_win - s_lose
            loss = criterion(diff, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            
            total_loss += loss.item()
            
        if rank == 0:
            avg_loss = total_loss / len(train_loader)
            strict, relaxed = validate(model.module, val_df, cfg, device)
            
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Strict: {strict:.2%} | Relaxed: {relaxed:.2%}")
            
            if relaxed > 0.65:
                torch.save(model.module.state_dict(), f"{cfg.train.save_dir}/epoch_{epoch+1}.pth")

    cleanup_ddp()

if __name__ == "__main__":
    main()