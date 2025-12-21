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

CSV_PATH = "dataset.csv" 
MODEL_NAME = 'mobileclip_s2'
BATCH_SIZE = 32
LR = 1e-4 
EPOCHS = 10

def setup_ddp():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    dist.destroy_process_group()

def bradley_terry_loss(win_scores, lose_scores, weights):
    diff = win_scores - lose_scores
    loss = -F.logsigmoid(diff)
    weighted_loss = loss * weights.view(-1, 1)
    return weighted_loss.mean()

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = torch.zeros(1).to(device)
    
    for win_img, lose_img, weights in loader:
        win_img, lose_img = win_img.to(device), lose_img.to(device)
        weights = weights.to(device)
        
        optimizer.zero_grad()
        
        s_win = model(win_img)
        s_lose = model(lose_img)
        
        loss = bradley_terry_loss(s_win, s_lose, weights)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.detach()
        
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    return total_loss.item() / (len(loader) * dist.get_world_size())

def validate(model, df_val, device):
    model.eval()
    correct = 0
    total = 0
    
    ds_helper = PropertyPreferenceDataset(pd.DataFrame(), model_name=MODEL_NAME)
    
    grouped = df_val.groupby(['group_id', 'label'])
    
    with torch.no_grad():
        for _, group in grouped:
            if len(group) < 2: continue
            
            recs = group.to_dict('records')
            batch_tensors = []
            gt_scores = []
            
            for r in recs:
                img = ds_helper._download_image(r['url'])
                tensor = ds_helper._letterbox_process(img)
                batch_tensors.append(tensor)
                gt_scores.append(r['score'])
            
            batch = torch.stack(batch_tensors).to(device)
            
            pred_scores = model(batch).squeeze().cpu().numpy()
            
            if batch.shape[0] < 2: continue
            
            pred_winner_idx = np.argmax(pred_scores)
            best_gt_score = max(gt_scores)
            
            if gt_scores[pred_winner_idx] == best_gt_score:
                correct += 1
            total += 1
            
    return correct / total if total > 0 else 0

def main():
    setup_ddp()
    
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    
    if local_rank == 0:
        print(f"--- DDP Training Started on {torch.cuda.device_count()} GPUs ---")
        df = pd.read_csv(CSV_PATH)
    else:
        df = pd.read_csv(CSV_PATH)

    gkf = GroupKFold(n_splits=5)
    groups = df['group_id'].values
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups=groups)):
        if local_rank == 0:
            print(f"--- Fold {fold+1} ---")
            
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        train_ds = PropertyPreferenceDataset(train_df, model_name=MODEL_NAME)
        
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        
        train_loader = DataLoader(
            train_ds, 
            batch_size=BATCH_SIZE, 
            sampler=train_sampler, 
            num_workers=4, 
            pin_memory=True
        )
        
        model = MobileCLIPRanker(model_name=MODEL_NAME).to(device)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
        
        for epoch in range(EPOCHS):
            train_sampler.set_epoch(epoch)
            
            loss = train_epoch(model, train_loader, optimizer, device)
            
            if local_rank == 0:
                print(f"Epoch {epoch+1} | Train Loss: {loss:.4f}")
                
                acc = validate(model, val_df, device)
                print(f"Epoch {epoch+1} | Val Accuracy: {acc:.4f}")
                
                if (epoch + 1) % 5 == 0:
                    torch.save(model.module.state_dict(), f"mobileclip_ddp_fold{fold+1}.pth")

        break
        
    cleanup_ddp()

if __name__ == "__main__":
    main()
