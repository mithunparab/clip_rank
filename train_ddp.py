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
from torchvision import transforms 
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

def dynamic_margin_loss(pred_scores, gt_scores, valid_len):
    device = pred_scores.device
    loss = torch.tensor(0.0, device=device)
    n_pairs = 0
    
    for b in range(pred_scores.shape[0]):
        n_imgs = int(valid_len[b].item())
        p_scores = pred_scores[b, :n_imgs]
        g_scores = gt_scores[b, :n_imgs]
        
        p_diff_mat = p_scores.unsqueeze(0) - p_scores.unsqueeze(1)
        g_diff_mat = g_scores.unsqueeze(0) - g_scores.unsqueeze(1)
        
        pair_mask = g_diff_mat > 0
        if pair_mask.sum() == 0: continue
            
        dynamic_margins = g_diff_mat[pair_mask] * 0.1
        preds = p_diff_mat[pair_mask]
        
        pair_losses = torch.relu(dynamic_margins - preds)
        loss += pair_losses.mean()
        n_pairs += 1
        
    return loss / n_pairs if n_pairs > 0 else loss

def validate(model, df_val, cfg, device):
    model.eval()
    ds_helper = PropertyPreferenceDataset(
        pd.DataFrame({'group_id':[], 'score':[]}), 
        images_dir="images", is_train=False, img_size=cfg.data.img_size
    )
    
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
            
            images, gt_scores = [], []
            for _, row in group.iterrows():
                path = row['file_path']
                if not os.path.exists(path): continue
                img = ds_helper._letterbox_image(path)
                t_img = ds_helper.normalize(transforms.functional.to_tensor(img))
                images.append(t_img)
                gt_scores.append(row['score'])
                
            if len(images) < 2: continue
            
            # Create Batch [1, N, 3, H, W]
            batch = torch.stack(images).unsqueeze(0).to(device)
            valid_len = torch.tensor([len(images)]).to(device)
            
            # Predict with context
            pred_scores = model(batch, valid_lens=valid_len).view(-1).cpu().numpy()
            
            best_pred_idx = np.argmax(pred_scores)
            score_of_model_choice = gt_scores[best_pred_idx]
            max_gt_score = max(gt_scores)
            
            if score_of_model_choice == max_gt_score: strict_wins += 1
            if score_of_model_choice >= (max_gt_score - 1.0): relaxed_wins += 1
            total_groups += 1
            
    if total_groups == 0: return 0.0, 0.0
    return strict_wins / total_groups, relaxed_wins / total_groups

def main():
    rank, local_rank = setup_ddp()
    cfg = load_config("config.yml")
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    
    df = pd.read_csv(cfg.data.csv_path)
    gkf = GroupKFold(n_splits=cfg.data.n_splits)
    train_idx, val_idx = next(gkf.split(df, groups=df['group_id']))
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    
    train_ds = PropertyPreferenceDataset(train_df, images_dir="images", is_train=True, img_size=cfg.data.img_size, max_len=cfg.data.max_images_per_group)
    sampler = DistributedSampler(train_ds, shuffle=True) if dist.is_initialized() else None
    
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, sampler=sampler, 
        num_workers=cfg.system.num_workers, pin_memory=cfg.system.pin_memory, 
        shuffle=(sampler is None), drop_last=True
    )
    
    device = torch.device(f"cuda:{local_rank}")
    model = MobileCLIPRanker(cfg).to(device)
    
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    optimizer = optim.AdamW(model.module.head.parameters(), lr=cfg.train.lr_head, weight_decay=cfg.train.weight_decay)
    
    if rank == 0:
        print(f"Training Group-Centered Ranker on {len(train_ds)} properties.")

    for epoch in range(cfg.train.epochs):
        if dist.is_initialized(): sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        
        iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}") if rank == 0 else train_loader
        
        for img_stack, score_stack, valid_len in iterator:
            img_stack, score_stack, valid_len = img_stack.to(device), score_stack.to(device), valid_len.to(device)
            
            optimizer.zero_grad()
            pred_scores = model(img_stack, valid_lens=valid_len)
            
            loss = dynamic_margin_loss(pred_scores, score_stack, valid_len)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            total_loss += loss.item()
        
        if rank == 0:
            avg_loss = total_loss / len(train_loader)
            strict, relaxed = validate(model.module, val_df, cfg, device)
            
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Strict: {strict:.2%} | Relaxed: {relaxed:.2%}")
            
            if relaxed > 0.65:
                os.makedirs(cfg.train.save_dir, exist_ok=True)
                torch.save(model.module.state_dict(), f"{cfg.train.save_dir}/epoch_{epoch+1}.pth")

    cleanup_ddp()

if __name__ == "__main__":
    main()