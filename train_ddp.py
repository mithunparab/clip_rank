import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import pandas as pd
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

def ordinal_margin_loss(pred_scores, gt_scores, valid_len):
    """
    Enforces the Caste System: Tier 2 >> Tier 1 >> Tier 0.
    1. Convert raw scores to Tiers (0, 1, 2).
    2. Penalize if Higher Tier Score < Lower Tier Score + Margin.
    """
    loss = 0.0
    n_pairs = 0
    
    # Define Tiers based on your logic
    # Tier 2: 8-10 (Gold)
    # Tier 1: 3-5 (Silver)
    # Tier 0: 0-2 (Trash)
    tiers = torch.zeros_like(gt_scores)
    tiers[gt_scores >= 8] = 2.0
    tiers[(gt_scores >= 3) & (gt_scores < 8)] = 1.0
    
    for b in range(pred_scores.shape[0]):
        n = int(valid_len[b].item())
        if n < 2: continue
        
        p = pred_scores[b, :n].view(-1)
        t = tiers[b, :n]
        
        # Broadcast differences
        tier_diff = t.unsqueeze(0) - t.unsqueeze(1)
        
        # Find pairs where i is a higher tier than j
        pairs = torch.nonzero(tier_diff > 0)
        
        if len(pairs) == 0: continue
            
        for idx in pairs:
            i, j = idx[0], idx[1]
            
            # Dynamic Margin based on Tier Distance
            # Gap of 1 tier (2 vs 1, or 1 vs 0) -> Margin 1.0
            # Gap of 2 tiers (2 vs 0)          -> Margin 2.5
            dist = tier_diff[i, j].item()
            margin = 1.0 if dist == 1.0 else 2.5
            
            # We want: Score_i > Score_j + Margin
            # Loss = ReLU(Margin - (Score_i - Score_j))
            current_gap = p[i] - p[j]
            loss += torch.relu(margin - current_gap)
            n_pairs += 1
            
    if n_pairs > 0:
        return loss / n_pairs
    return torch.tensor(0.0, device=pred_scores.device, requires_grad=True)

def validate(model, df_val, cfg, device):
    model.eval()
    ds = PropertyPreferenceDataset(pd.DataFrame({'group_id':[], 'score':[]}), images_dir="images", is_train=False, img_size=cfg.data.img_size)
    if 'file_path' not in df_val.columns: df_val['file_path'] = df_val.index.map(lambda x: f"images/{x}.jpg")
    
    grouped = df_val.groupby('group_id')
    strict_wins = 0
    total_groups = 0
    
    debug_printed = False
    
    with torch.no_grad():
        for _, group in grouped:
            if len(group) < 2: continue
            images, scores = [], []
            for _, row in group.iterrows():
                if not os.path.exists(row['file_path']): continue
                images.append(ds._process(row['file_path']))
                scores.append(row['score'])
            
            if len(images) < 2: continue
            batch = torch.stack(images).unsqueeze(0).to(device)
            
            preds = model(batch, valid_lens=torch.tensor([len(images)])).view(-1).cpu().numpy()
            
            # Validation Metric: Did we pick a Tier 2 image?
            best_idx = preds.argmax()
            best_score = scores[best_idx]
            
            # Ideally we want a Tier 2 (>=8). If none exist, we want Tier 1 (>=3).
            max_possible = max(scores)
            
            # Simple Strict Accuracy: Did we pick the highest available tier?
            if best_score >= 8:
                strict_wins += 1 # We picked a Gold image
            elif max_possible < 8 and best_score >= 3:
                strict_wins += 1 # No Gold existed, so we picked Silver. Good job.
            elif max_possible < 3:
                strict_wins += 1 # Only trash existed, we picked trash. Fine.
                
            total_groups += 1
            
            if not debug_printed:
                print(f"[DEBUG] Preds: {preds[:5]} | GT: {scores[:5]}")
                debug_printed = True
                
    return strict_wins / total_groups if total_groups > 0 else 0.0

def save_checkpoint(model, optimizer, epoch, path):
    raw_model = model.module if hasattr(model, "module") else model
    torch.save({'state_dict': raw_model.state_dict()}, path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    rank, local_rank = setup_ddp()
    cfg = load_config("config.yml")
    
    os.makedirs(cfg.train.save_dir, exist_ok=True)
    torch.manual_seed(42)
    
    df = pd.read_csv(cfg.data.csv_path)
    # Simple split
    val_groups = df['group_id'].unique()[:int(len(df['group_id'].unique()) * 0.1)]
    train_df = df[~df['group_id'].isin(val_groups)]
    val_df = df[df['group_id'].isin(val_groups)]
    
    train_ds = PropertyPreferenceDataset(train_df, images_dir="images", is_train=True, img_size=cfg.data.img_size)
    sampler = DistributedSampler(train_ds, shuffle=True) if dist.is_initialized() else None
    
    train_loader = DataLoader(train_ds, batch_size=1, sampler=sampler, num_workers=4, pin_memory=True)
    
    device = torch.device(f"cuda:{local_rank}")
    model = MobileCLIPRanker(cfg).to(device)
    if dist.is_initialized(): model = DDP(model, device_ids=[local_rank])
    
    optimizer = optim.AdamW(model.parameters(), lr=cfg.train.lr_head, weight_decay=0.01)
    
    for epoch in range(cfg.train.epochs):
        model.train()
        total_loss = 0.0
        
        iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}") if rank == 0 else train_loader
        
        for imgs, scores, vlen in iterator:
            imgs, scores = imgs.to(device), scores.to(device)
            optimizer.zero_grad()
            preds = model(imgs, vlen)
            loss = ordinal_margin_loss(preds, scores, vlen)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if rank == 0:
            acc = validate(model.module if hasattr(model, 'module') else model, val_df, cfg, device)
            print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Accuracy: {acc:.2%}")
            save_checkpoint(model, optimizer, epoch, f"{cfg.train.save_dir}/last.pth")

    cleanup_ddp()

if __name__ == "__main__":
    main()