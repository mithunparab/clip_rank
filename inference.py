import torch
import requests
import yaml
import os
import time
import glob
from PIL import Image
from io import BytesIO
from types import SimpleNamespace
from torchvision import transforms
import numpy as np
from model import MobileCLIPRanker, OrdinalRanker, LDLRanker, get_norm_stats
from train_binary import BinaryClassifier

def load_config(path="config.yml"):
    with open(path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    def recursive_namespace(d):
        if isinstance(d, dict):
            for k, v in d.items():
                d[k] = recursive_namespace(v)
            return SimpleNamespace(**d)
        return d
    return recursive_namespace(cfg_dict)

class PropertyRanker:
    def __init__(self, model_path, config_path='config.yml', device=None):
        self.cfg = load_config(config_path)
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        print(f"--- Loading Ranker ---")
        print(f"Device: {self.device}")
        
        # 1. Detect model type from checkpoint
        print(f"Loading Weights: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Strip 'module.' prefix if trained with DDP
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k.replace("module.", "")] = v

        # Auto-detect model type from checkpoint keys
        self.is_ordinal = any('head.biases' in k for k in new_state_dict)
        self.is_ldl = any('score_values' in k for k in new_state_dict)
        # Binary models now use MobileCLIPRanker architecture (same as listwise)
        self.is_binary = False

        if self.is_ldl:
            print("Detected LDL model")
            self.model = LDLRanker(self.cfg)
        elif self.is_ordinal:
            print("Detected ordinal (CORAL) model")
            self.model = OrdinalRanker(self.cfg)
        else:
            self.model = MobileCLIPRanker(self.cfg)

        self.model.load_state_dict(new_state_dict)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully.\n")
        
        # 3. CRITICAL: EXACT MATCH TO TRAINING TRANSFORMS
        norm_mean, norm_std = get_norm_stats(self.cfg.model.name)
        self.process = transforms.Compose([
            transforms.Resize(self.cfg.data.img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.cfg.data.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std)
        ])

    def rank(self, image_list):
        valid_tensors = []
        valid_indices = []
        clean_urls = []
        
        print(f"Processing {len(image_list)} images...")
        
        for i, src in enumerate(image_list):
            if not src or not isinstance(src, str): continue
            try:
                if src.startswith("http"):
                    resp = requests.get(src, timeout=5)
                    img = Image.open(BytesIO(resp.content)).convert('RGB')
                else:
                    img = Image.open(src).convert('RGB')
                
                # Apply Transform
                tensor = self.process(img)
                
                valid_tensors.append(tensor)
                valid_indices.append(i)
                clean_urls.append(src)
            except Exception as e:
                print(f"Error loading image {i}: {e}")
        
        if not valid_tensors:
            return []

        with torch.no_grad():
            if self.is_ldl or self.is_ordinal:
                # Pointwise: score each image independently
                batch = torch.stack(valid_tensors).to(self.device)  # (N, C, H, W)
                raw_scores = self.model.score(batch).cpu().numpy()
            else:
                # Group-based ranking model
                batch = torch.stack(valid_tensors).unsqueeze(0).to(self.device)
                valid_len = torch.tensor([len(valid_tensors)]).to(self.device)
                raw_scores = self.model(batch, valid_lens=valid_len).view(-1).cpu().numpy()
            
        results = []
        for i, score in enumerate(raw_scores):
            results.append({'source': clean_urls[i], 'score': float(score)})
            
        # Sort descending (Higher score = Better)
        results.sort(key=lambda x: x['score'], reverse=True)
        return results

if __name__ == "__main__":
    if os.path.exists("checkpoints/best_model.pth"):
        model_path = "checkpoints/best_model.pth"
        print("Using Best Model.")
    elif os.path.exists("checkpoints/last.pth"):
        model_path = "checkpoints/last.pth"
        print("Using Last Epoch.")
    else:
        # Fallback search
        checkpoints = sorted(glob.glob("checkpoints/*.pth"), key=os.path.getmtime)
        model_path = checkpoints[-1] if checkpoints else None

    if not model_path:
        print("No model found!")
        exit()

    ranker = PropertyRanker(model_path=model_path)
    
    test_urls = [
        "https://ap.rdcpix.com/69fe76be4fd818c9b1e25b8b6c79432el-m3865337706s-w2048_h1536.jpg",
        "https://ap.rdcpix.com/69fe76be4fd818c9b1e25b8b6c79432el-m1211374265s-w2048_h1536.jpg",
        "https://ap.rdcpix.com/69fe76be4fd818c9b1e25b8b6c79432el-m713883090s-w2048_h1536.jpg",
        "https://ap.rdcpix.com/c3065cb0efd74e0e69c634c4e7926ed0l-m3456441259s-w2048_h1536.jpg"
    ]
    
    start_time = time.time()
    
    results = ranker.rank(test_urls)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print("\n" + "="*50)
    print(f"RANKING RESULTS")
    print("="*50)
    
    for i, res in enumerate(results):
        print(f"{i+1}. Score: {res['score']:.4f} | {res['source']}")
        
    print("="*50)
    print(f"Total Time Taken: {elapsed:.4f} seconds")
    print("="*50)