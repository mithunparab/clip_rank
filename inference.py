import torch
import requests
import yaml
import os
import glob
from PIL import Image
from io import BytesIO
from types import SimpleNamespace
from torchvision import transforms
import numpy as np
from model import MobileCLIPRanker

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
        
        self.model = MobileCLIPRanker(self.cfg)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
        
        self.model.load_state_dict(new_state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        self.process = transforms.Compose([
            transforms.Resize(self.cfg.data.img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.cfg.data.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.481, 0.457, 0.408), std=(0.268, 0.261, 0.275))
        ])

    def rank(self, image_list):
        valid_tensors = []
        valid_indices = []
        
        for i, src in enumerate(image_list):
            try:
                if src.startswith("http"):
                    resp = requests.get(src, timeout=5)
                    img = Image.open(BytesIO(resp.content)).convert('RGB')
                else:
                    img = Image.open(src).convert('RGB')
                
                tensor = self.process(img)
                valid_tensors.append(tensor)
                valid_indices.append(i)
            except Exception as e:
                print(f"Skipping {i}: {e}")
        
        if not valid_tensors:
            return []

        batch = torch.stack(valid_tensors).unsqueeze(0).to(self.device)
        valid_len = torch.tensor([len(valid_tensors)]).to(self.device)
        
        with torch.no_grad():
            raw_scores = self.model(batch, valid_lens=valid_len).view(-1).cpu().numpy()
            
        results = []
        for idx, score in zip(valid_indices, raw_scores):
            results.append({'source': image_list[idx], 'score': float(score)})
            
        results.sort(key=lambda x: x['score'], reverse=True)
        return results

if __name__ == "__main__":
    if os.path.exists("checkpoints/best_model.pth"):
        model_path = "checkpoints/best_model.pth"
    elif os.path.exists("checkpoints/last.pth"):
        model_path = "checkpoints/last.pth"
    else:
        checkpoints = sorted(glob.glob("checkpoints/*.pth"), key=os.path.getmtime)
        model_path = checkpoints[-1] if checkpoints else None

    if not model_path:
        print("No model found.")
        exit()

    ranker = PropertyRanker(model_path=model_path)
    
    test_urls = [
        "https://ap.rdcpix.com/69fe76be4fd818c9b1e25b8b6c79432el-m3865337706s-w2048_h1536.jpg",
        "https://ap.rdcpix.com/69fe76be4fd818c9b1e25b8b6c79432el-m1211374265s-w2048_h1536.jpg",
        "https://ap.rdcpix.com/69fe76be4fd818c9b1e25b8b6c79432el-m713883090s-w2048_h1536.jpg",
        "https://ap.rdcpix.com/c3065cb0efd74e0e69c634c4e7926ed0l-m3456441259s-w2048_h1536.jpg"
    ]
    
    ranked_results = ranker.rank(test_urls)
    
    print("\nRANKING RESULTS:")
    for i, res in enumerate(ranked_results):
        print(f"{i+1}. Score: {res['score']:.4f} | {res['source']}")