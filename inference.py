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
import mobileclip

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
        print(f"Weights: {model_path}")
        
        self.model = MobileCLIPRanker(self.cfg)
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                raw_state_dict = checkpoint['model_state_dict']
            else:
                raw_state_dict = checkpoint

            state_dict = {k.replace("module.", ""): v for k, v in raw_state_dict.items()}
            
            self.model.load_state_dict(state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully.\n")
            
        except Exception as e:
            print(f"\nCRITICAL ERROR LOADING WEIGHTS: {e}")
            print("Ensure model.py architecture matches the checkpoint.")
            exit()
        
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), 
            std=(0.229, 0.224, 0.225)
        )

    def _letterbox_process(self, img):
        target_size = self.cfg.data.img_size
        w, h = img.size
        scale = target_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
        background = Image.new('RGB', (target_size, target_size), (0, 0, 0))
        offset = ((target_size - new_w) // 2, (target_size - new_h) // 2)
        background.paste(img_resized, offset)
        t_img = transforms.functional.to_tensor(background)
        return self.normalize(t_img)

    def rank(self, image_list):
        valid_tensors = []
        valid_indices = []
        
        print(f"Processing {len(image_list)} images...")
        
        for i, src in enumerate(image_list):
            try:
                if src.startswith("http"):
                    resp = requests.get(src, timeout=5)
                    img = Image.open(BytesIO(resp.content)).convert('RGB')
                else:
                    img = Image.open(src).convert('RGB')
                tensor = self._letterbox_process(img)
                valid_tensors.append(tensor)
                valid_indices.append(i)
            except Exception as e:
                print(f"Error loading image {i}: {e}")
        
        if not valid_tensors:
            return []

        batch = torch.stack(valid_tensors).unsqueeze(0).to(self.device)
        valid_len = torch.tensor([len(valid_tensors)]).to(self.device)
        
        with torch.no_grad():
            scores = self.model(batch, valid_lens=valid_len).view(-1).cpu().numpy()
            
        results = []
        for idx, score in zip(valid_indices, scores):
            results.append({'source': image_list[idx], 'score': float(score)})
            
        results.sort(key=lambda x: x['score'], reverse=True)
        return results

if __name__ == "__main__":
    best_model_path = "checkpoints/best_model.pth"
    
    if os.path.exists(best_model_path):
        model_path = best_model_path
        print(f"Using Best Saved Model: {model_path}")
    else:
        checkpoints = sorted(glob.glob("checkpoints/*.pth"), key=os.path.getmtime)
        if not checkpoints:
            print("Error: No checkpoints found in 'checkpoints/' directory.")
            exit()
        model_path = checkpoints[-1]
        print(f"Using Latest Checkpoint: {model_path}")

    ranker = PropertyRanker(model_path=model_path)
    
    test_urls = [
        "https://ap.rdcpix.com/69fe76be4fd818c9b1e25b8b6c79432el-m3865337706s-w2048_h1536.jpg",
        "https://ap.rdcpix.com/69fe76be4fd818c9b1e25b8b6c79432el-m1211374265s-w2048_h1536.jpg",
        "https://ap.rdcpix.com/69fe76be4fd818c9b1e25b8b6c79432el-m713883090s-w2048_h1536.jpg",
        "https://ap.rdcpix.com/c3065cb0efd74e0e69c634c4e7926ed0l-m3456441259s-w2048_h1536.jpg"
    ]
    
    ranked_results = ranker.rank(test_urls)
    
    print("\n" + "="*50)
    print(f"RANKING RESULTS (Best to Worst)")
    print("="*50)
    
    if ranked_results:
        print(f"\n🏆 WINNER (Score: {ranked_results[0]['score']:.4f})")
        print(f"   Source: {ranked_results[0]['source']}")
        
        print("\nRunners Up:")
        for i, res in enumerate(ranked_results[1:], 1):
            print(f"{i}. Score: {res['score']:.4f} | {res['source']}")
            
    print("="*50)