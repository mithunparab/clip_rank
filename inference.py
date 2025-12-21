import torch
import requests
import yaml
import os
import argparse
from PIL import Image
from io import BytesIO
from types import SimpleNamespace
from model import MobileCLIPRanker
import mobileclip

def load_config(path="config.yml"):
    with open(path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    cfg = SimpleNamespace(**cfg_dict)
    cfg.system = SimpleNamespace(**cfg_dict['system'])
    cfg.data = SimpleNamespace(**cfg_dict['data'])
    cfg.model = SimpleNamespace(**cfg_dict['model'])
    if not hasattr(cfg.model, 'num_anchors'):
        cfg.model.num_anchors = 1
    return cfg

class PropertyRanker:
    def __init__(self, model_path, config_path='config.yml', device=None):
        self.cfg = load_config(config_path)
        
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        print(f"Loading model on {self.device}...")

        self.model = MobileCLIPRanker(self.cfg)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        state_dict = {}
        for k, v in checkpoint.items():
            if k.startswith("module."):
                state_dict[k[7:]] = v
            else:
                state_dict[k] = v
                
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        _, _, self.preprocess = mobileclip.create_model_and_transforms(self.cfg.model.name)

    def _download_and_process(self, input_source):
        """
        Handles URL string OR local path string.
        """
        try:
            if input_source.startswith("http"):
                resp = requests.get(input_source, timeout=3)
                img = Image.open(BytesIO(resp.content)).convert('RGB')
            else:
                img = Image.open(input_source).convert('RGB')
        except Exception as e:
            print(f"Error loading {input_source}: {e}")
            return None

        target_size = self.cfg.data.img_size
        w, h = img.size
        scale = target_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        
        background = Image.new('RGB', (target_size, target_size), (0, 0, 0))
        offset = ((target_size - new_w) // 2, (target_size - new_h) // 2)
        background.paste(img_resized, offset)
        
        return self.preprocess(background)

    def rank(self, image_list):
        """
        Takes a list of URLs/Paths, returns sorted list of dicts.
        """
        valid_tensors = []
        valid_indices = []
        
        print(f"Processing {len(image_list)} images...")
        
        for i, src in enumerate(image_list):
            tensor = self._download_and_process(src)
            if tensor is not None:
                valid_tensors.append(tensor)
                valid_indices.append(i)
        
        if not valid_tensors:
            return []

        batch = torch.stack(valid_tensors).to(self.device)
        
        with torch.no_grad():
            scores = self.model(batch).cpu().numpy()
            
        results = []
        for idx, score in zip(valid_indices, scores):
            display_score = float(score) * 10.0
            
            results.append({
                'source': image_list[idx],
                'score': display_score,
                'raw_sim': float(score)
            })
            
        results.sort(key=lambda x: x['score'], reverse=True)
        return results

if __name__ == "__main__":
    import glob
    checkpoints = sorted(glob.glob("checkpoint_epoch_*.pth"), key=os.path.getmtime)
    if not checkpoints:
        print("No checkpoints found! Run training first.")
        exit()
    CHECKPOINT = checkpoints[-1]
    
    ranker = PropertyRanker(model_path=CHECKPOINT)
    
    test_inputs = [
        "https://ap.rdcpix.com/69fe76be4fd818c9b1e25b8b6c79432el-m3865337706s-w2048_h1536.jpg", 
        "https://ap.rdcpix.com/69fe76be4fd818c9b1e25b8b6c79432el-m1211374265s-w2048_h1536.jpg", 
        "https://ap.rdcpix.com/69fe76be4fd818c9b1e25b8b6c79432el-m713883090s-w2048_h1536.jpg",  
        "https://ap.rdcpix.com/c3065cb0efd74e0e69c634c4e7926ed0l-m3456441259s-w2048_h1536.jpg"  
    ]
    
    ranked_results = ranker.rank(test_inputs)
    
    print("\n--- RANKING RESULTS ---")
    for r in ranked_results:
        print(f"Score: {r['score']:.2f} | {r['source']}")