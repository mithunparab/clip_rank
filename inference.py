import torch
import requests
import yaml
import os
from PIL import Image
from io import BytesIO
from types import SimpleNamespace
from model import MobileCLIPRanker
import mobileclip
from torchvision import transforms

def load_config(path="config.yml"):
    # Simple loader to avoid recursive namespace issues
    with open(path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    # Basic namespace conversion
    cfg = SimpleNamespace(**cfg_dict)
    cfg.system = SimpleNamespace(**cfg_dict['system'])
    cfg.data = SimpleNamespace(**cfg_dict['data'])
    cfg.model = SimpleNamespace(**cfg_dict['model'])
    return cfg

class PropertyRanker:
    def __init__(self, model_path, config_path='config.yml', device=None):
        self.cfg = load_config(config_path)
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        print(f"Loading Ranker on {self.device}...")
        self.model = MobileCLIPRanker(self.cfg)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        # Handle DDP prefixes
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
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
        
        for i, src in enumerate(image_list):
            try:
                if src.startswith("http"):
                    resp = requests.get(src, timeout=3)
                    img = Image.open(BytesIO(resp.content)).convert('RGB')
                else:
                    img = Image.open(src).convert('RGB')
                
                tensor = self._letterbox_process(img)
                valid_tensors.append(tensor)
                valid_indices.append(i)
            except Exception as e:
                print(f"Error loading {src}: {e}")
        
        if not valid_tensors: return []

        # Create Batch: [1, N, 3, H, W]
        # This matches the training input shape exactly
        batch = torch.stack(valid_tensors).unsqueeze(0).to(self.device)
        valid_len = torch.tensor([len(valid_tensors)]).to(self.device)
        
        with torch.no_grad():
            # Pass valid_len so mean subtraction works correctly
            scores = self.model(batch, valid_lens=valid_len).view(-1).cpu().numpy()
            
        results = []
        for idx, score in zip(valid_indices, scores):
            results.append({
                'source': image_list[idx],
                'score': float(score)
            })
            
        results.sort(key=lambda x: x['score'], reverse=True)
        return results

if __name__ == "__main__":
    import glob
    checkpoints = sorted(glob.glob("checkpoints/*.pth"), key=os.path.getmtime)
    if checkpoints:
        print(f"Using: {checkpoints[-1]}")
        ranker = PropertyRanker(checkpoints[-1])
        # Test with dummy data or real URLs here
    else:
        print("No checkpoints found.")