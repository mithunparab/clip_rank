import torch
import requests
from PIL import Image
from io import BytesIO
import mobileclip
from model import MobileCLIPRanker

class PropertyRanker:
    def __init__(self, model_path, model_name='mobileclip_s2', device='cpu'):
        self.device = device
        self.img_size = 224
        
        self.model = MobileCLIPRanker(model_name=model_name)
        state_dict = torch.load(model_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()
        
        _, _, self.preprocess = mobileclip.create_model_and_transforms(model_name)

    def _download_and_process(self, url):
        try:
            resp = requests.get(url, timeout=2)
            img = Image.open(BytesIO(resp.content)).convert('RGB')
        except:
            return None

        w, h = img.size
        scale = self.img_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        
        background = Image.new('RGB', (self.img_size, self.img_size), (0, 0, 0))
        offset = ((self.img_size - new_w) // 2, (self.img_size - new_h) // 2)
        background.paste(img_resized, offset)
        
        return self.preprocess(background)

    def rank_urls(self, urls):
        valid_tensors = []
        valid_indices = []
        
        for i, url in enumerate(urls):
            tensor = self._download_and_process(url)
            if tensor is not None:
                valid_tensors.append(tensor)
                valid_indices.append(i)
        
        if not valid_tensors:
            return []

        batch = torch.stack(valid_tensors).to(self.device)
        
        with torch.no_grad():
            scores = self.model(batch).squeeze(-1).cpu().numpy()
            
        results = []
        for idx, score in zip(valid_indices, scores):
            results.append({
                'url': urls[idx],
                'score': float(score),
                'original_index': idx
            })
            
        results.sort(key=lambda x: x['score'], reverse=True)
        return results

if __name__ == "__main__":
    ranker = PropertyRanker(model_path="mobileclip_ranker_fold1.pth")
    
    test_urls = [
        "https://ap.rdcpix.com/69fe76be4fd818c9b1e25b8b6c79432el-m3865337706s-w2048_h1536.jpg",
        "https://ap.rdcpix.com/69fe76be4fd818c9b1e25b8b6c79432el-m713883090s-w2048_h1536.jpg"
    ]
    
    ranked = ranker.rank_urls(test_urls)
    
    print(f"Winner: {ranked[0]['url']}")
    print(f"Score: {ranked[0]['score']:.4f}")