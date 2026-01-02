import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os
import pandas as pd

class PropertyPreferenceDataset(Dataset):
    def __init__(self, df, images_dir="images", is_train=False, img_size=224):
        self.img_size = img_size
        self.df = df.copy()
        
        if 'file_path' not in self.df.columns:
            self.df['file_path'] = self.df.index.map(lambda x: os.path.join(images_dir, f"{x}.jpg"))
        self.df = self.df[self.df['file_path'].apply(os.path.exists)]
 
        self.process = transforms.Compose([
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.481, 0.457, 0.408), std=(0.268, 0.261, 0.275))
        ])

        self.groups = []
        if not self.df.empty:
            grouped = self.df.groupby('group_id')
            for _, group in grouped:
                if len(group) < 2: continue
                self.groups.append(group.to_dict('records'))

    def _process(self, path):
        try:
            with Image.open(path) as img:
                return self.process(img.convert('RGB'))
        except:
            return torch.zeros(3, self.img_size, self.img_size)

    def __len__(self): return len(self.groups)

    def __getitem__(self, idx):
        records = self.groups[idx]
        selected = records[:15]
        
        tensors = []
        scores = []
        
        for r in selected:
            tensors.append(self._process(r['file_path']))
            
            raw = float(r['score'])
            lbl = str(r.get('label', '')).lower()
            
            if raw >= 8:
                if lbl in ['outdoor','bathroom','other','balcony']: raw = 0.0
                elif lbl == 'bedroom': raw = 3.0
            
            scores.append(raw)
        
        pad = 15 - len(tensors)
        if pad > 0:
            tensors += [torch.zeros(3, self.img_size, self.img_size)] * pad
            scores += [-100.0] * pad
            
        return torch.stack(tensors), torch.tensor(scores, dtype=torch.float32), torch.tensor(len(selected))