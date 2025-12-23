import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torchvision import transforms
import os
import numpy as np

class PropertyPreferenceDataset(Dataset):
    def __init__(self, df, images_dir="images", is_train=False, img_size=336, max_len=15):
        self.img_size = img_size
        self.max_len = max_len
        self.images_dir = images_dir
        
        self.df = df.copy()
        if 'file_path' not in self.df.columns:
            self.df['file_path'] = self.df.index.map(lambda x: os.path.join(self.images_dir, f"{x}.jpg"))

        valid_mask = self.df['file_path'].apply(os.path.exists)
        self.df = self.df[valid_mask]

        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), 
            std=(0.229, 0.224, 0.225)
        )

        self.groups = []
        if not self.df.empty:
            grouped = self.df.groupby('group_id')
            for g_id, group in grouped:
                if len(group) < 2: continue
                self.groups.append(group.to_dict('records'))

    def _letterbox_image(self, img_path):
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                w, h = img.size
                scale = self.img_size / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                img_resized = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
                canvas = Image.new('RGB', (self.img_size, self.img_size), (0, 0, 0))
                x_offset = (self.img_size - new_w) // 2
                y_offset = (self.img_size - new_h) // 2
                canvas.paste(img_resized, (x_offset, y_offset))
                return canvas
        except:
            return Image.new('RGB', (self.img_size, self.img_size), (0, 0, 0))

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        records = self.groups[idx]
        current_len = min(len(records), self.max_len)
        selected = records[:current_len]
        
        image_tensors = []
        scores = []
        
        for r in selected:
            img = self._letterbox_image(r['file_path'])
            t_img = transforms.functional.to_tensor(img)
            t_img = self.normalize(t_img)
            image_tensors.append(t_img)
            scores.append(float(r['score']))
            
        pad_len = self.max_len - current_len
        if pad_len > 0:
            for _ in range(pad_len):
                image_tensors.append(torch.zeros(3, self.img_size, self.img_size))
                scores.append(-100.0)
        
        return torch.stack(image_tensors), torch.tensor(scores, dtype=torch.float32), torch.tensor(current_len)