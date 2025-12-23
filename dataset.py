import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torchvision import transforms
import random
import os

class PropertyPreferenceDataset(Dataset):
    def __init__(self, df, is_train=False, img_size=336):
        self.img_size = img_size
        self.is_train = is_train
        self.pairs = []
        
        self.normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), 
            std=(0.229, 0.224, 0.225)
        )

        if not df.empty:
            grouped = df.groupby('group_id')
            
            for _, group in grouped:
                if len(group) < 2: continue
                
                records = group.to_dict('records')
                n = len(records)
                
                for i in range(n):
                    for j in range(n):
                        if i == j: continue
                        
                        s_i = records[i]['score']
                        s_j = records[j]['score']
                        
                        if s_i > s_j:
                            self.pairs.append({
                                'win_path': records[i]['file_path'],
                                'lose_path': records[j]['file_path'],
                                'score_diff': s_i - s_j
                            })

    def _letterbox_image(self, img_path):
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            return torch.zeros(3, self.img_size, self.img_size)

        w, h = img.size
        scale = self.img_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)

        canvas = Image.new('RGB', (self.img_size, self.img_size), (0, 0, 0))
        
        x_offset = (self.img_size - new_w) // 2
        y_offset = (self.img_size - new_h) // 2
        canvas.paste(img, (x_offset, y_offset))

        return canvas

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        
        img_win = self._letterbox_image(item['win_path'])
        img_lose = self._letterbox_image(item['lose_path'])

        if self.is_train:
            if random.random() > 0.5:
                img_win = transforms.functional.hflip(img_win)
            if random.random() > 0.5:
                img_lose = transforms.functional.hflip(img_lose)

        t_win = transforms.functional.to_tensor(img_win)
        t_lose = transforms.functional.to_tensor(img_lose)
        
        t_win = self.normalize(t_win)
        t_lose = self.normalize(t_lose)

        return t_win, t_lose, torch.tensor(1.0, dtype=torch.float32)
