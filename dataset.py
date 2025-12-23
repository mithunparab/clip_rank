import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torchvision import transforms
import random
import os

class PropertyPreferenceDataset(Dataset):
    def __init__(self, df, images_dir="images", is_train=False, img_size=336):
        self.img_size = img_size
        self.is_train = is_train
        self.pairs = []
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

        # 2. ANCHOR-BEST STRATEGY
        if not self.df.empty:
            grouped = self.df.groupby('group_id')
            
            for group_id, group in grouped:
                if len(group) < 2: continue
                
                records = group.to_dict('records')
                
                max_score = max(r['score'] for r in records)
                
                anchors = [r for r in records if r['score'] >= (max_score - 0.5)]
                

                others = [r for r in records if r['score'] <= (max_score - 2.0)]
                
                if not anchors or not others:
                    continue
                
                for win in anchors:
                    for lose in others:
                        self.pairs.append({
                            'win_path': win['file_path'],
                            'lose_path': lose['file_path'],
                            'score_diff': win['score'] - lose['score']
                        })

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
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        img_win = self._letterbox_image(item['win_path'])
        img_lose = self._letterbox_image(item['lose_path'])

        if self.is_train:
            if random.random() > 0.5: img_win = transforms.functional.hflip(img_win)
            if random.random() > 0.5: img_lose = transforms.functional.hflip(img_lose)

        t_win = self.normalize(transforms.functional.to_tensor(img_win))
        t_lose = self.normalize(transforms.functional.to_tensor(img_lose))

        return t_win, t_lose, torch.tensor(1.0, dtype=torch.float32)