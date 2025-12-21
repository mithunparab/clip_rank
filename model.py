import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import mobileclip
import requests
from io import BytesIO

class PropertyPreferenceDataset(Dataset):
    def __init__(self, csv_path_or_df, model_name='mobileclip_s2', pretrained_path=None, img_size=224):
        _, _, self.preprocess = mobileclip.create_model_and_transforms(model_name, pretrained=pretrained_path)
        self.img_size = img_size
        self.pairs = []
        
        if isinstance(csv_path_or_df, str):
            df = pd.read_csv(csv_path_or_df)
        else:
            df = csv_path_or_df

        if df.empty:
            return

        if 'group_id' in df.columns and 'label' in df.columns:
            groups = df.groupby(['group_id', 'label'])
            
            for _, group in groups:
                records = group.to_dict('records')
                n = len(records)
                if n < 2: continue
                    
                for i in range(n):
                    for j in range(n):
                        if i == j: continue
                        
                        score_a = records[i]['score']
                        score_b = records[j]['score']
                        
                        if score_a > score_b:
                            diff = score_a - score_b
                            self.pairs.append({
                                'win': records[i],
                                'lose': records[j],
                                'weight': diff
                            })

    def __len__(self):
        return len(self.pairs)

    def _download_image(self, url):
        try:
            resp = requests.get(url, timeout=3)
            img = Image.open(BytesIO(resp.content)).convert('RGB')
            return img
        except:
            return Image.new('RGB', (self.img_size, self.img_size))

    def _letterbox_process(self, img_pil):
        w, h = img_pil.size
        scale = self.img_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        
        img_resized = img_pil.resize((new_w, new_h), Image.Resampling.BILINEAR)
        
        background = Image.new('RGB', (self.img_size, self.img_size), (0, 0, 0))
        offset = ((self.img_size - new_w) // 2, (self.img_size - new_h) // 2)
        background.paste(img_resized, offset)
        
        return self.preprocess(background)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        
        win_pil = self._download_image(item['win']['url'])
        lose_pil = self._download_image(item['lose']['url'])
        
        win_tensor = self._letterbox_process(win_pil)
        lose_tensor = self._letterbox_process(lose_pil)
        
        weight = torch.tensor(item['weight'], dtype=torch.float32)
        
        return win_tensor, lose_tensor, weight