import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import mobileclip

class PropertyPreferenceDataset(Dataset):
    def __init__(self, csv_path_or_df, model_name='mobileclip_s2', pretrained_path=None, img_size=224):
        _, _, self.preprocess = mobileclip.create_model_and_transforms(model_name, pretrained=pretrained_path)
        self.img_size = img_size
        self.pairs = []
        
        if isinstance(csv_path_or_df, str):
            df = pd.read_csv(csv_path_or_df)
        else:
            df = csv_path_or_df

        if df.empty: return

        df['file_path'] = df.index.map(lambda x: f"images/{x}.jpg")
        
        initial_count = len(df)
        df = df[df['file_path'].apply(os.path.exists)]
        final_count = len(df)
        
        if final_count == 0:
            print(f"WARNING: No local images found! Expected 'images/0.jpg', etc.")
            print(f"Current Working Directory: {os.getcwd()}")
            print(f"Did you run prepare_data.py?")
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
                                'win_path': records[i]['file_path'],
                                'lose_path': records[j]['file_path'],
                                'weight': diff
                            })
        
        print(f"Dataset Loaded: {len(self.pairs)} pairs from {final_count} images.")

    def __len__(self):
        return len(self.pairs)

    def _load_local(self, path):
        try:
            img = Image.open(path).convert('RGB')
        except:
            return Image.new('RGB', (self.img_size, self.img_size))
            
        w, h = img.size
        scale = self.img_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        background = Image.new('RGB', (self.img_size, self.img_size), (0, 0, 0))
        offset = ((self.img_size - new_w) // 2, (self.img_size - new_h) // 2)
        background.paste(img_resized, offset)
        
        return self.preprocess(background)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        
        win_tensor = self._load_local(item['win_path'])
        lose_tensor = self._load_local(item['lose_path'])
        weight = torch.tensor(item['weight'], dtype=torch.float32)
        
        return win_tensor, lose_tensor, weight