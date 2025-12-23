import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
from torchvision import transforms

class PropertyPreferenceDataset(Dataset):
    def __init__(self, csv_path_or_df, model_name='mobileclip_b', pretrained_path=None, img_size=336, is_train=False):
        self.img_size = img_size
        self.is_train = is_train
        self.pairs = []
        
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        
        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

        if isinstance(csv_path_or_df, str):
            df = pd.read_csv(csv_path_or_df)
        else:
            df = csv_path_or_df

        if df.empty: return

        df['file_path'] = df.index.map(lambda x: f"images/{x}.jpg")
        df = df[df['file_path'].apply(os.path.exists)]
        
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
                        
                        # Only pairs where A > B
                        if score_a > score_b:
                            diff = (score_a - score_b) / 10.0 # Normalize 0.0-1.0
                            
                            self.pairs.append({
                                'win_path': records[i]['file_path'],
                                'lose_path': records[j]['file_path'],
                                'diff': diff
                            })
                            
    def _load_local(self, path):
        try:
            img = Image.open(path).convert('RGB')
        except:
            return torch.zeros(3, self.img_size, self.img_size)
            
        w, h = img.size
        scale = self.img_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
        background = Image.new('RGB', (self.img_size, self.img_size), (0, 0, 0))
        offset = ((self.img_size - new_w) // 2, (self.img_size - new_h) // 2)
        background.paste(img_resized, offset)
        
        return self.transform(background)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        win = self._load_local(item['win_path'])
        lose = self._load_local(item['lose_path'])
        diff = torch.tensor(item['diff'], dtype=torch.float32)
        return win, lose, diff

    def __len__(self):
        return len(self.pairs)