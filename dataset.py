import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
from torchvision import transforms

class PropertyPreferenceDataset(Dataset):
    def __init__(self, csv_path_or_df, model_name='mobileclip_b', pretrained_path=None, img_size=336, is_train=False, max_group_size=6):
        self.img_size = img_size
        self.is_train = is_train
        self.max_group_size = max_group_size
        self.groups = []
        
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        
        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
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
            grouped = df.groupby(['group_id', 'label'])
            for _, g in grouped:
                recs = g.to_dict('records')
                if len(recs) < 2: continue
                
                self.groups.append(recs)

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
        group_recs = self.groups[idx]
        
        scores = [r['score'] for r in group_recs]
        max_score = max(scores)
        
        if self.is_train:
            import random
            random.shuffle(group_recs)
            
        selected_recs = group_recs[:self.max_group_size]
        
        images = []
        target_idx = -1
        
        current_max = -1
        
        for i, rec in enumerate(selected_recs):
            img = self._load_local(rec['file_path'])
            images.append(img)
            
            if rec['score'] > current_max:
                current_max = rec['score']
                target_idx = i
                
        real_len = len(images)
        pad_len = self.max_group_size - real_len
        if pad_len > 0:
            pad_tensor = torch.zeros(3, self.img_size, self.img_size)
            for _ in range(pad_len):
                images.append(pad_tensor)
                
        img_stack = torch.stack(images)
        
        mask = torch.cat([torch.ones(real_len), torch.zeros(pad_len)])
        
        return img_stack, torch.tensor(target_idx, dtype=torch.long), mask

    def __len__(self):
        return len(self.groups)