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
        self.data = []
        
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
        
        for _, row in df.iterrows():
            self.data.append({
                'path': row['file_path'],
                'score': float(row['score']) / 10.0
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
        item = self.data[idx]
        img_tensor = self._load_local(item['path'])
        return img_tensor, torch.tensor(item['score'], dtype=torch.float32)

    def __len__(self):
        return len(self.data)