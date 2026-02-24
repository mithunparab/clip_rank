import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os
import random
import pandas as pd


def _remap_score(raw, label):
    """Apply label-based score remapping."""
    lbl = str(label).lower()
    if raw >= 8:
        if lbl in ['outdoor', 'bathroom', 'other', 'balcony']:
            return 0.0
        if lbl == 'bedroom':
            return 3.0
    return raw


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
                if len(group) < 2:
                    continue
                self.groups.append(group.to_dict('records'))

    def _process(self, path):
        try:
            with Image.open(path) as img:
                return self.process(img.convert('RGB'))
        except:
            return torch.zeros(3, self.img_size, self.img_size)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        records = self.groups[idx]
        selected = records[:15]

        tensors = []
        scores = []

        for r in selected:
            tensors.append(self._process(r['file_path']))
            scores.append(_remap_score(float(r['score']), r.get('label', '')))

        pad = 15 - len(tensors)
        if pad > 0:
            tensors += [torch.zeros(3, self.img_size, self.img_size)] * pad
            scores += [-100.0] * pad

        return torch.stack(tensors), torch.tensor(scores, dtype=torch.float32), torch.tensor(len(selected))


class CachedFeatureDataset(Dataset):
    """Loads precomputed backbone features instead of images. ~10-50x faster."""

    def __init__(self, df, cache_dir="cached_features", max_per_group=15):
        self.cache_dir = cache_dir
        self.max_per_group = max_per_group
        self.feat_dim = None

        df = df.copy()
        if 'file_path' not in df.columns:
            df['file_path'] = df.index.map(lambda x: f"images/{x}.jpg")

        # Filter to rows with cached features
        df['_cache_path'] = df['file_path'].apply(self._cache_path)
        df = df[df['_cache_path'].apply(os.path.exists)]

        self.groups = []
        for _, group in df.groupby('group_id'):
            if len(group) < 2:
                continue
            self.groups.append(group.to_dict('records'))

        # Detect feature dim from first cached file
        if self.groups:
            first_path = self.groups[0][0]['_cache_path']
            self.feat_dim = torch.load(first_path, weights_only=True).shape[-1]

    def _cache_path(self, file_path):
        basename = os.path.splitext(os.path.basename(file_path))[0]
        return os.path.join(self.cache_dir, f"{basename}.pt")

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        records = self.groups[idx][:self.max_per_group]

        features = []
        scores = []

        for r in records:
            try:
                feat = torch.load(r['_cache_path'], weights_only=True)
                features.append(feat)
                scores.append(_remap_score(float(r['score']), r.get('label', '')))
            except Exception:
                continue

        valid_len = len(features)
        if valid_len < 2:
            # Return a dummy that will be skipped
            pad_feat = torch.zeros(self.max_per_group, self.feat_dim or 768)
            pad_scores = torch.full((self.max_per_group,), -100.0)
            return pad_feat, pad_scores, torch.tensor(0)

        pad = self.max_per_group - valid_len
        if pad > 0:
            features += [torch.zeros_like(features[0])] * pad
            scores += [-100.0] * pad

        return torch.stack(features), torch.tensor(scores, dtype=torch.float32), torch.tensor(valid_len)