import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os
import numpy as np

# Collapsed bins: 8 bins with ~900+ images each
# Maps raw scores (-10 to 10) to bin centers
BIN_EDGES = [
    (-10, -10),  # bin 0: reject (5037)
    (-9, -9),    # bin 1: bad (1306)
    (-8, -8),    # bin 2: poor (1601)
    (-7, -1),    # bin 3: below avg (902)
    (0, 0),      # bin 4: neutral (3320)
    (1, 6),      # bin 5: above avg (1237: 24+1105+23+5+6+49+25)
    (7, 7),      # bin 6: good (1439)
    (8, 10),     # bin 7: gold (834)
]
BIN_CENTERS = np.array([-10, -9, -8, -4, 0, 2, 7, 9], dtype=np.float32)
NUM_BINS = len(BIN_EDGES)


def score_to_bin(score):
    """Map raw score to bin index."""
    for i, (lo, hi) in enumerate(BIN_EDGES):
        if lo <= score <= hi:
            return i
    return 0 if score < -10 else NUM_BINS - 1


def make_target_distribution(score, sigma=1.0, gold_sigma=0.5):
    """Gaussian soft target over collapsed bins, centered at true score.

    Uses bin centers for distance calculation so the Gaussian spreads
    naturally across neighboring bins.
    """
    s = gold_sigma if score >= 7 else sigma
    dist = np.exp(-0.5 * ((BIN_CENTERS - score) / s) ** 2)
    dist /= dist.sum()
    return dist.astype(np.float32)


class LDLImageDataset(Dataset):
    """Pointwise dataset with Gaussian soft label distributions over collapsed bins."""

    def __init__(self, df, images_dir="images", is_train=False, img_size=224,
                 norm_mean=(0.481, 0.457, 0.408), norm_std=(0.268, 0.261, 0.275),
                 sigma=1.0, gold_sigma=0.5):
        self.img_size = img_size
        self.sigma = sigma
        self.gold_sigma = gold_sigma

        df = df.copy()
        if 'file_path' not in df.columns:
            df['file_path'] = df.index.map(lambda x: os.path.join(images_dir, f"{x}.jpg"))

        df = df[df['file_path'].apply(os.path.exists)]
        self.records = df[['file_path', 'score']].to_dict('records')

        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0),
                                             interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=norm_mean, std=norm_std)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=norm_mean, std=norm_std)
            ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        try:
            with Image.open(r['file_path']) as img:
                tensor = self.transform(img.convert('RGB'))
        except Exception:
            tensor = torch.zeros(3, self.img_size, self.img_size)

        score = float(r['score'])
        target_dist = make_target_distribution(score, self.sigma, self.gold_sigma)
        return tensor, torch.from_numpy(target_dist), torch.tensor(score, dtype=torch.float32)
