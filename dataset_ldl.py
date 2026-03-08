import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os
import numpy as np

# Score bins: -10, -9, ..., 10
SCORE_VALUES = np.arange(-10, 11, dtype=np.float32)  # 21 bins
NUM_BINS = 21


def make_target_distribution(score, sigma=1.0, gold_sigma=0.5):
    """Gaussian soft target centered at true score.

    Tighter sigma for gold tier (score >= 7) forces the model to be
    precise about hero image ordering. Wider sigma for lower scores
    allows more tolerance for annotator disagreement.
    """
    s = gold_sigma if score >= 7 else sigma
    dist = np.exp(-0.5 * ((SCORE_VALUES - score) / s) ** 2)
    dist /= dist.sum()
    return dist.astype(np.float32)


class LDLImageDataset(Dataset):
    """Pointwise dataset with Gaussian soft label distributions."""

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
        return tensor, torch.from_numpy(target_dist), score
