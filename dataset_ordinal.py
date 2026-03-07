import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os

# Thresholds at every integer from -9 to 9 (K=19)
# For score s, ordinal_label[k] = 1 if s > THRESHOLDS[k] else 0
THRESHOLDS = list(range(-9, 10))  # [-9, -8, ..., 9]
NUM_THRESHOLDS = len(THRESHOLDS)   # 19


class OrdinalImageDataset(Dataset):
    """Pointwise dataset — each image is an independent sample with ordinal labels."""

    def __init__(self, df, images_dir="images", is_train=False, img_size=224,
                 norm_mean=(0.481, 0.457, 0.408), norm_std=(0.268, 0.261, 0.275)):
        self.img_size = img_size

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
        # Ordinal labels: 1 if score > threshold_k
        ordinal = torch.tensor([1.0 if score > t else 0.0 for t in THRESHOLDS])
        return tensor, ordinal, score
