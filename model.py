import torch
import torch.nn as nn
import mobileclip
import open_clip
from huggingface_hub import hf_hub_download

# MobileCLIP2 variant registry: name -> (repo_id, filename, arch, embed_dim, loader)
VARIANTS = {
    "mobileclip2_s0": {
        "repo": "apple/MobileCLIP2-S0",
        "file": "mobileclip2_s0.pt",
        "arch": "MobileCLIP2-S0",
        "dim": 512,
        "loader": "open_clip",
    },
    "mobileclip2_s3": {
        "repo": "apple/MobileCLIP2-S3",
        "file": "mobileclip2_s3.pt",
        "arch": "MobileCLIP2-S3",
        "dim": 512,
        "loader": "open_clip",
    },
    "mobileclip2_b": {
        "repo": "apple/MobileCLIP2-B",
        "file": "mobileclip2_b.pt",
        "arch": "MobileCLIP2-B",
        "dim": 512,
        "loader": "open_clip",
    },
    "mobileclip2_l14": {
        "repo": "apple/MobileCLIP2-L-14",
        "file": "mobileclip2_l14.pt",
        "arch": "MobileCLIP2-L-14",
        "dim": 768,
        "loader": "open_clip",
    },
}


class RankingHead(nn.Module):
    """2-layer MLP head with dropout. More capacity than a single linear."""
    def __init__(self, in_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class MobileCLIPRanker(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        name = cfg.model.name.lower()
        if name not in VARIANTS:
            raise ValueError(f"Unknown model '{name}'. Choose from: {list(VARIANTS.keys())}")

        v = VARIANTS[name]
        print(f"Initializing {name} backbone...")
        ckpt_path = hf_hub_download(repo_id=v["repo"], filename=v["file"])

        if v["loader"] == "open_clip":
            model, _, _ = open_clip.create_model_and_transforms(v["arch"], pretrained=ckpt_path)
            self.backbone = model.visual
        else:
            model, _, _ = mobileclip.create_model_and_transforms(v["arch"], pretrained=ckpt_path)
            self.backbone = model.image_encoder

        # Auto-detect backbone output dim instead of trusting registry
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            self.backbone_dim = self.backbone(dummy).shape[-1]
        print(f"  backbone_dim={self.backbone_dim}")

        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        unfreeze = getattr(cfg.model, "unfreeze_last", 60)
        params_to_train = list(self.backbone.named_parameters())[-unfreeze:]
        for name, param in params_to_train:
            param.requires_grad = True

        head_hidden = getattr(cfg.model, "head_hidden_dim", 256)
        head_dropout = getattr(cfg.model, "head_dropout", 0.1)
        self.head = RankingHead(self.backbone_dim, head_hidden, head_dropout)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x, valid_lens=None):
        # x can be images (5D) or precomputed features (3D)
        if x.dim() == 5:
            b, g, c, h, w = x.shape
            x_flat = x.view(b * g, c, h, w)
            features = self.backbone(x_flat)
            features = features.view(b, g, -1)
        else:
            features = x

        scores = self.head(features)
        return scores
