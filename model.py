import torch
import torch.nn as nn
import open_clip
from huggingface_hub import hf_hub_download

# Normalization constants
CLIP_NORM = ((0.481, 0.457, 0.408), (0.268, 0.261, 0.275))
IMAGENET_NORM = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

# Variant registry: name -> (repo_id, filename, arch, embed_dim, loader)
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
    "dinov3_convnext_large": {
        "repo": "facebook/dinov3-convnext-large-pretrain-lvd1689m",
        "loader": "transformers",
    },
    "mobileclip2_s4": {
        "repo": "apple/MobileCLIP2-S4",
        "file": "mobileclip2_s4.pt",
        "arch": "MobileCLIP2-S4",
        "dim": 512,
        "loader": "open_clip",
    },
    # Standard OpenCLIP ViT-B models — proper ViTs, ~4x faster than L14 on CPU
    "vit_b16": {
        "arch": "ViT-B-16",
        "pretrained": "laion2b_s34b_b88k",
        "dim": 512,
        "loader": "open_clip_hub",
    },
    "vit_b32": {
        "arch": "ViT-B-32",
        "pretrained": "laion2b_s34b_b79k",
        "dim": 512,
        "loader": "open_clip_hub",
    },
    "vit_b16_dfn": {
        "arch": "ViT-B-16",
        "pretrained": "dfn2b",
        "dim": 512,
        "loader": "open_clip_hub",
    },
    # ConvNeXt — pure convolution, much faster than ViT on CPU
    "convnext_large_22k": {
        "repo": "facebook/convnext-large-224-22k-1k",
        "dim": 1536,
        "loader": "transformers",
    },
    # MetaCLIP 2 — Meta's ViT-L/14 with worldwide data curation
    "metaclip2_l14": {
        "repo": "facebook/metaclip-2-worldwide-l14",
        "dim": 768,
        "loader": "transformers_clip",
    },
}


def get_norm_stats(model_name):
    """Return (mean, std) normalization tuples appropriate for the model."""
    v = VARIANTS.get(model_name.lower(), {})
    if v.get("loader") == "transformers":
        return IMAGENET_NORM
    # CLIP and CLIP-derived models (open_clip, transformers_clip)
    return CLIP_NORM


class HFBackboneWrapper(nn.Module):
    """Wraps a HuggingFace model to match the CLIP backbone interface:
    (B, C, H, W) tensor -> (B, dim) embeddings via pooler_output."""

    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        return outputs.pooler_output


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

        if v["loader"] == "transformers_clip":
            from transformers import CLIPVisionModel
            vision_model = CLIPVisionModel.from_pretrained(v["repo"])
            self.backbone = HFBackboneWrapper(vision_model)
        elif v["loader"] == "transformers":
            from transformers import AutoModel
            hf_model = AutoModel.from_pretrained(v["repo"])
            self.backbone = HFBackboneWrapper(hf_model)
        elif v["loader"] == "open_clip_hub":
            # Standard OpenCLIP models — downloads pretrained weights automatically
            model, _, _ = open_clip.create_model_and_transforms(v["arch"], pretrained=v["pretrained"])
            self.backbone = model.visual
        elif v["loader"] == "open_clip":
            ckpt_path = hf_hub_download(repo_id=v["repo"], filename=v["file"])
            model, _, _ = open_clip.create_model_and_transforms(v["arch"], pretrained=ckpt_path)
            self.backbone = model.visual
        else:
            import mobileclip
            ckpt_path = hf_hub_download(repo_id=v["repo"], filename=v["file"])
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
