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
    # OpenAI CLIP ViT-L/14 — the original, canonical CLIP
    "clip_vit_l14": {
        "repo": "openai/clip-vit-large-patch14",
        "dim": 768,
        "loader": "transformers_clip",
    },
    # Aesthetics predictor V2 — uses same CLIP ViT-L/14 backbone (linearMSE
    # only trains a head, does not fine-tune CLIP weights). Equivalent to
    # clip_vit_l14 for our purposes.
    "aesthetics_l14": {
        "repo": "openai/clip-vit-large-patch14",
        "dim": 768,
        "loader": "transformers_clip",
    },
    # Birder CLIP ViT-L/14 — OpenAI CLIP loaded via birder framework
    "birder_vit_l14": {
        "birder_name": "vit_l14_pn_quick_gelu_openai-clip",
        "dim": 768,
        "loader": "birder",
    },
    # Pixio ViT-L/16 — Facebook's MAE-based vision encoder (2B images)
    "pixio_vitl16": {
        "repo": "facebook/pixio-vitl16",
        "dim": 1024,
        "loader": "pixio",
    },
}


def get_norm_stats(model_name):
    """Return (mean, std) normalization tuples appropriate for the model."""
    v = VARIANTS.get(model_name.lower(), {})
    if v.get("loader") in ("transformers", "pixio"):
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


class PixioBackboneWrapper(nn.Module):
    """Wraps Pixio model: mean-pool over 8 class tokens -> (B, 768)."""

    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        # First 8 tokens are class tokens
        class_tokens = outputs.last_hidden_state[:, :8, :]
        return class_tokens.mean(dim=1)


class BirderBackboneWrapper(nn.Module):
    """Wraps a birder model: (B, C, H, W) tensor -> (B, dim) projections."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)


def _build_backbone(cfg):
    """Shared backbone construction for all ranker variants."""
    name = cfg.model.name.lower()
    if name not in VARIANTS:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(VARIANTS.keys())}")

    v = VARIANTS[name]
    print(f"Initializing {name} backbone...")

    if v["loader"] == "transformers_clip":
        from transformers import CLIPVisionModel
        vision_model = CLIPVisionModel.from_pretrained(v["repo"])
        backbone = HFBackboneWrapper(vision_model)
    elif v["loader"] == "transformers":
        from transformers import AutoModel
        hf_model = AutoModel.from_pretrained(v["repo"])
        backbone = HFBackboneWrapper(hf_model)
    elif v["loader"] == "open_clip_hub":
        model, _, _ = open_clip.create_model_and_transforms(v["arch"], pretrained=v["pretrained"])
        backbone = model.visual
    elif v["loader"] == "pixio":
        from transformers import AutoModel
        hf_model = AutoModel.from_pretrained(v["repo"])
        backbone = PixioBackboneWrapper(hf_model)
    elif v["loader"] == "birder":
        import birder
        from pathlib import Path
        models_dir = Path("models")
        if models_dir.exists() and not models_dir.is_dir():
            models_dir.unlink()
        models_dir.mkdir(parents=True, exist_ok=True)
        net, _ = birder.load_pretrained_model(v["birder_name"], inference=True)
        backbone = BirderBackboneWrapper(net)
    elif v["loader"] == "open_clip":
        ckpt_path = hf_hub_download(repo_id=v["repo"], filename=v["file"])
        model, _, _ = open_clip.create_model_and_transforms(v["arch"], pretrained=ckpt_path)
        backbone = model.visual
    else:
        import mobileclip
        ckpt_path = hf_hub_download(repo_id=v["repo"], filename=v["file"])
        model, _, _ = mobileclip.create_model_and_transforms(v["arch"], pretrained=ckpt_path)
        backbone = model.image_encoder

    # Auto-detect backbone output dim
    img_size = getattr(cfg.data, "img_size", 224)
    with torch.no_grad():
        dummy = torch.zeros(1, 3, img_size, img_size)
        backbone_dim = backbone(dummy).shape[-1]
    print(f"  backbone_dim={backbone_dim}")

    # Freeze all, then unfreeze last N params
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False

    unfreeze = getattr(cfg.model, "unfreeze_last", 60)
    params_to_train = list(backbone.named_parameters())[-unfreeze:]
    for pname, param in params_to_train:
        param.requires_grad = True

    return backbone, backbone_dim


def _find_unfrozen_modules(backbone):
    """Find the highest-level backbone submodules containing unfrozen params."""
    unfrozen = set()
    for name, param in backbone.named_parameters():
        if param.requires_grad:
            top = name.split('.')[0]
            child = getattr(backbone, top, None)
            if child is not None and isinstance(child, nn.Module):
                unfrozen.add(child)
    return list(unfrozen)


class OrdinalHead(nn.Module):
    """CORAL ordinal regression head.

    Shared linear weights with per-threshold biases enforce ordinal
    consistency: P(score > k) >= P(score > k+1) is guaranteed because
    all thresholds share the same projection direction, differing only
    in bias (intercept).

    Score = sum(sigmoid(logits)) gives a continuous ranking score in [0, K].
    """
    def __init__(self, in_dim, num_thresholds=19, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.num_thresholds = num_thresholds
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # CORAL: shared linear projection + per-threshold bias
        self.linear = nn.Linear(hidden_dim, 1, bias=False)
        self.biases = nn.Parameter(torch.zeros(num_thresholds))

    def forward(self, x):
        """x: (B, in_dim) -> (B, K) logits"""
        h = self.proj(x)
        return self.linear(h) + self.biases  # (B, K) via broadcast


class OrdinalRanker(nn.Module):
    """Pointwise ordinal regression ranker using CORAL."""

    def __init__(self, cfg):
        super().__init__()
        self.backbone, self.backbone_dim = _build_backbone(cfg)
        self._unfrozen_modules = _find_unfrozen_modules(self.backbone)

        num_thresholds = getattr(cfg.model, "num_thresholds", 19)
        head_hidden = getattr(cfg.model, "head_hidden_dim", 256)
        head_dropout = getattr(cfg.model, "head_dropout", 0.1)
        self.head = OrdinalHead(self.backbone_dim, num_thresholds, head_hidden, head_dropout)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        if mode:
            for module in self._unfrozen_modules:
                module.train()
                for sub in module.modules():
                    if isinstance(sub, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
                        sub.eval()
        return self

    def forward(self, x):
        """x: (B, C, H, W) or (B, G, C, H, W) -> logits"""
        if x.dim() == 5:
            b, g, c, h, w = x.shape
            x = x.view(b * g, c, h, w)
            features = self.backbone(x)
            logits = self.head(features)
            return logits.view(b, g, -1)  # (B, G, K)
        else:
            features = self.backbone(x)
            return self.head(features)  # (B, K)

    def score(self, x):
        """Continuous ranking score = sum of sigmoid(logits)."""
        logits = self.forward(x)
        return torch.sigmoid(logits).sum(dim=-1)  # (B,) or (B, G)


# ---- Label Distribution Learning (LDL) ----

# Score values: -10, -9, ..., 10 (21 bins)
SCORE_VALUES = torch.arange(-10, 11, dtype=torch.float32)  # (21,)
NUM_BINS = 21


class LDLHead(nn.Module):
    """Label Distribution Learning head.

    Predicts a probability distribution over 21 score bins (-10 to 10).
    Final ranking score = expected value = sum(score_k * P(score_k)).
    """
    def __init__(self, in_dim, num_bins=21, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.num_bins = num_bins
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_bins),
        )

    def forward(self, x):
        """x: (B, in_dim) -> (B, 21) raw logits"""
        return self.net(x)


class LDLRanker(nn.Module):
    """Pointwise Label Distribution Learning ranker.

    Predicts a full probability distribution over score values.
    Trained with KL divergence against Gaussian soft targets.
    """

    def __init__(self, cfg):
        super().__init__()
        self.backbone, self.backbone_dim = _build_backbone(cfg)
        self._unfrozen_modules = _find_unfrozen_modules(self.backbone)

        head_hidden = getattr(cfg.model, "head_hidden_dim", 256)
        head_dropout = getattr(cfg.model, "head_dropout", 0.1)
        self.head = LDLHead(self.backbone_dim, NUM_BINS, head_hidden, head_dropout)
        # Register score values as buffer so they move with the model
        self.register_buffer('score_values', SCORE_VALUES.clone())

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        if mode:
            for module in self._unfrozen_modules:
                module.train()
                for sub in module.modules():
                    if isinstance(sub, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
                        sub.eval()
        return self

    def forward(self, x):
        """x: (B, C, H, W) or (B, G, C, H, W) -> (B, 21) or (B, G, 21) logits"""
        if x.dim() == 5:
            b, g, c, h, w = x.shape
            x = x.view(b * g, c, h, w)
            features = self.backbone(x)
            logits = self.head(features)
            return logits.view(b, g, -1)
        else:
            features = self.backbone(x)
            return self.head(features)

    def score(self, x):
        """Expected value: sum(score_k * P(score_k)) -> continuous ranking score."""
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        return (probs * self.score_values).sum(dim=-1)


class RankingHead(nn.Module):
    """Cross-image self-attention + MLP head.

    Scores each image relative to the group context rather than
    independently. Equivalent to one transformer encoder layer over
    the group, followed by a per-image MLP projection.

    Human annotators judge images comparatively — this makes the model
    do the same. A mediocre photo looks worse when the group contains
    a gold image; the attention lets the model detect this contrast.
    """
    def __init__(self, in_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        # All backbone dims (512, 768, 1024, 1536) are divisible by 8
        n_heads = 8
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = nn.MultiheadAttention(
            in_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Zero-init the attention output projection so attn_out=0 at epoch 0.
        # The residual x + attn_out = x initially → identical to the old
        # independent MLP head. Attention activates gradually as it learns
        # meaningful cross-image comparisons. Without this, random attention
        # corrupts backbone features and causes a performance drop.
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, x, valid_lens=None):
        # x: (B, G, in_dim)
        B, G, _ = x.shape

        # Mask padded positions so valid images don't attend to padding
        key_padding_mask = None
        if valid_lens is not None:
            key_padding_mask = (
                torch.arange(G, device=x.device).unsqueeze(0)
                >= valid_lens.to(x.device).unsqueeze(1)
            )  # True = ignore

        normed = self.norm1(x)
        attn_out, _ = self.attn(
            normed, normed, normed, key_padding_mask=key_padding_mask
        )
        x = x + attn_out           # residual
        return self.mlp(self.norm2(x))  # (B, G, 1)


class MobileCLIPRanker(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone, self.backbone_dim = _build_backbone(cfg)
        self._unfrozen_modules = _find_unfrozen_modules(self.backbone)

        head_hidden = getattr(cfg.model, "head_hidden_dim", 256)
        head_dropout = getattr(cfg.model, "head_dropout", 0.1)
        self.head = RankingHead(self.backbone_dim, head_hidden, head_dropout)

    def train(self, mode=True):
        super().train(mode)
        # Keep entire backbone in eval first (frozen layers stay eval)
        self.backbone.eval()
        if mode:
            # Re-enable dropout in unfrozen blocks by setting them to train,
            # then selectively put their norm layers back to eval.
            for module in self._unfrozen_modules:
                module.train()
                for sub in module.modules():
                    if isinstance(sub, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
                        sub.eval()
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

        scores = self.head(features, valid_lens=valid_lens)
        return scores
