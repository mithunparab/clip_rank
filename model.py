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


class IndependentHead(nn.Module):
    """Per-image MLP head — scores each image independently.

    No cross-image interaction. Each image gets a score based only on
    its own features. Transfers better to production because the score
    doesn't depend on what other images happen to be in the group.
    """
    def __init__(self, in_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, valid_lens=None):
        # x: (B, G, in_dim) → (B, G, 1)
        return self.mlp(self.norm(x))


class RankingHead(nn.Module):
    """Cross-image self-attention + MLP head.

    Scores each image relative to the group context rather than
    independently. Equivalent to one transformer encoder layer over
    the group, followed by a per-image MLP projection.

    Warning: boosts validation GoldAcc but hurts real-world accuracy
    because attention patterns are group-composition-dependent and
    don't transfer to production groups with different characteristics.
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

        name = cfg.model.name.lower()
        if name not in VARIANTS:
            raise ValueError(f"Unknown model '{name}'. Choose from: {list(VARIANTS.keys())}")

        v = VARIANTS[name]
        img_size = getattr(cfg.data, "img_size", 224)
        print(f"Initializing {name} backbone at {img_size}x{img_size}...")

        if v["loader"] == "transformers_clip":
            from transformers import CLIPVisionModel
            vision_model = CLIPVisionModel.from_pretrained(v["repo"])
            self.backbone = HFBackboneWrapper(vision_model)
        elif v["loader"] == "transformers":
            from transformers import AutoModel
            hf_model = AutoModel.from_pretrained(v["repo"])
            self.backbone = HFBackboneWrapper(hf_model)
        elif v["loader"] == "open_clip_hub":
            # Standard OpenCLIP models — downloads pretrained weights automatically.
            # force_image_size triggers position-embedding interpolation for non-224 input.
            model, _, _ = open_clip.create_model_and_transforms(
                v["arch"], pretrained=v["pretrained"], force_image_size=img_size
            )
            self.backbone = model.visual
        elif v["loader"] == "pixio":
            from transformers import AutoModel
            hf_model = AutoModel.from_pretrained(v["repo"])
            self.backbone = PixioBackboneWrapper(hf_model)
        elif v["loader"] == "birder":
            import birder
            from pathlib import Path
            # Ensure birder's model cache dir exists as a directory
            models_dir = Path("models")
            if models_dir.exists() and not models_dir.is_dir():
                models_dir.unlink()
            models_dir.mkdir(parents=True, exist_ok=True)
            net, _ = birder.load_pretrained_model(v["birder_name"], inference=True)
            self.backbone = BirderBackboneWrapper(net)
        elif v["loader"] == "open_clip":
            ckpt_path = hf_hub_download(repo_id=v["repo"], filename=v["file"])
            # force_image_size: resize patch_embed + interpolate pos_embed to the target.
            # MobileCLIP's timm ViT has strict_img_size asserts, so this is required for
            # anything other than 224.
            model, _, _ = open_clip.create_model_and_transforms(
                v["arch"], pretrained=ckpt_path, force_image_size=img_size
            )
            self.backbone = model.visual
        else:
            import mobileclip
            ckpt_path = hf_hub_download(repo_id=v["repo"], filename=v["file"])
            model, _, _ = mobileclip.create_model_and_transforms(v["arch"], pretrained=ckpt_path)
            self.backbone = model.image_encoder

        # Auto-detect backbone output dim instead of trusting registry
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            self.backbone_dim = self.backbone(dummy).shape[-1]
        print(f"  backbone_dim={self.backbone_dim}")

        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        unfreeze = getattr(cfg.model, "unfreeze_last", 60)
        params_to_train = list(self.backbone.named_parameters())[-unfreeze:]
        for pname, param in params_to_train:
            param.requires_grad = True

        # Track parent modules of unfrozen params so train() can re-enable dropout
        self._unfrozen_modules = self._find_unfrozen_modules()

        head_hidden = getattr(cfg.model, "head_hidden_dim", 256)
        head_dropout = getattr(cfg.model, "head_dropout", 0.1)
        use_attention = getattr(cfg.model, "use_attention", False)

        if use_attention:
            self.head = RankingHead(self.backbone_dim, head_hidden, head_dropout)
        else:
            self.head = IndependentHead(self.backbone_dim, head_hidden, head_dropout)

    def _find_unfrozen_modules(self):
        """Find the highest-level backbone submodules containing unfrozen params."""
        unfrozen = set()
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                # Walk up to the top-level child of backbone
                top = name.split('.')[0]
                child = getattr(self.backbone, top, None)
                if child is not None and isinstance(child, nn.Module):
                    unfrozen.add(child)
        return list(unfrozen)

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
