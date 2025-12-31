import torch
import torch.nn as nn
import torch.nn.functional as F
import mobileclip
from huggingface_hub import hf_hub_download
import numpy as np

class MobileCLIPRanker(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        ckpt = self._download_weights(cfg.model.name)
        print(f"Loading Backbone from {ckpt}...")
        model, _, _ = mobileclip.create_model_and_transforms(cfg.model.name, pretrained=ckpt)
        self.backbone = model.image_encoder
        self.backbone_dim = 512 
        
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.projector = nn.Sequential(
            nn.Dropout(cfg.model.dropout), 
            nn.Linear(self.backbone_dim, self.backbone_dim),
            nn.LayerNorm(self.backbone_dim),
            nn.GELU()
        )
        
        self.ideal_vector = nn.Parameter(torch.randn(1, 1, self.backbone_dim))
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.apply(self._init_weights)

    def _download_weights(self, model_name):
        return hf_hub_download(repo_id="apple/MobileCLIP-B", filename="mobileclip_b.pt")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight) 
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval() 
        return self

    def forward(self, x, valid_lens=None):
        # x: [Batch, GroupSize, 3, H, W]
        b, g, c, h, w = x.shape
        x_flat = x.view(b * g, c, h, w)
        
        with torch.no_grad():
            features = self.backbone(x_flat) 
            
        features = features.view(b, g, -1)
        
        if valid_lens is not None:
            mask = torch.arange(g, device=x.device).expand(b, g) < valid_lens.unsqueeze(1)
            mask = mask.unsqueeze(-1).float()
            sum_features = (features * mask).sum(dim=1, keepdim=True)
            mean_features = sum_features / (valid_lens.view(b, 1, 1) + 1e-6)
        else:
            mean_features = features.mean(dim=1, keepdim=True)
            
        centered_features = features - mean_features 
        
        projected = self.projector(centered_features)
        
        projected_norm = F.normalize(projected, p=2, dim=2)
        ideal_norm = F.normalize(self.ideal_vector, p=2, dim=2)
        
        similarity = (projected_norm * ideal_norm).sum(dim=2)
        
        scores = similarity * self.logit_scale.exp()
        
        return scores