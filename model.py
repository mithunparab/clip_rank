import torch
import torch.nn as nn
import os
import mobileclip
from huggingface_hub import hf_hub_download

class MobileCLIPRanker(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        ckpt_path = self._download_weights(cfg.model.name)
        print(f"Loading Backbone from {ckpt_path}...")
        model, _, _ = mobileclip.create_model_and_transforms(
            cfg.model.name, 
            pretrained=ckpt_path
        )
        self.backbone = model.image_encoder
        self.backbone_dim = 512 
        
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.head = nn.Sequential(
            nn.Dropout(p=cfg.model.dropout),
            nn.Linear(self.backbone_dim, cfg.model.head_hidden_dim),
            nn.LayerNorm(cfg.model.head_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.model.head_hidden_dim, 1) 
        )
        
        self.apply(self._init_weights)

    def _download_weights(self, model_name):
        repo_id = "apple/MobileCLIP-B"
        filename = "mobileclip_b.pt"
        try:
            return hf_hub_download(repo_id=repo_id, filename=filename)
        except:
            raise RuntimeError("Failed to download MobileCLIP weights")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x):
        # Listwise Input: [Batch, GroupSize, 3, H, W]
        b, g, c, h, w = x.shape
        
        # Flatten: [B*G, 3, H, W]
        x_flat = x.view(b * g, c, h, w)
        
        with torch.no_grad():
            features = self.backbone(x_flat) # [B*G, 512]
            
        scores = self.head(features) # [B*G, 1]
        
        # Reshape: [Batch, GroupSize]
        scores = scores.view(b, g)
        return scores