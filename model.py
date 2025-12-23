import torch
import torch.nn as nn
import mobileclip
from huggingface_hub import hf_hub_download

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
            
        self.head = nn.Sequential(
            nn.Dropout(p=cfg.model.dropout),
            nn.Linear(self.backbone_dim, cfg.model.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=cfg.model.dropout),
            nn.Linear(cfg.model.head_hidden_dim, 1)
        )
        
        self.apply(self._init_weights)

    def _download_weights(self, model_name):
        return hf_hub_download(repo_id="apple/MobileCLIP-B", filename="mobileclip_b.pt")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval() 
        return self

    def forward(self, x):
        # x: [Batch, GroupSize, 3, H, W]
        b, g, c, h, w = x.shape
        x_flat = x.view(b * g, c, h, w)
        
        with torch.no_grad():
            features = self.backbone(x_flat)
            
        scores = self.head(features)
        return scores.view(b, g)