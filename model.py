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
            nn.Linear(self.backbone_dim * 2, cfg.model.head_hidden_dim),
            nn.LayerNorm(cfg.model.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=cfg.model.dropout),
            nn.Linear(cfg.model.head_hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1)
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

    def forward(self, x, valid_lens=None):
        b, g, c, h, w = x.shape
        x_flat = x.view(b * g, c, h, w)
        
        with torch.no_grad():
            features = self.backbone(x_flat)
            
        features = features.view(b, g, -1)
        
        if valid_lens is not None:
            mask = torch.arange(g, device=x.device).expand(b, g) < valid_lens.unsqueeze(1)
            mask = mask.unsqueeze(-1).float()
            
            sum_features = (features * mask).sum(dim=1, keepdim=True)
            mean_features = sum_features / valid_lens.view(b, 1, 1)
        else:
            mean_features = features.mean(dim=1, keepdim=True)
            
        diff_features = features - mean_features
        
        combined = torch.cat([features, diff_features], dim=2)
        
        scores = self.head(combined)
        return scores.squeeze(-1)
