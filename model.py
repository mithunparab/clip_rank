import torch
import torch.nn as nn
import mobileclip

class MobileCLIPRanker(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        print(f"Loading {cfg.model.name}...")
        model, _, _ = mobileclip.create_model_and_transforms(cfg.model.name, pretrained='s13b')
        self.backbone = model.image_encoder
        
        self.backbone_dim = 512 
        self.head = nn.Sequential(
            nn.Dropout(p=cfg.model.dropout),
            nn.Linear(self.backbone_dim, cfg.model.head_hidden_dim),
            nn.LayerNorm(cfg.model.head_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.model.head_hidden_dim, 1)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def train(self, mode=True):
        """
        CRITICAL: We must override train() to keep the backbone in eval mode.
        This freezes BatchNorm statistics. With batch_size=16, BN stats 
        would be too noisy and destroy the pre-trained features.
        """
        super().train(mode)
        self.backbone.eval() 
        return self

    def forward(self, x):
        features = self.backbone(x)
        
        score = self.head(features)
        
        return score