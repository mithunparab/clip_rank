import torch
import torch.nn as nn
import mobileclip

class MobileCLIPRanker(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        full_model, _, _ = mobileclip.create_model_and_transforms(cfg.model.name)
        self.backbone = full_model.image_encoder
        
        for param in self.backbone.parameters():
            param.requires_grad = True
            
        with torch.no_grad():
            dummy = torch.zeros(1, 3, cfg.data.img_size, cfg.data.img_size)
            dim = self.backbone(dummy).shape[1]
            
        self.score_head = nn.Sequential(
            nn.Dropout(0.3), 
            nn.Linear(dim, cfg.model.head_hidden_dim),
            nn.LayerNorm(cfg.model.head_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.model.head_hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.backbone(x)
        return self.score_head(features) * 10.0