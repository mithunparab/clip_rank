import torch
import torch.nn as nn
import mobileclip

class MobileCLIPRanker(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        full_model, _, _ = mobileclip.create_model_and_transforms(cfg.model.name)
        self.backbone = full_model.image_encoder
        
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        with torch.no_grad():
            self.backbone.eval()
            dummy = torch.zeros(1, 3, cfg.data.img_size, cfg.data.img_size)
            dim = self.backbone(dummy).shape[1]
            
        self.score_head = nn.Sequential(
            nn.Linear(dim, cfg.model.head_hidden_dim),
            nn.LayerNorm(cfg.model.head_hidden_dim), 
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(cfg.model.head_hidden_dim, 1)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval() 
        return self

    def forward(self, x):
        self.backbone.eval()
        with torch.no_grad():
            features = self.backbone(x)
            features = features / features.norm(dim=-1, keepdim=True)
        
        return self.score_head(features)