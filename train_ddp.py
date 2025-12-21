import torch
import torch.nn as nn
import torch.nn.functional as F
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
            
        self.anchors = nn.Parameter(torch.randn(cfg.model.num_anchors, dim))
        
        nn.init.orthogonal_(self.anchors)

    def forward(self, x):
        img_feats = self.backbone(x)
        img_feats = F.normalize(img_feats, dim=-1)
        
        anchor_feats = F.normalize(self.anchors, dim=-1)
        
        sims = img_feats @ anchor_feats.T
        
        best_sim, _ = sims.max(dim=1)
        
        return best_sim