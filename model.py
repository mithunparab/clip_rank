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
            
        self.quality_anchor = nn.Parameter(torch.randn(1, dim))
        
        self.logit_scale = nn.Parameter(torch.ones([]) * 4.6052) 

    def forward(self, x):
        image_features = self.backbone(x)
        image_features = F.normalize(image_features, dim=-1)
        
        anchor = F.normalize(self.quality_anchor, dim=-1)

        similarity = (image_features @ anchor.T).squeeze(1)
        return similarity