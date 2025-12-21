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
            
        parameters_to_train = []
        for name, param in list(self.backbone.named_parameters())[::-1]:
            if any(x in name for x in ['head', 'projector', 'fc', 'layer_scale', 'norm', 'resblocks.11']):
                param.requires_grad = True
                parameters_to_train.append(name)
            
            if len(parameters_to_train) > 20: 
                break
        
        with torch.no_grad():
            self.backbone.eval()
            dummy = torch.zeros(1, 3, cfg.data.img_size, cfg.data.img_size)
            dim = self.backbone(dummy).shape[1]
            
        self.score_head = nn.Sequential(
            nn.Linear(dim, cfg.model.head_hidden_dim),
            nn.BatchNorm1d(cfg.model.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(cfg.model.head_hidden_dim, 1)
        )

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval() 
        return self

    def forward(self, x):
        self.backbone.eval()
        features = self.backbone(x)
        return self.score_head(features)