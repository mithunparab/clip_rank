import torch
import torch.nn as nn
import mobileclip

class MobileCLIPRanker(nn.Module):
    def __init__(self, head_hidden_dim=512, dropout=0.2):
        super().__init__()
        
        model, _, _ = mobileclip.create_model_and_transforms('mobileclip_b', pretrained='s13b')
        self.backbone = model.image_encoder
        
        self.backbone_dim = 512 
        
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.backbone_dim, head_hidden_dim),
            nn.LayerNorm(head_hidden_dim),
            nn.GELU(),
            nn.Linear(head_hidden_dim, 1)
        )
        
    def train(self, mode=True):
        """
        Overridden to keep Backbone in eval mode (Frozen BatchNorm)
        while allowing Gradients for fine-tuning.
        """
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x):

        features = self.backbone(x)
        
        score = self.head(features)
        
        return score