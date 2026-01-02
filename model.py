import torch
import torch.nn as nn
import mobileclip
from huggingface_hub import hf_hub_download

class MobileCLIPRanker(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        repo = "apple/MobileCLIP-B" if "b" in cfg.model.name else "apple/MobileCLIP-S0"
        filename = f"{cfg.model.name}.pt"
        ckpt = hf_hub_download(repo_id=repo, filename=filename)
        
        print(f"Loading {filename}...")
        model, _, _ = mobileclip.create_model_and_transforms(cfg.model.name, pretrained=ckpt)
        self.backbone = model.image_encoder
        self.backbone_dim = 512 
        
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        params_to_train = list(self.backbone.named_parameters())[-30:] 
        print(f"Unfreezing {len(params_to_train)} parameters...")
        for name, param in params_to_train:
            param.requires_grad = True
            
        self.head = nn.Linear(self.backbone_dim, 1)
        
    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x, valid_lens=None):
        b, g, c, h, w = x.shape
        x_flat = x.view(b * g, c, h, w)
        
        features = self.backbone(x_flat) 
        features = features.view(b, g, -1)
        scores = self.head(features)
        
        return scores