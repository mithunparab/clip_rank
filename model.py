import torch
import torch.nn as nn
import mobileclip
import open_clip
from huggingface_hub import hf_hub_download

class MobileCLIPRanker(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.model_type = "mobileclip"
        
        if "l14" in cfg.model.name.lower() or "l-14" in cfg.model.name.lower():
            self.model_type = "open_clip"
            repo_id = "apple/MobileCLIP2-L-14"
            filename = "mobileclip2_l14.pt"
            ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
            
            model, _, _ = open_clip.create_model_and_transforms(
                'MobileCLIP2-L-14', 
                pretrained=ckpt_path
            )
            self.backbone = model.visual
            self.backbone_dim = 768
            
        else:
            if "s3" in cfg.model.name.lower():
                repo_id = "apple/MobileCLIP-S3"
                filename = "mobileclip_s3.pt"
                arch = "mobileclip_s3"
            elif "s0" in cfg.model.name.lower():
                repo_id = "apple/MobileCLIP-S0"
                filename = "mobileclip_s0.pt"
                arch = "mobileclip_s0"
            else:
                repo_id = "apple/MobileCLIP2-B"
                filename = "mobileclip2_b.pt"
                arch = "mobileclip_b"
                
            ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
            model, _, _ = mobileclip.create_model_and_transforms(arch, pretrained=ckpt_path)
            self.backbone = model.image_encoder
            self.backbone_dim = 512

        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        params_to_train = list(self.backbone.named_parameters())[-60:] 
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
        
        with torch.no_grad():
            if self.model_type == "open_clip":
                features = self.backbone(x_flat) 
            else:
                features = self.backbone(x_flat)
            
        features = features.view(b, g, -1)
        scores = self.head(features)
        
        return scores