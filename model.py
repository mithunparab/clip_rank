import torch
import torch.nn as nn
import mobileclip
from huggingface_hub import hf_hub_download

class MobileCLIPRanker(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        ckpt_path, self.model_name = self._download_weights(cfg.model.name)
        
        print(f"Loading {self.model_name} from {ckpt_path}...")
        
        try:
            model, _, _ = mobileclip.create_model_and_transforms(self.model_name, pretrained=ckpt_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            alt_name = self.model_name.replace("-", "_").lower()
            model, _, _ = mobileclip.create_model_and_transforms(alt_name, pretrained=ckpt_path)

        self.backbone = model.image_encoder
        self.backbone_dim = 512 
        
        if "l14" in self.model_name.lower() or "l-14" in self.model_name.lower():
            self.backbone_dim = 768 
            
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        params_to_train = list(self.backbone.named_parameters())[-40:] 
        print(f"Unfreezing {len(params_to_train)} backbone parameters...")
        for name, param in params_to_train:
            param.requires_grad = True
            
        self.head = nn.Linear(self.backbone_dim, 1)
        
    def _download_weights(self, config_name):
        """
        Maps simple config names to HuggingFace Repos & Filenames
        """
        name = config_name.lower()
        
        if "mobileclip2_l14" in name or "mobileclip2-l-14" in name:
            repo_id = "apple/MobileCLIP2-L-14"
            filename = "mobileclip2_l14.pt"
            model_arch_name = "mobileclip2_l14"
            
        elif "mobileclip_s3" in name:
            repo_id = "apple/MobileCLIP-S3"
            filename = "mobileclip_s3.pt"
            model_arch_name = "mobileclip_s3"
            
        elif "mobileclip2_b" in name:
            repo_id = "apple/MobileCLIP2-B"
            filename = "mobileclip2_b.pt"
            model_arch_name = "mobileclip2_b"
            
        elif "s0" in name:
            repo_id = "apple/MobileCLIP-S0"
            filename = "mobileclip_s0.pt"
            model_arch_name = "mobileclip_s0"
            
        else:
            repo_id = "apple/MobileCLIP-B"
            filename = "mobileclip_b.pt"
            model_arch_name = "mobileclip_b"
            
        print(f"Downloading {filename} from {repo_id}...")
        return hf_hub_download(repo_id=repo_id, filename=filename), model_arch_name

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