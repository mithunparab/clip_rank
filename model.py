import torch
import torch.nn as nn
import os
import mobileclip
from huggingface_hub import hf_hub_download

class MobileCLIPRanker(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        checkpoint_path = self._download_weights(cfg.model.name)
        
        print(f"Loading {cfg.model.name} from {checkpoint_path}...")
        model, _, _ = mobileclip.create_model_and_transforms(
            cfg.model.name, 
            pretrained=checkpoint_path
        )
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

    def _download_weights(self, model_name):
        """
        Downloads official Apple MobileCLIP weights from Hugging Face.
        Returns the local path to the .pt file.
        """
        repo_id = "apple/MobileCLIP-B"
        filename = "mobileclip_b.pt"
        
        if model_name != "mobileclip_b":
            raise ValueError(f"Auto-download not implemented for {model_name}. Use mobileclip_b.")

        try:
            print(f"Checking for weights: {repo_id}/{filename}...")
            cached_path = hf_hub_download(repo_id=repo_id, filename=filename)
            return cached_path
        except Exception as e:
            print(f"Error downloading weights: {e}")
            raise

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def train(self, mode=True):
        """
        Override the default train() to freeze the backbone during training.
        """
        super().train(mode)
        self.backbone.eval() 
        return self

    def forward(self, x):
        features = self.backbone(x)
        score = self.head(features)
        return score