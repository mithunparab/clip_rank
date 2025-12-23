import torch
import torch.nn as nn
import mobileclip
from huggingface_hub import hf_hub_download

class MobileCLIPRanker(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        ckpt = self._download_weights(cfg.model.name)
        print(f"Loading Backbone from {ckpt}...")
        model, _, _ = mobileclip.create_model_and_transforms(cfg.model.name, pretrained=ckpt)
        self.backbone = model.image_encoder
        self.backbone_dim = 512 
        
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.project = nn.Linear(self.backbone_dim, 256)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, 
            nhead=4, 
            dim_feedforward=512, 
            dropout=0.1, 
            batch_first=True
        )
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.scorer = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        
        self.apply(self._init_weights)

    def _download_weights(self, model_name):
        return hf_hub_download(repo_id="apple/MobileCLIP-B", filename="mobileclip_b.pt")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval() 
        return self

    def forward(self, x, mask=None):
        b, g, c, h, w = x.shape
        
        x_flat = x.view(b * g, c, h, w)
        with torch.no_grad():
            features = self.backbone(x_flat)
            
        features = features.view(b, g, -1)
        x_ctx = self.project(features)
        
        if mask is not None:
            padding_mask = (mask == 0)
        else:
            padding_mask = None
            
        x_ctx = self.context_encoder(x_ctx, src_key_padding_mask=padding_mask)
        scores = self.scorer(x_ctx)
        
        return scores.squeeze(-1)
