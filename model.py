import torch
import torch.nn as nn
import torch.nn.functional as F
import mobileclip

class MobileCLIPRanker(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        full_model, _, _ = mobileclip.create_model_and_transforms(cfg.model.name)
        tokenizer = mobileclip.get_tokenizer(cfg.model.name)
        
        self.backbone = full_model.image_encoder
        
        for param in self.backbone.parameters():
            param.requires_grad = True
            
        prompts = [
            "wide angle photo of an empty room with large open floor space",
            "room with large blank empty walls and visible corners",
            "empty bedroom with a large window and clear floor area",
            "spacious living room with a fireplace and high ceiling",
            "perspective view of a long room with depth and hardwood floor"
        ]
        
        with torch.no_grad():
            text_tokens = tokenizer(prompts)
            anchor_feats = full_model.text_encoder(text_tokens)
            anchor_feats = F.normalize(anchor_feats, dim=-1)
            
        self.register_buffer("anchors", anchor_feats.float())
        

        self.score_head = nn.Sequential(
            nn.Linear(len(prompts), 1),
            nn.Sigmoid() 
        )
        
        del full_model.text_encoder
        
        self.logit_scale = nn.Parameter(torch.ones([]) * 100.0) 

    def forward(self, x):
        img_feats = self.backbone(x)
        img_feats = F.normalize(img_feats, dim=-1)
        
        anchor_feats = self.anchors
        
        sims = img_feats @ anchor_feats.T
        
        score = self.score_head(sims * 10.0) 
        
        return score