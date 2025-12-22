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
            
        good_prompts = [
            "wide angle photo of an empty room with large open floor space",
            "room with large blank empty walls and visible corners",
            "perspective view of a long room with depth",
            "empty bedroom with a large window and clear floor area"
        ]
        
        bad_prompts = [
            "close up photo of furniture texture",
            "cluttered messy room with too many objects",
            "corner of a room facing a wall",
            "blurry low quality dark photo",
            "exterior photo of a house"
        ]
        
        with torch.no_grad():
            good_tokens = tokenizer(good_prompts)
            bad_tokens = tokenizer(bad_prompts)
            
            good_feats = full_model.text_encoder(good_tokens)
            bad_feats = full_model.text_encoder(bad_tokens)
            
            good_feats = F.normalize(good_feats, dim=-1)
            bad_feats = F.normalize(bad_feats, dim=-1)
            
        self.register_buffer("good_anchors", good_feats.float())
        self.register_buffer("bad_anchors", bad_feats.float())
        
        del full_model.text_encoder
        
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        img_feats = self.backbone(x)
        img_feats = F.normalize(img_feats, dim=-1)
        
        sim_good = (img_feats @ self.good_anchors.T)
        best_good, _ = sim_good.max(dim=1)
        
        sim_bad = (img_feats @ self.bad_anchors.T)
        best_bad, _ = sim_bad.max(dim=1)
        
        raw_score = best_good - best_bad
        
        final_score = torch.sigmoid((raw_score * self.scale) + self.bias)
        
        return final_score