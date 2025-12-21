import torch
import torch.nn as nn
import mobileclip

class MobileCLIPRanker(nn.Module):
    def __init__(self, model_name='mobileclip_s2', pretrained_path=None):
        super().__init__()
        full_model, _, _ = mobileclip.create_model_and_transforms(model_name, pretrained=pretrained_path)
        
        self.backbone = full_model.image_encoder

        for param in self.backbone.parameters():
            param.requires_grad = False
            
        parameters_to_train = []
        for name, param in list(self.backbone.named_parameters())[::-1]:
            if any(x in name for x in ['head', 'projector', 'fc', 'layer_scale', 'norm']):
                param.requires_grad = True
                parameters_to_train.append(name)
            
            if len(parameters_to_train) > 10: 
                break
        
        with torch.no_grad():
            self.backbone.eval()
            dummy = torch.zeros(1, 3, 224, 224)
            dim = self.backbone(dummy).shape[1]
            
        self.score_head = nn.Linear(dim, 1, bias=False)

    def train(self, mode=True):
        """
        Critical Override:
        When train() is called, we keep the backbone in eval mode 
        to freeze BatchNorm statistics.
        """
        super().train(mode)
        if mode:
            self.backbone.eval()
        return self

    def forward(self, x):
        self.backbone.eval()
        
        features = self.backbone(x)
        features = features / features.norm(dim=-1, keepdim=True)
        score = self.score_head(features)
        return score