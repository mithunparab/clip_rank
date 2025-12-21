import torch
import torch.nn as nn
import mobileclip

class MobileCLIPRanker(nn.Module):
    def __init__(self, model_name='mobileclip_s2', pretrained_path=None):
        super().__init__()
        
        self.backbone, _, _ = mobileclip.create_model_and_transforms(model_name, pretrained=pretrained_path)
        
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
            dummy = torch.zeros(1, 3, 224, 224)
            dim = self.backbone.encode_image(dummy).shape[1]
            
        self.score_head = nn.Linear(dim, 1, bias=False)

    def forward(self, x):
        self.backbone.train() 
        
        features = self.backbone.encode_image(x)
        features = features / features.norm(dim=-1, keepdim=True)
        
        score = self.score_head(features)
        return score
