import torch
import torch.nn as nn
from torchvision import models

class CNNBackbone(nn.Module):
    """
    DenseNet121 feature extractor. Removes the final linear classification 
    layer and replaces it with Global Average Pooling to yield a 1024-d 
    feature vector for any given standard image input.
    """
    def __init__(self, use_pretrained=True):
        super(CNNBackbone, self).__init__()
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if use_pretrained else None
        densenet = models.densenet121(weights=weights)
        
        # DenseNet feature extraction backbone
        self.features = densenet.features
        
        # Global Average Pooling to reduce (batch, 1024, 7, 7) -> (batch, 1024, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        features = self.features(x)
        out = self.pool(features)
        out = torch.flatten(out, 1)
        return out
