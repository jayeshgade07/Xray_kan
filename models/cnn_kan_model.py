import torch.nn as nn
from .cnn_backbone import CNNBackbone
from .kan_layer import KANLinear

class CNNBaseline(nn.Module):
    """ Model 1: Standard CNN + Linear Layer """
    def __init__(self, num_classes=14, use_pretrained=True):
        super().__init__()
        self.backbone = CNNBackbone(use_pretrained)
        self.classifier = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

class CNNDense(nn.Module):
    """ Model 2: Standard CNN + Dense MLP Layer (High param count) """
    def __init__(self, num_classes=14, use_pretrained=True):
        super().__init__()
        self.backbone = CNNBackbone(use_pretrained)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

class CNNKAN(nn.Module):
    """ Model 3: CNN + Kolmogorov-Arnold Network Layer (Proposed) """
    def __init__(self, num_classes=14, use_pretrained=True):
        super().__init__()
        self.backbone = CNNBackbone(use_pretrained)
        # Use KAN Linear to map directly to logits
        self.classifier = KANLinear(in_features=1024, out_features=num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
