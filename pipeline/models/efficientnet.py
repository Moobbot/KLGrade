from torch import nn
import torchvision.models as models
from .base import BaseClassifier


class EfficientNetB0Classifier(BaseClassifier):
    def __init__(self, num_classes: int, in_channels: int = 3, pretrained: bool = True):
        super().__init__(num_classes=num_classes, in_channels=in_channels)
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        # EfficientNet-B0 expects 3ch; custom in_channels not handled here beyond 3ch
        in_feats = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_feats, num_classes)

    def forward(self, x):
        return self.backbone(x)
