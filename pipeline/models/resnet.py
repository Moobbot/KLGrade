from torch import nn
import torchvision.models as models
from .base import BaseClassifier


class ResNet50Classifier(BaseClassifier):
    def __init__(self, num_classes: int, in_channels: int = 3, pretrained: bool = True):
        super().__init__(num_classes=num_classes, in_channels=in_channels)
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        if in_channels == 1:
            self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feats, num_classes)

    def forward(self, x):
        return self.backbone(x)


class ResNet101Classifier(BaseClassifier):
    def __init__(self, num_classes: int, in_channels: int = 3, pretrained: bool = True):
        super().__init__(num_classes=num_classes, in_channels=in_channels)
        weights = models.ResNet101_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet101(weights=weights)
        if in_channels == 1:
            self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feats, num_classes)

    def forward(self, x):
        return self.backbone(x)
