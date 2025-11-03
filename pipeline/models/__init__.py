import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import CLASSES
from .cnn_basic import CNNBasic
from .resnet import ResNet50Classifier, ResNet101Classifier
from .efficientnet import EfficientNetB0Classifier


REGISTRY = {
    "cnn_basic": CNNBasic,
    "resnet50": ResNet50Classifier,
    "resnet101": ResNet101Classifier,
    "efficientnet_b0": EfficientNetB0Classifier,
}


def build_model(name: str = "resnet50", pretrained: bool = True, in_channels: int = 3):
    num_classes = len(CLASSES)
    cls = REGISTRY.get(name)
    if cls is None:
        # fallback to resnet50
        cls = ResNet50Classifier
    # Not all models use 'pretrained' or 'in_channels' the same way; pass what they accept
    try:
        return cls(num_classes=num_classes, in_channels=in_channels, pretrained=pretrained)
    except TypeError:
        try:
            return cls(num_classes=num_classes, in_channels=in_channels)
        except TypeError:
            return cls(num_classes=num_classes)
