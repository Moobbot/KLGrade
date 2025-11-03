"""
Base classifier definition to enforce a common interface across models.
"""
from torch import nn


class BaseClassifier(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

    def forward(self, x):  # pragma: no cover - interface only
        raise NotImplementedError("Forward must be implemented by subclasses")
