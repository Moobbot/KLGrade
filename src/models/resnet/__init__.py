"""ResNet-based KIOCMIL models.

Base models using ResNet backbones.
"""

from .kiocmil_resnet import KiocmilModel, AttentionPool

__all__ = [
    "KiocmilModel",
    "AttentionPool",
]
