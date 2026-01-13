"""
CDT-CAD Package: Context-Aware Deformable Transformers for Computer-Aided Detection

This package implements the CDT-CAD architecture for knee osteoarthritis detection.
"""

from .deformable_transformer import (
    MultiScaleDeformableAttention,
    DeformableTransformerEncoder,
    DeformableTransformerDecoder,
    PositionEmbeddingSine
)
from .feature_extractor import (
    DCEBlock,
    FPBlock,
    IterativeContextAwareFeatureExtractor
)
from .cdt_cad_model import CDTCAD

__all__ = [
    'MultiScaleDeformableAttention',
    'DeformableTransformerEncoder',
    'DeformableTransformerDecoder',
    'PositionEmbeddingSine',
    'DCEBlock',
    'FPBlock',
    'IterativeContextAwareFeatureExtractor',
    'CDTCAD'
]
