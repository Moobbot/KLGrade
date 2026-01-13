"""
CDT-CAD: Context-Aware Deformable Transformers for Computer-Aided Detection

Main model architecture integrating:
- ResNet-50 backbone
- Iterative Context-Aware Feature Extractor
- Deformable Transformer Encoder/Decoder
- Classification and BBox regression heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Tuple, Optional
import torchvision
from torchvision.models import resnet50, ResNet50_Weights

from .deformable_transformer import (
    DeformableTransformerEncoder,
    DeformableTransformerDecoder,
    DeformableTransformerEncoderLayer,
    DeformableTransformerDecoderLayer,
    PositionEmbeddingSine,
)
from .feature_extractor import IterativeContextAwareFeatureExtractor


class CDTCAD(nn.Module):
    """
    CDT-CAD: Context-Aware Deformable Transformers for Computer-Aided Detection

    End-to-end object detection model for medical image analysis.
    """

    def __init__(
        self,
        num_classes: int = 5,
        num_queries: int = 100,
        hidden_dim: int = 256,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_feature_levels: int = 4,
        n_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        n_points: int = 4,
        dilation_rates: List[int] = [1, 2, 4, 8],
        num_iterations: int = 3,
        wavelet: str = "haar",
        pretrained_backbone: bool = True,
    ):
        """
        Args:
            num_classes: Number of object classes (e.g., 5 for KL0-KL4)
            num_queries: Number of object queries
            hidden_dim: Hidden dimension for transformer
            num_encoder_layers: Number of transformer encoder layers
            num_decoder_layers: Number of transformer decoder layers
            num_feature_levels: Number of feature pyramid levels
            n_heads: Number of attention heads
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            n_points: Number of sampling points for deformable attention
            dilation_rates: Dilation rates for DCE blocks
            num_iterations: Number of iterative refinements
            wavelet: Wavelet type for FP blocks
            pretrained_backbone: Use ImageNet pretrained weights
        """
        super().__init__()

        self.num_classes = num_classes
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.num_feature_levels = num_feature_levels

        # ===== 1. Backbone (ResNet-50) =====
        if pretrained_backbone:
            weights = ResNet50_Weights.IMAGENET1K_V2
        else:
            weights = None

        backbone = resnet50(weights=weights)

        # Extract feature extraction layers (C2, C3, C4, C5)
        self.backbone_layer1 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,  # C2: stride 4, channels 256
        )
        self.backbone_layer2 = backbone.layer2  # C3: stride 8, channels 512
        self.backbone_layer3 = backbone.layer3  # C4: stride 16, channels 1024
        self.backbone_layer4 = backbone.layer4  # C5: stride 32, channels 2048

        backbone_channels = [256, 512, 1024, 2048]

        # Freeze backbone batch norm layers (optional, recommended for fine-tuning)
        for module in [
            self.backbone_layer1,
            self.backbone_layer2,
            self.backbone_layer3,
            self.backbone_layer4,
        ]:
            for m in module.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    for param in m.parameters():
                        param.requires_grad = False

        # ===== 2. Iterative Context-Aware Feature Extractor =====
        self.feature_extractor = IterativeContextAwareFeatureExtractor(
            backbone_channels=backbone_channels,
            hidden_dim=hidden_dim,
            num_iterations=num_iterations,
            dilation_rates=dilation_rates,
            wavelet=wavelet,
        )

        # ===== 3. Positional Encoding =====
        self.position_embedding = PositionEmbeddingSine(hidden_dim // 2, normalize=True)

        # ===== 4. Deformable Transformer =====
        encoder_layer = DeformableTransformerEncoderLayer(
            d_model=hidden_dim,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            n_levels=num_feature_levels,
            n_points=n_points,
        )
        self.transformer_encoder = DeformableTransformerEncoder(
            encoder_layer, num_encoder_layers
        )

        decoder_layer = DeformableTransformerDecoderLayer(
            d_model=hidden_dim,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            n_levels=num_feature_levels,
            n_points=n_points,
        )
        self.transformer_decoder = DeformableTransformerDecoder(
            decoder_layer, num_decoder_layers
        )

        # ===== 5. Object Queries =====
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # ===== 6. Prediction Heads =====
        # Classification head
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for "no object"

        # Bounding box regression head
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)  # 4 coords (cx, cy, w, h)

        # Initialize prediction heads
        nn.init.constant_(self.class_embed.bias.data, 0)
        nn.init.xavier_uniform_(self.class_embed.weight.data)

    def forward(self, images: Tensor) -> Dict[str, Tensor]:
        """
        Forward pass

        Args:
            images: Input images [B, 3, H, W]

        Returns:
            outputs: Dictionary containing:
                - 'pred_logits': Class predictions [B, num_queries, num_classes+1]
                - 'pred_boxes': Bbox predictions [B, num_queries, 4]
        """
        B = images.shape[0]

        # ===== Extract backbone features =====
        c2 = self.backbone_layer1(images)
        c3 = self.backbone_layer2(c2)
        c4 = self.backbone_layer3(c3)
        c5 = self.backbone_layer4(c4)

        backbone_features = [c2, c3, c4, c5]

        # ===== Context-aware feature extraction =====
        refined_features = self.feature_extractor(backbone_features)

        # ===== Prepare features for transformer =====
        src_flatten = []
        spatial_shapes = []
        pos_embeds = []

        for feat in refined_features:
            # Flatten spatial dimensions
            B, C, H, W = feat.shape
            spatial_shapes.append((H, W))

            # Positional encoding
            pos_embed = self.position_embedding(feat)
            pos_embeds.append(pos_embed)

            # Flatten: [B, C, H, W] -> [B, H*W, C]
            feat_flat = feat.flatten(2).transpose(1, 2)
            src_flatten.append(feat_flat)

        # Concatenate all levels
        src = torch.cat(src_flatten, dim=1)  # [B, sum(H_i*W_i), C]
        pos = torch.cat([p.flatten(2).transpose(1, 2) for p in pos_embeds], dim=1)

        # Spatial shapes tensor
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=images.device
        )
        level_start_index = torch.cat(
            [spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]]
        )

        # Reference points (normalized grid centers for each level)
        reference_points = self._get_reference_points(
            spatial_shapes, device=images.device
        )

        # ===== Transformer Encoder =====
        memory = self.transformer_encoder(
            src + pos,  # Add positional encoding
            reference_points,
            spatial_shapes,
            level_start_index,
        )

        # ===== Transformer Decoder =====
        # Object queries
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(
            B, 1, 1
        )  # [B, num_queries, C]

        # Query reference points (learnable or initialized)
        tgt = torch.zeros_like(query_embed)
        query_reference_points = reference_points[
            :, : self.num_queries
        ]  # Use first N reference points

        hs = self.transformer_decoder(
            tgt, query_reference_points, memory, spatial_shapes, level_start_index
        )

        # ===== Prediction Heads =====
        outputs_class = self.class_embed(hs)  # [B, num_queries, num_classes+1]
        outputs_coord = self.bbox_embed(hs).sigmoid()  # [B, num_queries, 4]

        return {"pred_logits": outputs_class, "pred_boxes": outputs_coord}

    def _get_reference_points(
        self, spatial_shapes: Tensor, device: torch.device
    ) -> Tensor:
        """
        Generate reference points for multi-scale features

        Args:
            spatial_shapes: [num_levels, 2] (H, W for each level)
            device: Device to place tensors

        Returns:
            reference_points: [1, sum(H_i*W_i), num_levels, 2]
        """
        reference_points_list = []

        for lvl, (H, W) in enumerate(spatial_shapes):
            # Grid of reference points
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
                indexing="ij",
            )

            # Normalize to [0, 1]
            ref_y = ref_y.reshape(-1) / H
            ref_x = ref_x.reshape(-1) / W

            ref = torch.stack((ref_x, ref_y), dim=-1)  # [H*W, 2]
            reference_points_list.append(ref)

        reference_points = torch.cat(reference_points_list, dim=0)  # [sum(H_i*W_i), 2]
        reference_points = reference_points.unsqueeze(0)  # [1, sum(H_i*W_i), 2]

        # Expand to all levels
        reference_points = reference_points.unsqueeze(2).repeat(
            1, 1, len(spatial_shapes), 1
        )

        return reference_points


class MLP(nn.Module):
    """Multi-Layer Perceptron for bbox regression"""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
