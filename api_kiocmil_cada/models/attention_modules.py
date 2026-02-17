"""
Context-Aware Deformable Attention Modules

Implements deformable attention mechanisms for KIOCMIL architecture,
inspired by CDT-CAD paper for medical imaging.

Key components:
1. DeformableAttention: Learnable spatial sampling with deformable offsets
2. PositionalEncoding: Encode spatial information from bboxes
3. CrossAttentionWithDeformable: Cross-attention with adaptive sampling
4. ContextEncoder: Multi-scale spatial feature extraction
5. LesionInstanceAggregation: Learned attention pooling for lesion instances
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List


class PositionalEncoding(nn.Module):
    """
    Encode spatial bbox information into embeddings.

    Supports:
    - Absolute position encoding (bbox center)
    - Relative position encoding (bbox size relative to image)
    - Sinusoidal positional encoding (similar to Transformer PE)
    """

    def __init__(self, feature_dim: int, max_seq_len: int = 1000):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_seq_len = max_seq_len

        # Create sinusoidal encoding table
        pe = torch.zeros(max_seq_len, feature_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, feature_dim, 2).float() * -(math.log(10000.0) / feature_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        if feature_dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, bbox: torch.Tensor) -> torch.Tensor:
        """
        Encode bbox information into embedding.

        Args:
            bbox: (batch_size, 4) or (4,) tensor [cx, cy, w, h] (normalized 0-1)

        Returns:
            encoding: (batch_size, feature_dim) or (feature_dim,)
        """
        if bbox.dim() == 1:
            bbox = bbox.unsqueeze(0)

        batch_size = bbox.shape[0]

        # Normalize bbox values to 0-999 range for PE lookup
        bbox_normalized = (bbox * (self.max_seq_len - 1)).long()
        bbox_normalized = torch.clamp(bbox_normalized, 0, self.max_seq_len - 1)

        # Get PE for each bbox component
        encodings = []
        for i in range(4):  # cx, cy, w, h
            idx = bbox_normalized[:, i]
            encodings.append(self.pe[idx])  # (batch_size, feature_dim)

        # Stack and average across components
        stacked = torch.stack(encodings, dim=1)  # (batch_size, 4, feature_dim)
        encoding = stacked.mean(dim=1)  # (batch_size, feature_dim)

        return encoding


class DeformableAttention(nn.Module):
    """
    Deformable Attention Mechanism.

    Learns adaptive spatial sampling offsets to focus on relevant regions.

    Reference:
    - Deformable DETR (Zhu et al., ICLR 2021)
    - CDT-CAD for medical imaging
    """

    def __init__(
        self,
        feature_dim: int,
        num_points: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_points = num_points
        self.num_heads = num_heads

        # Generate offset grids (learnable)
        self.offset_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(
                feature_dim, num_points * 2
            ),  # (num_points, 2) for (dy, dx) offsets
        )

        # Attention weight generation
        self.weight_net = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, num_points),  # (num_points,)
        )

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)

    def bilinear_sample(
        self,
        feature_map: torch.Tensor,
        sampling_locations: torch.Tensor,
    ) -> torch.Tensor:
        """
        Bilinear interpolation sampling from feature map.

        Args:
            feature_map: (B, C, H, W) feature maps
            sampling_locations: (B, num_points, 2) normalized coordinates in [-1, 1]

        Returns:
            sampled_features: (B, num_points, C)
        """
        # Use grid_sample for bilinear interpolation
        # Convert normalized [-1, 1] to grid format
        grid = sampling_locations.unsqueeze(1)  # (B, 1, num_points, 2)

        # grid_sample expects (N, Tout, Hin, Win, 2) format
        sampled = F.grid_sample(
            feature_map,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )  # (B, C, 1, num_points)

        sampled = sampled.squeeze(2).permute(0, 2, 1)  # (B, num_points, C)
        return sampled

    def forward(
        self,
        query: torch.Tensor,
        feature_map: torch.Tensor,
        bbox_center: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Deformable attention forward pass.

        Args:
            query: (B, feature_dim) or (B, 1, feature_dim) query embedding
            feature_map: (B, C, H, W) spatial feature map
            bbox_center: (B, 2) normalized bbox center [cx, cy]

        Returns:
            output: (B, feature_dim) attended output
            weights: (B, num_points) attention weights
        """
        if query.dim() == 2:
            query = query.unsqueeze(1)  # (B, 1, feature_dim)

        B, _, C = query.shape
        H, W = feature_map.shape[-2:]

        # Generate offset grids relative to bbox center
        offsets = self.offset_net(query.squeeze(1))  # (B, num_points * 2)
        offsets = offsets.view(B, self.num_points, 2)  # (B, num_points, 2)

        # Normalize offsets to [-0.5, 0.5] of image dimensions
        offsets = torch.tanh(offsets) * 0.3  # Clip to reasonable range

        # Compute sampling locations: bbox_center + offsets
        # Convert bbox_center from [0, 1] to [-1, 1] for grid_sample
        bbox_center_normalized = (bbox_center * 2) - 1  # (B, 2)
        bbox_center_expanded = bbox_center_normalized.unsqueeze(1)  # (B, 1, 2)

        sampling_locations = bbox_center_expanded + offsets  # (B, num_points, 2)

        # Clamp to valid range [-1, 1]
        sampling_locations = torch.clamp(sampling_locations, -1, 1)

        # Sample features at computed locations
        sampled_features = self.bilinear_sample(
            feature_map, sampling_locations
        )  # (B, num_points, C)

        # Generate attention weights
        query_flat = query.squeeze(1)  # (B, C)
        sampled_mean = sampled_features.mean(dim=1)  # (B, C)
        combined = torch.cat([query_flat, sampled_mean], dim=-1)  # (B, 2C)
        weights = self.weight_net(combined)  # (B, num_points)
        weights = F.softmax(weights, dim=-1)  # (B, num_points)

        # Weighted aggregation
        weighted_features = (sampled_features * weights.unsqueeze(-1)).sum(
            dim=1
        )  # (B, C)

        return weighted_features, weights


class CrossAttentionWithDeformable(nn.Module):
    """
    Cross-attention between query (lesion) and context (feature map)
    with deformable spatial sampling.

    Allows lesion embeddings to adaptively attend to context regions.
    """

    def __init__(
        self,
        feature_dim: int,
        num_points: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_points = num_points
        self.num_heads = num_heads

        # Deformable attention for context sampling
        self.deformable_attn = DeformableAttention(
            feature_dim=feature_dim,
            num_points=num_points,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Cross-attention between lesion and context
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Residual connection and layer norm
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        lesion_query: torch.Tensor,
        context_feature_map: torch.Tensor,
        lesion_bbox: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with cross-attention and deformable sampling.

        Args:
            lesion_query: (B, feature_dim) lesion embedding
            context_feature_map: (B, C, H, W) spatial context features
            lesion_bbox: (B, 2) normalized lesion bbox center [cx, cy]

        Returns:
            output: (B, feature_dim) contextualized lesion embedding
            attn_weights: (B, num_points) deformable attention weights
        """
        # Step 1: Deformable attention to sample relevant context
        sampled_context, deform_weights = self.deformable_attn(
            query=lesion_query,
            feature_map=context_feature_map,
            bbox_center=lesion_bbox,
        )  # (B, C), (B, num_points)

        # Step 2: Residual connection
        lesion_query_expanded = lesion_query.unsqueeze(1)  # (B, 1, C)

        # Step 3: Cross-attention with context
        attn_output, attn_weights = self.cross_attn(
            query=lesion_query_expanded,
            key=lesion_query_expanded,  # Self-attention for now
            value=lesion_query_expanded,
        )

        # Residual + LayerNorm
        x = self.norm1(lesion_query + attn_output.squeeze(1))

        # Step 4: Feedforward
        ffn_output = self.ffn(x)

        # Residual + LayerNorm
        output = self.norm2(x + ffn_output)

        return output, deform_weights


class ContextEncoder(nn.Module):
    """
    Extract multi-scale spatial context features from full image.

    Produces feature maps at multiple scales for deformable attention.
    """

    def __init__(
        self,
        backbone: nn.Module,
        backbone_dim: int,
        feature_dim: int = 256,
        num_scales: int = 3,
    ):
        super().__init__()
        self.backbone = backbone
        self.backbone_dim = backbone_dim
        self.feature_dim = feature_dim
        self.num_scales = num_scales

        # Feature pyramid layers for multi-scale output
        self.pyramid_layers = nn.ModuleList(
            [
                nn.Conv2d(
                    backbone_dim,
                    feature_dim,
                    kernel_size=1,
                )
                for _ in range(num_scales)
            ]
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale context features.

        Args:
            x: (B, 3, H, W) input image

        Returns:
            features: List of (B, feature_dim, H_i, W_i) at different scales
        """
        # Extract backbone features
        backbone_features = self.backbone(x)

        if isinstance(backbone_features, (list, tuple)):
            # Multi-scale output
            features = backbone_features
        else:
            # Single-scale output, create pyramid
            features = [backbone_features]
            for _ in range(self.num_scales - 1):
                features.append(
                    F.adaptive_avg_pool2d(
                        features[-1],
                        output_size=(
                            features[-1].shape[2] // 2,
                            features[-1].shape[3] // 2,
                        ),
                    )
                )

        # Normalize to feature_dim and apply pyramid layers
        output_features = []
        for i, feat in enumerate(features[: self.num_scales]):
            feat_normalized = self.pyramid_layers[i](feat)
            output_features.append(feat_normalized)

        return output_features


class LesionInstanceAggregation(nn.Module):
    """
    Aggregate multiple lesion instances using learned attention weights.

    Instead of max pooling, learns adaptive weights for each instance.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim

        # Learnable query token for aggregation
        self.query_token = nn.Parameter(torch.randn(1, 1, feature_dim))

        # Attention mechanism
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Feature refinement
        self.refinement = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
        )

        self.norm = nn.LayerNorm(feature_dim)

    def forward(
        self, lesion_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregate multiple lesion instances.

        Args:
            lesion_features: (K, feature_dim) lesion embeddings (variable K)
                            or (B, K, feature_dim) for batched

        Returns:
            aggregated: (feature_dim,) or (B, feature_dim) aggregated features
            weights: (K,) or (B, K) attention weights
        """
        # Handle both single and batched inputs
        if lesion_features.dim() == 2:
            lesion_features = lesion_features.unsqueeze(0)  # (1, K, C)
            squeeze_output = True
        else:
            squeeze_output = False

        B, K, C = lesion_features.shape

        if K == 0:
            # No lesions, return zero
            device = lesion_features.device
            if squeeze_output:
                return torch.zeros(C, device=device), torch.zeros(1, device=device)
            else:
                return torch.zeros(B, C, device=device), torch.zeros(
                    B, 1, device=device
                )

        # Expand query token
        query = self.query_token.expand(B, -1, -1)  # (B, 1, C)

        # Multi-head attention
        attn_output, attn_weights = self.multihead_attn(
            query=query,
            key=lesion_features,
            value=lesion_features,
            need_weights=True,
            average_attn_weights=True,
        )  # (B, 1, C), (B, 1, K)

        # Refine features
        refined = self.refinement(attn_output)  # (B, 1, C)

        # Residual connection
        output = self.norm(attn_output + refined)  # (B, 1, C)

        # Squeeze and return
        output = output.squeeze(1)  # (B, C)
        weights = attn_weights.squeeze(1)  # (B, K)

        if squeeze_output:
            output = output.squeeze(0)  # (C,)
            weights = weights.squeeze(0)  # (K,)

        return output, weights


class FusionTransformer(nn.Module):
    """
    Fusion transformer to combine context and attended lesion embeddings.

    Uses self-attention to learn relationships between context and lesions.
    """

    def __init__(
        self,
        feature_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_layers = num_layers

        # Multi-layer transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    def forward(
        self,
        context: torch.Tensor,
        js_lesions: torch.Tensor,
        ost_lesions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse context and lesion embeddings.

        Args:
            context: (B, feature_dim) context embedding
            js_lesions: (B, feature_dim) aggregated JS lesion embedding
            ost_lesions: (B, feature_dim) aggregated Ost lesion embedding

        Returns:
            fused: (B, feature_dim) fused embedding
        """
        B = context.shape[0]

        # Stack into sequence: [context, js_lesions, ost_lesions]
        x = torch.stack([context, js_lesions, ost_lesions], dim=1)  # (B, 3, C)

        # Apply transformer
        output = self.transformer_encoder(x)  # (B, 3, C)

        # Average pooling across sequence
        fused = output.mean(dim=1)  # (B, C)

        return fused
