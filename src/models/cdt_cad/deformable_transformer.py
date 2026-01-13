"""
Deformable Transformer components for CDT-CAD

Implements:
- Multi-Scale Deformable Attention
- Deformable Transformer Encoder/Decoder
- Positional Encoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, List
import math


class PositionEmbeddingSine(nn.Module):
    """
    Sine-cosine positional encoding for spatial features.
    """

    def __init__(
        self, num_pos_feats=128, temperature=10000, normalize=True, scale=None
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Feature tensor [B, C, H, W]
        Returns:
            pos: Positional encoding [B, num_pos_feats*2, H, W]
        """
        B, C, H, W = x.shape
        mask = torch.zeros((B, H, W), dtype=torch.bool, device=x.device)

        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos


class MultiScaleDeformableAttention(nn.Module):
    """
    Multi-Scale Deformable Attention Module

    This module computes attention over deformable sampling points instead of
    the entire feature map, making it computationally efficient.
    """

    def __init__(
        self, d_model: int = 256, n_levels: int = 4, n_heads: int = 8, n_points: int = 4
    ):
        """
        Args:
            d_model: Channel dimension of input features
            n_levels: Number of feature pyramid levels
            n_heads: Number of attention heads
            n_points: Number of sampling points per head per level
        """
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        # Learnable parameters
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters"""
        nn.init.constant_(self.sampling_offsets.weight.data, 0.0)

        # Initialize offsets in a grid pattern
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.n_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )

        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1

        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

        nn.init.constant_(self.attention_weights.weight.data, 0.0)
        nn.init.constant_(self.attention_weights.bias.data, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query: Tensor,
        reference_points: Tensor,
        input_flatten: Tensor,
        input_spatial_shapes: Tensor,
        input_level_start_index: Tensor,
    ) -> Tensor:
        """
        Args:
            query: Query features [B, Len_q, C]
            reference_points: Reference points [B, Len_q, n_levels, 2]
            input_flatten: Flattened multi-scale features [B, Len_in, C]
            input_spatial_shapes: Spatial shapes [n_levels, 2]
            input_level_start_index: Start index for each level [n_levels]

        Returns:
            output: Attended features [B, Len_q, C]
        """
        B, Len_q, _ = query.shape
        B, Len_in, _ = input_flatten.shape

        value = self.value_proj(input_flatten)
        value = value.view(B, Len_in, self.n_heads, self.d_model // self.n_heads)

        # Compute sampling offsets
        sampling_offsets = self.sampling_offsets(query).view(
            B, Len_q, self.n_heads, self.n_levels, self.n_points, 2
        )

        # Compute attention weights
        attention_weights = self.attention_weights(query).view(
            B, Len_q, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            B, Len_q, self.n_heads, self.n_levels, self.n_points
        )

        # Compute sampling locations
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        else:
            raise ValueError(
                f"Unsupported reference_points shape: {reference_points.shape}"
            )

        # Sample features
        output = self._sample_features(
            value,
            input_spatial_shapes,
            input_level_start_index,
            sampling_locations,
            attention_weights,
        )

        output = self.output_proj(output)

        return output

    def _sample_features(
        self,
        value: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        sampling_locations: Tensor,
        attention_weights: Tensor,
    ) -> Tensor:
        """Sample features at deformable locations using grid_sample"""
        B, _, n_heads, head_dim = value.shape
        _, Len_q, _, n_levels, n_points, _ = sampling_locations.shape

        # Split value by levels
        value_list = value.split([H * W for H, W in spatial_shapes], dim=1)
        sampling_grids = 2 * sampling_locations - 1

        sampling_value_list = []
        for level, (H, W) in enumerate(spatial_shapes):
            # Reshape value for grid_sample
            value_l = (
                value_list[level]
                .flatten(2)
                .view(B, H, W, n_heads * head_dim)
                .permute(0, 3, 1, 2)
            )

            # Get sampling grid for this level
            sampling_grid_l = (
                sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
            )
            # sampling_grid_l: [B * n_heads, Len_q, n_points, 2]
            sampling_grid_l = sampling_grid_l.view(B * n_heads, Len_q, n_points, 2)

            # Expand value for each head
            value_l = value_l.view(B, n_heads, head_dim, H, W).flatten(0, 1)

            # Sample using grid_sample
            sampling_value_l = F.grid_sample(
                value_l,
                sampling_grid_l,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            # sampling_value_l: [B * n_heads, head_dim, Len_q, n_points]
            sampling_value_list.append(sampling_value_l)

        # Aggregate across levels
        attention_weights = attention_weights.transpose(1, 2).reshape(
            B * n_heads, 1, Len_q, n_levels * n_points
        )

        output = torch.stack(sampling_value_list, dim=-2).flatten(-2)
        output = (output * attention_weights).sum(-1).view(B, n_heads * head_dim, Len_q)

        return output.transpose(1, 2).contiguous()


class DeformableTransformerEncoderLayer(nn.Module):
    """Single Deformable Transformer Encoder Layer"""

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        n_levels: int = 4,
        n_points: int = 4,
    ):
        super().__init__()

        # Multi-scale deformable attention
        self.self_attn = MultiScaleDeformableAttention(
            d_model, n_levels, n_heads, n_points
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        src: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
    ) -> Tensor:
        # Self-attention
        src2 = self.self_attn(
            src, reference_points, src, spatial_shapes, level_start_index
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feedforward
        src2 = self.linear2(self.dropout2(F.relu(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    """Deformable Transformer Encoder (stack of encoder layers)"""

    def __init__(
        self, encoder_layer: DeformableTransformerEncoderLayer, num_layers: int
    ):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(
        self,
        src: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
    ) -> Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, reference_points, spatial_shapes, level_start_index)
        return output


class DeformableTransformerDecoderLayer(nn.Module):
    """Single Deformable Transformer Decoder Layer"""

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        n_levels: int = 4,
        n_points: int = 4,
    ):
        super().__init__()

        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-attention (deformable)
        self.cross_attn = MultiScaleDeformableAttention(
            d_model, n_levels, n_heads, n_points
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: Tensor,
        reference_points: Tensor,
        memory: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
    ) -> Tensor:
        # Self-attention
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross-attention
        tgt2 = self.cross_attn(
            tgt, reference_points, memory, spatial_shapes, level_start_index
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Feedforward
        tgt2 = self.linear2(self.dropout3(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    """Deformable Transformer Decoder (stack of decoder layers)"""

    def __init__(
        self, decoder_layer: DeformableTransformerDecoderLayer, num_layers: int
    ):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(
        self,
        tgt: Tensor,
        reference_points: Tensor,
        memory: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
    ) -> Tensor:
        output = tgt
        for layer in self.layers:
            output = layer(
                output, reference_points, memory, spatial_shapes, level_start_index
            )
        return output
