"""
Context-Aware Feature Extractor components for CDT-CAD

Implements:
- DCE Block (Dilated Context Encoding)
- FP Block (Frequency Pooling with Wavelet Transform)
- Iterative Context-Aware Feature Extractor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple
import pywt
import numpy as np


class DCEBlock(nn.Module):
    """
    Dilated Context Encoding Block

    Uses dilated convolutions with multiple dilation rates to capture
    multi-scale context information without losing resolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation_rates: List[int] = [1, 2, 4, 8],
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            dilation_rates: List of dilation rates for multi-scale context
        """
        super().__init__()

        self.dilation_rates = dilation_rates

        # Dilated convolutions
        self.dilated_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels // len(dilation_rates),
                        kernel_size=3,
                        padding=rate,
                        dilation=rate,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels // len(dilation_rates)),
                    nn.ReLU(inplace=True),
                )
                for rate in dilation_rates
            ]
        )

        # 1x1 convolution to merge multi-scale features
        self.merge = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input features [B, C_in, H, W]
        Returns:
            out: Context-encoded features [B, C_out, H, W]
        """
        # Apply dilated convolutions in parallel
        features = [conv(x) for conv in self.dilated_convs]

        # Concatenate multi-scale features
        multi_scale = torch.cat(features, dim=1)

        # Merge features
        out = self.merge(multi_scale)

        # Add skip connection
        out = out + self.skip(x)

        return out


class FPBlock(nn.Module):
    """
    Frequency Pooling Block using Discrete Wavelet Transform

    Uses wavelet transform to decompose features into frequency components,
    helping to handle occlusions and scale variations.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        wavelet: str = "haar",
        pooling_factor: int = 2,
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            wavelet: Wavelet type ('haar', 'db1', 'db2', etc.)
            pooling_factor: Downsampling factor (2 for halving spatial dims)
        """
        super().__init__()

        self.wavelet = wavelet
        self.pooling_factor = pooling_factor

        # DWT produces 4 components: LL (low-low), LH, HL, HH
        # LL: low-frequency (approximation), others: high-frequency (details)
        dwt_channels = in_channels * 4

        # Channel reduction after DWT
        self.channel_reduce = nn.Sequential(
            nn.Conv2d(dwt_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Optional: Learnable weights for frequency components
        self.freq_weights = nn.Parameter(torch.ones(4) / 4)  # LL, LH, HL, HH

    def dwt2d(self, x: Tensor) -> Tensor:
        """
        Apply 2D Discrete Wavelet Transform

        Args:
            x: Input tensor [B, C, H, W]
        Returns:
            coeffs: Wavelet coefficients [B, C*4, H//2, W//2]
        """
        B, C, H, W = x.shape

        # Process each sample and channel
        coeffs_list = []

        for b in range(B):
            batch_coeffs = []
            for c in range(C):
                # Extract single channel
                channel_data = x[b, c].cpu().numpy()

                # Apply 2D DWT
                coeffs = pywt.dwt2(channel_data, self.wavelet)
                cA, (cH, cV, cD) = coeffs  # LL, LH, HL, HH

                # Stack frequency components
                freq_stack = np.stack([cA, cH, cV, cD], axis=0)  # [4, H//2, W//2]
                batch_coeffs.append(freq_stack)

            # Stack all channels
            batch_coeffs = np.stack(batch_coeffs, axis=0)  # [C, 4, H//2, W//2]
            coeffs_list.append(batch_coeffs)

        # Stack all batches
        coeffs_array = np.stack(coeffs_list, axis=0)  # [B, C, 4, H//2, W//2]

        # Convert back to tensor and reshape
        coeffs_tensor = torch.from_numpy(coeffs_array).to(x.device, dtype=x.dtype)
        coeffs_tensor = coeffs_tensor.reshape(B, C * 4, H // 2, W // 2)

        return coeffs_tensor

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input features [B, C_in, H, W]
        Returns:
            out: Frequency-pooled features [B, C_out, H//pooling_factor, W//pooling_factor]
        """
        # Apply DWT
        dwt_features = self.dwt2d(x)

        # Reduce channels
        out = self.channel_reduce(dwt_features)

        return out


class IterativeContextAwareFeatureExtractor(nn.Module):
    """
    Iterative Context-Aware Feature Extractor

    Combines DCE and FP blocks in an iterative manner to refine
    multi-scale features with both spatial context and frequency information.
    """

    def __init__(
        self,
        backbone_channels: List[int] = [
            256,
            512,
            1024,
            2048,
        ],  # ResNet-50 C2, C3, C4, C5
        hidden_dim: int = 256,
        num_iterations: int = 3,
        dilation_rates: List[int] = [1, 2, 4, 8],
        wavelet: str = "haar",
    ):
        """
        Args:
            backbone_channels: Channel dimensions from backbone feature maps
            hidden_dim: Hidden dimension for feature processing
            num_iterations: Number of iterative refinement stages
            dilation_rates: Dilation rates for DCE blocks
            wavelet: Wavelet type for FP blocks
        """
        super().__init__()

        self.num_iterations = num_iterations
        self.num_levels = len(backbone_channels)

        # Input projections to hidden_dim
        self.input_projections = nn.ModuleList(
            [nn.Conv2d(in_ch, hidden_dim, kernel_size=1) for in_ch in backbone_channels]
        )

        # Iterative refinement modules
        self.dce_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        DCEBlock(hidden_dim, hidden_dim, dilation_rates)
                        for _ in range(self.num_levels)
                    ]
                )
                for _ in range(num_iterations)
            ]
        )

        # FP blocks for downsampling (between levels)
        self.fp_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        FPBlock(hidden_dim, hidden_dim, wavelet)
                        for _ in range(self.num_levels - 1)  # No FP for finest level
                    ]
                )
                for _ in range(num_iterations)
            ]
        )

        # Top-down lateral connections
        self.lateral_convs = nn.ModuleList(
            [
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
                for _ in range(self.num_levels - 1)
            ]
        )

    def forward(self, features: List[Tensor]) -> List[Tensor]:
        """
        Args:
            features: List of feature maps from backbone [C2, C3, C4, C5]
                      Each: [B, C_i, H_i, W_i]

        Returns:
            refined_features: List of refined multi-scale features
                              Each: [B, hidden_dim, H_i, W_i]
        """
        # Project to hidden_dim
        current_features = [
            proj(feat) for proj, feat in zip(self.input_projections, features)
        ]

        # Iterative refinement
        for iter_idx in range(self.num_iterations):
            # Top-down pathway (coarse to fine)
            top_down = [None] * self.num_levels
            top_down[-1] = current_features[-1]  # Start from coarsest level

            for i in range(self.num_levels - 2, -1, -1):
                # Upsample coarser level
                upsampled = F.interpolate(
                    top_down[i + 1],
                    size=current_features[i].shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )

                # Lateral connection
                lateral = self.lateral_convs[i](current_features[i])

                # Merge
                top_down[i] = upsampled + lateral

            # Apply DCE blocks
            refined = []
            for level_idx, feat in enumerate(top_down):
                refined_feat = self.dce_blocks[iter_idx][level_idx](feat)
                refined.append(refined_feat)

            # Update current features
            current_features = refined

        return current_features
