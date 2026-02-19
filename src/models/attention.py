"""Attention modules that can be plugged into DNN or CNN backbones."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureAttention(nn.Module):
    """Soft feature-wise attention gate for tabular input.

    Produces a weight vector over input features, then returns the
    element-wise product with the input (attended representation).

    Args:
        input_dim: Number of input features.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.fc(x)
        return x * weights


class ChannelAttention1D(nn.Module):
    """Squeeze-and-Excitation style channel attention for 1-D CNN feature maps.

    Args:
        num_channels: Number of channels (C) in the feature map (B, C, L).
        reduction: Reduction ratio for the bottleneck.
    """

    def __init__(self, num_channels: int, reduction: int = 8):
        super().__init__()
        mid = max(num_channels // reduction, 1)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # (B, C, 1)
            nn.Flatten(),             # (B, C)
            nn.Linear(num_channels, mid),
            nn.ReLU(),
            nn.Linear(mid, num_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.se(x).unsqueeze(-1)  # (B, C, 1)
        return x * scale
