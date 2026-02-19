"""1-D CNN for IDS feature sequences.

Input shape: (batch, 1, num_features)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """1-D convolutional network with configurable depth.

    Args:
        input_dim: Length of the feature vector (F).
        num_classes: Number of traffic classes.
        channels: Sequence of output channel counts per conv block.
        kernel_size: Convolution kernel width.
        dropout: Dropout probability before the classifier head.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        channels: list[int] = (64, 128, 256),
        kernel_size: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        conv_blocks: list[nn.Module] = []
        in_ch = 1
        for out_ch in channels:
            conv_blocks += [
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*conv_blocks)

        # Compute flattened size after pooling
        dummy = torch.zeros(1, 1, input_dim)
        flat_dim = self.conv(dummy).numel()

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x))
