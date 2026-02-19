"""Feedforward deep neural network for tabular IDS data."""
from __future__ import annotations

import torch
import torch.nn as nn


class DNN(nn.Module):
    """Fully-connected DNN with BatchNorm and Dropout.

    Args:
        input_dim: Number of input features.
        num_classes: Number of traffic classes.
        hidden_dims: Sequence of hidden layer widths.
        dropout: Dropout probability applied after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list[int] = (256, 128, 64),
        dropout: float = 0.3,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
