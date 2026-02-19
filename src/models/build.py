"""Model factory: construct a model from a config dict."""
from __future__ import annotations

import torch.nn as nn

from .cnn1d import CNN1D
from .dnn import DNN


def build_model(cfg: dict) -> nn.Module:
    """Return an instantiated model based on ``cfg['arch']``.

    Expected config keys (all models):
        arch (str): 'dnn' | 'cnn1d'
        input_dim (int)
        num_classes (int)

    DNN-specific keys: hidden_dims, dropout
    CNN1D-specific keys: channels, kernel_size, dropout
    """
    arch = cfg["arch"].lower()
    if arch == "dnn":
        return DNN(
            input_dim=cfg["input_dim"],
            num_classes=cfg["num_classes"],
            hidden_dims=cfg.get("hidden_dims", [256, 128, 64]),
            dropout=cfg.get("dropout", 0.3),
        )
    if arch == "cnn1d":
        return CNN1D(
            input_dim=cfg["input_dim"],
            num_classes=cfg["num_classes"],
            channels=cfg.get("channels", [64, 128, 256]),
            kernel_size=cfg.get("kernel_size", 3),
            dropout=cfg.get("dropout", 0.3),
        )
    raise ValueError(f"Unknown architecture: {arch!r}")
