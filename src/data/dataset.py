"""PyTorch Dataset wrapper for pre-processed numpy arrays."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class IDSDataset(Dataset):
    """Wraps (X, y) numpy arrays as a PyTorch Dataset.

    Args:
        X: Feature matrix of shape (N, F) or (N, 1, F) for CNN input.
        y: Integer label vector of shape (N,).
        cnn_input: If True, inserts a channel dimension so X becomes (N, 1, F).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, cnn_input: bool = False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        if cnn_input and self.X.ndim == 2:
            self.X = self.X.unsqueeze(1)  # (N, 1, F)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]
