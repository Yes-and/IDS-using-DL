"""Shared preprocessing utilities: normalisation, encoding, and splitting."""
from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def encode_labels(
    labels: Union[pd.Series, np.ndarray],
) -> Tuple[np.ndarray, LabelEncoder]:
    """Encode string or integer class labels to contiguous integers.

    Args:
        labels: 1-D array-like of raw class labels (strings or integers).
            Accepts both ``pd.Series`` and ``np.ndarray`` (e.g. from Polars
            ``.to_numpy()``).

    Returns:
        Tuple of (encoded integer array, fitted LabelEncoder).
    """
    le = LabelEncoder()
    encoded = le.fit_transform(labels)
    return encoded, le


def scale_features(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return X_train, X_val, X_test, scaler


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.10,
    test_size: float = 0.10,
    seed: int = 42,
) -> Tuple[np.ndarray, ...]:
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    val_frac = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_frac, stratify=y_tmp, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
