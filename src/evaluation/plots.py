"""Visualisation utilities: confusion matrix, ROC/PR curves."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    save_path: str | Path | None = None,
    normalise: bool = True,
) -> plt.Figure:
    if normalise:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names) - 1)))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalise else "d",
                xticklabels=class_names, yticklabels=class_names,
                cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix" + (" (normalised)" if normalise else ""))
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_per_class_f1(
    per_class: dict,
    save_path: str | Path | None = None,
) -> plt.Figure:
    names = list(per_class.keys())
    f1s = [per_class[n]["f1"] for n in names]
    supports = [per_class[n]["support"] for n in names]

    fig, ax = plt.subplots(figsize=(max(8, len(names)), 4))
    bars = ax.bar(names, f1s, color="steelblue")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1-score")
    ax.set_title("Per-class F1-score")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    # Annotate with support counts
    for bar, sup in zip(bars, supports):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"n={sup}", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
