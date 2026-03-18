import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_confusion_matrix(cm, labels, save_path):

    plt.figure(figsize=(12,10))

    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path)
    plt.close()


def plot_per_class_f1(per_class, save_path):
    """Bar chart of per-class F1 scores.

    per_class: dict mapping class name (str) → f1 score (float)
    """
    labels = list(per_class.keys())
    scores = list(per_class.values())

    fig, ax = plt.subplots(figsize=(max(8, len(labels)), 5))
    ax.bar(labels, scores)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Class")
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-class F1 Score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    os.makedirs(os.path.dirname(str(save_path)), exist_ok=True)
    plt.savefig(save_path)
    plt.close()