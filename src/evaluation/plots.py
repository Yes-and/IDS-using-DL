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