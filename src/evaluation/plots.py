import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(cm, path):

    plt.figure(figsize=(8,6))

    sns.heatmap(cm, annot=True, fmt="d")

    plt.title("Confusion Matrix")

    plt.savefig(path)

    plt.close()