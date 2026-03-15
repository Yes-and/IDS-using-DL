import yaml
from pathlib import Path

from src.data.xiiotid import load_xiiotid_dataset
from src.data.preprocessing import preprocess_dataset
from src.models.build import build_model
from src.training.trainer import train_model
from src.evaluation.metrics import evaluate_model
from src.evaluation.plots import plot_confusion_matrix


def main():

    config = yaml.safe_load(open("configs/xiiotid_dnn.yaml"))

    print("Loading dataset")

    df = load_xiiotid_dataset(config["data"]["raw_path"])

    X_train, X_test, y_train, y_test, encoder = preprocess_dataset(
        df,
        config["data"]["label_column"],
        config["data"]["test_size"]
    )

    print("Building model")

    model = build_model(
        config,
        X_train.shape[1],
        len(encoder.classes_)
    )

    print("Training model")

    train_model(model, X_train, y_train, config)

    print("Evaluating")

    acc, report, cm = evaluate_model(model, X_test, y_test)

    print("Accuracy:", acc)
    print(report)

    plot_confusion_matrix(
        cm,
        Path(config["output"]["figures_dir"]) / "confusion_matrix.png"
    )


if __name__ == "__main__":
    main()