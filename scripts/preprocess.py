import yaml

from src.data.xiiotid import load_xiiotid_dataset
from src.data.preprocessing import preprocess_dataset


def main():

    config = yaml.safe_load(open("configs/xiiotid_dnn.yaml"))

    print("Loading dataset...")

    df = load_xiiotid_dataset(config["data"]["raw_path"])

    X_train, X_test, y_train, y_test, encoder = preprocess_dataset(
        df,
        config["data"]["label_column"],
        config["data"]["test_size"]
    )

    print("\nPreprocessing complete")

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    print("\nClasses detected:")
    print(encoder.classes_)


if __name__ == "__main__":
    main()