import pickle
from pathlib import Path

import numpy as np
import yaml

from src.data.xiiotid import load_xiiotid_dataset
from src.data.preprocessing import preprocess_dataset


def main():

    config = yaml.safe_load(open("configs/xiiotid_dnn.yaml"))

    print("Loading dataset...")

    df = load_xiiotid_dataset(config["data"]["raw_path"])

    X_train, X_test, yb_train, yb_test, ym_train, ym_test, le, scaler = preprocess_dataset(
        df,
        config["data"]["label_column"],
        config["data"]["test_size"]
    )

    print("\nPreprocessing complete")
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("\nClasses detected:")
    print(le.classes_)

    # -----------------------------------------
    # Save processed arrays and artefacts
    # -----------------------------------------
    out_dir = Path(config["data"]["processed_path"])
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "X_test.npy", X_test)
    np.save(out_dir / "yb_train.npy", yb_train)
    np.save(out_dir / "yb_test.npy", yb_test)
    np.save(out_dir / "ym_train.npy", ym_train)
    np.save(out_dir / "ym_test.npy", ym_test)

    with open(out_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    with open(out_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(f"\nSaved processed data to {out_dir}")


if __name__ == "__main__":
    main()
