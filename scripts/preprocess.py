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

    X, yb, ym, le = preprocess_dataset(df, config["data"]["label_column"])

    print("\nPreprocessing complete")
    print("X shape:", X.shape)
    print("Classes detected:")
    print(le.classes_)

    out_dir = Path(config["data"]["processed_path"])
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "yb.npy", yb)
    np.save(out_dir / "ym.npy", ym)

    with open(out_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    print(f"\nSaved processed data to {out_dir}")


if __name__ == "__main__":
    main()
