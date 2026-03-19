import pickle
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import train_test_split

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

    # Stratified held-out split — stratify on ym (multi-class) to guarantee
    # all attack types appear in both partitions
    test_size = config.get("evaluation", {}).get("test_size", 0.2)
    all_idx = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        all_idx, test_size=test_size, random_state=42, stratify=ym
    )
    print(f"\nHeld-out split: {len(train_idx)} train / {len(test_idx)} test ({test_size:.0%})")

    out_dir = Path(config["data"]["processed_path"])
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "yb.npy", yb)
    np.save(out_dir / "ym.npy", ym)
    np.save(out_dir / "test_idx.npy", test_idx)

    with open(out_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    print(f"Saved processed data to {out_dir}")


if __name__ == "__main__":
    main()
