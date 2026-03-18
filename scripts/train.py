"""Train the two-stage IDS classifier.

Usage:
    python scripts/train.py --config configs/xiiotid_dnn.yaml
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    proc_dir = ROOT / cfg["data"]["processed_path"]

    # -----------------------------------------
    # Load preprocessed data
    # -----------------------------------------
    X_train = np.load(proc_dir / "X_train.npy")
    X_test  = np.load(proc_dir / "X_test.npy")
    yb_train = np.load(proc_dir / "yb_train.npy")
    yb_test  = np.load(proc_dir / "yb_test.npy")
    ym_train = np.load(proc_dir / "ym_train.npy")
    ym_test  = np.load(proc_dir / "ym_test.npy")

    with open(proc_dir / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    # -----------------------------------------
    # Stage 1: Binary model
    # -----------------------------------------
    from src.training.trainer_binary import train_binary_model

    print("\n--- Training Binary Model ---")
    binary_model = train_binary_model(
        X_train, yb_train, X_test, yb_test, cfg
    )

    out_dir = ROOT / cfg["output"]["model_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    binary_model.save_weights(str(out_dir / "binary_model.weights.h5"))
    print(f"Binary model saved to {out_dir / 'binary_model.weights.h5'}")

    # -----------------------------------------
    # Stage 2: Attack-type model (attack samples only)
    # -----------------------------------------
    from sklearn.preprocessing import LabelEncoder as AttackLabelEncoder
    from sklearn.utils.class_weight import compute_class_weight
    from src.training.trainer_attack import train_attack_model

    print("\n--- Training Attack Model ---")

    attack_mask = yb_train == 1
    X_train_attack = X_train[attack_mask]

    attack_mask_test = yb_test == 1
    X_test_attack = X_test[attack_mask_test]

    # Re-encode attack labels to contiguous 0-indexed classes
    attack_le = AttackLabelEncoder()
    y_train_attack = attack_le.fit_transform(ym_train[attack_mask])
    y_test_attack  = attack_le.transform(ym_test[attack_mask_test])

    classes = np.unique(y_train_attack)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_attack)
    class_weights = dict(zip(classes, weights))
    print("Class weights:", class_weights)

    attack_model = train_attack_model(
        X_train_attack, y_train_attack,
        X_test_attack,  y_test_attack,
        class_weights, cfg
    )

    attack_model.save_weights(str(out_dir / "attack_model.weights.h5"))
    print(f"Attack model saved to {out_dir / 'attack_model.weights.h5'}")


if __name__ == "__main__":
    main()
