"""Preprocess a raw dataset and save splits to data/processed/.

Usage:
    python scripts/preprocess.py --dataset xiiotid
    python scripts/preprocess.py --dataset xiiotid --label class3 --no-alert-cols
    python scripts/preprocess.py --dataset cicids2019
"""
import argparse
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["xiiotid", "cicids2019"])
    # XIIOTID-specific options
    parser.add_argument(
        "--label",
        default="class2",
        choices=["class1", "class2", "class3"],
        help="(XIIOTID only) Label granularity: binary / attack-category / specific-type",
    )
    parser.add_argument(
        "--no-alert-cols",
        action="store_true",
        help="(XIIOTID only) Drop alert columns derived from external IDS tools",
    )
    args = parser.parse_args()

    DATA_PROC.mkdir(parents=True, exist_ok=True)

    from src.data.preprocessing import encode_labels, scale_features, split_data

    # ── Load & extract feature matrix ────────────────────────────────────────
    if args.dataset == "xiiotid":
        from src.data.xiiotid import feature_matrix, load_xiiotid

        df = load_xiiotid(
            DATA_RAW / "xiiotid",
            include_alert_cols=not args.no_alert_cols,
        )
        X, y_raw, feature_names = feature_matrix(df, label_col=args.label)

    else:
        from src.data.cicids2019 import feature_matrix, load_cicids2019

        df = load_cicids2019(DATA_RAW / "cicids2019")
        X, y_raw, feature_names = feature_matrix(df)

    print(f"[{args.dataset}] Loaded: {X.shape[0]:,} samples, {X.shape[1]} features")

    # ── Encode labels, split, scale ───────────────────────────────────────────
    y, le = encode_labels(y_raw)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    X_train, X_val, X_test, scaler = scale_features(X_train, X_val, X_test)

    # ── Save ──────────────────────────────────────────────────────────────────
    suffix = f"_{args.label}" if args.dataset == "xiiotid" else ""
    out_dir = DATA_PROC / f"{args.dataset}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "X_val.npy", X_val)
    np.save(out_dir / "X_test.npy", X_test)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "y_val.npy", y_val)
    np.save(out_dir / "y_test.npy", y_test)
    with open(out_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    with open(out_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(out_dir / "feature_names.txt", "w") as f:
        f.write("\n".join(feature_names))

    # ── Summary ───────────────────────────────────────────────────────────────
    counts = np.bincount(y)
    print(f"Split  — train: {len(y_train):,}  val: {len(y_val):,}  test: {len(y_test):,}")
    print(f"Classes ({len(le.classes_)}): {list(le.classes_)}")
    print("Class distribution (train):")
    for i, name in enumerate(le.classes_):
        n = int((y_train == i).sum())
        print(f"  {name:<35s} {n:>8,}  ({100*n/len(y_train):.2f}%)")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
