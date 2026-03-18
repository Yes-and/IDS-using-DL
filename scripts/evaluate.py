"""Evaluate a saved checkpoint on the test split.

Usage:
    python scripts/evaluate.py --config configs/xiiotid_dnn.yaml \
        --checkpoint results/xiiotid_dnn_best.weights.h5
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", default=None, help="Directory to save metrics and plots")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dataset = cfg["dataset"]
    proc_dir = ROOT / "data" / "processed" / dataset

    X_test = np.load(proc_dir / "X_test.npy")
    y_test = np.load(proc_dir / "y_test.npy")
    with open(proc_dir / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    cfg["input_dim"] = X_test.shape[1]
    cfg["num_classes"] = len(le.classes_)

    from src.evaluation.metrics import full_report
    from src.evaluation.plots import plot_confusion_matrix, plot_per_class_f1
    from src.models.build import build_model

    model = build_model(cfg)
    model.load_weights(args.checkpoint)

    y_pred = np.argmax(model.predict(X_test), axis=1)

    class_names = list(le.classes_)
    results = full_report(y_test, y_pred, class_names)
    print(results["report_str"])
    print(f"Macro F1:    {results['macro_f1']:.4f}")
    print(f"Weighted F1: {results['weighted_f1']:.4f}")

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    run = Path(args.checkpoint).stem
    with open(out_dir / f"{run}_metrics.json", "w") as f:
        json.dump({k: v for k, v in results.items() if k != "cm"}, f, indent=2)

    plot_confusion_matrix(results["cm"], class_names, save_path=out_dir / f"{run}_cm.png")
    plot_per_class_f1(results["per_class"], save_path=out_dir / f"{run}_f1.png")
    print(f"Saved metrics and plots to {out_dir}")


if __name__ == "__main__":
    main()
