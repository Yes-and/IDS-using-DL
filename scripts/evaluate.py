"""Evaluate the two-stage IDS classifier on the test split.

Usage:
    python scripts/evaluate.py --config configs/xiiotid_dnn.yaml \
        --checkpoint results/models/attack_model.weights.h5
"""
import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from sklearn.preprocessing import LabelEncoder as AttackLabelEncoder

ROOT = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to attack model weights")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    proc_dir = ROOT / cfg["data"]["processed_path"]

    X_test   = np.load(proc_dir / "X_test.npy")
    yb_test  = np.load(proc_dir / "yb_test.npy")
    ym_test  = np.load(proc_dir / "ym_test.npy")
    with open(proc_dir / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    from src.evaluation.metrics import full_report
    from src.evaluation.plots import plot_confusion_matrix, plot_per_class_f1
    from src.training.trainer_binary import build_binary_model
    from src.training.trainer_attack import build_attack_model

    # -----------------------------------------
    # Stage 1: Binary model evaluation
    # -----------------------------------------
    binary_model = build_binary_model(X_test.shape[1])
    binary_weights = Path(args.checkpoint).parent / "binary_model.weights.h5"
    binary_model.load_weights(str(binary_weights))

    yb_pred_prob = binary_model.predict(X_test, verbose=0)
    yb_pred = (yb_pred_prob.squeeze() >= 0.5).astype(int)

    print("=== Stage 1: Binary Classification (Normal vs Attack) ===")
    binary_results = full_report(yb_test, yb_pred, class_names=["Normal", "Attack"])
    print(binary_results["report_str"])

    # -----------------------------------------
    # Stage 2: Attack-type model evaluation
    # -----------------------------------------
    attack_mask = yb_test == 1
    X_test_attack = X_test[attack_mask]
    ym_test_attack = ym_test[attack_mask]

    attack_le = AttackLabelEncoder()
    y_test_attack = attack_le.fit_transform(ym_test_attack)
    attack_class_names = [le.classes_[i] for i in attack_le.classes_]

    attack_model = build_attack_model(X_test_attack.shape[1], len(attack_le.classes_))
    attack_model.load_weights(args.checkpoint)

    y_pred_attack = np.argmax(attack_model.predict(X_test_attack, verbose=0), axis=1)

    print("\n=== Stage 2: Attack-Type Classification ===")
    attack_results = full_report(y_test_attack, y_pred_attack, class_names=attack_class_names)
    print(attack_results["report_str"])
    print(f"Macro F1:    {attack_results['macro_f1']:.4f}")
    print(f"Weighted F1: {attack_results['weighted_f1']:.4f}")

    # -----------------------------------------
    # Save outputs
    # -----------------------------------------
    metrics_dir = ROOT / cfg["output"]["metrics_dir"]
    figures_dir = ROOT / cfg["output"]["figures_dir"]
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    run = Path(args.checkpoint).stem + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(metrics_dir / f"{run}_binary_metrics.json", "w") as f:
        json.dump({k: v for k, v in binary_results.items() if k != "cm"}, f, indent=2)

    with open(metrics_dir / f"{run}_attack_metrics.json", "w") as f:
        json.dump({k: v for k, v in attack_results.items() if k != "cm"}, f, indent=2)

    plot_confusion_matrix(binary_results["cm"], ["Normal", "Attack"],
                          save_path=figures_dir / f"{run}_binary_cm.png")
    plot_confusion_matrix(attack_results["cm"], attack_class_names,
                          save_path=figures_dir / f"{run}_attack_cm.png")
    plot_per_class_f1(attack_results["per_class"],
                      save_path=figures_dir / f"{run}_attack_f1.png")

    print(f"\nMetrics saved to {metrics_dir}")
    print(f"Plots saved to {figures_dir}")


if __name__ == "__main__":
    main()
