"""Evaluate the two-stage IDS classifier using saved fold artefacts.

Usage:
    # Evaluate a single fold (0-indexed):
    python scripts/evaluate.py --config configs/xiiotid_dnn.yaml --fold 0

    # Evaluate all folds and aggregate:
    python scripts/evaluate.py --config configs/xiiotid_dnn.yaml
"""
import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).parent.parent


def _evaluate_fold(fold, X, yb, ym, le, model_dir, cfg, figures_dir, run_tag):
    from sklearn.metrics import accuracy_score, f1_score

    from src.evaluation.metrics import full_report
    from src.evaluation.plots import plot_confusion_matrix, plot_per_class_f1
    from src.training.trainer_binary import build_binary_model
    from src.training.trainer_attack import build_attack_model

    # Rebuild val indices using the same seed / fold as training
    from sklearn.model_selection import StratifiedKFold
    n_folds = cfg["training"]["n_folds"]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    splits = list(skf.split(X, yb))
    _, val_idx = splits[fold]

    # Load scaler and scale val set
    with open(model_dir / f"scaler_fold{fold}.pkl", "rb") as f:
        scaler = pickle.load(f)
    X_val  = scaler.transform(X[val_idx])
    yb_val = yb[val_idx]
    ym_val = ym[val_idx]

    # -----------------------------------------
    # Stage 1: Binary model evaluation
    # -----------------------------------------
    binary_model = build_binary_model(X_val.shape[1])
    binary_model.load_weights(str(model_dir / f"binary_model_fold{fold}.weights.h5"))

    yb_pred_prob = binary_model.predict(X_val, verbose=0)
    yb_pred = (yb_pred_prob.squeeze() >= 0.5).astype(int)

    print(f"\n=== Fold {fold} — Stage 1: Binary Classification ===")
    binary_results = full_report(yb_val, yb_pred, class_names=["Normal", "Attack"])
    print(binary_results["report_str"])

    # -----------------------------------------
    # Stage 2: Attack-type model evaluation
    # -----------------------------------------
    attack_mask = yb_val == 1
    X_val_attack = X_val[attack_mask]
    ym_val_attack = ym_val[attack_mask]

    with open(model_dir / f"attack_label_encoder_fold{fold}.pkl", "rb") as f:
        attack_le = pickle.load(f)
    y_val_attack = attack_le.transform(ym_val_attack)
    attack_class_names = [le.classes_[i] for i in attack_le.classes_]

    attack_model = build_attack_model(X_val_attack.shape[1], len(attack_le.classes_))
    attack_model.load_weights(str(model_dir / f"attack_model_fold{fold}.weights.h5"))

    y_pred_attack = np.argmax(attack_model.predict(X_val_attack, verbose=0), axis=1)

    print(f"=== Fold {fold} — Stage 2: Attack-Type Classification ===")
    attack_results = full_report(y_val_attack, y_pred_attack, class_names=attack_class_names)
    print(attack_results["report_str"])
    print(f"Macro F1:    {attack_results['macro_f1']:.4f}")
    print(f"Weighted F1: {attack_results['weighted_f1']:.4f}")

    # Plots
    tag = f"{run_tag}_fold{fold}"
    plot_confusion_matrix(binary_results["cm"], ["Normal", "Attack"],
                          save_path=figures_dir / f"{tag}_binary_cm.png")
    plot_confusion_matrix(attack_results["cm"], attack_class_names,
                          save_path=figures_dir / f"{tag}_attack_cm.png")
    plot_per_class_f1(attack_results["per_class"],
                      save_path=figures_dir / f"{tag}_attack_f1.png")

    return {
        "fold": fold,
        "binary_accuracy": float(accuracy_score(yb_val, yb_pred)),
        "binary_macro_f1": binary_results["macro_f1"],
        "attack_accuracy": float(accuracy_score(y_val_attack, y_pred_attack)),
        "attack_macro_f1": attack_results["macro_f1"],
        "attack_weighted_f1": attack_results["weighted_f1"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, default=None,
                        help="Fold index (0-indexed). Omit to evaluate all folds.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    proc_dir = ROOT / cfg["data"]["processed_path"]
    model_dir = ROOT / cfg["output"]["model_dir"]
    metrics_dir = ROOT / cfg["output"]["metrics_dir"]
    figures_dir = ROOT / cfg["output"]["figures_dir"]
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    X  = np.load(proc_dir / "X.npy")
    yb = np.load(proc_dir / "yb.npy")
    ym = np.load(proc_dir / "ym.npy")
    with open(proc_dir / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    n_folds = cfg["training"]["n_folds"]
    folds_to_eval = [args.fold] if args.fold is not None else list(range(n_folds))

    all_results = []
    for fold in folds_to_eval:
        result = _evaluate_fold(fold, X, yb, ym, le, model_dir, cfg, figures_dir, run_tag)
        all_results.append(result)

    # -----------------------------------------
    # Aggregate and save
    # -----------------------------------------
    print(f"\n{'='*50}")
    print("  Aggregated Evaluation Results")
    print(f"{'='*50}")
    for metric in ["binary_accuracy", "binary_macro_f1", "attack_accuracy", "attack_macro_f1", "attack_weighted_f1"]:
        vals = [r[metric] for r in all_results]
        print(f"  {metric}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    summary = {
        "run": run_tag,
        "folds_evaluated": folds_to_eval,
        "per_fold": all_results,
        "mean": {k: float(np.mean([r[k] for r in all_results]))
                 for k in all_results[0] if k != "fold"},
        "std":  {k: float(np.std([r[k] for r in all_results]))
                 for k in all_results[0] if k != "fold"},
    }

    out_path = metrics_dir / f"{run_tag}_cv_metrics.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nMetrics saved to {out_path}")
    print(f"Plots saved to {figures_dir}")


if __name__ == "__main__":
    main()
