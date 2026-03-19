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


def _evaluate_fold(fold, X, yb, ym, le, model_dir, cfg, figures_dir, run_tag, eval_class_names,
                   train_pool_idx):
    from sklearn.metrics import accuracy_score, f1_score

    from src.evaluation.metrics import full_report
    from src.evaluation.plots import plot_confusion_matrix, plot_per_class_f1
    from src.training.trainer_binary import build_binary_model
    from src.training.trainer_attack import build_attack_model

    # Rebuild val indices using the same seed / fold as training
    from sklearn.model_selection import StratifiedKFold
    n_folds = cfg["training"]["n_folds"]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    splits = list(skf.split(X[train_pool_idx], yb[train_pool_idx]))
    _, local_val_idx = splits[fold]
    val_idx = train_pool_idx[local_val_idx]

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

    # Filter evaluation to classes meeting the eval_min_samples threshold
    keep_idx = [i for i, name in enumerate(attack_class_names) if name in eval_class_names]
    keep_mask = np.isin(y_val_attack, keep_idx)
    label_map = {old: new for new, old in enumerate(keep_idx)}
    n_kept = len(keep_idx)
    y_true_eval = np.array([label_map[y] for y in y_val_attack[keep_mask]])
    y_pred_eval = np.array([label_map.get(y, n_kept) for y in y_pred_attack[keep_mask]])
    eval_names = [attack_class_names[i] for i in keep_idx]

    excluded = [n for n in attack_class_names if n not in eval_class_names]
    if excluded:
        print(f"  (excluded from metrics — below threshold: {', '.join(excluded)})")

    print(f"=== Fold {fold} — Stage 2: Attack-Type Classification ===")
    attack_results = full_report(y_true_eval, y_pred_eval, class_names=eval_names)
    print(attack_results["report_str"])
    print(f"Macro F1:    {attack_results['macro_f1']:.4f}")
    print(f"Weighted F1: {attack_results['weighted_f1']:.4f}")

    # Plots
    tag = f"{run_tag}_fold{fold}"
    plot_confusion_matrix(binary_results["cm"], ["Normal", "Attack"],
                          save_path=figures_dir / f"{tag}_binary_cm.png")
    plot_confusion_matrix(attack_results["cm"], eval_names,
                          save_path=figures_dir / f"{tag}_attack_cm.png")
    plot_per_class_f1(attack_results["per_class"],
                      save_path=figures_dir / f"{tag}_attack_f1.png")

    # -----------------------------------------
    # End-to-end evaluation (Stage 1 gates Stage 2)
    # -----------------------------------------
    fn_count = int(np.sum((yb_val == 1) & (yb_pred == 0)))
    fp_count = int(np.sum((yb_val == 0) & (yb_pred == 1)))

    # Run Stage 2 on what Stage 1 actually passed through (may include Normal FPs,
    # excludes Attack FNs that were silently dropped at Stage 1).
    stage2_gate = yb_pred == 1
    y_pred_stage2_gated = np.full(len(yb_val), -1, dtype=int)  # -1 = predicted Normal
    if stage2_gate.any():
        s2_out = np.argmax(attack_model.predict(X_val[stage2_gate], verbose=0), axis=1)
        y_pred_stage2_gated[stage2_gate] = s2_out

    # Score end-to-end over true attack samples only (same keep filter as above).
    # Predictions of -1 (Stage 1 FN) map to n_kept → counted as misses by full_report.
    y_pred_e2e_attack = y_pred_stage2_gated[attack_mask]
    y_pred_e2e_eval = np.array([
        n_kept if p == -1 else label_map.get(p, n_kept)
        for p in y_pred_e2e_attack[keep_mask]
    ])

    print(f"\n=== Fold {fold} — End-to-End: Attack Classification (Stage 1 gate) ===")
    print(f"  Stage 1 false negatives (attacks silently missed): {fn_count}")
    print(f"  Stage 1 false positives (Normal routed to Stage 2): {fp_count}")
    e2e_results = full_report(y_true_eval, y_pred_e2e_eval, class_names=eval_names)
    print(e2e_results["report_str"])
    print(f"End-to-end Macro F1:    {e2e_results['macro_f1']:.4f}")
    print(f"End-to-end Weighted F1: {e2e_results['weighted_f1']:.4f}")

    plot_confusion_matrix(e2e_results["cm"], eval_names,
                          save_path=figures_dir / f"{tag}_e2e_attack_cm.png")

    return {
        "fold": fold,
        "binary_accuracy": float(accuracy_score(yb_val, yb_pred)),
        "binary_macro_f1": binary_results["macro_f1"],
        "attack_accuracy": float(accuracy_score(y_true_eval, y_pred_eval)),
        "attack_macro_f1": attack_results["macro_f1"],
        "attack_weighted_f1": attack_results["weighted_f1"],
        "e2e_attack_macro_f1": e2e_results["macro_f1"],
        "e2e_attack_weighted_f1": e2e_results["weighted_f1"],
        "stage1_fn_count": fn_count,
        "stage1_fp_count": fp_count,
    }


def _evaluate_test(X, yb, ym, le, test_idx, model_dir, cfg, figures_dir, run_tag, eval_class_names):
    """Ensemble evaluation on the held-out test set (averages predictions across all CV folds)."""
    from sklearn.metrics import accuracy_score
    from src.evaluation.metrics import full_report
    from src.evaluation.plots import plot_confusion_matrix, plot_per_class_f1
    from src.training.trainer_binary import build_binary_model
    from src.training.trainer_attack import build_attack_model

    n_folds = cfg["training"]["n_folds"]
    X_test  = X[test_idx]
    yb_test = yb[test_idx]
    ym_test = ym[test_idx]

    # --- Stage 1: Binary ensemble ---
    binary_probs = np.zeros(len(test_idx))
    for fold in range(n_folds):
        with open(model_dir / f"scaler_fold{fold}.pkl", "rb") as f:
            scaler = pickle.load(f)
        binary_model = build_binary_model(scaler.transform(X_test).shape[1])
        binary_model.load_weights(str(model_dir / f"binary_model_fold{fold}.weights.h5"))
        binary_probs += binary_model.predict(scaler.transform(X_test), verbose=0).squeeze()
    binary_probs /= n_folds
    yb_pred = (binary_probs >= 0.5).astype(int)

    print("\n=== TEST SET — Stage 1: Binary Classification ===")
    binary_results = full_report(yb_test, yb_pred, class_names=["Normal", "Attack"])
    print(binary_results["report_str"])

    # --- Stage 2: Attack ensemble ---
    # attack_le classes are sorted alphabetically and consistent across folds
    with open(model_dir / "attack_label_encoder_fold0.pkl", "rb") as f:
        attack_le = pickle.load(f)
    attack_class_names = [le.classes_[i] for i in attack_le.classes_]
    n_attack_classes = len(attack_le.classes_)

    attack_mask = yb_test == 1
    X_test_attack = X_test[attack_mask]
    ym_test_attack = ym_test[attack_mask]
    y_true_attack = attack_le.transform(ym_test_attack)

    attack_probs = np.zeros((len(X_test_attack), n_attack_classes))
    for fold in range(n_folds):
        with open(model_dir / f"scaler_fold{fold}.pkl", "rb") as f:
            scaler = pickle.load(f)
        X_scaled = scaler.transform(X_test_attack)
        attack_model = build_attack_model(X_scaled.shape[1], n_attack_classes)
        attack_model.load_weights(str(model_dir / f"attack_model_fold{fold}.weights.h5"))
        attack_probs += attack_model.predict(X_scaled, verbose=0)
    attack_probs /= n_folds
    y_pred_attack = np.argmax(attack_probs, axis=1)

    # Filter to eval classes
    keep_idx = [i for i, name in enumerate(attack_class_names) if name in eval_class_names]
    keep_mask = np.isin(y_true_attack, keep_idx)
    label_map = {old: new for new, old in enumerate(keep_idx)}
    n_kept = len(keep_idx)
    y_true_eval = np.array([label_map[y] for y in y_true_attack[keep_mask]])
    y_pred_eval = np.array([label_map.get(y, n_kept) for y in y_pred_attack[keep_mask]])
    eval_names  = [attack_class_names[i] for i in keep_idx]

    excluded = [n for n in attack_class_names if n not in eval_class_names]
    if excluded:
        print(f"  (excluded from metrics — below threshold: {', '.join(excluded)})")

    print("=== TEST SET — Stage 2: Attack-Type Classification ===")
    attack_results = full_report(y_true_eval, y_pred_eval, class_names=eval_names)
    print(attack_results["report_str"])
    print(f"Macro F1:    {attack_results['macro_f1']:.4f}")
    print(f"Weighted F1: {attack_results['weighted_f1']:.4f}")

    tag = f"{run_tag}_test"
    plot_confusion_matrix(binary_results["cm"], ["Normal", "Attack"],
                          save_path=figures_dir / f"{tag}_binary_cm.png")
    plot_confusion_matrix(attack_results["cm"], eval_names,
                          save_path=figures_dir / f"{tag}_attack_cm.png")
    plot_per_class_f1(attack_results["per_class"],
                      save_path=figures_dir / f"{tag}_attack_f1.png")

    # -----------------------------------------
    # End-to-end evaluation (Stage 1 gates Stage 2)
    # -----------------------------------------
    fn_count = int(np.sum((yb_test == 1) & (yb_pred == 0)))
    fp_count = int(np.sum((yb_test == 0) & (yb_pred == 1)))

    stage2_gate = yb_pred == 1
    y_pred_stage2_gated = np.full(len(yb_test), -1, dtype=int)  # -1 = predicted Normal
    if stage2_gate.any():
        e2e_attack_probs = np.zeros((int(stage2_gate.sum()), n_attack_classes))
        for fold in range(n_folds):
            with open(model_dir / f"scaler_fold{fold}.pkl", "rb") as f:
                scaler = pickle.load(f)
            X_scaled = scaler.transform(X_test[stage2_gate])
            attack_model = build_attack_model(X_scaled.shape[1], n_attack_classes)
            attack_model.load_weights(str(model_dir / f"attack_model_fold{fold}.weights.h5"))
            e2e_attack_probs += attack_model.predict(X_scaled, verbose=0)
        e2e_attack_probs /= n_folds
        y_pred_stage2_gated[stage2_gate] = np.argmax(e2e_attack_probs, axis=1)

    y_pred_e2e_attack = y_pred_stage2_gated[attack_mask]
    y_pred_e2e_eval = np.array([
        n_kept if p == -1 else label_map.get(p, n_kept)
        for p in y_pred_e2e_attack[keep_mask]
    ])

    print("\n=== TEST SET — End-to-End: Attack Classification (Stage 1 gate) ===")
    print(f"  Stage 1 false negatives (attacks silently missed): {fn_count}")
    print(f"  Stage 1 false positives (Normal routed to Stage 2): {fp_count}")
    e2e_results = full_report(y_true_eval, y_pred_e2e_eval, class_names=eval_names)
    print(e2e_results["report_str"])
    print(f"End-to-end Macro F1:    {e2e_results['macro_f1']:.4f}")
    print(f"End-to-end Weighted F1: {e2e_results['weighted_f1']:.4f}")

    plot_confusion_matrix(e2e_results["cm"], eval_names,
                          save_path=figures_dir / f"{tag}_e2e_attack_cm.png")

    return {
        "binary_accuracy":        float(accuracy_score(yb_test, yb_pred)),
        "binary_macro_f1":        binary_results["macro_f1"],
        "attack_accuracy":        float(accuracy_score(y_true_eval, y_pred_eval)),
        "attack_macro_f1":        attack_results["macro_f1"],
        "attack_weighted_f1":     attack_results["weighted_f1"],
        "e2e_attack_macro_f1":    e2e_results["macro_f1"],
        "e2e_attack_weighted_f1": e2e_results["weighted_f1"],
        "stage1_fn_count":        fn_count,
        "stage1_fp_count":        fp_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, default=None,
                        help="Fold index (0-indexed). Omit to evaluate all folds.")
    parser.add_argument("--test", action="store_true",
                        help="Evaluate on the held-out test set using an ensemble of all fold models.")
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
    test_idx = np.load(proc_dir / "test_idx.npy")
    with open(proc_dir / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    # Build train pool (mirrors train.py)
    test_mask = np.zeros(len(X), dtype=bool)
    test_mask[test_idx] = True
    train_pool_idx = np.where(~test_mask)[0]

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Compute which attack classes meet the eval threshold
    eval_min = cfg.get("evaluation", {}).get("eval_min_samples", 0)
    class_counts = {name: int(np.sum(ym == i)) for i, name in enumerate(le.classes_) if name != "Normal"}
    eval_class_names = {name for name, count in class_counts.items() if count >= eval_min}
    excluded_globally = sorted(name for name in class_counts if name not in eval_class_names)
    if excluded_globally:
        print(f"Classes excluded from evaluation (below {eval_min} samples): {', '.join(excluded_globally)}")

    # --test: held-out test set evaluation
    if args.test:
        result = _evaluate_test(X, yb, ym, le, test_idx, model_dir, cfg, figures_dir, run_tag, eval_class_names)
        print(f"\n{'='*50}")
        print("  Held-Out Test Set Results")
        print(f"{'='*50}")
        for metric, val in result.items():
            print(f"  {metric}: {val:.4f}")
        out_path = metrics_dir / f"{run_tag}_test_metrics.json"
        with open(out_path, "w") as f:
            json.dump({"run": run_tag, "eval_type": "held_out_test", "results": result}, f, indent=2)
        print(f"\nMetrics saved to {out_path}")
        print(f"Plots saved to {figures_dir}")
        return

    n_folds = cfg["training"]["n_folds"]
    folds_to_eval = [args.fold] if args.fold is not None else list(range(n_folds))

    all_results = []
    for fold in folds_to_eval:
        result = _evaluate_fold(fold, X, yb, ym, le, model_dir, cfg, figures_dir, run_tag, eval_class_names,
                                train_pool_idx=train_pool_idx)
        all_results.append(result)

    # -----------------------------------------
    # Aggregate and save
    # -----------------------------------------
    print(f"\n{'='*50}")
    print("  Aggregated Evaluation Results")
    print(f"{'='*50}")
    for metric in [
        "binary_accuracy", "binary_macro_f1",
        "attack_accuracy", "attack_macro_f1", "attack_weighted_f1",
        "e2e_attack_macro_f1", "e2e_attack_weighted_f1",
        "stage1_fn_count", "stage1_fp_count",
    ]:
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
