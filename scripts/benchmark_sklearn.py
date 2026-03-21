"""Benchmark sklearn classifiers against the two-stage DNN on XIIOTID.

Each classifier is trained as a two-stage pipeline mirroring the DNN:
  Stage 1: Binary classifier (Normal vs Attack) trained on all train-pool samples.
  Stage 2: Attack-type classifier trained on attack samples only.

Evaluation mirrors evaluate.py exactly:
  - Stage 1 binary accuracy
  - Stage 2 attack-type F1 with ground-truth gate
  - End-to-end attack-type F1 with Stage 1 gate (FN/FP counts reported)

Usage:
    python scripts/benchmark_sklearn.py --config configs/xiiotid_dnn.yaml
"""
import argparse
import json
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder as AttackLabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).parent.parent


def _evaluate_classifier(name, stage1_clf, stage2_clf,
                          X_train, yb_train, ym_train,
                          X_test, yb_test, ym_test,
                          le, attack_le, eval_class_names,
                          figures_dir, run_tag):
    from src.evaluation.metrics import full_report
    from src.evaluation.plots import plot_confusion_matrix, plot_per_class_f1

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")

    # ------------------------------------------------------------------
    # Stage 1: Binary classifier
    # ------------------------------------------------------------------
    print("\n--- Stage 1: Binary classifier ---")
    t0 = time.time()
    stage1_clf.fit(X_train, yb_train)
    stage1_time = time.time() - t0
    print(f"Training time: {stage1_time:.1f}s")

    yb_pred = stage1_clf.predict(X_test)

    binary_results = full_report(yb_test, yb_pred, class_names=["Normal", "Attack"])
    print(binary_results["report_str"])

    fn_count = int(np.sum((yb_test == 1) & (yb_pred == 0)))
    fp_count = int(np.sum((yb_test == 0) & (yb_pred == 1)))

    # ------------------------------------------------------------------
    # Stage 2: Attack-type classifier (train on attack samples only)
    # ------------------------------------------------------------------
    print("\n--- Stage 2: Attack-type classifier ---")
    attack_mask_train = yb_train == 1
    y_train_attack = attack_le.transform(ym_train[attack_mask_train])

    t0 = time.time()
    stage2_clf.fit(X_train[attack_mask_train], y_train_attack)
    stage2_time = time.time() - t0
    print(f"Training time: {stage2_time:.1f}s")

    # Kept classes filter — mirrors evaluate.py exactly
    attack_class_names = [le.classes_[i] for i in attack_le.classes_]
    keep_idx = [i for i, n in enumerate(attack_class_names) if n in eval_class_names]
    eval_names = [attack_class_names[i] for i in keep_idx]
    n_kept = len(keep_idx)
    label_map = {old: new for new, old in enumerate(keep_idx)}

    excluded = [n for n in attack_class_names if n not in eval_class_names]
    if excluded:
        print(f"  (excluded from metrics — below threshold: {', '.join(sorted(excluded))})")

    # Ground-truth gate: evaluate Stage 2 on true attack samples
    attack_mask_test = yb_test == 1
    ym_test_attack = ym_test[attack_mask_test]
    y_true_attack = attack_le.transform(ym_test_attack)
    y_pred_attack = stage2_clf.predict(X_test[attack_mask_test])

    keep_mask = np.isin(y_true_attack, keep_idx)
    y_true_eval = np.array([label_map[y] for y in y_true_attack[keep_mask]])
    y_pred_eval = np.array([label_map.get(int(y), n_kept) for y in y_pred_attack[keep_mask]])

    print(f"\n=== {name} — Stage 2: Attack-Type Classification (ground-truth gate) ===")
    attack_results = full_report(y_true_eval, y_pred_eval, class_names=eval_names)
    print(attack_results["report_str"])
    print(f"Macro F1:    {attack_results['macro_f1']:.4f}")
    print(f"Weighted F1: {attack_results['weighted_f1']:.4f}")

    # ------------------------------------------------------------------
    # End-to-end: Stage 1 gates Stage 2 — mirrors evaluate.py exactly
    # ------------------------------------------------------------------
    stage2_gate = yb_pred == 1
    y_pred_stage2_gated = np.full(len(yb_test), -1, dtype=int)
    if stage2_gate.any():
        y_pred_stage2_gated[stage2_gate] = stage2_clf.predict(X_test[stage2_gate])

    y_pred_e2e_attack = y_pred_stage2_gated[attack_mask_test]
    y_pred_e2e_eval = np.array([
        n_kept if p == -1 else label_map.get(int(p), n_kept)
        for p in y_pred_e2e_attack[keep_mask]
    ])

    print(f"\n=== {name} — End-to-End: Attack Classification (Stage 1 gate) ===")
    print(f"  Stage 1 false negatives (attacks silently missed): {fn_count}")
    print(f"  Stage 1 false positives (Normal routed to Stage 2): {fp_count}")
    e2e_results = full_report(y_true_eval, y_pred_e2e_eval, class_names=eval_names)
    print(e2e_results["report_str"])
    print(f"End-to-end Macro F1:    {e2e_results['macro_f1']:.4f}")
    print(f"End-to-end Weighted F1: {e2e_results['weighted_f1']:.4f}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    tag = f"{run_tag}_{name.lower().replace(' ', '_')}"
    plot_confusion_matrix(binary_results["cm"], ["Normal", "Attack"],
                          save_path=figures_dir / f"{tag}_binary_cm.png")
    plot_confusion_matrix(attack_results["cm"], eval_names,
                          save_path=figures_dir / f"{tag}_attack_cm.png")
    plot_per_class_f1(attack_results["per_class"],
                      save_path=figures_dir / f"{tag}_attack_f1.png")
    plot_confusion_matrix(e2e_results["cm"], eval_names,
                          save_path=figures_dir / f"{tag}_e2e_attack_cm.png")

    return {
        "name": name,
        "stage1_train_time_s": round(stage1_time, 1),
        "stage2_train_time_s": round(stage2_time, 1),
        "binary_accuracy": float(accuracy_score(yb_test, yb_pred)),
        "binary_macro_f1": float(binary_results["macro_f1"]),
        "attack_macro_f1": float(attack_results["macro_f1"]),
        "attack_weighted_f1": float(attack_results["weighted_f1"]),
        "e2e_attack_macro_f1": float(e2e_results["macro_f1"]),
        "e2e_attack_weighted_f1": float(e2e_results["weighted_f1"]),
        "stage1_fn_count": fn_count,
        "stage1_fp_count": fp_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    proc_dir = ROOT / cfg["data"]["processed_path"]
    metrics_dir = ROOT / cfg["output"]["metrics_dir"]
    figures_dir = ROOT / cfg["output"]["figures_dir"]
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    X  = np.load(proc_dir / "X.npy")
    ym = np.load(proc_dir / "ym.npy")
    yb = np.load(proc_dir / "yb.npy")
    test_idx = np.load(proc_dir / "test_idx.npy")
    with open(proc_dir / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    # Build train pool — mirrors train.py
    test_mask = np.zeros(len(X), dtype=bool)
    test_mask[test_idx] = True
    train_pool_idx = np.where(~test_mask)[0]
    print(f"Train pool: {len(train_pool_idx)} samples | Test: {len(test_idx)} samples")

    # Scale — fit on train pool only to avoid leakage
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_pool_idx])
    X_test  = scaler.transform(X[test_idx])
    yb_train = yb[train_pool_idx]
    ym_train = ym[train_pool_idx]
    yb_test  = yb[test_idx]
    ym_test  = ym[test_idx]

    # Fit attack label encoder on train pool attack samples — mirrors trainer_attack.py
    attack_mask_train = yb_train == 1
    attack_le = AttackLabelEncoder()
    attack_le.fit(ym_train[attack_mask_train])

    # Eval class filter — mirrors evaluate.py
    eval_min = cfg.get("evaluation", {}).get("eval_min_samples", 0)
    class_counts = {name: int(np.sum(ym == i)) for i, name in enumerate(le.classes_) if name != "Normal"}
    eval_class_names = {name for name, count in class_counts.items() if count >= eval_min}
    excluded_globally = sorted(name for name in class_counts if name not in eval_class_names)
    if excluded_globally:
        print(f"Classes excluded from evaluation (below {eval_min} samples): {', '.join(excluded_globally)}")

    dt_params  = dict(class_weight="balanced", random_state=42, max_depth=20, min_samples_leaf=5)
    rf_params  = dict(class_weight="balanced", random_state=42, max_depth=20, min_samples_leaf=5,
                      n_estimators=100, n_jobs=-1)
    lr_params  = dict(class_weight="balanced", solver="saga", max_iter=2000, tol=1e-3, random_state=42)

    classifiers = [
        ("Decision Tree",
         DecisionTreeClassifier(**dt_params),
         DecisionTreeClassifier(**dt_params)),
        ("Random Forest",
         RandomForestClassifier(**rf_params),
         RandomForestClassifier(**rf_params)),
        ("Logistic Regression",
         LogisticRegression(**lr_params),
         LogisticRegression(**lr_params)),
    ]

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []

    for name, stage1_clf, stage2_clf in classifiers:
        result = _evaluate_classifier(
            name, stage1_clf, stage2_clf,
            X_train, yb_train, ym_train,
            X_test, yb_test, ym_test,
            le, attack_le, eval_class_names,
            figures_dir, run_tag,
        )
        all_results.append(result)

    # Summary table
    print(f"\n{'='*70}")
    print("  Benchmark Summary")
    print(f"{'='*70}")
    header = (f"{'Method':<22} {'Bin Acc':>8} {'Atk Macro F1':>13} "
              f"{'E2E Macro F1':>13} {'Atk Wtd F1':>11}")
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(f"{r['name']:<22} {r['binary_accuracy']:>8.4f} {r['attack_macro_f1']:>13.4f} "
              f"{r['e2e_attack_macro_f1']:>13.4f} {r['attack_weighted_f1']:>11.4f}")

    out_path = metrics_dir / f"{run_tag}_sklearn_benchmark.json"
    with open(out_path, "w") as f:
        json.dump({"run": run_tag, "results": all_results}, f, indent=2)
    print(f"\nMetrics saved to {out_path}")
    print(f"Plots saved to {figures_dir}")


if __name__ == "__main__":
    main()
