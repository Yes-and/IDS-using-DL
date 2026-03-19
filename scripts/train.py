"""Train the two-stage IDS classifier with stratified k-fold CV.

Usage:
    python scripts/train.py --config configs/xiiotid_dnn.yaml
"""
import argparse
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).parent.parent


def _save_report(cfg, config_path, le, fold_results, out_dir):
    now = datetime.now()
    n_folds = len(fold_results)

    def _fold_row(k, r):
        return (
            f"| {k+1} "
            f"| {r['binary_val_acc']:.4f} "
            f"| {r['attack_val_acc']:.4f} "
            f"| {r['attack_macro_f1']:.4f} |"
        )

    fold_rows = "\n".join(_fold_row(k, r) for k, r in enumerate(fold_results))

    bin_accs   = [r["binary_val_acc"]   for r in fold_results]
    atk_accs   = [r["attack_val_acc"]   for r in fold_results]
    atk_f1s    = [r["attack_macro_f1"]  for r in fold_results]

    summary_row = (
        f"| **Mean ± Std** "
        f"| {np.mean(bin_accs):.4f} ± {np.std(bin_accs):.4f} "
        f"| {np.mean(atk_accs):.4f} ± {np.std(atk_accs):.4f} "
        f"| {np.mean(atk_f1s):.4f} ± {np.std(atk_f1s):.4f} |"
    )

    report = f"""# Training Run Report

**Date:** {now.strftime("%Y-%m-%d %H:%M:%S")}
**Config:** {config_path}
**Folds:** {n_folds}

---

## Dataset

**Classes ({len(le.classes_)}):** {", ".join(le.classes_)}

---

## Per-Fold Validation Metrics

| Fold | Binary Acc | Attack Acc | Attack Macro F1 |
|------|-----------|-----------|----------------|
{fold_rows}
{summary_row}

---

## Notes

<!-- Add observations, what changed, known issues, next steps -->
"""

    reports_dir = out_dir.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"run_{now.strftime('%Y%m%d_%H%M%S')}.md"
    report_path.write_text(report)
    print(f"Report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    proc_dir = ROOT / cfg["data"]["processed_path"]

    # -----------------------------------------
    # Load preprocessed data (full arrays)
    # -----------------------------------------
    X  = np.load(proc_dir / "X.npy")
    yb = np.load(proc_dir / "yb.npy")
    ym = np.load(proc_dir / "ym.npy")

    with open(proc_dir / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    # Exclude held-out test indices from CV pool
    test_idx = np.load(proc_dir / "test_idx.npy")
    test_mask = np.zeros(len(X), dtype=bool)
    test_mask[test_idx] = True
    train_pool_idx = np.where(~test_mask)[0]
    print(f"CV pool: {len(train_pool_idx)} samples ({len(test_idx)} held out for test)")

    out_dir = ROOT / cfg["output"]["model_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------
    # Imports
    # -----------------------------------------
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.preprocessing import LabelEncoder as AttackLabelEncoder
    from sklearn.metrics import f1_score

    from src.training.trainer_binary import train_binary_model
    from src.training.trainer_attack import train_attack_model

    n_folds = cfg["training"]["n_folds"]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = []

    for fold, (local_train_idx, local_val_idx) in enumerate(skf.split(X[train_pool_idx], yb[train_pool_idx])):
        train_idx = train_pool_idx[local_train_idx]
        val_idx   = train_pool_idx[local_val_idx]
        print(f"\n{'='*50}")
        print(f"  Fold {fold + 1} / {n_folds}")
        print(f"{'='*50}")

        X_train, X_val = X[train_idx], X[val_idx]
        yb_train, yb_val = yb[train_idx], yb[val_idx]
        ym_train, ym_val = ym[train_idx], ym[val_idx]

        # Per-fold scaling (fit only on train)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)

        # -----------------------------------------
        # Stage 1: Binary model
        # -----------------------------------------
        class_weight = None
        if cfg["training"].get("class_weight"):
            classes = np.unique(yb_train)
            weights = compute_class_weight(class_weight="balanced", classes=classes, y=yb_train)
            class_weight = dict(zip(classes.tolist(), weights.tolist()))
            print("Binary class weights:", class_weight)

        print("\n--- Training Binary Model ---")
        binary_model, binary_history = train_binary_model(
            X_train, yb_train, X_val, yb_val, cfg, class_weight=class_weight
        )

        binary_model.save_weights(str(out_dir / f"binary_model_fold{fold}.weights.h5"))
        print(f"Binary model (fold {fold}) saved.")

        bh = binary_history.history
        binary_val_acc = max(bh["val_accuracy"])

        # -----------------------------------------
        # Stage 2: Attack-type model
        # -----------------------------------------
        attack_mask_train = yb_train == 1
        attack_mask_val   = yb_val   == 1

        X_train_attack = X_train[attack_mask_train]
        X_val_attack   = X_val[attack_mask_val]

        attack_le = AttackLabelEncoder()
        y_train_attack = attack_le.fit_transform(ym_train[attack_mask_train])
        y_val_attack   = attack_le.transform(ym_val[attack_mask_val])

        atk_classes = np.unique(y_train_attack)
        atk_weights = compute_class_weight(class_weight="balanced", classes=atk_classes, y=y_train_attack)
        attack_class_weights = dict(zip(atk_classes.tolist(), atk_weights.tolist()))

        print("\n--- Training Attack Model ---")
        attack_model, attack_history = train_attack_model(
            X_train_attack, y_train_attack,
            X_val_attack,   y_val_attack,
            attack_class_weights, cfg
        )

        attack_model.save_weights(str(out_dir / f"attack_model_fold{fold}.weights.h5"))
        print(f"Attack model (fold {fold}) saved.")

        with open(out_dir / f"attack_label_encoder_fold{fold}.pkl", "wb") as f:
            pickle.dump(attack_le, f)
        with open(out_dir / f"scaler_fold{fold}.pkl", "wb") as f:
            pickle.dump(scaler, f)

        # Per-fold val metrics
        ah = attack_history.history
        attack_val_acc = max(ah["val_accuracy"])
        y_pred_attack = np.argmax(attack_model.predict(X_val_attack, verbose=0), axis=1)
        attack_macro_f1 = f1_score(y_val_attack, y_pred_attack, average="macro", zero_division=0)

        fold_results.append({
            "binary_val_acc": binary_val_acc,
            "attack_val_acc": attack_val_acc,
            "attack_macro_f1": attack_macro_f1,
        })

        print(f"\nFold {fold + 1} results:")
        print(f"  Binary val acc:   {binary_val_acc:.4f}")
        print(f"  Attack val acc:   {attack_val_acc:.4f}")
        print(f"  Attack macro F1:  {attack_macro_f1:.4f}")

    # -----------------------------------------
    # Aggregate summary
    # -----------------------------------------
    print(f"\n{'='*50}")
    print("  Cross-Validation Summary")
    print(f"{'='*50}")
    for metric, key in [("Binary val acc", "binary_val_acc"), ("Attack val acc", "attack_val_acc"), ("Attack macro F1", "attack_macro_f1")]:
        vals = [r[key] for r in fold_results]
        print(f"  {metric}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # -----------------------------------------
    # Save training report
    # -----------------------------------------
    _save_report(cfg, args.config, le, fold_results, out_dir)


if __name__ == "__main__":
    main()
