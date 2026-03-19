# CLAUDE.md — IDS-using-DL

Developer guide for humans and AI coding agents. Describes the current state of the codebase honestly.

---

## What this project does

Trains deep learning classifiers for network intrusion detection (IDS) on the **XIIOTID** dataset (IoT traffic, 820k samples). Uses a two-stage approach: a binary classifier (Normal vs Attack) followed by a multi-class attack-type classifier. Addresses class imbalance via inverse-frequency class weighting and stratified k-fold CV.

---

## Current state

**Pipeline is fully functional and validated.** All three scripts run end-to-end. 5-fold CV and held-out test evaluation are implemented and verified.

### Final results (2026-03-19, 15 evaluated classes — 3 excluded below 1000 samples)

**5-fold CV (80% train pool):**

| Metric | Mean | Std |
|--------|------|-----|
| Binary accuracy | 98.65% | ±0.06% |
| Attack macro F1 | 98.52% | ±0.14% |
| Attack weighted F1 | 99.69% | ±0.03% |

**Held-out test set (20%, never seen during training):**

| Metric | Value |
|--------|-------|
| Binary accuracy | 98.71% |
| Attack macro F1 | 98.41% |
| Attack weighted F1 | 99.69% |

CV and test agree within ~0.1% — no overfitting, CV estimates were not optimistically biased.

**Caveats:**
- XIIOTID is a lab dataset with highly separable features. Near-perfect metrics are expected and are not a bug.
- `Fake_notification`, `MitM`, and `crypto-ransomware` are excluded from evaluation metrics (below 1000 samples) but the model trains on all 18 attack classes.

---

## Architecture

### Two-stage classification

```
Input → [Binary model] → Normal / Attack
                               ↓
                         [Attack model] → Attack type (18 classes trained, 15 evaluated)
```

- **Stage 1** (`trainer_binary.py`): 2-layer sigmoid classifier. Labels: `0 = Normal`, `1 = Attack`.
- **Stage 2** (`trainer_attack.py`): 3-layer softmax classifier. Trained only on attack samples. Attack labels are re-encoded with a fresh `LabelEncoder` per fold to produce contiguous 0-indexed classes with Normal excluded. The fitted encoder is saved and loaded at eval time — do not re-fit on val data.

### Framework

TensorFlow / Keras only. PyTorch has been removed.

### Model builders

The training pipeline does **not** go through `build_model()` in `src/models/build.py`. Each trainer has its own builder:
- `trainer_binary.py` → `build_binary_model(input_dim, lr)`
- `trainer_attack.py` → `build_attack_model(input_dim, num_classes, lr)`

`build_model()` exists for future use but is currently unused.

### Config system

All hyperparameters live in `configs/xiiotid_dnn.yaml`:

```yaml
dataset: xiiotid
arch: dnn

data:
  raw_path: data/raw/xiiotid
  processed_path: data/processed/xiiotid
  label_column: class1

model:
  hidden_dims: [256, 128, 64, 32]   # NOTE: currently hardcoded in trainers, not read from config
  dropout: 0.3                       # NOTE: same — hardcoded

training:
  n_folds: 5
  batch_size: 256
  epochs: 40
  learning_rate: 0.001
  class_weight: true

evaluation:
  eval_min_samples: 1000            # Classes below this are trained on but excluded from metrics
  test_size: 0.2                    # Fraction held out at preprocess time; CV runs on the rest

output:
  model_dir: results/models
  metrics_dir: results/metrics
  figures_dir: results/figures
```

---

## File map

```
pyproject.toml                  Editable install — run `pip install -e .` once after cloning
configs/
  xiiotid_dnn.yaml              Active config
data/
  raw/xiiotid/                  Raw CSV files (gitignored)
  processed/xiiotid/            X.npy, yb.npy, ym.npy, label_encoder.pkl, test_idx.npy
scripts/
  preprocess.py                 Saves full arrays + test_idx.npy to data/processed/xiiotid/
  train.py                      5-fold CV on train split only; saves per-fold artefacts
  evaluate.py                   default: evaluate all CV folds (filtered classes only)
                                --fold N: evaluate a single fold (0-indexed)
                                --test: evaluate held-out test set (ensemble of all folds)
src/
  data/
    xiiotid.py                  CSV loader
    preprocessing.py            Returns raw unscaled X (float32), yb, ym, le. No split, no scaling.
                                Drops: Date, Timestamp, Scr_IP, Des_IP, class2, class3
  models/
    dnn.py                      TF DNN builder (architecture currently hardcoded)
    build.py                    Model factory (unused by current pipeline)
  training/
    trainer_binary.py           Stage 1 trainer — supports class_weight kwarg
    trainer_attack.py           Stage 2 trainer
  evaluation/
    metrics.py                  full_report() → report_str, macro_f1, weighted_f1, cm, per_class
    plots.py                    plot_confusion_matrix(), plot_per_class_f1()
results/
  models/                       Per-fold weights + scalers + encoders (gitignored)
  metrics/                      Timestamped JSON outputs (gitignored)
  figures/                      Confusion matrix + F1 plots (gitignored)
  reports/                      Markdown training reports (gitignored)
```

---

## How to run

```bash
# 1. Preprocess — saves arrays + test_idx.npy; only needed once or after raw data changes
python scripts/preprocess.py

# 2. Train — 5-fold CV on train split only
python scripts/train.py --config configs/xiiotid_dnn.yaml

# 3a. Evaluate CV folds (filtered classes, eval_min_samples threshold applied)
python scripts/evaluate.py --config configs/xiiotid_dnn.yaml
python scripts/evaluate.py --config configs/xiiotid_dnn.yaml --fold 0

# 3b. Evaluate held-out test set (run after training is finalised — do not use to tune)
python scripts/evaluate.py --config configs/xiiotid_dnn.yaml --test
```

Per-fold artefacts saved by `train.py`:
- `binary_model_fold{k}.weights.h5`
- `attack_model_fold{k}.weights.h5`
- `attack_label_encoder_fold{k}.pkl`
- `scaler_fold{k}.pkl`

---

## Key decisions

**Why two-stage?** A single multi-class model is dominated by the Normal majority. Separating the binary decision from attack-type classification lets each model focus on its own distribution.

**Why stratified k-fold instead of time-based split?** A temporal split was tried but abandoned: rare attack classes only appear in certain time windows, so some classes were entirely absent from val/test, breaking `attack_le.transform`. XIIOTID timestamps are simulation artifacts with no temporal signal.

**Why scaler is per-fold?** Fitting `StandardScaler` on the full dataset before splitting would leak val/test statistics into training. The scaler is fit on `X[train_idx]` only and saved alongside the model for use at eval time.

**Why high scores?** Investigated and confirmed: XIIOTID is a lab dataset with highly separable features. Near-perfect metrics are expected. Identity columns (`Date`, `Timestamp`, `Scr_IP`, `Des_IP`) were dropped but had minimal impact (~0.3%). This is a known dataset limitation.

**Why are 3 classes excluded from evaluation?** `Fake_notification` (~15 samples total), `MitM` (~110), and `crypto-ransomware` (~440) fall below `eval_min_samples: 1000`. A single misclassification swings their F1 by 30%+, making the metric meaningless. The model is trained on all 18 classes — exclusion is evaluation-only and is disclosed explicitly. The threshold is configurable.

**Why 80/20 for the held-out split?** The rarest class (`Fake_notification`, ~15 samples) needs at least 3 test samples for `sklearn`'s stratified split to work. 80/20 achieves this; 90/10 risks a stratification failure. The split is stratified on `ym` (multi-class labels) to guarantee all 18 attack types appear in both partitions.

**Why stratify the held-out split on `ym` not `yb`?** Stratifying on binary labels would guarantee Normal/Attack proportions but could leave some rare attack types entirely in one partition. Stratifying on `ym` is the stricter constraint.

**Why inverse-frequency class weighting?** XIIOTID is severely imbalanced — RDOS alone is ~28k samples while `Reverse_shell` is ~200. `compute_class_weight('balanced')` is applied to both the binary model and the attack-type model so rare classes are not drowned out during training.

**How does `--test` ensemble work?** Averages sigmoid/softmax probabilities across all 5 fold models (each fold's own scaler is applied before prediction), then argmax. Valid because `attack_le` is fit with `LabelEncoder` which sorts alphabetically — classes are in the same order across all folds. Do not use `--test` results to tune hyperparameters; it is a final, one-shot evaluation.

**Why `ym` stores integers, not class name strings?** `preprocessing.py` returns `ym` as the output of `LabelEncoder.transform()` — integer codes 0..18. Comparisons like `ym == "BruteForce"` will silently return all-False. Always use `enumerate(le.classes_)` to get `(index, name)` pairs when computing per-class counts from `ym`.

**How are excluded-class predictions handled in evaluation?** `keep_mask` filters rows where the *true* label is a kept class. The model can still predict an excluded class for those rows. These predictions are mapped to a dummy index `n_kept` (one beyond the kept range) via `label_map.get(y, n_kept)`, so they count as misclassifications. `full_report()` receives `labels=list(range(n_kept))` to restrict sklearn's reporting to the kept classes only.

---

## Known issues / TODO

1. **`dnn.py` architecture is hardcoded** — `model.hidden_dims` and `model.dropout` in the config are not used; trainers build a fixed architecture.
2. **CICIDS-2019 unsupported** — config exists but no data loader.
