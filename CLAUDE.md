# CLAUDE.md — IDS-using-DL

Developer guide for humans and AI coding agents. Describes the **current state** of the codebase honestly, including what works and what doesn't.

---

## What this project does

Trains deep learning classifiers for network intrusion detection (IDS) on the **XIIOTID** dataset (IoT traffic). Uses a two-stage approach: a binary classifier (Normal vs Attack) followed by a multi-class classifier (attack type). Addresses severe class imbalance via class weighting and stratified splits.

---

## Current state

> **Pipeline is partially functional.** Preprocessing, training, and evaluation all run, but the
> splitting strategy is in transition — see "Next task" below before running anything.

### What works
- Data loading and preprocessing for XIIOTID (`src/data/xiiotid.py`, `src/data/preprocessing.py`)
- `scripts/preprocess.py` — saves `.npy` arrays and `.pkl` artefacts to `data/processed/xiiotid/`
- `scripts/train.py` — loads processed data, runs two-stage TF training, saves model weights + `attack_label_encoder.pkl`, auto-generates a markdown training report to `results/reports/`
- `scripts/evaluate.py` — two-stage TF inference, evaluates binary and attack-type models separately, saves timestamped metrics JSON + plots
- Two-stage TensorFlow training: binary model then attack-type model (`src/training/trainer_binary.py`, `src/training/trainer_attack.py`)
- Model factory (`src/models/build.py`) — TF only, extensible
- Evaluation metrics and plots (`src/evaluation/metrics.py`, `src/evaluation/plots.py`)

### What is broken / in progress
- `preprocessing.py` currently does a **time-based 80/10/10 split** which is being replaced with
  stratified k-fold CV (see "Next task"). Do not treat current preprocessed `.npy` files as valid.

---

## Architecture

### Two-stage classification

```
Input flow → [Binary model] → Normal / Attack
                                    ↓
                              [Attack model] → Attack type label
```

- **Stage 1** (`trainer_binary.py`): 2-layer TF sigmoid classifier. Labels: `0 = Normal`, `1 = Attack`.
- **Stage 2** (`trainer_attack.py`): 3-layer TF softmax classifier. Trained only on attack samples. After filtering to attack samples, labels are re-encoded with a fresh `LabelEncoder` to produce contiguous 0-indexed classes (18 classes, Normal excluded).

### Framework

The entire codebase uses **TensorFlow / Keras**. PyTorch has been removed.

### Model builders vs model factory

**Important:** the training pipeline does **not** go through `build_model()` in `src/models/build.py`. Each trainer has its own builder:
- `trainer_binary.py` → `build_binary_model(input_dim, lr)`
- `trainer_attack.py` → `build_attack_model(input_dim, num_classes, lr)`

`build_model()` / `src/models/build.py` is used only by `evaluate.py` (and currently unused there too, since `evaluate.py` calls the trainer builders directly). It exists for future use if a unified builder is needed.

### Adding a new model architecture

1. Add a new `build_<name>(input_dim, lr)` / `build_<name>(input_dim, num_classes, lr)` function — either in the relevant trainer file or in a new `src/models/<name>.py`.
2. Wire it into the trainer that needs it.
3. If you want it accessible via `build_model()`, also import it in `src/models/build.py` and add an `elif` branch.
4. Add a corresponding config file in `configs/`.

### Config system

All hyperparameters are in `configs/*.yaml`. Key fields:

```yaml
dataset: xiiotid
arch: dnn                        # top-level key; maps to build_model() in src/models/build.py

data:
  raw_path: data/raw/xiiotid
  processed_path: data/processed/xiiotid    # directory, not a file
  label_column: class1

training:
  n_folds: 5                   # stratified k-fold CV

model:
  hidden_dims: [256, 128, 64, 32]
  dropout: 0.3

training:
  batch_size: 256
  epochs: 40
  learning_rate: 0.001
  class_weight: true

output:
  model_dir: results/models
  metrics_dir: results/metrics
  figures_dir: results/figures
```

Active config: `configs/xiiotid_dnn.yaml`. (`configs/cicids2019_dnn.yaml` exists but dataset is unsupported.)

---

## File map

```
pyproject.toml                  Editable install config — run `pip install -e .` once after cloning
configs/
  xiiotid_dnn.yaml              Experiment config for XIIOTID + DNN
  cicids2019_dnn.yaml           Experiment config for CICIDS-2019 + DNN (dataset not supported)
data/
  raw/xiiotid/                  Place raw CSV files here (gitignored)
  processed/xiiotid/            X.npy, yb.npy, ym.npy, label_encoder.pkl
                                (splits and per-fold scalers generated at training time, not saved here)
notebooks/
  explore_data.ipynb            EDA
  test_loading.ipynb            Data loading experiments
scripts/
  preprocess.py                 Load + preprocess + save all outputs
  train.py                      Two-stage training (requires --config); auto-saves report to results/reports/
  evaluate.py                   Two-stage inference + metrics + plots (requires --config + --checkpoint)
src/
  data/
    xiiotid.py                  XIIOTID CSV loader
    preprocessing.py            Encodes labels, drops identity + label columns, returns raw unscaled
                                arrays: X, yb, ym, le. Scaling happens per-fold in train.py.
                                Note: drops Date, Timestamp, Scr_IP, Des_IP, class2, class3 from features
  models/
    dnn.py                      TF DNN builder
    build.py                    Model factory: routes config["arch"] to the right builder
  training/
    trainer_binary.py           Stage 1 binary TF trainer — returns (model, history)
    trainer_attack.py           Stage 2 attack-type TF trainer — returns (model, history)
  evaluation/
    metrics.py                  full_report(): accuracy, F1, confusion matrix, per-class F1
    plots.py                    plot_confusion_matrix(), plot_per_class_f1()
results/
  models/                       Saved model weights (gitignored)
  metrics/                      Metrics JSON per evaluation run (gitignored)
  figures/                      Confusion matrix + F1 plots (gitignored)
  reports/                      Auto-generated markdown training reports (gitignored)
```

---

## How to run

```bash
# 1. Preprocess — saves processed data to data/processed/xiiotid/
python scripts/preprocess.py

# 2. Train — two-stage: binary then attack-type; saves weights to results/models/
python scripts/train.py --config configs/xiiotid_dnn.yaml

# 3. Evaluate
python scripts/evaluate.py --config configs/xiiotid_dnn.yaml --checkpoint results/models/attack_model.weights.h5
```

---

## Investigation findings — suspiciously good results (2026-03-18)

The pipeline was producing near-perfect metrics. Root causes investigated:

| Fix | Impact | Status |
|-----|--------|--------|
| Drop identity features (`Date`, `Timestamp`, `Scr_IP`, `Des_IP`) | Small drop (~0.3–0.4%) — not the main driver | Done |
| Separate val set from test set (was using test set for early stopping) | Minimal impact — model was not over-fitting to test set via callbacks | Done |
| Time-based split | **Abandoned** — see decision below | Reverted |
| Add class weighting to binary model | Pending — subsumed into CV work | Not started |

**Conclusion:** scores remain high (~99%) after fixes. The dataset itself appears to be the cause —
XIIOTID is a lab dataset where traffic features are highly separable by class. This is a known
limitation of the dataset, not a bug in the pipeline.

### Decision: stratified k-fold CV instead of time-based split

A time-based split was implemented but exposed a hard constraint: rare attack classes only appear
in certain time windows, so a temporal cut leaves some classes entirely absent from val/test. This
makes evaluation unreliable for those classes and breaks `attack_le.transform` on unseen labels.

**Decision (2026-03-18):** Replace the time-based split with **stratified k-fold cross-validation**.
This guarantees all classes appear in every fold's train and val sets, giving reliable per-class
metrics across all 18 attack types. The tradeoff (no temporal ordering) is acceptable because
XIIOTID is a lab dataset — timestamps are simulation artifacts, not organic traffic evolution.

---

## Next task — implement stratified k-fold CV `[ ]`

### What needs to change

**`src/data/preprocessing.py`**
- Remove the time-based sort and index-based split
- Remove scaling entirely — return raw unscaled `X`, `yb`, `ym`, `le`
- Scaler must be fit per-fold in `train.py` to avoid leaking val/test statistics

**`scripts/preprocess.py`**
- Save `X.npy`, `yb.npy`, `ym.npy` (full arrays, no splits)
- Save `label_encoder.pkl`; do not save `scaler.pkl` (scaler is now per-fold)

**`scripts/train.py`**
- Read `n_folds` from config (`training.n_folds`)
- Use `StratifiedKFold(n_splits=n_folds)` stratified on `yb`
- For each fold:
  1. Split indices into train/val
  2. Fit `StandardScaler` on `X[train_idx]`, transform both train and val
  3. If `training.class_weight: true`, compute balanced weights for binary labels and pass to binary `model.fit()`
  4. Train binary model on fold's train, validate on fold's val
  5. Filter to attack samples, re-encode attack labels with fresh `attack_le`
  6. Train attack model on fold's attack-train, validate on fold's attack-val
  7. Save `binary_model_fold{k}.weights.h5`, `attack_model_fold{k}.weights.h5`,
     `attack_label_encoder_fold{k}.pkl`, `scaler_fold{k}.pkl` to `results/models/`
  8. Record per-fold val metrics (binary acc, attack acc, macro F1)
- After loop: aggregate and report mean ± std for key metrics

**`scripts/evaluate.py`**
- Accept `--fold N` to evaluate a single fold, or iterate all folds if omitted
- For each fold being evaluated: load `scaler_fold{k}.pkl`, scale the full dataset,
  then run the two-stage inference on the fold's val indices
- Aggregate metrics across folds and save a single timestamped JSON + plots

**`results/reports/`**
- Training report should show a per-fold metrics table and a mean ± std summary row

### Checklist for implementing (follow in order)

- [ ] Rewrite `preprocessing.py` — remove split and scaling; return `X`, `yb`, `ym`, `le`
- [ ] Update `preprocess.py` — save `X.npy`, `yb.npy`, `ym.npy` and `label_encoder.pkl` only
- [ ] Update config `xiiotid_dnn.yaml` — remove `test_size`, add `n_folds: 5`
- [ ] Rewrite `train.py` — k-fold loop with per-fold scaling, binary class weighting, training, and saving
- [ ] Update `_save_report()` in `train.py` — per-fold table + mean ± std summary
- [ ] Rewrite `evaluate.py` — load fold artefacts, run inference on val indices, aggregate
- [ ] Run pipeline end-to-end and verify all folds complete without class-coverage errors
- [ ] Compare aggregated CV metrics to previous single-run metrics; document findings

---

## Known issues / TODO

1. **CICIDS-2019** — config exists (`configs/cicids2019_dnn.yaml`) but no data loader; implement `src/data/cicids2019.py` if needed
2. **`dnn.py` ignores `model.hidden_dims` and `model.dropout` from config** — architecture is hardcoded to 256→128→64→32; could be made config-driven
