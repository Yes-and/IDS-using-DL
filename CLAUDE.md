# CLAUDE.md — IDS-using-DL

Developer guide for humans and AI coding agents. Describes the **current state** of the codebase honestly, including what works and what doesn't.

---

## What this project does

Trains deep learning classifiers for network intrusion detection (IDS) on the **XIIOTID** dataset (IoT traffic). Uses a two-stage approach: a binary classifier (Normal vs Attack) followed by a multi-class classifier (attack type). Addresses severe class imbalance via class weighting and stratified splits.

---

## Current state

> The pipeline is end-to-end functional. All known blocking bugs have been fixed.

### What works
- Data loading and preprocessing for XIIOTID (`src/data/xiiotid.py`, `src/data/preprocessing.py`)
- `scripts/preprocess.py` — saves all `.npy` splits and both `.pkl` artefacts to `data/processed/xiiotid/`
- `scripts/train.py` — loads processed data, runs two-stage TF training, saves model weights, auto-generates a markdown training report to `results/reports/`
- `scripts/evaluate.py` — two-stage TF inference, evaluates binary and attack-type models separately, saves metrics JSON + plots
- Two-stage TensorFlow training: binary model then attack-type model (`src/training/trainer_binary.py`, `src/training/trainer_attack.py`)
- Model factory (`src/models/build.py`) — TF only, extensible
- Evaluation metrics and plots (`src/evaluation/metrics.py`, `src/evaluation/plots.py`)

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
  test_size: 0.2

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
  processed/xiiotid/            X_train.npy, X_test.npy, yb_train.npy, yb_test.npy,
                                ym_train.npy, ym_test.npy, label_encoder.pkl, scaler.pkl
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
    preprocessing.py            Encoding, scaling, stratified train/test split
                                Returns: X_train, X_test, yb_train, yb_test, ym_train, ym_test, le, scaler
                                Note: drops class2, class3 from features; applies pd.to_numeric safety net
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

## Investigation plan — suspiciously good results

The pipeline produces metrics that are too good to trust. Fixing the issues below one at a time
(retrain + evaluate after each step) will reveal how much each one inflates scores.

### Step 1 — Drop identity features `[x]`

**File:** `src/data/preprocessing.py`

`Date`, `Timestamp`, `Scr_IP`, `Des_IP` are factorized to integers and kept in `X`.
In this lab dataset specific IP addresses and timestamps likely map 1-to-1 to attack types,
so the model can memorise identity instead of learning traffic behaviour.

**Change:** Explicitly drop these columns before building `X` (after factorizing object columns,
before the train/test split).

**Expected signal:** If accuracy drops substantially, label leakage via identity features was
the main driver of inflated scores.

---

### Step 2 — Separate validation set from test set `[ ]`

**Files:** `src/data/preprocessing.py`, `scripts/train.py`

Both `EarlyStopping` and `ReduceLROnPlateau` currently use `X_test`/`y_test` as
`validation_data`, so the model is implicitly optimised against the held-out set.

**Change:** Carve out a dedicated validation split from the training data (e.g. 80/10/10
train/val/test). Pass the val split to trainers and keep the test split untouched until
final evaluation in `evaluate.py`.

**Expected signal:** Metrics on the true test set should be somewhat lower and more honest.

---

### Step 3 — Switch to a time-based split `[ ]`

**File:** `src/data/preprocessing.py`

`train_test_split` randomly shuffles rows. For IDS, training on earlier traffic and testing
on later traffic is more representative of real deployment.

**Change:** Sort by `Timestamp` (or `Date`) before splitting, then cut at a fixed index
instead of using `train_test_split`. (Do this after Step 1 removes `Timestamp` from features —
it can still be used for ordering before being dropped.)

**Expected signal:** Metrics may drop further if the model was benefiting from seeing future
traffic patterns during training.

---

### Step 4 — Add class weighting to the binary model `[ ]`

**File:** `src/training/trainer_binary.py`

The attack-type model uses `class_weight` but the binary model does not. If Normal >> Attack
in the dataset the binary model may be biased toward predicting Normal.

**Change:** Compute balanced class weights for the binary labels (same pattern as `train.py`
does for the attack model) and pass them to `model.fit()`.

**Expected signal:** Binary recall on the Attack class should improve; overall accuracy may
dip slightly.

---

## Other known issues / TODO

1. **CICIDS-2019** — config exists (`configs/cicids2019_dnn.yaml`) but no data loader; implement `src/data/cicids2019.py` if needed
2. **`dnn.py` ignores `model.hidden_dims` and `model.dropout` from config** — architecture is hardcoded to 256→128→64→32; could be made config-driven
