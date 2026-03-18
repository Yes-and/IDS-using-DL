# CLAUDE.md — IDS-using-DL

Developer guide for humans and AI coding agents. Describes the **current state** of the codebase honestly, including what works and what doesn't.

---

## What this project does

Trains deep learning classifiers for network intrusion detection (IDS) on the **XIIOTID** dataset (IoT traffic). Uses a two-stage approach: a binary classifier (Normal vs Attack) followed by a multi-class classifier (attack type). Addresses severe class imbalance via class weighting and stratified splits.

---

## Current state (as of March 2026)

> The pipeline is end-to-end functional. All known blocking bugs have been fixed.

### What works
- Data loading and preprocessing for XIIOTID (`src/data/xiiotid.py`, `src/data/preprocessing.py`)
- `scripts/preprocess.py` — saves all `.npy` splits and both `.pkl` artefacts to `data/processed/xiiotid/`
- `scripts/train.py` — loads processed data, runs two-stage TF training, saves model weights
- `scripts/evaluate.py` — TF inference via `model.load_weights()` + `model.predict()`, saves metrics JSON + plots
- Two-stage TensorFlow training: binary model then attack-type model (`src/training/trainer_binary.py`, `src/training/trainer_attack.py`)
- Model factory (`src/models/build.py`) — TF only, extensible
- Evaluation metrics and plots (`src/evaluation/metrics.py`, `src/evaluation/plots.py`)

### What is incomplete or unsupported

| Item | Status |
|------|--------|
| CICIDS-2019 | Config exists (`configs/cicids2019_dnn.yaml`) but no data loader. Not supported. |

---

## Architecture

### Two-stage classification

```
Input flow → [Binary model] → Normal / Attack
                                    ↓
                              [Attack model] → Attack type label
```

- **Stage 1** (`trainer_binary.py`): 2-layer TF sigmoid classifier. Labels: `0 = Normal`, `1 = Attack`.
- **Stage 2** (`trainer_attack.py`): 3-layer TF softmax classifier. Trained only on attack samples. Labels are offset: `y - 1` so class 0 = first attack type.

### Framework

The entire codebase uses **TensorFlow / Keras**. PyTorch has been removed.

### Adding a new model architecture

1. Implement `build_<name>(input_dim, num_classes)` in `src/models/<name>.py` — return a compiled TF model.
2. Import it in `src/models/build.py` and add an `elif arch == "<name>"` branch.
3. Add a corresponding config file in `configs/`.

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
  train.py                      Two-stage training (requires --config)
  evaluate.py                   Inference + metrics + plots (requires --config + --checkpoint)
src/
  data/
    xiiotid.py                  XIIOTID CSV loader
    preprocessing.py            Encoding, scaling, stratified train/test split
                                Returns: X_train, X_test, yb_train, yb_test, ym_train, ym_test, le, scaler
  models/
    dnn.py                      TF DNN builder
    build.py                    Model factory: routes config["arch"] to the right builder
  training/
    trainer_binary.py           Stage 1 binary TF trainer (reads config)
    trainer_attack.py           Stage 2 attack-type TF trainer (reads config)
  evaluation/
    metrics.py                  full_report(): accuracy, F1, confusion matrix, per-class F1
    plots.py                    plot_confusion_matrix(), plot_per_class_f1()
results/                        Checkpoints, metrics JSON, plots (gitignored)
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

## Known issues / TODO

1. **CICIDS-2019** — implement dataset loader (`src/data/cicids2019.py`) if that dataset is needed
2. **`dnn.py` ignores `model.hidden_dims` and `model.dropout` from config** — architecture is hardcoded to 256→128→64→32; could be made config-driven
3. **No class weighting in binary trainer** — `trainer_binary.py` doesn't apply class weights; may matter if Normal/Attack ratio is very skewed
