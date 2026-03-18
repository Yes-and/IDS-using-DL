# CLAUDE.md — IDS-using-DL

Developer guide for humans and AI coding agents. Describes the **current state** of the codebase honestly, including what works and what doesn't.

---

## What this project does

Trains deep learning classifiers for network intrusion detection (IDS) on the **XIIOTID** dataset (IoT traffic). Uses a two-stage approach: a binary classifier (Normal vs Attack) followed by a multi-class classifier (attack type). Addresses severe class imbalance via class weighting and Focal Loss.

---

## Current state (as of March 2026)

> The codebase is a work-in-progress. The training pipeline is functional but not config-driven. The evaluation pipeline is written but blocked by two remaining bugs.

### What works
- Data loading and preprocessing for XIIOTID (`src/data/xiiotid.py`, `src/data/preprocessing.py`)
- Two-stage TensorFlow training: binary model then attack-type model (`src/training/trainer_binary.py`, `src/training/trainer_attack.py`)
- Model factory (`src/models/build.py`) — TF only, extensible
- Evaluation metrics and plots (`src/evaluation/metrics.py`, `src/evaluation/plots.py`)
- Evaluation script (`scripts/evaluate.py`) — TF inference via `model.load_weights()` + `model.predict()`

### What is broken or incomplete

| File | Issue |
|------|-------|
| `scripts/preprocess.py` | Never saves `.npy` / `.pkl` files — only prints results. `evaluate.py` depends on those files. |
| `scripts/train.py` | Imports `load_dataset` from `src.data.dataset`, which does not exist. Crashes on import. Also ignores YAML config. |
| `src/training/trainer.py` | WIP generic trainer — accepts `config` but still uses hardcoded `epochs=40, batch_size=256`. Not yet wired into any script. |
| CICIDS-2019 | Dataset loader was removed (was a stub). Configs exist but the dataset is not supported. |

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
arch: dnn             # maps to build_model() in src/models/build.py
input_dim: null       # filled at runtime
num_classes: null     # filled at runtime
hidden_dims: [...]    # DNN layer sizes
dropout: 0.3
epochs: 50
batch_size: 1024
lr: 1.0e-3
loss: focal | cross_entropy
use_class_weights: true
```

Active config files: `configs/xiiotid_dnn.yaml`, `configs/cicids2019_dnn.yaml`.

---

## File map

```
configs/
  xiiotid_dnn.yaml              Experiment config for XIIOTID + DNN
  cicids2019_dnn.yaml           Experiment config for CICIDS-2019 + DNN (dataset not yet supported)
data/
  raw/xiiotid/                  Place raw CSV files here (gitignored)
  processed/xiiotid/            X_train.npy, X_test.npy, y_train.npy, y_test.npy,
                                label_encoder.pkl, scaler.pkl  (not yet generated — see BUG-1)
notebooks/
  explore_data.ipynb            EDA
  test_loading.ipynb            Data loading experiments
scripts/
  preprocess.py                 Load + preprocess  (broken: doesn't save output — BUG-1)
  train.py                      Two-stage training (broken: bad import + no config — BUG-2)
  evaluate.py                   Inference + metrics + plots
src/
  data/
    xiiotid.py                  XIIOTID CSV loader
    preprocessing.py            Encoding, scaling, stratified train/test split
  models/
    dnn.py                      TF DNN builder + Focal Loss
    build.py                    Model factory: routes config["arch"] to the right builder
  training/
    trainer.py                  Generic TF trainer (WIP: hardcoded hyperparams — BUG-3)
    trainer_binary.py           Stage 1 binary TF trainer
    trainer_attack.py           Stage 2 attack-type TF trainer
  evaluation/
    metrics.py                  full_report(): accuracy, F1, confusion matrix
    plots.py                    plot_confusion_matrix(), plot_per_class_f1()
results/                        Checkpoints, metrics JSON, plots (gitignored)
```

---

## How to run (what actually works today)

```bash
# 1. Preprocess (note: does not yet save files — prints only)
python scripts/preprocess.py

# 2. Training must be invoked directly from a notebook or custom script:
#    trainer_binary.py: train_binary_model(X_train, yb_train, X_val, yb_val)
#    trainer_attack.py: train_attack_model(X_train, ym_train, X_val, ym_val, class_weights)

# 3. Evaluate (blocked until preprocess.py saves files)
python scripts/evaluate.py --config configs/xiiotid_dnn.yaml --checkpoint <path.weights.h5>
```

---

## Known issues / TODO

1. **BUG-1** `scripts/preprocess.py` — add `np.save` / `pickle.dump` to persist processed splits to `data/processed/`
2. **BUG-2** `scripts/train.py` — fix broken `load_dataset` import; wire in YAML config via `--config` argument
3. **BUG-3** `src/training/trainer.py` — replace hardcoded `epochs`/`batch_size` with values read from `config`
4. **CICIDS-2019** — re-implement dataset loader (`src/data/cicids2019.py`) if that dataset is needed
