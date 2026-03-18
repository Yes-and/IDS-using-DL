# CLAUDE.md — IDS-using-DL

Developer guide for humans and AI coding agents. Describes the **current state** of the codebase honestly, including what works and what doesn't.

---

## What this project does

Trains deep learning classifiers for network intrusion detection (IDS) on the **XIIOTID** dataset (IoT traffic). Uses a two-stage approach: a binary classifier (Normal vs Attack) followed by a multi-class classifier (attack type). Addresses severe class imbalance via class weighting and Focal Loss.

---

## Current state (as of March 2026)

> The codebase is a work-in-progress. The training pipeline is functional but not config-driven. The evaluation pipeline is written but not yet connected to the trained models.

### What works
- Data loading and preprocessing for XIIOTID (`src/data/xiiotid.py`, `src/data/preprocessing.py`)
- Two-stage TensorFlow training: binary model then attack-type model (`src/training/trainer_binary.py`, `src/training/trainer_attack.py`)
- Evaluation metrics and plots (`src/evaluation/metrics.py`, `src/evaluation/plots.py`)

### What is broken or incomplete

| File | Issue |
|------|-------|
| `scripts/preprocess.py` | Never saves `.npy` files — only prints results. `evaluate.py` depends on those files. |
| `scripts/train.py` | Imports `load_dataset` from `src.data.dataset`, which does not exist (only `IDSDataset` class is there). Crashes on import. |
| `src/models/build.py` | Signature is `build_model(config, input_dim, num_classes)` but `evaluate.py` calls `build_model(cfg)`. Also only routes to DNN; CNN is unhandled. Key lookup uses `config["model"]["type"]` but configs use `config["arch"]`. |
| `evaluate.py` ↔ trainers | **Framework mismatch**: `evaluate.py` does `torch.load()` PyTorch inference, but the trainers save TensorFlow models. These two pipelines do not connect. |
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

### Models

| Model | Framework | File |
|-------|-----------|------|
| DNN (main) | TensorFlow / Keras | `src/models/dnn.py` |
| 1D-CNN | PyTorch | `src/models/cnn1d.py` |
| Binary classifier | TensorFlow / Keras | `src/training/trainer_binary.py` |
| Attack classifier | TensorFlow / Keras | `src/training/trainer_attack.py` |
| Attention modules | PyTorch | `src/models/attention.py` |
| Focal Loss | PyTorch | `src/training/losses.py` |

> Note: TF and PyTorch both exist in the repo. The active training pipeline is TF. The PyTorch components (CNN, attention, Focal Loss, `evaluate.py`) form a second pipeline that is not yet complete.

### Config system

All hyperparameters are in `configs/*.yaml`. Key fields:

```yaml
dataset: xiiotid
arch: dnn | cnn1d
input_dim: null        # filled at runtime
num_classes: null      # filled at runtime
hidden_dims: [...]     # DNN layer sizes
channels: [...]        # CNN channel sizes
dropout: 0.3
epochs: 50
batch_size: 1024
lr: 1.0e-3
loss: focal | cross_entropy
use_class_weights: true
```

Config files: `configs/xiiotid_dnn.yaml`, `configs/xiiotid_cnn.yaml`, `configs/cicids2019_dnn.yaml`, `configs/cicids2019_cnn.yaml`.

---

## File map

```
configs/                        YAML experiment configs
data/
  raw/xiiotid/                  Place raw CSV files here
  processed/xiiotid/            X_train.npy, X_test.npy, y_train.npy, y_test.npy,
                                label_encoder.pkl, scaler.pkl  (not yet generated)
notebooks/
  explore_data.ipynb            EDA
  test_loading.ipynb            Data loading experiments
scripts/
  preprocess.py                 Entry point: load + preprocess (broken: doesn't save)
  train.py                      Entry point: two-stage training (broken: bad import)
  evaluate.py                   Entry point: inference + metrics (blocked by build.py bug)
src/
  data/
    xiiotid.py                  XIIOTID CSV loader
    preprocessing.py            Encoding, scaling, train/test split
    dataset.py                  PyTorch IDSDataset wrapper
  models/
    dnn.py                      TF DNN builder
    cnn1d.py                    PyTorch 1D-CNN builder
    attention.py                FeatureAttention, ChannelAttention1D (PyTorch)
    build.py                    Model factory (broken: wrong signature + key)
  training/
    trainer.py                  Generic TF trainer (WIP: not wired in)
    trainer_binary.py           Stage 1 binary TF trainer
    trainer_attack.py           Stage 2 attack-type TF trainer
    losses.py                   PyTorch FocalLoss
  evaluation/
    metrics.py                  full_report(): accuracy, F1, confusion matrix
    plots.py                    plot_confusion_matrix(), plot_per_class_f1()
results/                        Saved model checkpoints, metrics JSON, plots
```

---

## How to run (what actually works today)

```bash
# 1. Preprocess (note: does not yet save files — prints only)
python scripts/preprocess.py

# 2. Training can be invoked by calling the trainers directly from a notebook or script:
#    trainer_binary.py: train_binary_model(X_train, yb_train, X_val, yb_val)
#    trainer_attack.py: train_attack_model(X_train, ym_train, X_val, ym_val, class_weights)

# 3. Evaluate — blocked until build.py signature is fixed and preprocess saves files
python scripts/evaluate.py --config configs/xiiotid_dnn.yaml --checkpoint <path>
```

---

## Known issues / TODO

1. `scripts/preprocess.py` — add `np.save` / `pickle.dump` to persist processed splits
2. `scripts/train.py` — fix broken `load_dataset` import; wire in YAML config
3. `src/models/build.py` — fix signature to `build_model(config)`; fix key from `config["model"]["type"]` to `config["arch"]`; add CNN routing
4. Framework split — decide whether to unify on TF or PyTorch; `evaluate.py` currently only works with PyTorch checkpoints
5. `src/training/trainer.py` — replace hardcoded hyperparams with values from `config`
6. CICIDS-2019 — re-implement loader if that dataset is needed
