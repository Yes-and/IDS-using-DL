# CLAUDE.md — IDS-using-DL

Developer guide for humans and AI coding agents. Describes the current state of the codebase honestly.

---

## What this project does

Trains deep learning classifiers for network intrusion detection (IDS) on the **XIIOTID** dataset (IoT traffic). Uses a two-stage approach: a binary classifier (Normal vs Attack) followed by a multi-class attack-type classifier. Addresses class imbalance via class weighting and stratified k-fold CV.

---

## Current state

**Pipeline is fully functional.** All three scripts run end-to-end. 5-fold CV is implemented and validated.

### Verified results (2026-03-18, 5-fold stratified CV)

| Metric | Mean | Std |
|--------|------|-----|
| Binary accuracy | 98.66% | ±0.09% |
| Attack accuracy (weighted F1) | 99.68% | ±0.08% |
| Attack macro F1 | 98.12% | ±0.23% |

**Caveats:**
- The gap between weighted F1 (99.68%) and macro F1 (98.12%) is driven by a few very rare classes (`Fake_notification`: 3 samples/fold, `MitM`: 22, `crypto-ransomware`: 88). Their per-class F1 scores are unreliable at this sample size.
- XIIOTID is a lab dataset with highly separable features. These scores reflect the dataset as much as the model — high results are expected and are not a bug.
- Metrics are on CV validation folds only. There is no held-out test set; a locked-away test partition would be needed for a final paper result.

---

## Architecture

### Two-stage classification

```
Input → [Binary model] → Normal / Attack
                               ↓
                         [Attack model] → Attack type (18 classes)
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
  cicids2019_dnn.yaml           Unused (no data loader for CICIDS-2019)
data/
  raw/xiiotid/                  Raw CSV files (gitignored)
  processed/xiiotid/            X.npy, yb.npy, ym.npy, label_encoder.pkl
scripts/
  preprocess.py                 Saves full arrays to data/processed/xiiotid/
  train.py                      5-fold CV training; saves per-fold artefacts to results/models/
  evaluate.py                   Loads per-fold artefacts; aggregates metrics to results/metrics/
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
# 1. Preprocess — only needed once, or after raw data changes
python scripts/preprocess.py

# 2. Train — runs 5-fold CV; saves per-fold artefacts to results/models/
python scripts/train.py --config configs/xiiotid_dnn.yaml

# 3. Evaluate — all folds aggregated; or --fold N for a single fold
python scripts/evaluate.py --config configs/xiiotid_dnn.yaml
python scripts/evaluate.py --config configs/xiiotid_dnn.yaml --fold 0
```

Per-fold artefacts saved by `train.py`:
- `binary_model_fold{k}.weights.h5`
- `attack_model_fold{k}.weights.h5`
- `attack_label_encoder_fold{k}.pkl`
- `scaler_fold{k}.pkl`

---

## Key decisions

**Why two-stage?** A single multi-class model is dominated by the Normal majority. Separating the binary decision from attack-type classification lets each model focus on its own distribution.

**Why stratified k-fold instead of time-based split?** A temporal split was tried but abandoned: rare attack classes only appear in certain time windows, so some classes were entirely absent from val/test, breaking `attack_le.transform`. XIIOTID timestamps are simulation artifacts anyway, so temporal ordering has no semantic value.

**Why scaler is per-fold?** Fitting `StandardScaler` on the full dataset before splitting would leak val/test statistics into training. The scaler is fit on `X[train_idx]` only and saved alongside the model for use at eval time.

**Why high scores?** Investigated and confirmed: XIIOTID is a lab dataset with highly separable features. Near-perfect metrics are expected. Identity columns (`Date`, `Timestamp`, `Scr_IP`, `Des_IP`) were dropped but had minimal impact (~0.3%). This is a known dataset limitation.

---

## Known issues / TODO

1. **`dnn.py` architecture is hardcoded** — `model.hidden_dims` and `model.dropout` in the config are not used; trainers build a fixed 256→128→64→32 network.
2. **No held-out test set** — all reported metrics are on CV val folds. Add a locked test split before publishing results.
3. **Rare class metrics are unreliable** — `Fake_notification` (3 samples/fold), `MitM` (22), `crypto-ransomware` (88) have too few samples for meaningful per-class F1.
4. **CICIDS-2019 unsupported** — config exists but no data loader.
