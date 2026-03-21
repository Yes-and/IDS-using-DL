# IDS using Deep Learning

Deep learning models for network intrusion detection on the **XIIOTID** dataset, using a two-stage classifier (Normal vs Attack, then attack type). Handles class imbalance via inverse-frequency class weighting.

> For full details on current state, known issues, and file statuses, see [CLAUDE.md](CLAUDE.md).
> For methodology and results, see [METHODOLOGY.md](METHODOLOGY.md).

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

---

## How to run

### DNN pipeline

```bash
# 1. Preprocess — only needed once, or after raw data changes
python scripts/preprocess.py

# 2. Train — 5-fold CV on the 80% training pool
python scripts/train.py --config configs/xiiotid_dnn.yaml

# 3a. Evaluate all CV folds (aggregated mean ± std)
python scripts/evaluate.py --config configs/xiiotid_dnn.yaml

# 3b. Evaluate a single fold (0-indexed)
python scripts/evaluate.py --config configs/xiiotid_dnn.yaml --fold 0

# 3c. Evaluate the held-out test set (probability ensemble of all 5 folds)
#     Run this only once, after training is finalised — do not use for tuning
python scripts/evaluate.py --config configs/xiiotid_dnn.yaml --test
```

### Sklearn benchmark

```bash
# Train and evaluate Decision Tree, Random Forest, and Logistic Regression
# as two-stage pipelines on the same train/test split used by the DNN
python scripts/benchmark_sklearn.py --config configs/xiiotid_dnn.yaml
```

Each evaluation run (DNN or sklearn) outputs:
- Timestamped JSON to `results/metrics/`
- Confusion matrix and per-class F1 plots to `results/figures/`

---

## Repository structure

```
configs/                    YAML experiment configs (one per dataset × architecture)
data/
  raw/xiiotid/              Place raw CSV files here (gitignored)
  processed/xiiotid/        Auto-generated arrays + split indices (gitignored)
scripts/
  preprocess.py             Feature extraction and train/test split
  train.py                  5-fold CV training of the two-stage DNN
  evaluate.py               DNN evaluation (CV folds, single fold, or held-out test)
  benchmark_sklearn.py      Two-stage sklearn benchmark (DT, RF, LR)
src/
  data/                     Data loading and preprocessing
  models/                   TensorFlow DNN architecture and model factory
  training/                 Two-stage training loops (binary + attack-type)
  evaluation/               Metrics and plots
results/
  models/                   Saved model weights and scalers (gitignored)
  metrics/                  Timestamped metrics JSON per run (gitignored)
  figures/                  Confusion matrix and F1 plots (gitignored)
  reports/                  Auto-generated markdown training reports (gitignored)
```

---

## Datasets

Only **XIIOTID** is currently supported. Place raw CSV files under `data/raw/xiiotid/`.

CICIDS-2019 configs exist but the dataset loader has not been implemented yet.
