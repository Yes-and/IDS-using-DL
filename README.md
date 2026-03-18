# IDS using Deep Learning

Deep learning models for network intrusion detection on the **XIIOTID** dataset, using a two-stage classifier (Normal vs Attack, then attack type). Handles class imbalance via class weighting and Focal Loss.

> For full details on current state, known issues, and file statuses, see [CLAUDE.md](CLAUDE.md).

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## How to run

```bash
# 1. Preprocess
python scripts/preprocess.py

# 2. Train (two-stage: binary then attack-type)
python scripts/train.py --config configs/xiiotid_dnn.yaml

# 3. Evaluate
python scripts/evaluate.py --config configs/xiiotid_dnn.yaml --checkpoint <path>
```

---

## Repository structure

```
configs/            YAML experiment configs (one per dataset × architecture)
data/
  raw/xiiotid/      Place raw CSV files here (gitignored)
  processed/        Auto-generated splits (gitignored)
notebooks/          EDA and data loading experiments
scripts/            Entry-point scripts (preprocess, train, evaluate)
src/
  data/             Data loading and preprocessing
  models/           TensorFlow DNN architecture and model factory
  training/         Two-stage training loops (binary + attack-type)
  evaluation/       Metrics and plots
results/            Checkpoints, metrics JSON, plots (gitignored)
```

---

## Datasets

Only **XIIOTID** is currently supported. Place raw CSV files under `data/raw/xiiotid/`.

CICIDS-2019 configs exist but the dataset loader has not been implemented yet.
