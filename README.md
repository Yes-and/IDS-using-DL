# IDS using Deep Learning

Deep learning models (DNN / 1-D CNN with optional attention) for network intrusion detection, evaluated on the **XIIOTID** and **CICIDS-2019** datasets with a focus on minority-class performance.

---

## Repository Structure

```
IDS-using-DL/
├── configs/                      # YAML experiment configs (one per dataset × architecture)
│   ├── xiiotid_dnn.yaml
│   ├── xiiotid_cnn.yaml
│   ├── cicids2019_dnn.yaml
│   └── cicids2019_cnn.yaml
│
├── data/
│   ├── raw/                      # Place original dataset files here (gitignored)
│   │   ├── xiiotid/
│   │   └── cicids2019/
│   └── processed/                # Auto-generated preprocessed splits (gitignored)
│
├── notebooks/                    # Jupyter notebooks for EDA and results analysis
│   ├── 01_eda_xiiotid.ipynb
│   ├── 02_eda_cicids2019.ipynb
│   └── 03_results_analysis.ipynb
│
├── scripts/                      # Entry-point scripts
│   ├── preprocess.py             # Load raw data → normalise → save splits
│   ├── train.py                  # Train a model from a config file
│   └── evaluate.py               # Evaluate checkpoint, save metrics and plots
│
├── src/                          # Library code (importable package)
│   ├── data/
│   │   ├── xiiotid.py            # Raw data loader for XIIOTID
│   │   ├── cicids2019.py         # Raw data loader for CICIDS-2019
│   │   ├── preprocessing.py      # Label encoding, standard scaling, stratified split
│   │   └── dataset.py            # PyTorch Dataset wrapper (supports DNN + CNN input shapes)
│   │
│   ├── models/
│   │   ├── dnn.py                # Feedforward DNN with BatchNorm + Dropout
│   │   ├── cnn1d.py              # 1-D CNN (configurable depth and channels)
│   │   ├── attention.py          # FeatureAttention (tabular) + ChannelAttention1D (SE-block)
│   │   └── build.py              # Model factory: build from config dict
│   │
│   ├── training/
│   │   ├── trainer.py            # Training loop + validation, TensorBoard logging, checkpointing
│   │   └── losses.py             # Focal Loss with optional class weights
│   │
│   └── evaluation/
│       ├── metrics.py            # Per-class precision / recall / F1, macro & weighted aggregates
│       └── plots.py              # Confusion matrix, per-class F1 bar chart, ROC/PR curves
│
├── results/                      # Saved checkpoints, metrics JSON, and plots (gitignored)
├── requirements.txt
└── .gitignore
```

---

## Datasets

| Dataset | Description |
|---|---|
| **XIIOTID** | IoT network traffic dataset with both binary and multi-class attack labels. |
| **CICIDS-2019** | CIC Intrusion Detection dataset (2019) containing various modern attack types. |

Place the raw files under `data/raw/xiiotid/` and `data/raw/cicids2019/` respectively, then run the preprocessing script (see below).

---

## Quickstart

### 1. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Preprocess a dataset

```bash
python scripts/preprocess.py --dataset xiiotid
python scripts/preprocess.py --dataset cicids2019
```

Processed `.npy` splits and `label_encoder.pkl` / `scaler.pkl` are saved to `data/processed/<dataset>/`.

### 3. Train

```bash
python scripts/train.py --config configs/xiiotid_dnn.yaml
python scripts/train.py --config configs/cicids2019_cnn.yaml --run-name exp1
```

Monitor training with TensorBoard:

```bash
tensorboard --logdir results/tb_logs
```

### 4. Evaluate

```bash
python scripts/evaluate.py \
    --config configs/xiiotid_dnn.yaml \
    --checkpoint results/xiiotid_dnn_best.pt
```

Outputs a per-class classification report, saves `*_metrics.json`, confusion matrix PNG, and per-class F1 bar chart to `results/`.

---

## Models

| Architecture | Module | Notes |
|---|---|---|
| DNN | `src/models/dnn.py` | Fully-connected, configurable depth, BatchNorm + Dropout |
| 1-D CNN | `src/models/cnn1d.py` | Treats feature vector as a 1-D sequence |
| Feature Attention | `src/models/attention.py` | Soft gate over input features (plug into DNN) |
| Channel Attention | `src/models/attention.py` | SE-block for CNN feature maps |

Architecture and hyperparameters are controlled entirely through YAML configs in `configs/`.

---

## Handling Class Imbalance

Both datasets contain heavily imbalanced traffic classes. The following mechanisms are available:

- **Focal Loss** (`src/training/losses.py`) — down-weights easy majority-class examples.
- **Inverse-frequency class weights** — passed to the loss function at training time.
- **SMOTE / resampling** — via `imbalanced-learn` (add to `scripts/preprocess.py` as needed).
- **Per-class metrics** — `src/evaluation/metrics.py` reports per-class F1, precision, and recall alongside macro/weighted aggregates.

---

## Configuration

Each YAML file in `configs/` controls all aspects of an experiment:

```yaml
dataset: xiiotid          # which dataset to load from data/processed/
arch: dnn                 # 'dnn' | 'cnn1d'
hidden_dims: [256, 128, 64]
dropout: 0.3
epochs: 50
batch_size: 1024
lr: 1.0e-3
loss: focal               # 'focal' | 'cross_entropy'
use_class_weights: true
```

`input_dim` and `num_classes` are filled automatically at runtime from the processed data.
