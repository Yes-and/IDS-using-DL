# CLAUDE.md — IDS-using-DL

Developer guide for humans and AI coding agents. Describes the current state of the codebase honestly.

---

## What this project does

Trains deep learning classifiers for network intrusion detection (IDS) on the **XIIOTID** dataset (IoT traffic, 820k samples). Uses a two-stage approach: a binary classifier (Normal vs Attack) followed by a multi-class attack-type classifier. Addresses class imbalance via inverse-frequency class weighting and stratified k-fold CV.

---

## Current state

**Pipeline is fully functional and validated.** All three scripts run end-to-end. 5-fold CV, held-out test evaluation, and end-to-end cascade evaluation are implemented and verified.

### Final results (2026-03-19, 15 evaluated classes — 3 excluded below 1000 samples)

**5-fold CV (80% train pool) — Stage 2 evaluated with ground-truth gate:**

| Metric | Mean | Std |
|--------|------|-----|
| Binary accuracy | 98.65% | ±0.06% |
| Attack macro F1 | 98.52% | ±0.14% |
| Attack weighted F1 | 99.69% | ±0.03% |

**Held-out test set (20%, never seen during training) — Stage 2 evaluated with ground-truth gate:**

| Metric | Value |
|--------|-------|
| Binary accuracy | 98.71% |
| Attack macro F1 | 98.41% |
| Attack weighted F1 | 99.69% |

CV and test agree within ~0.1% — no overfitting, CV estimates were not optimistically biased.

**End-to-end evaluation (Stage 1 gates Stage 2) — fold 0 only, full CV run pending:**

Stage 2 is now also evaluated using Stage 1's predictions as the gate (see Key decisions). Fold 0 results:

| Metric | Stage 2 (ground-truth gate) | End-to-end (Stage 1 gate) | Delta |
|--------|------|-----|-----|
| Attack macro F1 | 98.28% | 94.79% | −3.5pp |
| Attack weighted F1 | 99.64% | 98.68% | −1.0pp |
| Stage 1 FNs (attacks missed) | — | 1261 / 63907 (2.0%) | — |
| Stage 1 FPs (Normal → Stage 2) | — | 413 / 67427 (0.6%) | — |

The macro F1 drop is concentrated in small classes. `Discovering_resources` alone accounts for ~65% of Stage 1 false negatives (recall 0.98 → 0.76); `TCP Relay` (0.97 → 0.66) and `fuzzing` (0.97 → 0.68) are also significantly affected. This indicates Stage 1 struggles to distinguish `Discovering_resources` from Normal traffic. Large classes (BruteForce, RDOS, Generic_scanning, Scanning_vulnerability) are unaffected. The 413 Stage 1 FPs have negligible impact on attack metrics.

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

- **Stage 1** (`trainer_binary.py`): 2 hidden layers (128→64, ReLU, BatchNorm, Dropout) + sigmoid output. Labels: `0 = Normal`, `1 = Attack`.
- **Stage 2** (`trainer_attack.py`): 3 hidden layers (256→128→64, ReLU, BatchNorm, Dropout) + softmax output. Trained only on attack samples. Attack labels are re-encoded with a fresh `LabelEncoder` per fold to produce contiguous 0-indexed classes with Normal excluded. The fitted encoder is saved and loaded at eval time — re-fitting on val data would renumber classes differently and may fail if val lacks rare classes.

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

Architecture (hidden layer dims, dropout rate) is hardcoded directly in the trainers — there is no `model:` config section to avoid stale/misleading values.

---

## File map

```
pyproject.toml                  Editable install — run `pip install -e .` once after cloning
configs/
  xiiotid_dnn.yaml              Active config
  cicids2019_dnn.yaml           Placeholder — no data loader exists yet (see Known issues)
data/
  raw/xiiotid/                  Raw CSV files (gitignored)
  processed/xiiotid/            X.npy, yb.npy, ym.npy, label_encoder.pkl, test_idx.npy
scripts/
  preprocess.py                 Saves full arrays + test_idx.npy to data/processed/xiiotid/
  train.py                      5-fold CV on train split only; saves per-fold artefacts
  evaluate.py                   default: evaluate all CV folds (filtered classes only)
                                --fold N: evaluate one fold's model against its own val split (0-indexed)
                                --test: evaluate held-out test set (probability ensemble of all 5 fold models)
                                Each mode produces three evaluation blocks: Stage 1 binary, Stage 2 attack
                                (ground-truth gate), and end-to-end attack (Stage 1 gate). End-to-end block
                                also reports Stage 1 FN/FP counts and saves a separate e2e confusion matrix.
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
    trainer_attack.py           Stage 2 trainer — also uses ReduceLROnPlateau (factor 0.5, patience 2)
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

**Why two-stage?** A single multi-class model is dominated by the Normal majority. Separating the binary decision from attack-type classification lets each model focus on its own distribution. Both independent (ground-truth gate) and end-to-end (Stage 1 gate) evaluation are now implemented — see "How does end-to-end evaluation work?" below.

**Why stratified k-fold instead of time-based split?** A temporal split was tried but abandoned: rare attack classes only appear in certain time windows, so some classes were entirely absent from val/test, breaking `attack_le.transform`. XIIOTID timestamps are simulation artifacts with no temporal signal.

**Why scaler is per-fold?** Fitting `StandardScaler` on the full dataset before splitting would leak val/test statistics into training. The scaler is fit on `X[train_idx]` only and saved alongside the model for use at eval time.

**Why high scores?** Investigated and confirmed: XIIOTID is a lab dataset with highly separable features. Near-perfect metrics are expected. Identity columns (`Date`, `Timestamp`, `Scr_IP`, `Des_IP`) were dropped but had minimal impact (~0.3%). This is a known dataset limitation.

**Why are 3 classes excluded from evaluation?** `Fake_notification` (~15 samples total), `MitM` (~110), and `crypto-ransomware` (~440) fall below `eval_min_samples: 1000`. A single misclassification swings their F1 by 30%+, making the metric meaningless. The model is trained on all 18 classes — exclusion is evaluation-only and is disclosed explicitly. The threshold is configurable.

**Why 80/20 for the held-out split?** The rarest class (`Fake_notification`, ~15 samples) needs at least 3 test samples for `sklearn`'s stratified split to work. 80/20 achieves this; 90/10 risks a stratification failure. The split is stratified on `ym` (multi-class labels) to guarantee all 18 attack types appear in both partitions.

**Why stratify the held-out split on `ym` not `yb`?** Stratifying on binary labels would guarantee Normal/Attack proportions but could leave some rare attack types entirely in one partition. Stratifying on `ym` is the stricter constraint.

**Why inverse-frequency class weighting?** XIIOTID is severely imbalanced — RDOS alone is ~28k samples while `Reverse_shell` is ~200. `compute_class_weight('balanced')` is applied to both the binary model and the attack-type model so rare classes are not drowned out during training.

**How does `--test` ensemble work?** Averages sigmoid/softmax probabilities across all 5 fold models (each fold's own scaler is applied before prediction), then argmax. Valid because `attack_le` is fit with `LabelEncoder` which sorts alphabetically — classes are in the same order across all folds. Do not use `--test` results to tune hyperparameters; it is a final, one-shot evaluation.

**Two LabelEncoders — don't confuse them.** There are two distinct encoders in the pipeline:
- **Global `le`** — fit by `preprocessing.py` on all 19 classes (18 attacks + Normal). Saved as `data/processed/xiiotid/label_encoder.pkl`. Produces `ym` (integer codes 0..18). Never re-fit; it is the stable mapping for the full dataset.
- **Per-fold `attack_le`** — fit by `trainer_attack.py` on the 18 attack classes only (Normal excluded) for each fold's training split. Saved as `results/models/attack_label_encoder_fold{k}.pkl`. Produces the 0-indexed class labels the attack model is trained on and predicts.

`ym` stores integers, not strings — comparisons like `ym == "BruteForce"` silently return all-False. Use `enumerate(le.classes_)` to get `(index, name)` pairs when computing per-class counts from `ym`.

**How are excluded-class predictions handled in evaluation?** `keep_mask` filters rows where the *true* label is a kept class. The model can still predict an excluded class for those rows. These predictions are mapped to a dummy index `n_kept` (one beyond the kept range) via `label_map.get(y, n_kept)`, so they count as misclassifications. `full_report()` receives `labels=list(range(n_kept))` to restrict sklearn's reporting to the kept classes only.

**How does end-to-end evaluation work?** `evaluate.py` runs a second Stage 2 pass using `yb_pred == 1` as the gate instead of `yb_val == 1`. Stage 1 false negatives (true attacks predicted Normal) never reach Stage 2; they are assigned prediction `-1`, which maps to the dummy `n_kept` index and counts as a miss in `full_report`. Stage 1 false positives (Normal samples predicted Attack) do reach Stage 2 but have no valid attack ground-truth label, so any Stage 2 prediction for them is also counted as wrong. The end-to-end block reports Stage 1 FN/FP counts and produces a separate `_e2e_attack_cm.png` confusion matrix. The ground-truth-gate Stage 2 block is preserved alongside it for direct comparison. The same `keep_mask` and `label_map` are reused — only `y_pred_e2e_eval` changes.

---

## Known issues / TODO

### Evaluation / methodology limitations

1. **Val metrics in `train.py` summary are inflated** — `binary_val_acc` and `attack_val_acc` are recorded as `max(history["val_accuracy"])` across all epochs. Both trainers use `EarlyStopping(restore_best_weights=True)` which restores the best `val_loss` epoch — not necessarily the best `val_accuracy` epoch. The in-training summary can therefore be slightly optimistic. `evaluate.py` reloads saved weights and re-evaluates, so its figures are accurate.
2. **`pd.factorize` runs before the train/test split** — `preprocessing.py` factorizes categorical columns on the full dataset before `train_test_split`. The integer encoding is thus influenced by test-set values. In practice the effect is minimal (it is an integer assignment, not a statistical transform), but ideally the encoding would be fit on training data only and applied to test, so held-out-only categories are treated as unknown.
3. **`attack_le` fold-consistency is an implicit assumption** — the `--test` ensemble loads only fold 0's `attack_le` and assumes all folds share the same class ordering. `LabelEncoder` sorts alphabetically, so this holds in normal conditions. It would silently break if a rare class were entirely absent from one fold's training split, causing that fold's encoder to produce a shifted index mapping.

### Missing functionality

4. **Architecture is hardcoded in trainers** — hidden layer dims and dropout are not configurable via `configs/xiiotid_dnn.yaml`; they must be changed directly in `trainer_binary.py` / `trainer_attack.py`.
5. **CICIDS-2019 unsupported** — `configs/cicids2019_dnn.yaml` exists but there is no data loader or preprocessing script.
