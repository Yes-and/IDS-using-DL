# Methodology and Results

Network intrusion detection for IoT traffic using a two-stage deep neural network classifier trained on the XIIOTID dataset.

---

## Dataset

**XIIOTID** — a controlled lab dataset of IoT network traffic containing approximately 820,000 samples across 19 classes: one Normal class and 18 attack types. The dataset is severely imbalanced, with the largest class (RDOS, ~28k samples) outnumbering the smallest attack types (~200 samples) by over 100x.

Three attack classes — `Fake_notification` (~15 samples), `MitM` (~110), and `crypto-ransomware` (~440) — fall below the evaluation threshold of 1,000 samples. The models are trained on all 18 attack classes but these three are excluded from reported metrics, as a single misclassification moves their F1 by more than 30 percentage points.

---

## Architecture

A two-stage classifier is used rather than a single multi-class model. A single model trained on all classes is dominated by the Normal majority class, making rare attack types harder to learn. The two stages are:

```
Input → [Stage 1: Binary classifier] → Normal / Attack
                                              ↓
                                   [Stage 2: Attack-type classifier] → Attack type
```

**Stage 1 — Binary classifier**
- Distinguishes Normal traffic from any attack
- Architecture: 2 hidden layers (128 → 64 units), ReLU activations, BatchNorm, Dropout, sigmoid output
- Labels: `0 = Normal`, `1 = Attack`

**Stage 2 — Attack-type classifier**
- Trained only on attack samples; Normal traffic is excluded
- Architecture: 3 hidden layers (256 → 128 → 64 units), ReLU activations, BatchNorm, Dropout, softmax output
- Outputs one of 18 attack classes

Both models use the TensorFlow/Keras framework.

---

## Training procedure

**Preprocessing**
Raw CSV features are loaded and the following columns are dropped before training: `Date`, `Timestamp`, `Scr_IP`, `Des_IP`, `class2`, `class3`. Features are scaled with `StandardScaler` fit on the training split only.

**Train/test split**
20% of the dataset is held out as a final test set before any training begins, stratified on the multi-class label to guarantee all 18 attack types appear in both partitions. The remaining 80% is used for cross-validation.

**Cross-validation**
5-fold stratified k-fold CV is run on the 80% training pool. A time-based split was considered but abandoned: rare attack classes only appear in certain time windows, causing some classes to be entirely absent from validation folds.

**Class imbalance**
Inverse-frequency class weighting (`sklearn`'s `compute_class_weight('balanced')`) is applied to both models. This upweights rare classes during the loss computation without resampling the data.

**Optimiser and callbacks**
Both models use Adam (lr = 0.001). Stage 2 additionally uses `ReduceLROnPlateau` (factor 0.5, patience 2 epochs). Both use `EarlyStopping` with `restore_best_weights=True`.

**Scalers and encoders**
The `StandardScaler` is fit per fold on training data only. Stage 2 uses a per-fold `LabelEncoder` fit on the fold's attack training samples; this encoder is saved alongside the model weights for use at evaluation time.

**Test-set ensemble**
The held-out test set is evaluated by averaging sigmoid/softmax probabilities across all 5 fold models (each applying its own scaler before prediction), then taking argmax. This ensemble is used once as a final, one-shot evaluation and is not used for tuning.

---

## Evaluation

Two evaluation modes are reported for Stage 2:

- **Ground-truth gate**: Stage 2 receives only the true attack samples (ideal upper bound, isolates Stage 2 performance from Stage 1 errors).
- **End-to-end (Stage 1 gate)**: Stage 2 receives samples where Stage 1 predicted Attack. Stage 1 false negatives never reach Stage 2 and are counted as misclassifications; Stage 1 false positives (Normal samples passed to Stage 2) are also counted as wrong. This reflects real deployment performance.

Metrics are macro F1 and weighted F1. Classes below `eval_min_samples = 1000` are excluded from metric computation; model predictions for excluded classes are mapped to a dummy index and treated as misclassifications.

---

## Results

### 5-fold cross-validation (80% training pool)

| Metric | Mean | Std |
|--------|------|-----|
| Stage 1 binary accuracy | 98.67% | ±0.06% |
| Stage 2 attack macro F1 (ground-truth gate) | 98.59% | ±0.07% |
| Stage 2 attack weighted F1 (ground-truth gate) | 99.70% | ±0.03% |
| End-to-end attack macro F1 | 95.04% | ±0.36% |
| End-to-end attack weighted F1 | 98.73% | ±0.12% |
| Stage 1 false negatives (mean per fold) | 1,276 / ~64k attacks (2.0%) | |
| Stage 1 false positives (mean per fold) | 466 / ~67k Normal (0.7%) | |

Best individual fold: **Fold 0** (binary acc 98.77%, attack macro F1 98.60%, e2e macro F1 95.51%, fewest Stage 1 FNs at 1,086).

### Held-out test set (20%, never seen during training — probability ensemble of all 5 folds)

| Metric | Value |
|--------|-------|
| Stage 1 binary accuracy | 98.71% |
| Stage 2 attack macro F1 (ground-truth gate) | 98.51% |
| Stage 2 attack weighted F1 (ground-truth gate) | 99.70% |
| End-to-end attack macro F1 | 95.23% |
| End-to-end attack weighted F1 | 98.72% |
| Stage 1 false negatives | 1,635 / ~82k attacks (2.0%) |
| Stage 1 false positives | 475 / ~84k Normal (0.6%) |

CV and held-out test results agree within ~0.1% across all metrics, indicating no overfitting and unbiased CV estimates.

### Per-class observations

**Weakest classes (ground-truth gate)**
- **C&C** — lowest individual F1 at approximately 0.92; the only class clearly below 0.95
- **Reverse_shell**, **TCP Relay**, **fuzzing** — slightly below 1.0 (~0.97–0.98)

**Largest end-to-end degradation (Stage 1 gate)**
The macro F1 drop of ~3.3 percentage points in end-to-end evaluation is concentrated in small classes whose traffic characteristics overlap with Normal, causing Stage 1 to miss them:

| Class | Ground-truth gate recall | End-to-end recall | Drop |
|-------|--------------------------|-------------------|------|
| Discovering_resources | 0.98 | 0.76 | −22pp |
| TCP Relay | 0.97 | 0.66 | −31pp |
| fuzzing | 0.97 | 0.68 | −29pp |

These three classes account for the majority of Stage 1 false negatives. Large classes (BruteForce, RDOS, Generic_scanning, Scanning_vulnerability) are unaffected.

---

## Caveats

- **Dataset limitation**: XIIOTID is a controlled lab dataset with highly separable features. Near-perfect metrics are expected and reflect the dataset rather than real-world difficulty. Field performance on live IoT traffic would be lower.
- **Excluded classes**: `Fake_notification`, `MitM`, and `crypto-ransomware` are excluded from all reported metrics due to insufficient sample counts, but the models are trained on them.
- **Feature extraction not included**: This pipeline operates on pre-computed flow-level features. Deployment on IoT devices would require an additional streaming feature extraction layer.
