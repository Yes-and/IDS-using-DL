"""Loader for the X-IIoTID dataset (IoT IDS benchmark).

Reference:
    Al-Hawawreh et al., "X-IIoTID: A Connectivity-Agnostic and Device-Agnostic
    Intrusion Data Set for Industrial Internet of Things", IEEE IoT Journal, 2021.

Expected raw directory layout::

    data/raw/xiiotid/
        *.csv          # one or more CSV files

Label granularity (``class*`` columns):
    class1  – binary: Normal / Attack
    class2  – attack category  (e.g. DoS, Scanning, MitM …)
    class3  – specific attack type (finest granularity)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

# ── Column groups ─────────────────────────────────────────────────────────────

# Identifiers — no attack-pattern signal; source/dest IPs would cause leakage
_DROP_COLS = ["Date", "Timestamp", "Scr_IP", "Des_IP", "Scr_port"]

# Categorical columns to one-hot encode
_CATEGORICAL_COLS = ["Protocol", "Service", "Conn_state"]

# Binary flag columns — cast to UInt8 regardless of how they arrive in CSV
_BINARY_FLAG_COLS = [
    "is_syn_only", "Is_SYN_ACK", "is_pure_ack", "is_with_payload",
    "FIN or RST", "Bad_checksum", "is_SYN_with_RST",
    "Login_attempt", "Succesful_login", "File_activity",
    "Process_activity", "read_write_physical.process", "is_privileged",
]

# Alert columns from external IDS tools.
# Including them gives the model information from pre-existing IDS infrastructure
# rather than raw traffic/system features alone.  Exclude for ablation studies.
_ALERT_COLS = ["anomaly_alert", "OSSEC_alert", "OSSEC_alert_level"]

# All three label columns — caller chooses which granularity to train on
LABEL_COLS = ["class1", "class2", "class3"]


# ── Main loader ───────────────────────────────────────────────────────────────

def load_xiiotid(
    raw_dir: str | Path,
    *,
    include_alert_cols: bool = True,
) -> pl.DataFrame:
    """Load, clean, and one-hot encode the X-IIoTID dataset.

    Steps performed:
    1. Lazy-scan all CSVs in *raw_dir* via Polars (memory-efficient).
    2. Drop identifier columns and optionally the external-IDS alert columns.
    3. Replace ±Inf with null; drop any row containing a null.
    4. Cast binary flag columns to UInt8 (handles bool/int/string variants).
    5. One-hot encode categorical columns (Protocol, Service, Conn_state).
    6. Cast all feature columns to Float32.
    7. Move the three ``class*`` label columns to the end.

    Args:
        raw_dir: Path to the directory containing raw XIIOTID CSV file(s).
        include_alert_cols: If False, drop anomaly_alert, OSSEC_alert, and
            OSSEC_alert_level.  Useful for ablation experiments that should not
            rely on pre-existing IDS infrastructure.

    Returns:
        Polars DataFrame — numeric feature columns followed by class1/class2/class3.
        Pass to :func:`feature_matrix` to obtain numpy arrays ready for training.
    """
    raw_dir = Path(raw_dir)
    csvs = sorted(raw_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    drop_cols = list(_DROP_COLS)
    if not include_alert_cols:
        drop_cols.extend(_ALERT_COLS)

    # ── 1. Lazy scan ──────────────────────────────────────────────────────────
    # null_values covers common encodings of missing/infinite data in CSV exports
    df = (
        pl.scan_csv(
            csvs if len(csvs) > 1 else csvs[0],
            infer_schema_length=50_000,
            null_values=["", "NA", "N/A", "nan", "NaN", "inf", "-inf", "Infinity", "-Infinity"],
        )
        .drop(drop_cols, strict=False)
        .collect()
    )

    # ── 2. Replace ±Inf remaining in numeric columns, then drop null rows ─────
    float_cols = [c for c, t in zip(df.columns, df.dtypes) if t in (pl.Float32, pl.Float64)]
    if float_cols:
        df = df.with_columns([
            pl.when(pl.col(c).is_infinite()).then(None).otherwise(pl.col(c)).alias(c)
            for c in float_cols
        ])
    df = df.drop_nulls()

    # ── 3. Cast binary flag columns to UInt8 ──────────────────────────────────
    flag_exprs = []
    for col in _BINARY_FLAG_COLS:
        if col not in df.columns:
            continue
        dtype = df[col].dtype
        if dtype == pl.Boolean:
            flag_exprs.append(pl.col(col).cast(pl.UInt8))
        elif dtype == pl.Utf8:
            # Handles '0'/'1', 'true'/'false', 'True'/'False', 'yes'/'no'
            flag_exprs.append(
                pl.col(col)
                  .str.to_lowercase()
                  .is_in(["1", "true", "yes"])
                  .cast(pl.UInt8)
                  .alias(col)
            )
        # Already an integer type — leave as-is (cast to Float32 in step 5)
    if flag_exprs:
        df = df.with_columns(flag_exprs)

    # ── 4. One-hot encode categorical columns ─────────────────────────────────
    cat_present = [c for c in _CATEGORICAL_COLS if c in df.columns]
    if cat_present:
        df = df.to_dummies(columns=cat_present, separator="_")

    # ── 5. Cast all feature columns to Float32 ────────────────────────────────
    label_set = set(LABEL_COLS)
    cast_exprs = [
        pl.col(c).cast(pl.Float32)
        for c, t in zip(df.columns, df.dtypes)
        if c not in label_set and t not in (pl.Utf8, pl.Categorical, pl.Float32)
    ]
    if cast_exprs:
        df = df.with_columns(cast_exprs)

    # ── 6. Re-order: features first, label columns last ───────────────────────
    feat_cols = [c for c in df.columns if c not in label_set]
    present_labels = [c for c in LABEL_COLS if c in df.columns]
    df = df.select(feat_cols + present_labels)

    return df


# ── Convenience extractor ─────────────────────────────────────────────────────

def feature_matrix(
    df: pl.DataFrame,
    label_col: str = "class2",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract (X, y_raw, feature_names) from a loaded X-IIoTID DataFrame.

    Args:
        df: Output of :func:`load_xiiotid`.
        label_col: Granularity of the target variable.  One of:
            ``'class1'`` (binary Normal/Attack),
            ``'class2'`` (attack category — recommended default),
            ``'class3'`` (specific attack type, finest granularity).

    Returns:
        X: float32 numpy array of shape ``(N, F)``.
        y_raw: string numpy array of shape ``(N,)`` — pass to
            :func:`~src.data.preprocessing.encode_labels`.
        feature_names: list of F feature column names (matches X columns).
    """
    if label_col not in LABEL_COLS:
        raise ValueError(f"label_col must be one of {LABEL_COLS}, got {label_col!r}")

    feat_cols = [c for c in df.columns if c not in set(LABEL_COLS)]
    X = df.select(feat_cols).to_numpy().astype(np.float32)
    y_raw = df[label_col].to_numpy()
    return X, y_raw, feat_cols
