import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def preprocess_dataset(df, label_column="class1"):
    """Return raw unscaled arrays X, yb, ym, le.

    Scaling must be done per-fold in train.py to avoid data leakage.
    """
    print("Starting preprocessing...")

    # ==============================
    # 1. CREATE BINARY LABEL
    # ==============================
    df["binary_label"] = df[label_column].apply(
        lambda x: 0 if x == "Normal" else 1
    )

    # ==============================
    # 2. HANDLE NON-NUMERIC COLUMNS
    # ==============================
    for col in df.columns:
        if df[col].dtype == "object" and col not in [label_column]:
            df[col] = pd.factorize(df[col])[0]

    df = df.fillna(0)

    # ==============================
    # 3. SPLIT FEATURES / LABELS
    # ==============================
    identity_cols = ["Date", "Timestamp", "Scr_IP", "Des_IP"]
    label_cols = [label_column, "binary_label"] + [c for c in ["class2", "class3"] if c in df.columns]
    drop_cols = label_cols + [c for c in identity_cols if c in df.columns]
    X = df.drop(columns=drop_cols)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    yb = df["binary_label"].to_numpy()

    le = LabelEncoder()
    ym = le.fit_transform(df[label_column])

    print("Classes:", le.classes_)
    print(f"Preprocessing complete — samples: {len(X)}, features: {X.shape[1]}")

    return X.to_numpy(dtype=np.float32), yb.astype(np.int32), ym.astype(np.int32), le