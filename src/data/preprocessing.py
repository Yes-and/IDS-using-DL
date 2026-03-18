import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_dataset(df, label_column="class1", test_size=0.2, random_state=42):

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
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce").astype("int64") // 10**9
            except Exception:
                df[col] = pd.factorize(df[col])[0]

    df = df.fillna(0)

    # ==============================
    # 3. SPLIT FEATURES / LABELS
    # ==============================
    X = df.drop(columns=[label_column, "binary_label"])
    y_binary = df["binary_label"]

    # Encode multiclass
    le = LabelEncoder()
    y_multi = le.fit_transform(df[label_column])

    print("Classes:", le.classes_)

    # ==============================
    # 4. TRAIN TEST SPLIT
    # ==============================
    X_train, X_test, yb_train, yb_test, ym_train, ym_test = train_test_split(
        X, y_binary, y_multi, test_size=test_size, random_state=random_state, stratify=y_binary
    )

    # ==============================
    # 5. SCALE
    # ==============================
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Preprocessing complete")

    return X_train, X_test, yb_train, yb_test, ym_train, ym_test, le, scaler