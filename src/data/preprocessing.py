import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def preprocess_dataset(df, label_column, test_size):

    df = df.dropna()

    X = df.drop(columns=[label_column])
    y = df[label_column]

    X = X.select_dtypes(include=["number"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y_encoded,
        test_size=test_size,
        stratify=y_encoded,
        random_state=42
    )

    return X_train, X_test, y_train, y_test, encoder