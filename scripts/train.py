import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from src.data.dataset import load_dataset   # 👈 use your loader
from src.data.preprocessing import preprocess_dataset

from src.training.trainer_binary import train_binary_model
from src.training.trainer_attack import train_attack_model


def main():

    # =========================
    # LOAD DATA
    # =========================
    df = load_dataset()

    # =========================
    # PREPROCESS
    # =========================
    X_train, X_test, yb_train, yb_test, ym_train, ym_test, le = preprocess_dataset(df)

    # =========================
    # 🔥 STAGE 1: BINARY MODEL
    # =========================
    print("\n--- Training Binary Model ---")
    binary_model = train_binary_model(X_train, yb_train, X_test, yb_test)

    # =========================
    # 🔥 STAGE 2: ATTACK MODEL
    # =========================
    print("\n--- Training Attack Model ---")

    # Only attack samples
    attack_indices = yb_train == 1
    X_train_attack = X_train[attack_indices]
    y_train_attack = ym_train[attack_indices] - 1   # 🔥 IMPORTANT FIX

    attack_indices_test = yb_test == 1
    X_test_attack = X_test[attack_indices_test]
    y_test_attack = ym_test[attack_indices_test] - 1  # 🔥 IMPORTANT FIX

    # =========================
    # 🔥 CLASS WEIGHTS
    # =========================
    classes = np.unique(y_train_attack)

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train_attack
    )

    class_weights = dict(zip(classes, class_weights))

    print("Class weights:", class_weights)

    # =========================
    # TRAIN ATTACK MODEL
    # =========================
    attack_model = train_attack_model(
        X_train_attack, y_train_attack,
        X_test_attack, y_test_attack,
        class_weights
    )


if __name__ == "__main__":
    main()