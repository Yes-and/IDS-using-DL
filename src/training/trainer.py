import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight


def train_model(model, X_train, y_train, X_test, y_test, config):

    print("Training model...")

    # -----------------------------
    # Class Weights
    # -----------------------------
    classes = np.unique(y_train)

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )

    class_weights = dict(enumerate(class_weights))

    # Normalize weights (VERY IMPORTANT)
    max_weight = max(class_weights.values())
    class_weights = {k: v / max_weight for k, v in class_weights.items()}
    print("Class weights:", class_weights)

    # -----------------------------
    # Callbacks
    # -----------------------------
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-5
    )

    # -----------------------------
    # Train
    # -----------------------------
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=40,
        batch_size=256,
        class_weight=class_weights,
        callbacks=[early_stop, lr_scheduler],
        shuffle=True,
        verbose=1
    )

    return history