import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def build_attack_model(input_dim, num_classes):

    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.2),

        Dense(128, activation='relu'),
        Dropout(0.2),

        Dense(64, activation='relu'),

        Dense(num_classes, activation='softmax')
    ])

    # 🔥 STABLE OPTIMIZER
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.0005,
        clipnorm=1.0
    )

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_attack_model(X_train, y_train, X_val, y_val, class_weights):

    model = build_attack_model(X_train.shape[1], len(set(y_train)))

    # =========================
    # CALLBACKS
    # =========================
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-5
    )

    # =========================
    # TRAIN
    # =========================
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=40,
        batch_size=512,
        callbacks=[early_stop, lr_scheduler],
        class_weight=class_weights,   # 🔥 KEY FIX
        verbose=1
    )

    return model