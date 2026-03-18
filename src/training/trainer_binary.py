import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

def build_binary_model(input_dim):

    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.2),

        Dense(64, activation='relu'),
        Dropout(0.2),

        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_binary_model(X_train, y_train, X_val, y_val, config=None):

    model = build_binary_model(X_train.shape[1])

    epochs     = config["training"]["epochs"]     if config else 30
    batch_size = config["training"]["batch_size"] if config else 512
    lr         = config["training"]["learning_rate"] if config else 0.001

    tf.keras.backend.set_value(model.optimizer.learning_rate, lr)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )

    return model