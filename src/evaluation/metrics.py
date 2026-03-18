import os
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

from src.evaluation.plots import plot_confusion_matrix


def evaluate_model(model, X_test, y_test, encoder=None):
    """
    Evaluate trained model on test dataset.

    Returns:
        accuracy
        classification_report
        confusion_matrix
    """

    print("\nEvaluating model...")

    # -------------------------
    # Model Predictions
    # -------------------------
    y_pred_probs = model.predict(X_test)

    # Convert probabilities → predicted class index
    y_pred = np.argmax(y_pred_probs, axis=1)

    # -------------------------
    # Basic Metrics
    # -------------------------
    accuracy = accuracy_score(y_test, y_pred)

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro"
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted"
    )

    # -------------------------
    # Detailed Report
    # -------------------------
    report = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    # -------------------------
    # Confusion Matrix
    # -------------------------
    cm = confusion_matrix(y_test, y_pred)

    # -------------------------
    # Print Metrics
    # -------------------------
    print("\n==============================")
    print("Model Evaluation Results")
    print("==============================")

    print("\nAccuracy:", accuracy)
    print("Macro F1 Score:", macro_f1)
    print("Weighted F1 Score:", weighted_f1)

    print("\nClassification Report:\n")
    print(report)

    # -------------------------
    # Save Confusion Matrix
    # -------------------------
    labels = None

    if encoder is not None:
        labels = encoder.classes_

    save_path = "results/figures/confusion_matrix.png"

    os.makedirs("results/figures", exist_ok=True)

    plot_confusion_matrix(
        cm,
        labels,
        save_path
    )

    print("\nConfusion matrix saved to:", save_path)

    # -------------------------
    # Return results
    # -------------------------
    return accuracy, report, cm

