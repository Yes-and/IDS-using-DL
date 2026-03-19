import os
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src.evaluation.plots import plot_confusion_matrix


def full_report(y_true, y_pred, class_names):
    """Return a dict of metrics expected by evaluate.py.

    Keys: report_str, macro_f1, weighted_f1, cm, per_class
    per_class is a dict mapping class name → f1 score.
    """
    labels = list(range(len(class_names)))
    report_str   = classification_report(y_true, y_pred, labels=labels, target_names=class_names, zero_division=0)
    macro_f1     = f1_score(y_true, y_pred, labels=labels, average="macro",    zero_division=0)
    weighted_f1  = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    cm           = confusion_matrix(y_true, y_pred, labels=labels)

    _, _, per_class_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )
    per_class = dict(zip(class_names, per_class_f1.tolist()))

    return {
        "report_str": report_str,
        "macro_f1":   macro_f1,
        "weighted_f1": weighted_f1,
        "cm":         cm,
        "per_class":  per_class,
    }


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

