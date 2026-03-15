from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def evaluate_model(model, X_test, y_test):

    preds = model.predict(X_test)
    preds = preds.argmax(axis=1)

    acc = accuracy_score(y_test, preds)

    report = classification_report(y_test, preds)

    cm = confusion_matrix(y_test, preds)

    return acc, report, cm