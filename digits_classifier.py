"""
Classification model for scikit-learn's digits dataset.
Demonstrates loading, training, and evaluating multiple classifiers.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)


def load_and_prepare_data():
    """Load and prepare the digits dataset."""
    # Load the dataset
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target

    print("Dataset shape:", X.shape)
    print("Number of classes:", len(np.unique(y)))
    print("Features per sample:", X.shape[1])

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Normalize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def train_and_evaluate_svm(X_train, X_test, y_train, y_test):
    """Train and evaluate SVM classifier."""
    print("\n" + "=" * 50)
    print("Support Vector Machine (SVM)")
    print("=" * 50)

    clf = SVC(kernel="rbf", gamma="scale", random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return clf, y_pred


def train_and_evaluate_random_forest(X_train, X_test, y_train, y_test):
    """Train and evaluate Random Forest classifier."""
    print("\n" + "=" * 50)
    print("Random Forest Classifier")
    print("=" * 50)

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return clf, y_pred


def train_and_evaluate_mlp(X_train, X_test, y_train, y_test):
    """Train and evaluate Neural Network (MLP) classifier."""
    print("\n" + "=" * 50)
    print("Multi-layer Perceptron (Neural Network)")
    print("=" * 50)

    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return clf, y_pred


def plot_confusion_matrix(y_test, y_pred, title):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.tight_layout()
    return disp


def visualize_sample_digits():
    """Visualize sample digits from the dataset."""
    digits = datasets.load_digits()
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(digits.images[i], cmap="gray")
        ax.set_title(f"Label: {digits.target[i]}")
        ax.axis("off")
    plt.suptitle("Sample Digits from Dataset")
    plt.tight_layout()
    return fig


def main():
    """Main function to run the classification pipeline."""
    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test, scaler = load_and_prepare_data()

    # Train and evaluate different classifiers
    svm_clf, svm_pred = train_and_evaluate_svm(X_train, X_test, y_train, y_test)
    rf_clf, rf_pred = train_and_evaluate_random_forest(X_train, X_test, y_train, y_test)
    mlp_clf, mlp_pred = train_and_evaluate_mlp(X_train, X_test, y_train, y_test)

    # Create visualizations
    print("\n" + "=" * 50)
    print("Creating visualizations...")
    print("=" * 50)

    # Visualize sample digits
    visualize_sample_digits()

    # Plot confusion matrices for each classifier
    plot_confusion_matrix(y_test, svm_pred, "SVM Confusion Matrix")
    plot_confusion_matrix(y_test, rf_pred, "Random Forest Confusion Matrix")
    plot_confusion_matrix(y_test, mlp_pred, "MLP Confusion Matrix")

    plt.show()

    print("\nDone!")


if __name__ == "__main__":
    main()
