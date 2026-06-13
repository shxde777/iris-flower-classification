"""
train_model.py — Iris Flower Classification Model Training

This script:
  1. Loads the built-in Iris dataset from Scikit-learn.
  2. Splits it into training and testing sets.
  3. Trains a Random Forest classifier.
  4. Evaluates the model with accuracy and a classification report.
  5. Saves the trained model to `model/iris_model.pkl` using Joblib.

Run this script once before starting the Streamlit app:
    python train_model.py
"""

# ──────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────
import os
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


def main():
    """Train and save the Iris classification model."""

    print("=" * 55)
    print("  🌸  Iris Flower Classification — Model Training")
    print("=" * 55)

    # ── 1. Load dataset ──────────────────────────────────
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    print(f"\n📊  Dataset loaded successfully")
    print(f"    Samples  : {X.shape[0]}")
    print(f"    Features : {X.shape[1]}  {feature_names}")
    print(f"    Classes  : {list(target_names)}")

    # ── 2. Split into train / test ───────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,       # 80 / 20 split
        random_state=42,     # reproducible results
        stratify=y,          # keep class balance
    )

    print(f"\n🔀  Train / Test split")
    print(f"    Training samples : {X_train.shape[0]}")
    print(f"    Testing samples  : {X_test.shape[0]}")

    # ── 3. Train a Random Forest classifier ──────────────
    model = RandomForestClassifier(
        n_estimators=100,    # 100 decision trees
        random_state=42,
        max_depth=5,         # prevent overfitting
        n_jobs=-1,           # use all CPU cores
    )

    model.fit(X_train, y_train)
    print("\n✅  Model trained  (RandomForestClassifier, 100 trees)")

    # ── 4. Evaluate ──────────────────────────────────────
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Cross-validation for a more robust accuracy estimate
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

    print(f"\n📈  Evaluation Results")
    print(f"    Test Accuracy         : {accuracy * 100:.2f}%")
    print(f"    Cross-Val Accuracy    : {cv_scores.mean() * 100:.2f}% "
          f"(± {cv_scores.std() * 100:.2f}%)")

    print(f"\n📋  Classification Report\n")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # ── 5. Save the model ────────────────────────────────
    model_dir = os.path.join(os.path.dirname(__file__), "model")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "iris_model.pkl")
    joblib.dump(model, model_path)

    print(f"💾  Model saved → {model_path}")
    print("=" * 55)
    print("  🚀  Done!  Run the app with:  streamlit run app.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
