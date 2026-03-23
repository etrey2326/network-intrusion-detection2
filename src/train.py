import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pickle
import os
import time

# ── 1. Load processed data ────────────────────────────────────────────────────

def load_processed_data(processed_dir="data/processed"):
    processed_dir = os.path.normpath(processed_dir)
    print(f"Loading from: {os.path.abspath(processed_dir)}")
    
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
    X_test  = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv")).squeeze()
    y_test  = pd.read_csv(os.path.join(processed_dir, "y_test.csv")).squeeze()

    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_train value counts:\n{y_train.value_counts()}")
    return X_train, X_test, y_train, y_test


# ── 2. Define models ──────────────────────────────────────────────────────────

def get_models():
    """
    Returns the three classifiers we want to train and compare.

    - Logistic Regression: simple, fast, interpretable baseline
    - Random Forest: handles non-linear patterns, robust to noise
    - XGBoost: gradient boosting, typically best performance on tabular data
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
            verbosity=0
        ),
    }


# ── 3. Train all models ───────────────────────────────────────────────────────

def train_all_models(X_train, y_train, models):
    """
    Trains each model and records how long it takes.
    Returns a dictionary of trained model objects.
    """
    trained = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start
        print(f"  Done in {elapsed:.1f}s")
        trained[name] = model

    return trained


# ── 4. Save trained models ────────────────────────────────────────────────────

def save_models(trained_models, models_dir="models"):
    """
    Saves each trained model to disk as a .pkl file so we don't
    have to retrain every time we want to make predictions.
    """
    os.makedirs(models_dir, exist_ok=True)

    for name, model in trained_models.items():
        filename = name.lower().replace(" ", "_") + ".pkl"
        filepath = os.path.join(models_dir, filename)
        with open(filepath, "wb") as f:
            pickle.dump(model, f)
        print(f"  Saved: {filepath}")


# ── 5. Main pipeline ──────────────────────────────────────────────────────────

def run_training(processed_dir="data/processed"):
    print("\n── Loading processed data ──")
    X_train, X_test, y_train, y_test = load_processed_data(processed_dir)

    print("\n── Initializing models ──")
    models = get_models()

    print("\n── Training models ──")
    trained_models = train_all_models(X_train, y_train, models)

    print("\n── Saving models ──")
    save_models(trained_models)

    print("\nAll models trained and saved to models/")
    return trained_models, X_test, y_test


if __name__ == "__main__":
    run_training()