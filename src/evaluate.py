import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# ── 1. Load models ────────────────────────────────────────────────────────────

def load_models(models_dir="models"):
    trained = {}
    for fname in os.listdir(models_dir):
        if fname.endswith(".pkl"):
            name = fname.replace(".pkl", "").replace("_", " ").title()
            with open(os.path.join(models_dir, fname), "rb") as f:
                trained[name] = pickle.load(f)
            print(f"  Loaded: {name}")
    return trained


# ── 2. Load test data ─────────────────────────────────────────────────────────

def load_test_data(processed_dir="data/processed"):
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv")).squeeze()
    print(f"X_test: {X_test.shape} | y_test distribution:\n{y_test.value_counts()}")
    return X_test, y_test


# ── 3. Evaluate all models ────────────────────────────────────────────────────

def evaluate_all(models, X_test, y_test):
    """
    Runs each model on the test set and collects metrics.
    """
    results = []

    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        y_pred = model.predict(X_test)

        results.append({
            "Model":     name,
            "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "F1 Score":  round(f1_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred), 4),
            "Recall":    round(recall_score(y_test, y_pred), 4),
        })

        print(classification_report(y_test, y_pred, target_names=["Normal", "Attack"]))

    return pd.DataFrame(results).sort_values("F1 Score", ascending=False)


# ── 4. Plot confusion matrices ────────────────────────────────────────────────

def plot_confusion_matrices(models, X_test, y_test, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
    if len(models) == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Normal", "Attack"],
            yticklabels=["Normal", "Attack"]
        )
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrices.png")
    plt.savefig(path, dpi=150)
    print(f"\nSaved confusion matrices to {path}")
    plt.show()


# ── 5. Plot metric comparison ─────────────────────────────────────────────────

def plot_metrics(results_df, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    metrics = ["Accuracy", "F1 Score", "Precision", "Recall"]
    x = np.arange(len(results_df))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, results_df[metric], width, label=metric)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(results_df["Model"], rotation=10)
    ax.set_ylim(0, 1.1)
    ax.set_title("Model Comparison")
    ax.legend()
    plt.tight_layout()

    path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(path, dpi=150)
    print(f"Saved model comparison to {path}")
    plt.show()


# ── 6. Main ───────────────────────────────────────────────────────────────────

def run_evaluation():
    print("\n── Loading models ──")
    models = load_models()

    print("\n── Loading test data ──")
    X_test, y_test = load_test_data()

    print("\n── Evaluating models ──")
    results = evaluate_all(models, X_test, y_test)

    print("\n── Results Summary ──")
    print(results.to_string(index=False))

    print("\n── Plotting confusion matrices ──")
    plot_confusion_matrices(models, X_test, y_test)

    print("\n── Plotting metric comparison ──")
    plot_metrics(results)


if __name__ == "__main__":
    run_evaluation()