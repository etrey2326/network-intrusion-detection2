import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

# ── 1. Load raw data ──────────────────────────────────────────────────────────

def load_data(raw_dir="data/raw"):
    files = [
        os.path.join(os.path.normpath(raw_dir), f"UNSW-NB15_{i}.csv")
        for i in range(1, 5)
    ]

    df = pd.concat(
        [pd.read_csv(f, header=None, low_memory=False) for f in files],
        ignore_index=True
    )

    features = pd.read_csv(
        os.path.join(os.path.normpath(raw_dir), "UNSW-NB15_features.csv"),
        encoding="latin-1"
    )
    df.columns = features["Name"].tolist()

    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns.")
    print(f"Class distribution:\n{df['Label'].value_counts()}")
    return df


# ── 2. Split features and target ──────────────────────────────────────────────

def split_features_target(df, target_col="Label"):
    """
    Separates the DataFrame into X (features) and y (target label).
    We do this FIRST before any cleaning or encoding so the Label
    column is never accidentally dropped or transformed.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    return X, y


# ── 3. Clean the data ─────────────────────────────────────────────────────────

def clean_data(X, y):
    """
    Drops identifier columns, removes rows with missing or infinite
    values, and keeps X and y aligned throughout.
    """
    cols_to_drop = ["srcip", "dstip", "sport", "dsport"]
    X = X.drop(columns=[c for c in cols_to_drop if c in X.columns])

    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    before = len(X)
    mask = X.notna().all(axis=1)
    X = X[mask]
    y = y[mask]
    print(f"Dropped {before - len(X):,} rows with missing/infinite values.")
    print(f"Class distribution after cleaning:\n{y.value_counts()}")

    return X, y


# ── 4. Encode categorical columns ─────────────────────────────────────────────

def encode_categoricals(X):
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    le = LabelEncoder()
    for col in categorical_cols:
        X[col] = le.fit_transform(X[col].astype(str))
        print(f"  Encoded: {col}")
    return X


# ── 5. Scale features ─────────────────────────────────────────────────────────

def scale_features(X_train, X_test):
    """
    Applies StandardScaler so all features have mean=0 and std=1.
    We fit ONLY on training data to prevent data leakage.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


# ── 6. Main pipeline ──────────────────────────────────────────────────────────

def run_preprocessing(raw_dir="data/raw", processed_dir="data/processed"):
    os.makedirs(processed_dir, exist_ok=True)

    print("\n── Loading data ──")
    df = load_data(raw_dir)

    print("\n── Splitting features and target first ──")
    X, y = split_features_target(df, target_col="Label")

    print("\n── Cleaning data ──")
    X, y = clean_data(X, y)

    print("\n── Encoding categoricals ──")
    X = encode_categoricals(X)

    print("\n── Train/test split (80/20) ──")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
    print(f"y_train distribution:\n{y_train.value_counts()}")

    print("\n── Scaling features ──")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    print("\n── Saving to data/processed/ ──")
    pd.DataFrame(X_train_scaled, columns=X.columns).to_csv(
        os.path.join(processed_dir, "X_train.csv"), index=False
    )
    pd.DataFrame(X_test_scaled, columns=X.columns).to_csv(
        os.path.join(processed_dir, "X_test.csv"), index=False
    )
    y_train.to_csv(os.path.join(processed_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(processed_dir, "y_test.csv"), index=False)

    print("\nDone! Files saved to data/processed/")
    return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == "__main__":
    run_preprocessing()
