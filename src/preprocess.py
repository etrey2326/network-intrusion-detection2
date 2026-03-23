import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

# ── 1. Load raw data ──────────────────────────────────────────────────────────

def load_data(raw_dir="data/raw"):
    """
    Reads all four UNSW-NB15 CSV files and the features dictionary,
    combines them into a single DataFrame, and attaches column names.
    """
    files = [
        os.path.join(os.path.normpath(raw_dir), f"UNSW-NB15_{i}.csv")
        for i in range(1, 5)
    ]

    # The CSV files have no header row, so we read them with header=None
    df = pd.concat(
        [pd.read_csv(f, header=None, low_memory=False) for f in files],
        ignore_index=True
    )

    # Load the feature names from the data dictionary
    features = pd.read_csv(
        os.path.join(os.path.normpath(raw_dir), "UNSW-NB15_features.csv"),
        encoding="latin-1"
    )
    df.columns = features["Name"].tolist()

    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns.")
    return df


# ── 2. Clean the data ─────────────────────────────────────────────────────────

def clean_data(df):
    """
    Drops columns we don't need, handles missing values,
    and fixes any obviously wrong data types.
    """
    # Drop identifier columns that aren't useful for modeling
    cols_to_drop = ["srcip", "dstip", "sport", "dsport"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Replace infinite values with NaN, then drop rows with any NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    before = len(df)
    df.dropna(inplace=True)
    after = len(df)
    print(f"Dropped {before - after:,} rows with missing/infinite values.")

    return df


# ── 3. Encode categorical columns ─────────────────────────────────────────────

def encode_categoricals(df):
    """
    Converts text columns (like protocol type and service)
    into numbers so our models can work with them.
    """
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Remove the target column if it somehow ended up as text
    categorical_cols = [c for c in categorical_cols if c != "label"]

    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))
        print(f"  Encoded: {col}")

    return df


# ── 4. Split features and target ──────────────────────────────────────────────

def split_features_target(df, target_col="Label"):
    """
    Separates the DataFrame into X (features) and y (target label).
    'label' is 0 for normal traffic and 1 for an attack.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    print(f"Features: {X.shape[1]} columns | Classes: {y.value_counts().to_dict()}")
    return X, y


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
    """
    Runs the full preprocessing pipeline and saves the results
    to the data/processed/ folder as CSV files.
    """
    os.makedirs(processed_dir, exist_ok=True)

    print("\n── Loading data ──")
    df = load_data(raw_dir)

    print("\n── Cleaning data ──")
    df = clean_data(df)

    print("\n── Encoding categoricals ──")
    df = encode_categoricals(df)

    print("\n── Splitting features and target ──")
    X, y = split_features_target(df, target_col="Label")

    print("\n── Train/test split (80/20) ──")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")

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

    print("Done! Files saved to data/processed/")
    return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == "__main__":
    run_preprocessing()