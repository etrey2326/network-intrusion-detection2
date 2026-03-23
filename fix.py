import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

print("Loading raw data...")
files = [f'data/raw/UNSW-NB15_{i}.csv' for i in range(1, 5)]
features = pd.read_csv('data/raw/UNSW-NB15_features.csv', encoding='latin-1')
df = pd.concat(
    [pd.read_csv(f, header=None, low_memory=False) for f in files],
    ignore_index=True
)
df.columns = features['Name'].tolist()
print(f"Loaded {len(df):,} rows")

# Split target FIRST
y = df['Label'].astype(int)
X = df.drop(columns=['Label'])

# Drop identifiers
X = X.drop(columns=[c for c in ['srcip','dstip','sport','dsport'] if c in X.columns])

# Replace infinite values with NaN
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop columns where more than 50% of values are missing
before_cols = X.shape[1]
X = X.loc[:, X.isnull().mean() < 0.5]
print(f"Dropped {before_cols - X.shape[1]} columns with >50% missing values")

# Fill remaining missing values with column median
X = X.fillna(X.median(numeric_only=True))

# Fill any remaining categoricals with mode
for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].fillna(X[col].mode()[0])

print(f"After cleaning: {len(X):,} rows | y counts:\n{y.value_counts()}")

# Encode categoricals
le = LabelEncoder()
for col in X.select_dtypes(include=['object']).columns:
    X[col] = le.fit_transform(X[col].astype(str))

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"y_train:\n{y_train.value_counts()}")

# Scale
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test_scaled  = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

# Save
for fname in ['X_train.csv','X_test.csv','y_train.csv','y_test.csv']:
    path = os.path.join('data', 'processed', fname)
    if os.path.exists(path):
        os.remove(path)

X_train_scaled.to_csv('data/processed/X_train.csv', index=False)
X_test_scaled.to_csv('data/processed/X_test.csv',   index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)
y_test.to_csv('data/processed/y_test.csv',   index=False)

# Verify
y_check = pd.read_csv('data/processed/y_train.csv')
print(f"\nVerification - y_train counts:\n{y_check.value_counts()}")
print("Done!")