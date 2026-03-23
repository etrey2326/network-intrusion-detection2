import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

print("Loading raw data...")

# load each of the 4 csv files
df1 = pd.read_csv("data/raw/UNSW-NB15_1.csv", header=None, low_memory=False)
df2 = pd.read_csv("data/raw/UNSW-NB15_2.csv", header=None, low_memory=False)
df3 = pd.read_csv("data/raw/UNSW-NB15_3.csv", header=None, low_memory=False)
df4 = pd.read_csv("data/raw/UNSW-NB15_4.csv", header=None, low_memory=False)
df = pd.concat([df1, df2, df3, df4], ignore_index=True)

# add column names using the features file
features = pd.read_csv("data/raw/UNSW-NB15_features.csv", encoding="latin-1")
df.columns = features["Name"].tolist()
print(f"Loaded {len(df):,} rows")

# split target FIRST
y = df["Label"].astype(int)
X = df.drop(columns=["Label"])

# drop identifiers
X = X.drop(columns=[c for c in ["srcip", "dstip", "sport", "dsport"] if c in X.columns])

# replace infinite values with NaN
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# drop columns where more than 50% of values are missing
cols_before = X.shape[1]
X = X.loc[:, X.isnull().mean() < 0.5]
print(f"Dropped {cols_before - X.shape[1]} columns with >50% missing values")

# fill remaining missing numeric values with median
X = X.fillna(X.median(numeric_only=True))

# fill remaining categorical missing values with mode
for col in X.select_dtypes(include=["object"]).columns:
    X[col] = X[col].fillna(X[col].mode()[0])

print(f"After cleaning: {len(X):,} rows | y counts:\n{y.value_counts()}")

# encode categorical columns
le = LabelEncoder()
for col in X.select_dtypes(include=["object"]).columns:
    X[col] = le.fit_transform(X[col].astype(str))

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"y_train:\n{y_train.value_counts()}")

# scale features - only fit on training data
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

# delete old files if they exist
for fname in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]:
    path = os.path.join("data", "processed", fname)
    if os.path.exists(path):
        os.remove(path)

# save new files
X_train_scaled.to_csv("data/processed/X_train.csv", index=False)
X_test_scaled.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

# verify the save worked
y_check = pd.read_csv("data/processed/y_train.csv")
print(f"\nVerification - y_train counts:\n{y_check.value_counts()}")
print("Done!")
