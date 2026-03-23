import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

print("Loading raw data...")

# load all 4 csv files one by one
df1 = pd.read_csv("data/raw/UNSW-NB15_1.csv", header=None, low_memory=False)
df2 = pd.read_csv("data/raw/UNSW-NB15_2.csv", header=None, low_memory=False)
df3 = pd.read_csv("data/raw/UNSW-NB15_3.csv", header=None, low_memory=False)
df4 = pd.read_csv("data/raw/UNSW-NB15_4.csv", header=None, low_memory=False)

# combine them all together
df = pd.concat([df1, df2, df3, df4], ignore_index=True)

# get the column names from the features file
features = pd.read_csv("data/raw/UNSW-NB15_features.csv", encoding="latin-1")
df.columns = features["Name"].tolist()
print(f"Loaded {len(df):,} rows")

# split target FIRST before doing anything else
y = df["Label"].astype(int)
X = df.drop(columns=["Label"])

# drop identifier columns that we dont want to use as features
X = X.drop(columns=[c for c in ["srcip", "dstip", "sport", "dsport"] if c in X.columns])

# replace inf values with NaN so we can drop them
X.replace([np.inf, -np.inf], np.nan, inplace=True)
mask = X.notna().all(axis=1)
X = X[mask]
y = y[mask]
print(f"After cleaning: {len(X):,} rows")

# encode all text columns to numbers
le = LabelEncoder()
for col in X.select_dtypes(include=["object"]).columns:
    X[col] = le.fit_transform(X[col].astype(str))

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"y_train distribution:\n{y_train.value_counts()}")

# scale features - fit only on train to avoid leakage
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

# save everything
X_train_scaled.to_csv("data/processed/X_train.csv", index=False)
X_test_scaled.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)
print("All files saved to data/processed/")
