import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

# load the first csv file
df1 = pd.read_csv("data/raw/UNSW-NB15_1.csv", header=None, low_memory=False)
print("loaded file 1")

# load the second csv file
df2 = pd.read_csv("data/raw/UNSW-NB15_2.csv", header=None, low_memory=False)
print("loaded file 2")

# load the third csv file
df3 = pd.read_csv("data/raw/UNSW-NB15_3.csv", header=None, low_memory=False)
print("loaded file 3")

# load the fourth csv file
df4 = pd.read_csv("data/raw/UNSW-NB15_4.csv", header=None, low_memory=False)
print("loaded file 4")

# combine all 4 files into one big dataframe
df = pd.concat([df1, df2, df3, df4], ignore_index=True)
print("combined all files")
print("total rows:", len(df))

# load the column names from the features file
features = pd.read_csv("data/raw/UNSW-NB15_features.csv", encoding="latin-1")
col_names = features["Name"].tolist()
df.columns = col_names
print("added column names")

# separate the label column from the features
# we do this first so we dont accidentally mess up the label
y = df["Label"]
y = y.astype(int)
X = df.drop(columns=["Label"])
print("separated X and y")
print("X shape:", X.shape)
print("y value counts:")
print(y.value_counts())

# drop columns that are just IDs and not useful for prediction
# srcip and dstip are ip addresses
# sport and dsport are port numbers
if "srcip" in X.columns:
    X = X.drop(columns=["srcip"])
if "dstip" in X.columns:
    X = X.drop(columns=["dstip"])
if "sport" in X.columns:
    X = X.drop(columns=["sport"])
if "dsport" in X.columns:
    X = X.drop(columns=["dsport"])
print("dropped id columns")

# replace infinity values with NaN
X = X.replace(np.inf, np.nan)
X = X.replace(-np.inf, np.nan)

# drop columns where more than 50% of values are missing
# dropping entire rows would remove too much normal traffic
cols_before = X.shape[1]
X = X.loc[:, X.isnull().mean() < 0.5]
print("dropped", cols_before - X.shape[1], "columns with more than 50% missing values")

# fill remaining missing values with the column median
# this keeps all rows so both normal and attack traffic stays in the dataset
X = X.fillna(X.median(numeric_only=True))

# fill any remaining text columns with the most common value
for col in X.select_dtypes(include=["object"]).columns:
    X[col] = X[col].fillna(X[col].mode()[0])

print("rows after cleaning:", len(X))
print("y value counts after cleaning:")
print(y.value_counts())

# encode categorical columns (text columns) to numbers
# machine learning models cant handle text
le = LabelEncoder()
for col in X.columns:
    if X[col].dtype == "object":
        X[col] = le.fit_transform(X[col].astype(str))
        print("encoded column:", col)

# reset the index so row numbers are clean before saving
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# split data into training and test sets
# 80% for training, 20% for testing
# random_state=42 so results are reproducible
# stratify=y so both sets have the same class ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("train size:", len(X_train))
print("test size:", len(X_test))
print("y_train value counts:")
print(y_train.value_counts())

# scale the features so they all have mean 0 and std 1
# important: only fit the scaler on training data to avoid data leakage
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# convert back to dataframe so we can save as csv
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
print("scaled the features")

# save everything to the processed folder
# create the folder if it doesnt exist
if not os.path.exists("data/processed"):
    os.makedirs("data/processed")

X_train_scaled.to_csv("data/processed/X_train.csv", index=False)
print("saved X_train.csv")

X_test_scaled.to_csv("data/processed/X_test.csv", index=False)
print("saved X_test.csv")

y_train.to_csv("data/processed/y_train.csv", index=False)
print("saved y_train.csv")

y_test.to_csv("data/processed/y_test.csv", index=False)
print("saved y_test.csv")

print("done! all files saved to data/processed/")