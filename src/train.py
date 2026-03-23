import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pickle
import os

# load the training data
print("loading data...")
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv")
y_test = pd.read_csv("data/processed/y_test.csv")

# squeeze converts the dataframe to a series (needed for sklearn)
y_train = y_train.squeeze()
y_test = y_test.squeeze()

print("X_train shape:", X_train.shape)
print("y_train value counts:")
print(y_train.value_counts())

# create the models folder if it doesnt exist
if not os.path.exists("models"):
    os.makedirs("models")

# train logistic regression
# max_iter=1000 because the default of 100 wasnt enough and it gave a warning
print("training logistic regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
print("logistic regression done")

# save logistic regression model to a file
with open("models/logistic_regression.pkl", "wb") as f:
    pickle.dump(lr_model, f)
print("saved logistic_regression.pkl")

# train random forest
# 100 trees is a common starting point
print("training random forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
print("random forest done")

# save random forest model
with open("models/random_forest.pkl", "wb") as f:
    pickle.dump(rf_model, f)
print("saved random_forest.pkl")

# train xgboost
print("training xgboost...")
xgb_model = XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0)
xgb_model.fit(X_train, y_train)
print("xgboost done")

# save xgboost model
with open("models/xgboost.pkl", "wb") as f:
    pickle.dump(xgb_model, f)
print("saved xgboost.pkl")

print("all 3 models trained and saved!")
