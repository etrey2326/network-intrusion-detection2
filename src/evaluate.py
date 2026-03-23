import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# load the test data
print("loading test data...")
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
print("X_test shape:", X_test.shape)
print("y_test value counts:")
print(y_test.value_counts())

# load the trained models
print("loading models...")
with open("models/logistic_regression.pkl", "rb") as f:
    lr_model = pickle.load(f)
print("loaded logistic regression")

with open("models/random_forest.pkl", "rb") as f:
    rf_model = pickle.load(f)
print("loaded random forest")

with open("models/xgboost.pkl", "rb") as f:
    xgb_model = pickle.load(f)
print("loaded xgboost")

# make predictions with each model
print("making predictions...")
lr_preds = lr_model.predict(X_test)
rf_preds = rf_model.predict(X_test)
xgb_preds = xgb_model.predict(X_test)

# calculate accuracy for each model
lr_accuracy = accuracy_score(y_test, lr_preds)
rf_accuracy = accuracy_score(y_test, rf_preds)
xgb_accuracy = accuracy_score(y_test, xgb_preds)

# calculate f1 score for each model
lr_f1 = f1_score(y_test, lr_preds)
rf_f1 = f1_score(y_test, rf_preds)
xgb_f1 = f1_score(y_test, xgb_preds)

# calculate precision for each model
lr_precision = precision_score(y_test, lr_preds)
rf_precision = precision_score(y_test, rf_preds)
xgb_precision = precision_score(y_test, xgb_preds)

# calculate recall for each model
lr_recall = recall_score(y_test, lr_preds)
rf_recall = recall_score(y_test, rf_preds)
xgb_recall = recall_score(y_test, xgb_preds)

# print results for each model
print("\nLogistic Regression Results:")
print("accuracy:", round(lr_accuracy, 4))
print("f1 score:", round(lr_f1, 4))
print("precision:", round(lr_precision, 4))
print("recall:", round(lr_recall, 4))
print(classification_report(y_test, lr_preds, target_names=["Normal", "Attack"]))

print("\nRandom Forest Results:")
print("accuracy:", round(rf_accuracy, 4))
print("f1 score:", round(rf_f1, 4))
print("precision:", round(rf_precision, 4))
print("recall:", round(rf_recall, 4))
print(classification_report(y_test, rf_preds, target_names=["Normal", "Attack"]))

print("\nXGBoost Results:")
print("accuracy:", round(xgb_accuracy, 4))
print("f1 score:", round(xgb_f1, 4))
print("precision:", round(xgb_precision, 4))
print("recall:", round(xgb_recall, 4))
print(classification_report(y_test, xgb_preds, target_names=["Normal", "Attack"]))

# put all results in a table
results = {
    "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
    "Accuracy": [round(lr_accuracy, 4), round(rf_accuracy, 4), round(xgb_accuracy, 4)],
    "F1 Score": [round(lr_f1, 4), round(rf_f1, 4), round(xgb_f1, 4)],
    "Precision": [round(lr_precision, 4), round(rf_precision, 4), round(xgb_precision, 4)],
    "Recall": [round(lr_recall, 4), round(rf_recall, 4), round(xgb_recall, 4)]
}
results_df = pd.DataFrame(results)
print("\nSummary:")
print(results_df)

# make sure outputs folder exists
if not os.path.exists("outputs"):
    os.makedirs("outputs")

# plot confusion matrices for all 3 models side by side
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# confusion matrix for logistic regression
lr_cm = confusion_matrix(y_test, lr_preds)
sns.heatmap(lr_cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
axes[0].set_title("Logistic Regression")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

# confusion matrix for random forest
rf_cm = confusion_matrix(y_test, rf_preds)
sns.heatmap(rf_cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
            xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
axes[1].set_title("Random Forest")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

# confusion matrix for xgboost
xgb_cm = confusion_matrix(y_test, xgb_preds)
sns.heatmap(xgb_cm, annot=True, fmt="d", cmap="Blues", ax=axes[2],
            xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
axes[2].set_title("XGBoost")
axes[2].set_xlabel("Predicted")
axes[2].set_ylabel("Actual")

plt.tight_layout()
plt.savefig("outputs/confusion_matrices.png", dpi=150)
print("saved confusion_matrices.png")
plt.show()

# bar chart comparing all models on all metrics
x = np.arange(3)  # 3 models
width = 0.2  # width of each bar

fig, ax = plt.subplots(figsize=(10, 5))

# plot each metric as its own group of bars
ax.bar(x + 0 * width, results_df["Accuracy"], width, label="Accuracy")
ax.bar(x + 1 * width, results_df["F1 Score"], width, label="F1 Score")
ax.bar(x + 2 * width, results_df["Precision"], width, label="Precision")
ax.bar(x + 3 * width, results_df["Recall"], width, label="Recall")

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(["Logistic Regression", "Random Forest", "XGBoost"], rotation=10)
ax.set_ylim(0, 1.1)
ax.set_title("Model Comparison")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/model_comparison.png", dpi=150)
print("saved model_comparison.png")
plt.show()

print("done!")
