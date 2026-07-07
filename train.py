"""
train.py
--------
This script does Steps 3-9 from the project plan:
  - Load the data
  - Explore it briefly (print class balance)
  - Split into train/test
  - Train a baseline model (Logistic Regression)
  - Train the real model (XGBoost)
  - Evaluate both using the RIGHT metrics for imbalanced data
  - Explain the model with SHAP (saves a chart as an image)
  - Save the trained model to a file

HOW TO RUN THIS:
  1. Make sure creditcard.csv is in the same folder as this file
  2. Open a terminal in this folder
  3. Run:  python train.py
  4. Wait - training + SHAP can take a few minutes on the full dataset
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, average_precision_score, confusion_matrix
from xgboost import XGBClassifier
import joblib
import shap
import matplotlib.pyplot as plt

# -----------------------------
# STEP 3: Load and explore data
# -----------------------------
print("Loading data...")
df = pd.read_csv("creditcard.csv")

print("\nShape of dataset (rows, columns):", df.shape)
print("\nHow many fraud vs non-fraud transactions?")
print(df["Class"].value_counts())
print("\nFraud percentage: {:.4f}%".format(df["Class"].mean() * 100))
# This number is the whole reason this project is interesting.
# If fraud is ~0.17%, a lazy model that always predicts "not fraud"
# would be "99.83% accurate" while being completely useless.

# -----------------------------
# STEP 4: Split into train/test
# -----------------------------
X = df.drop("Class", axis=1)   # all columns except the answer
y = df["Class"]                 # the answer column (0 = normal, 1 = fraud)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
# stratify=y keeps the same fraud % in both train and test sets
# random_state=42 just makes the split reproducible every time you run this

print(f"\nTraining rows: {len(X_train)} | Test rows: {len(X_test)}")

# -----------------------------
# STEP 5: Baseline model
# -----------------------------
print("\nTraining baseline (Logistic Regression)...")
baseline = LogisticRegression(class_weight="balanced", max_iter=1000)
baseline.fit(X_train, y_train)

baseline_preds = baseline.predict(X_test)
print("\n--- Baseline (Logistic Regression) results ---")
print(classification_report(y_test, baseline_preds, digits=4))
baseline_pr_auc = average_precision_score(y_test, baseline.predict_proba(X_test)[:, 1])
print(f"Baseline PR-AUC: {baseline_pr_auc:.4f}")

# -----------------------------
# STEP 6: Real model - XGBoost
# -----------------------------
print("\nTraining XGBoost...")
fraud_ratio = (y_train == 0).sum() / (y_train == 1).sum()
model = XGBClassifier(
    scale_pos_weight=fraud_ratio,   # tells the model to pay more attention to rare fraud cases
    eval_metric="aucpr",
    random_state=42,
)
model.fit(X_train, y_train)

# -----------------------------
# STEP 7: Evaluate properly
# -----------------------------
preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]  # probability of being fraud

print("\n--- XGBoost results ---")
print(classification_report(y_test, preds, digits=4))

pr_auc = average_precision_score(y_test, probs)
print(f"XGBoost PR-AUC: {pr_auc:.4f}  (compare to baseline: {baseline_pr_auc:.4f})")

cm = confusion_matrix(y_test, preds)
print("\nConfusion matrix:")
print("                Predicted Normal | Predicted Fraud")
print(f"Actual Normal:  {cm[0][0]:>16} | {cm[0][1]}")
print(f"Actual Fraud:   {cm[1][0]:>16} | {cm[1][1]}")
# Read this as: how many frauds did we catch (bottom-right) vs miss (bottom-left)?
# and how many innocent transactions did we wrongly flag (top-right)?

# -----------------------------
# STEP 8: Explain with SHAP
# -----------------------------
print("\nCalculating SHAP values (this explains WHY the model decides what it decides)...")
# Using a sample of the test set to keep this fast
sample = X_test.sample(n=min(1000, len(X_test)), random_state=42)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample)

plt.figure()
shap.summary_plot(shap_values, sample, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150)
print("Saved shap_summary.png - this shows which features matter most overall")

# -----------------------------
# STEP 9: Save the trained model
# -----------------------------
joblib.dump(model, "fraud_model.pkl")
print("\nSaved trained model to fraud_model.pkl")
print("\nDone! Next: run the Streamlit app with:  streamlit run app.py")
