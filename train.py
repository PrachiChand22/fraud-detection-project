import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, average_precision_score, confusion_matrix
from xgboost import XGBClassifier
import joblib
import shap
import matplotlib.pyplot as plt

print("Loading data...")
df = pd.read_csv("creditcard.csv")

print("\nShape of dataset (rows, columns):", df.shape)
print("\nHow many fraud vs non-fraud transactions?")
print(df["Class"].value_counts())
print("\nFraud percentage: {:.4f}%".format(df["Class"].mean() * 100))

X = df.drop("Class", axis=1)    # all columns except the answer
y = df["Class"]                 # the answer column (0 = normal, 1 = fraud)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"\nTraining rows: {len(X_train)} | Test rows: {len(X_test)}")
print("\nTraining baseline (Logistic Regression)...")
baseline = LogisticRegression(class_weight="balanced", max_iter=1000)
baseline.fit(X_train, y_train)

baseline_preds = baseline.predict(X_test)
print("\n--- Baseline (Logistic Regression) results ---")
print(classification_report(y_test, baseline_preds, digits=4))
baseline_pr_auc = average_precision_score(y_test, baseline.predict_proba(X_test)[:, 1])
print(f"Baseline PR-AUC: {baseline_pr_auc:.4f}")
print("\nTraining XGBoost...")
fraud_ratio = (y_train == 0).sum() / (y_train == 1).sum()
model = XGBClassifier(
    scale_pos_weight=fraud_ratio,   # tells the model to pay more attention to rare fraud cases
    eval_metric="aucpr",
    random_state=42,
)
model.fit(X_train, y_train)
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
print("\nCalculating SHAP values (this explains WHY the model decides what it decides)...")

sample = X_test.sample(n=min(1000, len(X_test)), random_state=42)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample)

plt.figure()
shap.summary_plot(shap_values, sample, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150)
print("Saved shap_summary.png - this shows which features matter most overall")

joblib.dump(model, "fraud_model.pkl")
print("\nSaved trained model to fraud_model.pkl")
print("\nDone! Next: run the Streamlit app with:  streamlit run app.py")
