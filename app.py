"""
app.py
------
This is Steps 10-11: the Streamlit dashboard.
It loads the model you already trained (fraud_model.pkl) and lets
you interactively test transactions and see WHY they were flagged.

HOW TO RUN THIS:
  1. Make sure you've already run train.py at least once (so fraud_model.pkl exists)
  2. In your terminal, run:  streamlit run app.py
  3. It will open a browser tab automatically at localhost:8501
"""

import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fraud Detection Demo", layout="wide")

# -----------------------------
# Load the trained model
# -----------------------------
@st.cache_resource  # this means: only load the model once, not on every click
def load_model():
    return joblib.load("fraud_model.pkl")

model = load_model()

st.title("💳 Credit Card Fraud Detector")
st.write(
    "This model was trained on 284K+ real (anonymized) transactions, "
    "where only ~0.17% were fraud. Enter transaction details below, "
    "or upload a row from the test set, to see a live prediction."
)

# -----------------------------
# Option A: Upload a CSV row
# -----------------------------
st.subheader("Option 1: Upload a transaction (CSV row)")
uploaded_file = st.file_uploader(
    "Upload a single-row CSV with the same columns as the training data (minus 'Class')",
    type="csv",
)

# -----------------------------
# Option B: Manual entry (simplified - Amount and Time only,
# with the anonymized V1-V28 features defaulted to 0 for simplicity)
# -----------------------------
st.subheader("Option 2: Quick manual test (simplified)")
col1, col2 = st.columns(2)
with col1:
    amount = st.number_input("Transaction Amount ($)", value=100.0, min_value=0.0)
with col2:
    time = st.number_input("Time (seconds since first transaction)", value=50000.0)

run_manual = st.button("Predict (manual entry)")

# -----------------------------
# Prediction logic
# -----------------------------
def predict_and_explain(row_df):
    """Takes a single-row dataframe with the correct columns, returns prediction + SHAP explanation."""
    prediction = model.predict(row_df)[0]
    probability = model.predict_proba(row_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ FLAGGED AS FRAUD — confidence: {probability:.2%}")
    else:
        st.success(f"✅ Looks normal — fraud probability: {probability:.2%}")

    # SHAP explanation for this specific prediction
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(row_df)

    st.write("**Why did the model decide this?** (features pushing toward fraud vs. normal)")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.bar_plot(shap_values[0], feature_names=row_df.columns.tolist(), show=False)
    st.pyplot(fig)


if uploaded_file is not None:
    row_df = pd.read_csv(uploaded_file)
    st.write("Uploaded row:", row_df)
    if st.button("Predict (uploaded row)"):
        predict_and_explain(row_df)

if run_manual:
    # NOTE: this is simplified for demo purposes. The model expects all
    # 30 columns (Time, V1-V28, Amount). Here we default the anonymized
    # V1-V28 columns to 0 since we don't have real values for a manually
    # entered transaction, and only vary Time/Amount for illustration.
    # For a fully accurate manual demo, consider letting users pick a
    # real row from the test set instead and tweak Amount/Time on it.
    columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    values = [time] + [0.0] * 28 + [amount]
    row_df = pd.DataFrame([values], columns=columns)
    predict_and_explain(row_df)

st.divider()
st.subheader("Overall model performance")
st.image("shap_summary.png", caption="Which features matter most across all predictions")
