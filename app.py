import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fraud Detection Demo", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("fraud_model.pkl")

model = load_model()

st.title("💳 Credit Card Fraud Detector")
st.write(
    "This model was trained on 284K+ real (anonymized) transactions, "
    "where only ~0.17% were fraud. Enter transaction details below, "
    "or upload a row from the test set, to see a live prediction."
)

# Option A: Upload a CSV row
st.subheader("Option 1: Upload a transaction (CSV row)")
uploaded_file = st.file_uploader(
    "Upload a single-row CSV with the same columns as the training data (minus 'Class')",
    type="csv",
)

# Option B: Manual entry (simplified - Amount and Time only,

st.subheader("Option 2: Quick manual test (simplified)")
col1, col2 = st.columns(2)
with col1:
    amount = st.number_input("Transaction Amount ($)", value=100.0, min_value=0.0)
with col2:
    time = st.number_input("Time (seconds since first transaction)", value=50000.0)

run_manual = st.button("Predict (manual entry)")

# Prediction logic

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
    columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    values = [time] + [0.0] * 28 + [amount]
    row_df = pd.DataFrame([values], columns=columns)
    predict_and_explain(row_df)

st.divider()
st.subheader("Overall model performance")
st.image("shap_summary.png", caption="Which features matter most across all predictions")
