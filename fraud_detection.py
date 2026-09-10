import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="AI Financial Fraud Detection",
    page_icon="🔐",
    layout="centered"
)

# --------------------------------------------------
# Load Model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("fraud_detection_pipeline.pkl")


model = load_model()

# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("🔐 AI Financial Fraud Detection System")

st.markdown(
    """
    **Machine Learning powered transaction risk analysis**

    Enter the transaction details below to determine whether
    the transaction is likely to be fraudulent.
    """
)

st.divider()

# --------------------------------------------------
# Transaction Details
# --------------------------------------------------
st.subheader("💳 Transaction Details")

transaction_type = st.selectbox(
    "Transaction Type",
    ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT"]
)

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=1000.0,
        step=100.0
    )

    oldbalanceOrg = st.number_input(
        "Old Balance (Sender)",
        min_value=0.0,
        value=10000.0,
        step=100.0
    )

    oldbalanceDest = st.number_input(
        "Old Balance (Receiver)",
        min_value=0.0,
        value=0.0,
        step=100.0
    )

with col2:
    newbalanceOrig = st.number_input(
        "New Balance (Sender)",
        min_value=0.0,
        value=9000.0,
        step=100.0
    )

    newbalanceDest = st.number_input(
        "New Balance (Receiver)",
        min_value=0.0,
        value=0.0,
        step=100.0
    )

st.divider()

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🔍 Analyze Transaction", use_container_width=True):

    # Basic validation
    if amount <= 0:
        st.warning("Please enter a transaction amount greater than 0.")
        st.stop()

    if oldbalanceOrg < 0 or newbalanceOrig < 0:
        st.warning("Sender balances cannot be negative.")
        st.stop()

    if oldbalanceDest < 0 or newbalanceDest < 0:
        st.warning("Receiver balances cannot be negative.")
        st.stop()

    # Create input DataFrame
    input_data = pd.DataFrame([{
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest
    }])

    # Prediction
    prediction = model.predict(input_data)[0]

    # Try to obtain probability
    probability = None

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data)[0][1]

    # --------------------------------------------------
    # Results
    # --------------------------------------------------
    st.subheader("📊 Analysis Result")

    if prediction == 1:

        st.error("🚨 HIGH RISK — Potential Fraud Detected")

        if probability is not None:
            risk_score = probability * 100
            st.metric(
                "Fraud Risk Score",
                f"{risk_score:.2f}%"
            )

        st.warning(
            "This transaction has been classified as potentially fraudulent "
            "by the machine learning model."
        )

    else:

        st.success("✅ LOW RISK — Transaction Appears Genuine")

        if probability is not None:
            risk_score = probability * 100
            st.metric(
                "Fraud Risk Score",
                f"{risk_score:.2f}%"
            )

        st.info(
            "The machine learning model classified this transaction "
            "as likely genuine."
        )

    # --------------------------------------------------
    # Transaction Summary
    # --------------------------------------------------
    st.subheader("📋 Transaction Summary")

    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:
        st.write(f"**Transaction Type:** {transaction_type}")
        st.write(f"**Amount:** {amount:,.2f}")
        st.write(f"**Sender Old Balance:** {oldbalanceOrg:,.2f}")

    with summary_col2:
        st.write(f"**Sender New Balance:** {newbalanceOrig:,.2f}")
        st.write(f"**Receiver Old Balance:** {oldbalanceDest:,.2f}")
        st.write(f"**Receiver New Balance:** {newbalanceDest:,.2f}")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()

st.caption(
    "AI Financial Fraud Detection System | "
    "Machine Learning & Streamlit"
)