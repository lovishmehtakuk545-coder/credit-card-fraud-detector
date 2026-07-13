"""
Credit Card Fraud Detection - Streamlit App
--------------------------------------------
Loads the artifacts produced by train_model.ipynb:
    - best_fraud_model.pkl   (trained classifier)
    - scaler.pkl             (fitted StandardScaler)
    - feature_names.joblib   (exact column order the model expects)

Run locally with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="💳",
    layout="centered",
)

# ----------------------------------------------------------------------
# Load artifacts (cached so this only runs once per session, not on every
# widget interaction / rerun)
# ----------------------------------------------------------------------
try:
    cache_decorator = st.cache_resource
except AttributeError:
    # Older Streamlit versions (pre-1.18) don't have cache_resource.
    # Fall back to the legacy st.cache with allow_output_mutation=True,
    # which is needed for caching non-serializable objects like models.
    cache_decorator = lambda f: st.cache(allow_output_mutation=True)(f)


@cache_decorator
def load_artifacts():
    model = joblib.load("best_fraud_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.joblib")
    return model, scaler, feature_names

try:
    model, scaler, feature_names = load_artifacts()
except FileNotFoundError as e:
    st.error(
        "Could not find one or more required files "
        "(best_fraud_model.pkl, scaler.pkl, feature_names.joblib). "
        "Make sure they are in the same folder as app.py.\n\n"
        f"Details: {e}"
    )
    st.stop()

st.title("💳 Credit Card Fraud Detector")
st.write(
    "Enter transaction details below. The model will predict whether "
    "the transaction is likely **fraudulent** or **legitimate**."
)

# ----------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------
# NOTE: Adjust the category lists below (merchant_category, merchant_city,
# transaction_channel, payment_method, card_type, device_type) so they
# EXACTLY match the category values that existed in the training data.
# pd.get_dummies only creates columns for categories it saw during training,
# so a mismatched/misspelled category here will simply be dropped by the
# reindex step further down (it will not raise an error).

with st.form("transaction_form"):
    col1, col2 = st.columns(2)

    with col1:
        transaction_amount = st.number_input(
            "Transaction Amount", min_value=0.0, value=500.0, step=10.0
        )
        transaction_hour = st.slider("Transaction Hour (0-23)", 0, 23, 12)
        transaction_frequency_24h = st.number_input(
            "Transactions in last 24h", min_value=0, value=2, step=1
        )
        avg_transaction_amount_30d = st.number_input(
            "Avg Transaction Amount (30d)", min_value=0.0, value=1500.0, step=10.0
        )
        amount_deviation = st.number_input(
            "Amount Deviation from Avg", value=0.0, step=10.0
        )
        distance_from_home_km = st.number_input(
            "Distance from Home (km)", min_value=0.0, value=10.0, step=1.0
        )
        previous_fraud_count = st.number_input(
            "Previous Fraud Count", min_value=0, value=0, step=1
        )
        merchant_risk_score = st.slider(
            "Merchant Risk Score (0-100)", 0.0, 100.0, 25.0
        )

    with col2:
        merchant_category = st.selectbox(
            "Merchant Category",
            ["Grocery", "Electronics", "Travel", "Dining", "Fashion",
             "Fuel", "Entertainment", "Healthcare", "Utilities", "Other"],
        )
        merchant_city = st.selectbox(
            "Merchant City",
            ["Delhi", "Mumbai", "Bengaluru", "Chennai", "Hyderabad",
             "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Other"],
        )
        transaction_channel = st.selectbox(
            "Transaction Channel", ["Online", "POS", "ATM", "Mobile App"]
        )
        payment_method = st.selectbox("Payment Method", ["Credit Card", "Debit Card"])
        card_type = st.selectbox("Card Type", ["Visa", "Mastercard", "RuPay", "Amex"])
        device_type = st.selectbox(
            "Device Type", ["Android", "iPhone", "Windows", "macOS", "Other"]
        )
        international_transaction = st.checkbox("International Transaction")
        card_present = st.checkbox("Card Present", value=True)
        otp_verified = st.checkbox("OTP Verified", value=True)
        billing_shipping_match = st.checkbox("Billing/Shipping Match", value=True)
        device_trusted = st.checkbox("Device Trusted", value=True)

    submitted = st.form_submit_button("Predict")

# ----------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------
if submitted:
    # 1. Assemble a single-row DataFrame matching the raw column names
    #    used before preprocessing in the notebook.
    raw_input = pd.DataFrame([{
        "transaction_amount": transaction_amount,
        "transaction_hour": transaction_hour,
        "transaction_frequency_24h": transaction_frequency_24h,
        "avg_transaction_amount_30d": avg_transaction_amount_30d,
        "amount_deviation": amount_deviation,
        "distance_from_home_km": distance_from_home_km,
        "previous_fraud_count": previous_fraud_count,
        "merchant_category": merchant_category,
        "merchant_city": merchant_city,
        "transaction_channel": transaction_channel,
        "payment_method": payment_method,
        "card_type": card_type,
        "device_type": device_type,
        "international_transaction": int(international_transaction),
        "card_present": int(card_present),
        "otp_verified": int(otp_verified),
        "billing_shipping_match": int(billing_shipping_match),
        "device_trusted": int(device_trusted),
        "merchant_risk_score": merchant_risk_score,
    }])

    # 2. One-hot encode categorical columns the same way as training
    encoded = pd.get_dummies(raw_input, drop_first=True)

    # 3. Reindex to match the exact columns/order the model was trained on.
    #    Any column the model expects but that didn't appear here (because
    #    this single row doesn't cover every category) is filled with 0.
    aligned = encoded.reindex(columns=feature_names, fill_value=0)

    # 4. Scale using the SAME fitted scaler from training (transform, not fit)
    scaled = scaler.transform(aligned)

    # 5. Predict
    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]  # P(fraud)

    st.markdown("---")
    if prediction == 1:
        st.error(f"⚠️ Prediction: **FRAUDULENT** (probability: {probability:.2%})")
    else:
        st.success(f"✅ Prediction: **Legitimate** (fraud probability: {probability:.2%})")

    with st.expander("See processed feature vector"):
        st.dataframe(aligned)

st.markdown("---")
st.caption(
    "This app uses a model trained on synthetic data for demonstration "
    "purposes. Predictions should not be used for real financial decisions."
)