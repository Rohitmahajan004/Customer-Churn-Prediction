import streamlit as st
import pandas as pd
import joblib

# ---------------------------
# Load Model & Feature Names
# ---------------------------
model = joblib.load("churn_model.pkl")
feature_names = joblib.load("feature_names.pkl")  # saved from churn_model.ipynb

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide",
    page_icon="📊"
)

# ---------------------------
# Header Section
# ---------------------------
st.title("📊 Customer Churn Prediction System")
st.write(
    """
    This web app predicts whether a customer will churn or stay,  
    based on their demographics and service usage.  
    Powered by **Machine Learning** 🚀
    """
)

# ---------------------------
# Sidebar for Input
# ---------------------------
st.sidebar.header("🔎 Enter Customer Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
phone = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
total_charges = st.sidebar.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=1000.0)

# ---------------------------
# Single Customer Prediction
# ---------------------------
if st.sidebar.button("🔮 Predict Churn"):
    # Prepare DataFrame
    input_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "InternetService": internet,
        "Contract": contract,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    # Encode categorical columns
    input_data = pd.get_dummies(input_data)

    # Align columns with training features
    input_data = input_data.reindex(columns=feature_names, fill_value=0)

    # Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("📌 Prediction Result:")
    if prediction == 1:
        st.error(f"❌ This customer is **likely to churn**.\n\nProbability: {probability:.2%}")
    else:
        st.success(f"✅ This customer is **likely to stay**.\n\nProbability: {probability:.2%}")

# ---------------------------
# Bulk Upload Prediction
# ---------------------------
st.markdown("---")
st.subheader("📂 Bulk Prediction with CSV Upload")

uploaded_file = st.file_uploader("Upload a CSV file with customer details", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### 📄 Uploaded Data Preview")
    st.dataframe(data.head())

    # Encode categorical columns
    data = pd.get_dummies(data)

    # Align columns with training features
    data = data.reindex(columns=feature_names, fill_value=0)

    # Predict for all rows
    preds = model.predict(data)
    probs = model.predict_proba(data)[:, 1]

    data["Churn_Prediction"] = preds
    data["Churn_Probability"] = probs

    st.write("### 📊 Predictions")
    st.dataframe(data)

    # Download option
    csv = data.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Predictions",
        csv,
        "churn_predictions.csv",
        "text/csv"
    )
