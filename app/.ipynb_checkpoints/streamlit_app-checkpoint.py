import streamlit as st
import os
import requests
import pandas as pd
import plotly.express as px
import joblib
import sys
import shap

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- FastAPI URL (Docker service name or localhost) ---
API_URL = "http://127.0.0.1:8000"

# --- Page Config ---
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Customer Churn Prediction Demo")
st.markdown("""
Interactive demo predicting whether a customer is likely to **churn**.  
Fill in the sidebar and click **Predict**.  
Top SHAP features show which factors influenced the prediction.
""")

# --- Sidebar Inputs ---
st.sidebar.header("Customer Profile Inputs")
SeniorCitizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
tenure = st.sidebar.number_input("Tenure (months)", value=12, step=1)
MonthlyCharges = st.sidebar.number_input("Monthly Charges", value=70.0)
TotalCharges = st.sidebar.number_input("Total Charges", value=1500.0)
tenure_monthly_ratio = st.sidebar.number_input("Tenure / MonthlyCharges ratio", value=0.17)
services_count = st.sidebar.number_input("Number of services subscribed", value=3)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
Partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
Dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
PhoneService = st.sidebar.selectbox("PhoneService", ["Yes", "No"])
MultipleLines = st.sidebar.selectbox("MultipleLines", ["No", "Yes", "No phone service"])
InternetService = st.sidebar.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.sidebar.selectbox("OnlineSecurity", ["Yes", "No", "No internet service"])
OnlineBackup = st.sidebar.selectbox("OnlineBackup", ["Yes", "No", "No internet service"])
DeviceProtection = st.sidebar.selectbox("DeviceProtection", ["Yes", "No", "No internet service"])
TechSupport = st.sidebar.selectbox("TechSupport", ["Yes", "No", "No internet service"])
StreamingTV = st.sidebar.selectbox("StreamingTV", ["Yes", "No", "No internet service"])
StreamingMovies = st.sidebar.selectbox("StreamingMovies", ["Yes", "No", "No internet service"])
Contract = st.sidebar.selectbox("Contract", ["Month-to-Month", "One year", "Two year"])
PaperlessBilling = st.sidebar.selectbox("PaperlessBilling", ["Yes", "No"])
PaymentMethod = st.sidebar.selectbox(
    "PaymentMethod", 
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)
contract_payment_interaction = st.sidebar.selectbox(
    "Contract-Payment interaction", 
    ["Month-to-Month_Electronic check", "Month-to-Month_Mailed check", "Month-to-Month_Credit card (automatic)",
     "One year_Bank transfer (automatic)", "One year_Credit card (automatic)", "One year_Electronic check", "One year_Mailed check",
     "Two year_Bank transfer (automatic)", "Two year_Credit card (automatic)", "Two year_Electronic check", "Two year_Mailed check"]
)

# --- Build input dictionary ---
customer_dict = {
    "SeniorCitizen": SeniorCitizen,
    "tenure": tenure,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges,
    "tenure_monthly_ratio": tenure_monthly_ratio,
    "services_count": services_count,
    "gender": gender,
    "Partner": Partner,
    "Dependents": Dependents,
    "PhoneService": PhoneService,
    "MultipleLines": MultipleLines,
    "InternetService": InternetService,
    "OnlineSecurity": OnlineSecurity,
    "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection,
    "TechSupport": TechSupport,
    "StreamingTV": StreamingTV,
    "StreamingMovies": StreamingMovies,
    "Contract": Contract,
    "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod,
    "contract_payment_interaction": contract_payment_interaction
}

# --- Predict button ---
if st.button("Predict"):
    with st.spinner("Calling prediction API..."):
        response = requests.post(
            f"{API_URL}/predict",
            json=customer_dict,
            timeout=10
        )

    if response.status_code != 200:
        st.error("❌ API error – check FastAPI service")
    else:
        result = response.json()
        prob = result["probability"]
        pred = result["prediction"]

        # --- Show prediction result ---
        st.balloons()
        st.subheader("Prediction Result")
        if pred == 1:
            st.error("⚠️ Customer is likely to churn")
        else:
            st.success("✅ Customer is likely to stay")

        # --- Gauge for probability ---
        st.subheader("Churn Probability")
        fig_gauge = px.pie(
            values=[prob, 1-prob],
            names=["Churn Probability", "No Churn"],
            hole=0.6,
            color_discrete_sequence=["red", "green"]
        )
        fig_gauge.update_traces(textinfo='percent+label', textposition='inside')
        st.plotly_chart(fig_gauge, width="stretch")

        # --- SHAP values ---
        st.subheader("Top Features Influencing This Prediction")

        # --- Load model pipeline ---
        model_pipeline = joblib.load("models/churn_model_final_v1.joblib")

        # --- Transform input using preprocessor only ---
        preprocessor = model_pipeline.named_steps["preprocessing"]
        X_transformed = preprocessor.transform(pd.DataFrame([customer_dict]))

        # --- Extract trained XGB model ---
        xgb_model = model_pipeline.named_steps["classifier"]

        # --- Create SHAP explainer and values ---
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_transformed)

        # --- Get feature names ---
        feature_names = preprocessor.get_feature_names_out()

        # --- Build top features DataFrame ---
        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0]
        else:
            shap_vals = shap_values[0]   
        shap_df = pd.DataFrame({
            "feature": feature_names,
            "shap_value": shap_vals
        })
        shap_df["abs_shap"] = shap_df["shap_value"].abs()
        top_features = shap_df.sort_values("abs_shap", ascending=False).head(10).reset_index(drop=True)

        st.markdown("""
        Blue bars increase churn probability, red bars decrease it. The longer the bar, the stronger the impact.
        """)

        fig = px.bar(
            top_features,
            x='abs_shap',
            y='feature',
            orientation='h',
            color='shap_value',
            color_continuous_scale=['red', 'blue'],
            labels={'abs_shap': 'Impact', 'feature': 'Feature', 'shap_value': 'SHAP value'},
            title='Feature Importance (SHAP)'
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=400)
        st.plotly_chart(fig, width="stretch")

        st.write("Detailed SHAP values:")
        st.table(top_features[['feature', 'shap_value', 'abs_shap']])
