# 🚀 Customer Churn Prediction – End-to-End ML Project

## 🎯 Live Demo

* 🌐 Streamlit App: *(βάλε link όταν κάνεις deploy)*
* ⚡ FastAPI Docs (Swagger): *(βάλε link)*

---

## 💼 Business Impact

* **~18% customers flagged as high-risk** (250 / 1,407)
* **Month-to-month contracts:** ~60% churn vs **~13%** for long-term contracts
* **Estimated revenue impact:** ~€2M/year saved with 5% churn reduction
* **Top 20% high-risk customers → ~50% of churn**
* Enables **targeted retention strategies & ROI optimization**

---

## 📸 Demo Preview

*(βάλε screenshots εδώ)*

* Streamlit dashboard
* SHAP explainability plots
* FastAPI Swagger UI

---

## 🧠 Problem Statement

Customer churn is a major challenge in telecom/SaaS businesses.
The goal of this project is to:

* Predict which customers are likely to churn
* Explain **why** they churn
* Provide **actionable insights** for business decisions

---

## 🛠 Tech Stack

* **Python** (Pandas, NumPy, Scikit-learn)
* **XGBoost**
* **SHAP** (Explainability)
* **FastAPI** (Backend API)
* **Streamlit** (Interactive dashboard)
* **Joblib** (Model persistence)

---

## ⚙️ ML Pipeline Overview

1. **EDA** → Data understanding & churn patterns
2. **Data Cleaning & Preprocessing**
3. **Feature Engineering** (domain-driven features)
4. **Model Training & Optimization**
5. **Explainability (SHAP)**
6. **Deployment (FastAPI + Streamlit)**

---

## 🧪 Model Performance

* Model: **XGBoost (tuned)**
* Recall: **0.87** (optimized to catch churners)
* Threshold: **0.4** (business-driven tuning)

👉 Focus: **Minimize missed churners (false negatives)**

---

## 🔍 Key Features & Insights

* **Tenure / Monthly Charges ratio**
* **Contract Type impact**
* **Fiber Internet usage**
* **Service usage intensity**

These features explain **~55% of churn behavior**.

---

## 📊 Explainability

### Global

* SHAP summary plots highlight main churn drivers

### Local (per customer)

* SHAP force & waterfall plots explain predictions
* Helps answer: *"Why THIS customer will churn?"*

---

## 🧩 Project Structure

```text
customer-churn-end2end/
├── data/ # Raw and processed datasets
├── notebooks/
│   ├── 01_EDA.ipynb # Exploratory Data Analysis
│   ├── 02_Feature_engineering.ipynb # Domain-driven feature creation
│   ├── 03_Preprocessing.ipynb # Data cleaning & preprocessing
│   ├── 04_Model_Training.ipynb # Model Training, Tuning & Evaluation
│   ├── 05_Explainability.ipynb # Explainability & Business Storytelling
├── src/
│   ├── __init__.py # package initialization
│   ├── preprocessing.py # data cleaning and preprocessing functions
│   ├── feature_engineering.py # create derived or domain-specific features
│   ├── evaluation.py # model evaluation metrics and helper functions
│   ├── modeling.py # model training, fitting, and prediction logic
│   ├── explainability # SHAP-based explainability functions & visualizations
├── models/ # Trained model artifacts (joblib)
├── app/
│   ├── main.py # Fast API
│   ├── streamlit_app.py # Interactive UI Streamlit App
│   ├── Customer_Churn_Prediction_App_Guide.pdf # PDF with step-by-step setup and usage instructions
├── pyproject.toml # project build configuration and package metadata
├── README.md  # project overview, setup, and usage instructions
├── requirements.txt # list of Python dependencies for the project
├── run_my_apps.ps1 # PowerShell script to run FastAPI and Streamlit apps
```

---

## 🚀 How to Run Locally

### 1. Clone repo

```bash
git clone <your-repo>
cd customer-churn-end2end
```

### 2. Create environment

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate (Windows)
pip install -r requirements.txt
```

### 3. Run apps

```bash
# FastAPI
uvicorn app.main:app --reload

# Streamlit
streamlit run app/streamlit_app.py
```

---

## 🔌 API Endpoints

* `/predict` → returns prediction + probability
* `/health` → service status

---

## 📘 Documentation

A detailed setup and usage guide is available:

👉 **Customer_Churn_Prediction_App_Guide.pdf**

Includes:
- Environment setup
- Running FastAPI & Streamlit
- App usage & inputs
- Troubleshooting

---

## 📦 Production Features

* Modular `src/` structure
* Reusable preprocessing & modeling pipeline
* Unit tests for core functions
* Clean separation: notebooks vs production code

---

## 📈 Future Improvements

* Model monitoring & data drift detection
* Batch prediction endpoint (CSV upload)
* Docker containerization
* Cloud deployment (AWS / GCP)

---

## 🏗 Engineering & Production Practices

- Refactored codebase into a proper Python package (`src/`)
- Eliminated `sys.path` hacks → clean imports & reproducibility
- Separated notebooks (EDA) from production code
- Installed project in editable mode (`pip install -e .`)
- Designed modular, reusable ML pipeline

👉 Focus: **scalability, maintainability, production-readiness**

---

## 👤 Author

**Rizos Constantinos**  
- LinkedIn: www.linkedin.com/in/constantinos-rizos-0589b5254  
- GitHub: https://github.com/RizosConstantinos
  
---

## ⭐ If you found this useful, feel free to star the repo!
