# Customer Churn Prediction – End-to-End ML Project

This project focuses on predicting customer churn for a telecom/SaaS company using machine learning.
The goal is to identify customers who are likely to leave, allowing the business to take proactive retention actions.

## Project Structure
customer-churn-end2end/
├── data/
│ └── raw/ # Original dataset
│ └── processed/ # Processed dataset
├── notebooks/
│ └── 01_EDA.ipynb # Exploratory Data Analysis
│ └── 02_Feature_engineering.ipynb # Domain-driven feature creation
│ └── 03_Preprocessing.ipynb # Data cleaning & preprocessing
│ └── 04_Model_Training.ipynb # Model Training, Tuning & Evaluation
│ └── 05_Explainability.ipynb # Explainability & Business Storytelling
├── src/
│   └── customer_churn/
│       └── __init_.py
│       └── preprocessing.py
│       └── feature_engineering.py
│       └── evaluation.py
│       └── modeling.py
├── pyproject.toml
├── README.md
├── requirements.txt


## 01 – Exploratory Data Analysis (EDA)

- Dataset overview and basic statistics
- Missing value analysis
- Churn distribution analysis
- Visualizations (histograms, boxplots, correlations)

**Deliverable:** EDA notebook with visual insights and observations.


## 02 – Feature Engineering

- Designed domain-driven features to enrich the dataset
- Created tenure-based ratio feature (tenure / MonthlyCharges)
- Engineered service usage feature (services_count) based on subscribed services
- Added interaction feature between contract type and payment method
- Validated new features using correlation analysis and visual inspection
- Implemented reusable feature engineering functions under src/feature_engineering.py

**Deliverable:** Feature-engineered dataset with additional predictive signals + modular feature engineering code.


## 03 – Data Cleaning & Preprocessing

- Converted mixed-type columns (e.g. TotalCharges) to numeric
- Handled missing values (dropped negligible number of rows)
- One-hot encoded categorical features
- Scaled numerical features using StandardScaler
- Created modular preprocessing functions under `src/preprocessing.py`
- Tested preprocessing pipeline on sample data

**Deliverable:** Clean dataset ready for modeling + reusable preprocessing code.


## 04 – Model Training, Tuning & Evaluation

- Baseline & Advanced Models: Compared Logistic Regression with XGBoost to capture non-linear customer patterns.
- Optimization: Performed GridSearchCV with 5-fold Cross-Validation to fine-tune hyperparameters for maximum stability.
- Threshold Tuning: Adjusted decision threshold to 0.4 to prioritize Recall (0.87), ensuring fewer churners are missed.
- Pipeline & Export: Wrapped preprocessing and model into a single Pipeline and exported it using joblib.
- Feature Importance: Identified top churn drivers (Contract Type, Tenure) to guide business strategy.

**Deliverable:**  04_Modeling.ipynb with comparative analysis and performance metrics.
                  Optimized model file: models/churn_model_final_v1.joblib
                  Modular modeling and evaluation functions under src/customer_churn/modeling.py and evaluation.py


## 05 – Explainability & Business Storytelling

The focus of this stage is to move beyond model performance metrics and provide
clear, actionable business insights into customer churn.

### Global Explainability
- SHAP analysis to identify global churn drivers.
- Contract type, tenure, and monthly charges show the strongest overall impact.
- Summary plots highlight magnitude and direction of feature effects.

### Customer-Level Explainability
- Local SHAP explanations (force, decision, waterfall plots) for high-risk customers.
- Clear breakdown of feature contributions per customer.
- Explains why a customer is likely to churn, not just the prediction.

### Risk Segmentation & Cohort Analysis
- Customers segmented into Low / Medium / High risk buckets.
- Cohort analysis (contract type, tenure) reveals systematic churn patterns.
- Enables targeted retention strategies.

**Deliverable:**  
Explainability-focused notebook with global insights, customer-level explanations, and business storytelling.


## Business Impact

This project enables the business to shift from reactive to proactive churn management.

- Early identification of high-risk customers using churn probabilities.
- Clear visibility into key churn drivers.
- Supports personalized retention actions and efficient resource allocation.
- Risk-based prioritization improves business impact of retention efforts.


## Project Improvements

- Refactored the project into a proper Python package under `src/customer_churn/`.
- Added `__init__.py` to make modules importable.
- Removed all `sys.path` hacks; notebooks now work as clients of the reusable code.
- Installed the package in editable mode (`pip install -e .`) for clean imports and reproducibility.
- This improves maintainability, modularity, and aligns the project with industry best practices.


## Next Steps

- Deployment and productionization
