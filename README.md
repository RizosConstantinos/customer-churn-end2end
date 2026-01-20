# Customer Churn Prediction – End-to-End ML Project

This project focuses on predicting customer churn for a telecom/SaaS company using machine learning.
The goal is to identify customers who are likely to leave, allowing the business to take proactive retention actions.

## Project Structure
customer-churn-end2end/
├── data/
│ └── raw/ # Original dataset
│ └── processed/ # Processed dataset
├── notebooks/
│ ├── 01_EDA.ipynb # Exploratory Data Analysis
│ └── 02_Feature_engineering.ipynb # Domain-driven feature creation
│ └── 03_Preprocessing.ipynb # Data cleaning & preprocessing
├── src/
│   └── customer_churn/
│       ├── __init_.py
│       ├── preprocessing.py
│       └── feature_engineering.py
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

## Project Improvements

- Refactored the project into a proper Python package under `src/customer_churn/`.
- Added `__init__.py` to make modules importable.
- Removed all `sys.path` hacks; notebooks now work as clients of the reusable code.
- Installed the package in editable mode (`pip install -e .`) for clean imports and reproducibility.
- This improves maintainability, modularity, and aligns the project with industry best practices.




## Next Steps

- Model training and evaluation 
- Explainability and business insights
- Deployment and productionization
