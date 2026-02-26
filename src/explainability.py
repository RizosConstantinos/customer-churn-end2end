# src/explainability.py

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# --------------------------
# SHAP Extraction
# --------------------------
def get_shap_values(model, X):
    """
    Extract SHAP values, transformed features, preprocessor, and explainer from pipeline.
    """
    preprocessor = model.named_steps["preprocessing"]
    X_transformed = preprocessor.transform(X)
    explainer = shap.TreeExplainer(model.named_steps["classifier"])
    shap_values = explainer.shap_values(X_transformed)
    return X_transformed, explainer, shap_values, preprocessor

# --------------------------
# SHAP Global Importance Plot
# --------------------------
def plot_shap_global(shap_values, feature_names, top_n=15, title="Top Features - SHAP"):
    """
    Plot mean absolute SHAP values for top N features.
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = pd.Series(mean_abs_shap, index=feature_names).sort_values(ascending=False)
    plt.figure(figsize=(10,7))
    shap_importance.head(top_n).plot(kind='barh', color='skyblue')
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("Mean Absolute SHAP Value")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# --------------------------
# SHAP Summary Plot
# --------------------------
def plot_shap_summary(shap_values, X_transformed, feature_names):
    """
    Summary plot with original feature names.
    """
    plt.text(-1.1, -0.5, "Blue = low feature value\nLeft = decreases churn", fontsize=8, color="blue")
    plt.text(0.2, -0.5, "Red = high feature value\nRight = increases churn", fontsize=8, color="red")
    shap.summary_plot(shap_values, X_transformed, feature_names=feature_names)


#------------------------
# SHAP Waterfall Plot for Individual Customer
# --------------------------
def plot_shap_waterfall(explainer, shap_values, X_transformed, preprocessor,
                        high_risk_customers=None, X_test=None, top_n=5):
    """
    Generate waterfall plots for top-N high-risk customers.
    If high_risk_customers is provided, computes positions automatically using X_test index.
    """
    if high_risk_customers is not None and X_test is not None:
        # Map original indices to positions in X_transformed/shap_values
        positions = [X_test.index.get_loc(idx) for idx in high_risk_customers.index[:top_n]]
    else:
        positions = list(range(top_n))

    # Plot waterfall for each selected customer
    for pos in positions:
        shap.plots.waterfall(shap.Explanation(
            values=shap_values[pos],
            base_values=explainer.expected_value,
            data=X_transformed[pos],
            feature_names=preprocessor.get_feature_names_out()
        ))
# --------------------------
# Risk Buckets
# --------------------------
def compute_risk_buckets(probs, buckets={"Low Risk":(0,0.3),"Medium Risk":(0.3,0.7),"High Risk":(0.7,1.0)}):
    """
    Assign probabilities to risk buckets.
    """
    labels = [next(bucket for bucket, (low, high) in buckets.items() if low <= prob < high) for prob in probs]
    return labels

def get_high_risk_customers(model, X_test, threshold=0.5):
    """
    Identify high-risk customers based on predicted probabilities.
    Returns a DataFrame with churn probabilities sorted descending.
    """
    y_probs = model.predict_proba(X_test)[:,1]                 # predicted probabilities
    high_risk_idx = y_probs >= threshold                        # mask high-risk
    high_risk_customers = X_test[high_risk_idx].copy()          # subset DataFrame
    high_risk_customers["Churn_Probability"] = y_probs[high_risk_idx]  # add column
    return high_risk_customers.sort_values(by="Churn_Probability", ascending=False)


def explain_single_customer(model, X_test_transformed, shap_values, explainer, original_feature_map, i=0):
    """
    Explain churn probability for a single customer using SHAP.
    Returns churn probability and plots a force plot.
    """
    # select row and keep 2D
    x_row_transformed = X_test_transformed[i].reshape(1, -1)

    # compute log-odds and convert to probability
    f_x = explainer.expected_value + shap_values[i].sum()
    p_churn = 1 / (1 + np.exp(-f_x))
    print(f"Churn probability for customer {i}: {p_churn*100:.2f}%")

    # create force plot
    shap.force_plot(
        explainer.expected_value,        # expected model output
        shap_values[i],                  # SHAP values for this row
        x_row_transformed,               # transformed features
        feature_names=original_feature_map,  # readable feature names
        matplotlib=True                  # matplotlib backend
    )    

def explain_top_n_customers(high_risk_customers, X_test, X_test_transformed, shap_values, explainer, original_feature_map, top_n=5):
    """
    Plot force plots for top-N high-risk customers.
    Prints churn probability for each customer.
    """
    # get positions in X_test_transformed
    top_positions = [X_test.index.get_loc(idx) for idx in high_risk_customers.index[:top_n]]

    # loop over top customers
    for idx, pos in zip(high_risk_customers.index[:top_n], top_positions):
        print(f"Customer {idx} Churn Probability = {high_risk_customers['Churn_Probability'].loc[idx]:.3f}")
        shap.force_plot(
            explainer.expected_value,
            shap_values[pos],
            X_test_transformed[pos],
            feature_names=original_feature_map,
            matplotlib=True
        )