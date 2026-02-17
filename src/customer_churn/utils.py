import pandas as pd
import shap


def preprocess_input(model_pipeline, customer_dict):
    """
    Prepare raw input and transform it using the pipeline preprocessor.
    """
    X_raw = pd.DataFrame([customer_dict])
    preprocessor = model_pipeline.named_steps["preprocessing"]
    X_transformed = preprocessor.transform(X_raw)
    return X_transformed, X_raw


def get_shap_top_features(model_pipeline, X_transformed, top_n=10):
    """
    Compute SHAP values EXACTLY like the training notebook.
    """
    # Extract trained XGBoost model ONLY
    xgb_model = model_pipeline.named_steps["classifier"]

    # Create SHAP explainer on raw XGBoost model
    explainer = shap.TreeExplainer(xgb_model)

    # Compute SHAP values
    shap_values = explainer.shap_values(X_transformed)

    # Binary classification: shap_values shape (1, n_features)
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    # Feature names from preprocessor
    feature_names = (
        model_pipeline.named_steps["preprocessing"]
        .get_feature_names_out()
    )

    shap_df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_vals[0]
    })

    shap_df["abs_shap"] = shap_df["shap_value"].abs()
    shap_df = (
        shap_df.sort_values("abs_shap", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    return shap_df, shap_df
