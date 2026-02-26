import os
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

# =========================================================
# Feature split
# =========================================================
def split_features(df, target_col, id_col=None):
    """Split dataframe into X, y and infer numeric/categorical features"""
    drop_cols = [target_col]
    if id_col is not None:
        drop_cols.append(id_col)

    X = df.drop(columns=drop_cols, errors="ignore")
    y = df[target_col]

    num_features = X.select_dtypes(include="number").columns.tolist()
    cat_features = X.select_dtypes(include=["object", "bool"]).columns.tolist()

    return X, y, num_features, cat_features

# =========================================================
# Preprocessor helper
# =========================================================
def get_preprocessor(num_features, cat_features):
    """Return a ColumnTransformer with numeric and categorical pipelines"""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_features)
        ]
    )

# =========================================================
# Logistic Regression Pipeline
# =========================================================
def build_logistic_pipeline(num_features, cat_features):
    """Build Logistic Regression pipeline with preprocessing"""
    preprocessor = get_preprocessor(num_features, cat_features)
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    return pipeline

# =========================================================
# XGBoost Pipeline
# =========================================================
def build_xgboost_pipeline(num_features, cat_features, scale_pos_weight=1):
    """Build XGBoost pipeline with preprocessing"""
    preprocessor = get_preprocessor(num_features, cat_features)
    xgb_clf = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
        base_score=0.5,  # prevent SHAP crash
    )
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", xgb_clf)
    ])
    return pipeline

# =========================================================
# Hyperparameter tuning
# =========================================================
def tune_xgboost_hyperparameters(pipeline, X_train, y_train, param_grid=None):
    """Run GridSearchCV for XGBoost pipeline"""
    if param_grid is None:
        param_grid = {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [3, 4, 5],
            "classifier__learning_rate": [0.01, 0.1],
            "classifier__subsample": [0.8, 1.0],
            "classifier__base_score": [0.5]
        }
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    print(f"Best F1: {grid_search.best_score_:.4f}")
    print(f"Best params: {grid_search.best_params_}")
    return grid_search.best_estimator_

# =========================================================
# Model saving
# =========================================================
def save_model(model, filepath):
    """Save pipeline to disk safely"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    clf = model.named_steps.get("classifier")
    if getattr(clf, "base_score", None) is None:
        clf.base_score = 0.5

    joblib.dump(model, filepath)
    print(f"Model saved to: {filepath}")

# =========================================================
# Threshold helper
# =========================================================
def apply_threshold(y_proba, threshold=0.5):
    """Convert probabilities to binary predictions using custom threshold"""
    return (y_proba >= threshold).astype(int)

def plot_feature_importance(pipeline, top_n=15, title="Top Features"):
    """
    Plot top N feature importances with readable labels:
    - Categorical features: 'Feature: Value'
    - Numeric features: 'Feature'
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    # Extract feature names and importances
    feature_names = pipeline.named_steps['preprocessing'].get_feature_names_out()
    importances = pipeline.named_steps['classifier'].feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    def clean_label(name):
        # Detect categorical one-hot features (have 'cat_' prefix or contain '_')
        if name.lower().startswith("cat") or name.count("_") >= 2:
            # Split the prefix and value
            parts = name.split("_", 1)
            value = parts[1] if len(parts) > 1 else parts[0]
            # Capitalize first letter of feature/value for readability
            return value.replace("_", " ").title()
        else:
            # Numeric features: keep as is
            return name

    feat_imp.index = [clean_label(f) for f in feat_imp.index]

    # Plot top N features
    plt.figure(figsize=(10, 7))
    feat_imp.head(top_n).plot(kind='barh', color='skyblue').invert_yaxis()
    plt.title(title)
    plt.xlabel("Importance Score")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()