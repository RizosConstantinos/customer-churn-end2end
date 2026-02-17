import os
import joblib
import pandas as pd

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
    """
    Split dataframe into features (X), target (y),
    and infer numeric/categorical feature columns.
    """
    drop_cols = [target_col]
    if id_col is not None:
        drop_cols.append(id_col)

    X = df.drop(columns=drop_cols, errors="ignore")
    y = df[target_col]

    num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_features = X.select_dtypes(include=["object", "bool"]).columns.tolist()

    return X, y, num_features, cat_features


# =========================================================
# Logistic Regression Pipeline
# =========================================================

def build_logistic_pipeline(num_features, cat_features):
    """
    Logistic Regression pipeline with preprocessing.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_features)
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("classifier", LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            ))
        ]
    )

    return pipeline


# =========================================================
# XGBoost Pipeline (BASE_SCORE FIXED)
# =========================================================

def build_xgboost_pipeline(num_features, cat_features, scale_pos_weight=1):
    """
    XGBoost pipeline with preprocessing.
    base_score is EXPLICITLY set to float (0.5) to avoid [5E-1] SHAP crash.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_features)
        ]
    )

    xgb_clf = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
        base_score=0.5,              # <<< FIXED
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("classifier", xgb_clf)
        ]
    )

    return pipeline


# =========================================================
# Hyperparameter tuning (optional)
# =========================================================

def tune_xgboost_hyperparameters(pipeline, X_train, y_train):
    """
    GridSearchCV for XGBoost.
    """
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
# Save model
# =========================================================

def save_model(model, filepath):
    """
    Save trained pipeline to disk.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Safety check: ensure base_score is float
    clf = model.named_steps["classifier"]
    if hasattr(clf, "base_score"):
        if clf.base_score is None:
            clf.base_score = 0.5
        elif isinstance(clf.base_score, str):
            try:
                clf.base_score = float(clf.base_score.strip("[]"))
            except ValueError:
                clf.base_score = 0.5
    print("Final base_score:", clf.base_score)

    joblib.dump(model, filepath)
    print(f"Model saved to: {filepath}")
