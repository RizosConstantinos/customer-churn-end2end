def split_features(df, target_col, id_col=None):
    """
    Split dataframe into features (X), target (y), 
    and infer numeric/categorical feature columns.
    
    Returns:
    - X: DataFrame with feature columns
    - y: Series with target
    - num_features: list of numeric feature column names
    - cat_features: list of categorical feature column names
    """
    drop_cols = [target_col]
    if id_col is not None:
        drop_cols.append(id_col)

    X = df.drop(columns=drop_cols, errors="ignore")
    y = df[target_col]

    num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_features = X.select_dtypes(include=["object", "bool"]).columns.tolist()

    return X, y, num_features, cat_features


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression


def build_logistic_pipeline(num_features, cat_features):
    """
    Build a logistic regression pipeline with preprocessing.
    - Numeric features are scaled
    - Categorical features are one-hot encoded
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop='first'), cat_features)
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, class_weight='balanced'))
        ]
    )

    return pipeline

from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def build_xgboost_pipeline(num_features, cat_features, scale_pos_weight=1):
    """
    Build an XGBoost pipeline with preprocessing.
    scale_pos_weight handles the imbalanced target classes.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop='first'), cat_features)
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("classifier", XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                eval_metric='logloss'
            ))
        ]
    )
    return pipeline

from sklearn.model_selection import GridSearchCV

def tune_xgboost_hyperparameters(pipeline, X_train, y_train):
    """
    Finds the best hyperparameters for XGBoost using GridSearchCV.
    """
    # Set the Grid
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 4, 5],
        'classifier__learning_rate': [0.01, 0.1],
        'classifier__subsample': [0.8, 1.0]
    }

    # use of f1' to balance Precision & Recall
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,            # 5-fold Cross-Validation
        scoring='f1',
        n_jobs=-1,       
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    
    print(f"Best Score (F1): {grid_search.best_score_:.3f}")
    print(f"Best Parameters: {grid_search.best_params_}")
    
    return grid_search.best_estimator_    

import joblib
import os

def save_model(model, filepath):
    """
    Saves the trained pipeline to a specific path using joblib.
    """
    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    joblib.dump(model, filepath)
    print(f"Model successfully saved to: {filepath}")    