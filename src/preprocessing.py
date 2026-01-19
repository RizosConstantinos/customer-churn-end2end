import pandas as pd
from sklearn.preprocessing import StandardScaler
    
def handle_missing_values(df, subset=None):
    """
    Handle missing values in the dataset.
    Currently drops rows with missing values.
    """
    if subset is not None:
        return df.dropna(subset=subset)
    else:
        return df.dropna()

    
def encode_categorical_features(df, categorical_features):
    """
    One-hot encode categorical features.
    """
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    return df_encoded
    
def scale_numeric_features(df, numeric_features):
    """
    Scale numeric features using StandardScaler.
    """
    scaler = StandardScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])
    return df
    
def convert_to_numeric(df, columns):
    """
    Convert specified columns to numeric.
    Non-convertible values become NaN.
    """
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df