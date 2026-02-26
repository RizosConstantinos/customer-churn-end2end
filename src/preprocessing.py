import pandas as pd

# Default numeric columns
NUMERIC_COLUMNS = ["TotalCharges"]

def convert_to_numeric(df, columns=None):
    """
    Convert specified columns to numeric.
    Non-convertible values become NaN.
    """
    df = df.copy()
    if columns is None:
        columns = NUMERIC_COLUMNS
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def handle_missing_values(df, subset=None, strategy="drop"):
    """
    Handle missing values in the dataset.
    strategy: 'drop' to drop rows, 'fill' to fill with 0
    """
    df = df.copy()
    if strategy == "drop":
        if subset:
            return df.dropna(subset=subset)
        return df.dropna()
    elif strategy == "fill":
        return df.fillna(0)
    else:
        raise ValueError("strategy must be 'drop' or 'fill'")

def preprocessing_pipeline(df):
    """
    Apply full preprocessing: numeric conversion + missing values handling
    """
    df = convert_to_numeric(df)
    df = handle_missing_values(df)
    return df