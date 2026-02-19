import pandas as pd
       
def convert_to_numeric(df, columns):
    """
    Convert specified columns to numeric.
    Non-convertible values become NaN.
    """
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def handle_missing_values(df, subset=None):
    """
    Handle missing values in the dataset.
    Currently drops rows with missing values.
    """
    if subset is not None:
        return df.dropna(subset=subset)
    else:
        return df.dropna()

