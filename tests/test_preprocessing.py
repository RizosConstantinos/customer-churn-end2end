import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import pytest

# Import directly from your preprocessing.py
from src.preprocessing import convert_to_numeric, handle_missing_values

# -----------------------------
# Test convert_to_numeric function
# -----------------------------
def test_convert_to_numeric():
    df = pd.DataFrame({
        'col1': ['1', '2', 'three'],
        'col2': ['4.0', 'five', '6.1']
    })
    
    df2 = convert_to_numeric(df, ['col1', 'col2'])
    
    # 'three' and 'five' should become NaN
    assert pd.isna(df2.loc[2, 'col1'])
    assert pd.isna(df2.loc[1, 'col2'])
    
    # Numeric values should be converted properly
    assert df2.loc[0, 'col1'] == 1
    assert df2.loc[0, 'col2'] == 4.0
    assert df2.loc[2, 'col2'] == 6.1

# -----------------------------
# Test handle_missing_values function
# -----------------------------
def test_handle_missing_values():
    df = pd.DataFrame({
        'a': [1, None, 3],
        'b': [4, 5, None]
    })
    
    # Drop rows with missing in column 'a'
    df2 = handle_missing_values(df, subset=['a'])
    assert df2.shape[0] == 2
    assert df2['a'].isna().sum() == 0

    # Drop rows with missing in all columns
    df3 = handle_missing_values(df)
    assert df3.shape[0] == 1
    assert df3.isna().sum().sum() == 0
