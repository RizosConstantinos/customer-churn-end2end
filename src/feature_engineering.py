import pandas as pd

def add_tenure_monthly_ratio(df):
    """
    Create feature: tenure divided by monthly charges
    """
    df = df.copy()
    df['tenure_monthly_ratio'] = df['tenure'] / (df['MonthlyCharges'] + 1e-6) # Avoid division by zero
    return df

DEFAULT_SERVICE_COLUMNS = [
    'PhoneService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies'
]

def add_services_count(df, service_columns=DEFAULT_SERVICE_COLUMNS):
    """
    Count number of active services per customer.
    Treats 'Yes' as 1, everything else as 0.
    """
    df = df.copy()
    df['services_count'] = (
        df[service_columns]
        .apply(lambda col: col.eq('Yes'))
        .sum(axis=1)
    )
    return df

def add_interaction_contract_payment(df):
    """
    Interaction feature between Contract type and Payment Method
    """
    df = df.copy()
    df['contract_payment_interaction'] = (
        df['Contract'].fillna('Unknown') + "_" +
        df['PaymentMethod'].fillna('Unknown')
    )
    return df

