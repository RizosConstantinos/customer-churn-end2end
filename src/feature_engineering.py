import pandas as pd

def add_tenure_monthly_ratio(df):
    """
    Create feature: tenure divided by monthly charges
    """
    df = df.copy()
    df['tenure_monthly_ratio'] = df['tenure'] / (df['MonthlyCharges'] + 1e-6) # +1 to avoid divide with zero
    return df

def add_services_count(df, service_columns):
    """
    Count number of services a customer has subscribed to
    Converts Yes/No to 1/0 first.
    """
    df = df.copy()
    df_numeric = df[service_columns].apply(lambda x: x.map({'Yes': 1, 'No': 0}))
    df['services_count'] = df_numeric.sum(axis=1)
    return df

def add_interaction_contract_payment(df):
    """
    Create interaction feature between Contract type and Payment method
    """
    df = df.copy()
    df['contract_payment_interaction'] = df['Contract'] + "_" + df['PaymentMethod']
    return df
