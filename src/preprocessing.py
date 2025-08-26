import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df['SeniorCitizen'] = df['SeniorCitizen'].astype('object')
    return df

def encode_features(df_raw):
    # Target column
    y = df_raw['Churn'].map({'Yes': 1, 'No': 0})

    # Drop ID and target
    df_encoded = df_raw.drop(['customerID', 'Churn'], axis=1)

    # One-hot encoding for categorical variables
    df_encoded = pd.get_dummies(df_encoded, drop_first=True)

    # Add target back
    df_encoded['Churn'] = y

    return df_encoded
