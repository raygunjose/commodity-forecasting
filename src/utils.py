import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """
    Load Excel dataset and sort by Date
    """
    df = pd.read_excel(file_path, parse_dates=['Date'])
    df = df.sort_values('Date')
    return df

def preprocess_features(df, target_col='Aluminum'):
    """
    Preprocess dataset for ML
    """
    # Drop non-numeric or irrelevant columns
    X = df.drop(columns=['Instrument Type', 'Date', target_col])
    y = df[target_col]

    # Scaling features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler