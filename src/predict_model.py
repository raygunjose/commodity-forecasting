import joblib
import pandas as pd

# Load model and scaler
model = joblib.load('models/aluminum_model.pkl')
scaler = joblib.load('models/scaler.pkl')

def predict(features_dict):
    """
    features_dict: dictionary of feature values
    Example:
    {
        "Zinc": 35000,
        "Lead": 5000,
        "Nickel": 800,
        "Copper": 200000,
        "Crude Oil": 120000,
        "Natural Gas": 630000
    }
    """
    df = pd.DataFrame([features_dict])
    X_scaled = scaler.transform(df)
    prediction = model.predict(X_scaled)
    return float(prediction[0])