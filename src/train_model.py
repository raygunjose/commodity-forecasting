import joblib
import pandas as pd
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from utils import preprocess_features  # we'll handle load_data inline now

# -------------------------
# Load and preprocess data
# -------------------------

# Skip first two rows; use third row as header
df = pd.read_excel("data/commodities.xlsx", header=2)  # <-- header=2 because 0-indexed
df.columns = df.columns.str.strip()  # remove leading/trailing spaces

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

print("Columns:", df.columns.tolist())
print(df.head())

# -------------------------
# Feature preprocessing
# -------------------------

# Example preprocess_features function
# Make sure this exists in utils.py or inline
X, y, scaler = preprocess_features(df, target_col='Aluminum')

# -------------------------
# Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# -------------------------
# Train model
# -------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------
# Validate
# -------------------------
y_pred = model.predict(X_test)
rmse = sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# -------------------------
# Save model and scaler
# -------------------------
joblib.dump(model, 'models/aluminum_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
print("Model and scaler saved!")