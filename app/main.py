from fastapi import FastAPI
from pydantic import BaseModel
from src.predict_model import predict

app = FastAPI(title="Commodity Price Prediction API")

class CommodityFeatures(BaseModel):
    Zinc: float
    Lead: float
    Nickel: float
    Copper: float
    Crude_Oil: float
    Natural_Gas: float

@app.get("/")
def home():
    return {"message": "Welcome to Commodity Price Prediction API"}

@app.post("/predict")
def predict_aluminum(data: CommodityFeatures):
    features = {
        "Zinc": data.Zinc,
        "Lead": data.Lead,
        "Nickel": data.Nickel,
        "Copper": data.Copper,
        "Crude Oil": data.Crude_Oil,
        "Natural Gas": data.Natural_Gas
    }
    prediction = predict(features)
    return {"Aluminum_Price_Prediction": prediction}