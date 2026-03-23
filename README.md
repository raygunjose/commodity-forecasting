# commodity-forecasting
Commodity Price Prediction API

## Setup Instructions

1. Clone the repo:
```bash
git clone <repo_url>
cd commodity-forecasting

pip install -r requirements.txt

python src/train_model.py

uvicorn app.main:app --reload

CHROME: http://127.0.0.1:8000/docs
POST: http://127.0.0.1:8000/predict