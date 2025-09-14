### Crypto Price Prediction API — Local Run Guide

#### Prerequisites
- Python 3.11 (required for TensorFlow 2.20)
- Alpha Vantage API key

#### 1) Create and activate virtual environment
```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate
```

#### 2) Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 3) Set Alpha Vantage API key
```bash
export ALPHA_VANTAGE_API_KEY=YOUR_ALPHA_VANTAGE_KEY
```

#### 4) Start the server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

#### 5) Test endpoints
- Prediction (example for BTC):
```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol":"BTC-USD"}'
```

- History (last 90 days):
```bash
curl -s "http://127.0.0.1:8000/history?symbol=BTC-USD"
```

#### Notes
- Models are loaded from `backend/models`. If any model/scaler file is missing for a symbol, predictions for that symbol will be unavailable.
- The API fetches market data from Alpha Vantage and applies minimal in-memory caching to mitigate rate limits.
