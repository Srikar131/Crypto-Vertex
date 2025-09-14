import tensorflow as tf
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.optimizers import Adam
from pydantic import BaseModel
import numpy as np
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
import joblib
from sklearn.preprocessing import MinMaxScaler
import os
from contextlib import asynccontextmanager

# --- Configuration ---
SUPPORTED_SYMBOLS = ["BTC-USD", "ETH-USD", "ADA-USD", "DOGE-USD", "SOL-USD"]

# Alpha Vantage configuration
ALPHA_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY") or os.getenv("ALPHAVANTAGE_API_KEY") or os.getenv("ALPHA_API_KEY")
ALPHA_BASE_URL = "https://www.alphavantage.co/query"

# Simple in-memory cache to mitigate rate limits and reduce latency
alpha_cache = {}

def _symbol_to_alpha_params(symbol: str):
    try:
        base, quote = symbol.split("-")
    except ValueError:
        base, quote = symbol, "USD"
    return base.upper(), quote.upper()

def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    desired = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    lower_to_original = {str(c).lower(): c for c in df.columns}
    rename_map = {}
    # First pass: exact lowercase match
    for k_lower, proper in desired.items():
        if proper in df.columns:
            continue
        if k_lower in lower_to_original:
            rename_map[lower_to_original[k_lower]] = proper
    # Second pass: substring fallback (e.g., "1a. open (usd)")
    if len(rename_map) < 5:
        for k_lower, proper in desired.items():
            if proper in set(list(rename_map.values()) + list(df.columns)):
                continue
            candidates = [c for c in df.columns if k_lower in str(c).lower()]
            if candidates:
                rename_map[candidates[0]] = proper
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

def fetch_alpha_vantage_daily(symbol: str, min_days: int = 180) -> pd.DataFrame:
    if not ALPHA_API_KEY:
        raise ValueError("Missing Alpha Vantage API key. Set ALPHA_VANTAGE_API_KEY in environment.")

    # Cache key and TTL (seconds)
    cache_key = f"daily::{symbol}"
    now_ts = time.time()
    cached = alpha_cache.get(cache_key)
    if cached and (now_ts - cached["ts"]) < 120:  # 2 minutes TTL
        return cached["df"].copy()

    base, quote = _symbol_to_alpha_params(symbol)
    params = {
        "function": "DIGITAL_CURRENCY_DAILY",
        "symbol": base,
        "market": quote,
        "apikey": ALPHA_API_KEY,
    }
    try:
        resp = requests.get(ALPHA_BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        raise ValueError(f"Alpha Vantage request failed for {symbol}: {str(e)}")

    time_series = data.get("Time Series (Digital Currency Daily)")
    if not time_series:
        # Error message from API often under "Note" or "Error Message"
        err = data.get("Note") or data.get("Error Message") or str(data)
        raise ValueError(f"Alpha Vantage response invalid for {symbol}: {err}")

    # Build DataFrame
    records = []
    for date_str, values in time_series.items():
        try:
            records.append({
                "Date": datetime.strptime(date_str, "%Y-%m-%d"),
                "Open": float(values.get("1a. open (USD)") or values.get("1b. open (USD)")),
                "High": float(values.get("2a. high (USD)") or values.get("2b. high (USD)")),
                "Low": float(values.get("3a. low (USD)") or values.get("3b. low (USD)")),
                "Close": float(values.get("4a. close (USD)") or values.get("4b. close (USD)")),
                "Volume": float(values.get("5. volume", 0.0)),
            })
        except Exception:
            # Skip rows with malformed data
            continue

    if not records:
        raise ValueError(f"No valid records parsed from Alpha Vantage for {symbol}.")

    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise ValueError(f"No valid rows parsed from Alpha Vantage for {symbol}.")
    # Normalize potential column naming/case issues
    df = normalize_ohlcv_columns(df)

    # Ensure types
    for col in ["Open","High","Low","Close","Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.sort_values("Date", inplace=True)
    df.set_index("Date", inplace=True)

    # Ensure minimum number of days
    if len(df) < min_days:
        raise ValueError(f"Insufficient Alpha Vantage data for {symbol}. Got {len(df)} rows.")

    # Cache it
    alpha_cache[cache_key] = {"df": df.copy(), "ts": now_ts}
    return df

# --- Pydantic model for request body ---
class CryptoData(BaseModel):
    symbol: str

# --- Global variable to hold all loaded assets ---
loaded_assets = {}

# --- Helper Functions ---
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, slow=26, fast=12, signal=9):
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line

def inverse_scale_predictions(predictions, scaler_obj, num_features=11, close_price_index=3):
    dummy_array = np.zeros((len(predictions), num_features))
    dummy_array[:, close_price_index] = predictions.flatten()
    unscaled_array = scaler_obj.inverse_transform(dummy_array)
    return unscaled_array[:, close_price_index]

def preprocess_live_data(symbol: str, scaler_obj: MinMaxScaler):
    # Fetch more days to avoid insufficient rows after indicators
    df = fetch_alpha_vantage_daily(symbol, min_days=180)

    # Validate required columns exist, try to recover case differences
    required = {"Open","High","Low","Close","Volume"}
    df = normalize_ohlcv_columns(df)
    if not required.issubset(set(df.columns)):
        missing = sorted(list(required - set(df.columns)))
        raise ValueError(f"Required columns missing after fetch: {missing}; got {list(df.columns)}")

    # Feature engineering
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'] = calculate_macd(df['Close'])
    df['Price_Change'] = df['Close'].pct_change()
    df['Volume_Change'] = df['Volume'].pct_change()

    df.ffill(inplace=True)
    df.bfill(inplace=True)

    features = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'SMA_20', 'SMA_50', 'RSI', 'MACD',
        'Price_Change', 'Volume_Change'
    ]
    # Select features with validation for clearer error reporting
    missing_features = [c for c in features if c not in df.columns]
    if missing_features:
        raise ValueError(
            f"Feature columns missing: {missing_features}; available columns: {list(df.columns)}"
        )
    df = df[features]

    # Ensure at least 60 rows remain
    if len(df) < 60:
        raise ValueError(f"Not enough rows after feature engineering for {symbol}. Got {len(df)} rows.")

    last_60_days = df.tail(60)

    # Scale safely
    try:
        scaled_data = scaler_obj.transform(last_60_days)
    except Exception as e:
        raise ValueError(f"Scaling failed for {symbol}. Error: {str(e)}")

    X_live = np.reshape(scaled_data, (1, 60, 11))
    X_live_xgb = np.reshape(scaled_data, (1, 60 * 11))

    return X_live, X_live_xgb

# --- Lifespan manager for loading models on startup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading all ML models and scalers...")
    optimizer = Adam(learning_rate=0.001)
    loss = 'mse'
    for symbol in SUPPORTED_SYMBOLS:
        try:
            prefix = symbol.split('-')[0].lower()
            lstm_path = f"models/{prefix}_lstm_model.h5"
            bilstm_path = f"models/{prefix}_bilstm_model.h5"
            xgb_path = f"models/{prefix}_xgboost_model.json"
            scaler_path = f"models/{prefix}_scaler.gz"

            lstm_m = tf.keras.models.load_model(lstm_path, compile=False)
            bilstm_m = tf.keras.models.load_model(bilstm_path, compile=False)
            lstm_m.compile(optimizer=optimizer, loss=loss, metrics=['mae'])
            bilstm_m.compile(optimizer=optimizer, loss=loss, metrics=['mae'])
            
            xgb_m = xgb.XGBRegressor(); xgb_m.load_model(xgb_path)
            scaler_obj = joblib.load(scaler_path)
            
            loaded_assets[symbol] = {"lstm": lstm_m, "bilstm": bilstm_m, "xgboost": xgb_m, "scaler": scaler_obj}
            print(f"✅ Successfully loaded assets for {symbol}")
        except Exception as e:
            print(f"❌ FAILED to load assets for {symbol}. Error: {e}")
    
    print(f"Startup complete. Loaded assets for: {list(loaded_assets.keys())}")
    yield
    print("Cleaning up resources...")
    loaded_assets.clear()

# --- FastAPI App Initialization ---
app = FastAPI(title="Crypto Price Prediction API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "Cryptocurrency Price Prediction API"}

@app.post("/predict")
async def predict_price(crypto_data: CryptoData):
    symbol = crypto_data.symbol
    if symbol not in loaded_assets:
        raise HTTPException(status_code=404, detail=f"Predictions for symbol '{symbol}' are not supported or failed to load.")
    try:
        assets = loaded_assets[symbol]
        X_live_lstm, X_live_xgb = preprocess_live_data(symbol, assets["scaler"])
        lstm_pred_scaled = assets["lstm"].predict(X_live_lstm)
        bilstm_pred_scaled = assets["bilstm"].predict(X_live_lstm)
        xgb_pred_scaled = assets["xgboost"].predict(X_live_xgb).reshape(-1, 1)

        lstm_pred = inverse_scale_predictions(lstm_pred_scaled, assets["scaler"])
        bilstm_pred = inverse_scale_predictions(bilstm_pred_scaled, assets["scaler"])
        xgb_pred = inverse_scale_predictions(xgb_pred_scaled, assets["scaler"])

        ensemble_pred = (lstm_pred[0] * 0.3 + bilstm_pred[0] * 0.5 + xgb_pred[0] * 0.2)

        return {
            "predictions": {
                "lstm": float(lstm_pred[0]),
                "bidirectional_lstm": float(bilstm_pred[0]),
                "xgboost": float(xgb_pred[0]),
                "ensemble": float(ensemble_pred)
            },
            "message": f"Live prediction for {symbol} successful."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/history")
async def get_history(symbol: str):
    if symbol not in loaded_assets:
        raise HTTPException(status_code=404, detail=f"History for symbol '{symbol}' is not supported or failed to load.")
    try:
        df = fetch_alpha_vantage_daily(symbol, min_days=90)
        df = df.tail(90).copy()
        df.reset_index(inplace=True)
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        return df[["Date", "Close"]].to_dict('records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
