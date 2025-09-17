from dotenv import load_dotenv
load_dotenv()

import tensorflow as tf
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.optimizers import Adam
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
import os
from contextlib import asynccontextmanager
import requests
import time

SUPPORTED_SYMBOLS = ["BTC-USD", "ETH-USD", "ADA-USD", "DOGE-USD", "SOL-USD"]
COINGECKO_MAP = {
    "BTC-USD": "bitcoin",
    "ETH-USD": "ethereum",
    "ADA-USD": "cardano",
    "DOGE-USD": "dogecoin",
    "SOL-USD": "solana"
}

class ApiRequest(BaseModel):
    symbol: str

loaded_assets = {}

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

def get_coingecko_ohlc(symbol: str, days=90, max_retries=3, retry_delay=2):
    coin_id = COINGECKO_MAP[symbol]
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days}
    last_exception = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, params=params, timeout=10)
            if not r.ok:
                raise ValueError(f"CoinGecko data fetch failed! (status {r.status_code})")
            prices = r.json().get("prices")
            if not prices:
                raise ValueError("No price data received from CoinGecko.")
            df = pd.DataFrame(prices, columns=["Timestamp", "Close"])
            df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms").dt.strftime("%Y-%m-%d")
            df = df.groupby("Date").last().reset_index()
            df["Open"] = df["Close"]
            df["High"] = df["Close"]
            df["Low"] = df["Close"]
            df["Volume"] = 0
            return df
        except Exception as e:
            last_exception = e
            print(f"[CoinGecko Fetch Attempt {attempt}] Error: {e}")
            time.sleep(retry_delay)
    raise ValueError(f"CoinGecko data fetch failed (after retries): {last_exception}")

def preprocess_live_data(symbol: str, scaler_obj: MinMaxScaler):
    df = get_coingecko_ohlc(symbol, days=90).copy()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'] = calculate_macd(df['Close'])
    df['Price_Change'] = df['Close'].pct_change()
    df['Volume_Change'] = df['Volume'].pct_change() if "Volume" in df else 0
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    features = ['Open', 'High', 'Low', 'Close', 'Volume',
                'SMA_20', 'SMA_50', 'RSI', 'MACD', 'Price_Change', 'Volume_Change']
    df = df[features]
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    last_60_days = df.tail(60)
    if len(last_60_days) < 60:
        raise ValueError(f"Not enough historical data for {symbol}. Got {len(last_60_days)} days.")
    scaled_data = scaler_obj.transform(last_60_days)
    X_live = np.reshape(scaled_data, (1, 60, 11))
    X_live_xgb = np.reshape(scaled_data, (1, 60 * 11))
    return X_live, X_live_xgb

def get_history_data(symbol: str):
    df = get_coingecko_ohlc(symbol, days=90)
    df_history = df[-90:].copy().reset_index(drop=True)
    records = []
    for _, row in df_history.iterrows():
        records.append({"Date": row["Date"], "Close": float(row["Close"])})
    return records

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading all ML models and scalers...")
    optimizer = Adam(learning_rate=0.001)
    loss = 'mse'
    for symbol in SUPPORTED_SYMBOLS:
        try:
            prefix = symbol.split('-')[0].lower()
            assets = {}
            assets["lstm"] = tf.keras.models.load_model(f"models/{prefix}_lstm_model.h5", compile=False)
            assets["bilstm"] = tf.keras.models.load_model(f"models/{prefix}_bilstm_model.h5", compile=False)
            assets["lstm"].compile(optimizer=optimizer, loss=loss, metrics=['mae'])
            assets["bilstm"].compile(optimizer=optimizer, loss=loss, metrics=['mae'])
            assets["xgboost"] = xgb.XGBRegressor()
            assets["xgboost"].load_model(f"models/{prefix}_xgboost_model.json")
            assets["scaler"] = joblib.load(f"models/{prefix}_scaler.gz")
            loaded_assets[symbol] = assets
            print(f"Successfully loaded assets for {symbol}")
        except Exception as e:
            print(f"!!! FAILED to load assets for {symbol}. Error: {e}")
    print(f"Startup complete. Loaded assets for: {list(loaded_assets.keys())}")
    yield
    print("Cleaning up resources...")
    loaded_assets.clear()

app = FastAPI(title="Crypto Price Prediction API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/")
async def root():
    return {"message": "Cryptocurrency Price Prediction API"}

@app.post("/predict")
async def predict_price(request: ApiRequest):
    symbol = request.symbol
    if symbol not in loaded_assets:
        raise HTTPException(status_code=404, detail=f"Predictions for symbol '{symbol}' are not supported or failed to load.")
    try:
        assets = loaded_assets[symbol]
        X_live_lstm, X_live_xgb = preprocess_live_data(symbol, assets["scaler"])
        lstm_pred = inverse_scale_predictions(assets["lstm"].predict(X_live_lstm), assets["scaler"])
        bilstm_pred = inverse_scale_predictions(assets["bilstm"].predict(X_live_lstm), assets["scaler"])
        xgb_pred = inverse_scale_predictions(assets["xgboost"].predict(X_live_xgb).reshape(-1, 1), assets["scaler"])
        ensemble_pred = (lstm_pred[0] * 0.3 + bilstm_pred[0] * 0.5 + xgb_pred[0] * 0.2)
        return {
            "predictions": {
                "lstm": float(lstm_pred[0]),
                "bidirectional_lstm": float(bilstm_pred[0]),
                "xgboost": float(xgb_pred[0]),
                "ensemble": float(ensemble_pred)
            }
        }
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while generating predictions: {str(e)} (Try again in a moment if this is a CoinGecko error)")

@app.post("/history")
async def get_history(request: ApiRequest):
    symbol = request.symbol
    if symbol not in SUPPORTED_SYMBOLS:
        raise HTTPException(status_code=404, detail=f"History for {symbol} is not supported.")
    try:
        history_data = get_history_data(symbol)
        return history_data
    except Exception as e:
        print(f"History error: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while fetching history: {str(e)} (Try again in a moment if this is a CoinGecko error)")
