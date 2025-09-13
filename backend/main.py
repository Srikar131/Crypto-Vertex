import tensorflow as tf
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.optimizers import Adam
from pydantic import BaseModel
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from sklearn.preprocessing import MinMaxScaler
import os

# --- Configuration ---
SUPPORTED_SYMBOLS = ["BTC-USD", "ETH-USD", "ADA-USD", "DOGE-USD", "SOL-USD"]
# --- DEBUG PRINT ---
print(f"--- SCRIPT READ BY SERVER --- SUPPORTED_SYMBOLS = {SUPPORTED_SYMBOLS}")

class CryptoData(BaseModel):
    symbol: str

app = FastAPI(title="Crypto Price Prediction API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

loaded_assets = {}

# (Helper functions remain the same)
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

@app.on_event("startup")
async def load_all_assets():
    print("--- STARTUP EVENT TRIGGERED ---")
    optimizer = Adam(learning_rate=0.001)
    loss = 'mse'

    for symbol in SUPPORTED_SYMBOLS:
        try:
            prefix = symbol.split('-')[0].lower()
            print(f"--> Attempting to load assets for {symbol}...")
            
            lstm_path = f"models/{prefix}_lstm_model.h5"
            if not os.path.exists(lstm_path):
                print(f"!!! FILE NOT FOUND: {lstm_path}")
                continue # Skip this symbol if a file is missing
            
            bilstm_path = f"models/{prefix}_bilstm_model.h5"
            xgb_path = f"models/{prefix}_xgboost_model.json"
            scaler_path = f"models/{prefix}_scaler.gz"

            lstm_m = tf.keras.models.load_model(lstm_path, compile=False)
            bilstm_m = tf.keras.models.load_model(bilstm_path, compile=False)
            lstm_m.compile(optimizer=optimizer, loss=loss, metrics=['mae'])
            bilstm_m.compile(optimizer=optimizer, loss=loss, metrics=['mae'])
            
            xgb_m = xgb.XGBRegressor(); xgb_m.load_model(xgb_path)
            scaler_obj = joblib.load(scaler_path)
            
            loaded_assets[symbol] = { "lstm": lstm_m, "bilstm": bilstm_m, "xgboost": xgb_m, "scaler": scaler_obj }
            print(f"--- SUCCESS: Loaded assets for {symbol}")
        
        except Exception as e:
            print(f"!!! FAILED to load assets for {symbol}. Error: {e}")

    print(f"--- STARTUP COMPLETE --- Successfully loaded assets for: {list(loaded_assets.keys())}")

# (Other functions remain the same)
def preprocess_live_data(symbol: str, scaler_obj: MinMaxScaler):
    df = yf.Ticker(symbol).history(period="100d")
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df['MACD'] = calculate_macd(df['Close'])
    df['Price_Change'] = df['Close'].pct_change()
    df['Volume_Change'] = df['Volume'].pct_change()
    df.ffill(inplace=True); df.bfill(inplace=True)
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'Price_Change', 'Volume_Change']
    df = df[features]
    last_60_days = df.tail(60)
    scaled_data = scaler_obj.transform(last_60_days)
    X_live = np.reshape(scaled_data, (1, 60, 11))
    X_live_xgb = np.reshape(scaled_data, (1, 60 * 11))
    return X_live, X_live_xgb

@app.get("/")
async def root(): return {"message": "Cryptocurrency Price Prediction API"}
@app.post("/predict")
async def predict_price(crypto_data: CryptoData):
    symbol = crypto_data.symbol
    if symbol not in SUPPORTED_SYMBOLS:
        raise HTTPException(status_code=404, detail=f"Predictions for symbol '{symbol}' are not supported.")
    
    try:
        assets = loaded_assets[symbol]
        X_live_lstm, X_live_xgb = preprocess_live_data(symbol, assets["scaler"])
        
        lstm_pred_scaled = assets["lstm"].predict(X_live_lstm)
        bilstm_pred_scaled = assets["bilstm"].predict(X_live_lstm)
        
        # vvv THIS IS THE CORRECTED LINE vvv
        xgb_pred_scaled = assets["xgboost"].predict(X_live_xgb).reshape(-1, 1)
        # ^^^ THIS IS THE CORRECTED LINE ^^^

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
            }, "message": f"Live prediction for {symbol} successful."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
@app.get("/history")
async def get_history(symbol: str):
    if symbol not in loaded_assets:
        raise HTTPException(status_code=404, detail=f"History for symbol '{symbol}' is not supported or failed to load.")
    try:
        df = yf.Ticker(symbol).history(period="90d")
        df.reset_index(inplace=True); df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        return df[['Date', 'Close']].to_dict('records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")