import os, time
import pandas as pd
import ccxt
from flask import Flask
from threading import Thread
from datetime import datetime
from termcolor import colored

# ================== الإعدادات ==================
SYMBOL = "DOGE/USDT:USDT"
INTERVAL = "15m"
LEVERAGE = 10
RISK_ALLOC = 0.60

# API Keys
API_KEY = os.getenv("BINGX_API_KEY")
API_SECRET = os.getenv("BINGX_API_SECRET")

# ================== تهيئة ==================
exchange = ccxt.bingx({
    "apiKey": API_KEY,
    "secret": API_SECRET,
    "enableRateLimit": True,
    "options": {"defaultType": "swap"}
})

app = Flask(__name__)

# ================== دوال مساعدة ==================
def color_val(value, label, icon="📊", color="white"):
    return colored(f"{icon} {label:<10}: {value}", color)

def fetch_balance():
    try:
        balance = exchange.fetch_balance()
        return balance["total"]["USDT"]
    except Exception as e:
        return f"Error: {e}"

def fetch_ohlcv():
    try:
        bars = exchange.fetch_ohlcv(SYMBOL, timeframe=INTERVAL, limit=200)
        df = pd.DataFrame(bars, columns=["time", "open", "high", "low", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        return df
    except Exception as e:
        print(colored(f"❌ Error fetching OHLCV: {e}", "red"))
        return None

def calculate_indicators(df):
    try:
        df["ema20"] = df["close"].ewm(span=20).mean()
        df["ema50"] = df["close"].ewm(span=50).mean()
        df["ema200"] = df["close"].ewm(span=200).mean()
        df["rsi"] = df["close"].pct_change().rolling(14).mean() * 100
        df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
        df["adx"] = abs(df["ema20"] - df["ema50"]).rolling(14).mean()
        return df.iloc[-1]  # آخر صف
    except Exception as e:
        print(colored(f"❌ Error calculating indicators: {e}", "red"))
        return None

def log_snapshot():
    df = fetch_ohlcv()
    if df is None:
        return
    ind = calculate_indicators(df)
    if ind is None:
        return

    balance = fetch_balance()
    print(colored("===== SNAPSHOT =====", "cyan"))
    print(color_val(balance, "Balance (USDT)", "💰", "yellow"))
    print(color_val(round(float(ind['close']), 6), "Price", "💲", "green"))
    print(color_val(round(float(ind['ema20']), 6), "EMA20", "📈", "blue"))
    print(color_val(round(float(ind['ema50']), 6), "EMA50", "📉", "magenta"))
    print(color_val(round(float(ind['ema200']), 6), "EMA200", "📊", "cyan"))
    print(color_val(round(float(ind['rsi']), 2), "RSI", "📊", "green"))
    print(color_val(round(float(ind['adx']), 2), "ADX", "📊", "yellow"))
    print(color_val(round(float(ind['atr']), 6), "ATR", "📊", "red"))
    print(colored("=====================", "cyan"))

# ================== التشغيل ==================
def bot_loop():
    while True:
        log_snapshot()
        time.sleep(60)  # كل دقيقة

@app.route("/")
def home():
    return "✅ Bot Running — Dashboard Active"

if __name__ == "__main__":
    t = Thread(target=bot_loop)
    t.daemon = True
    t.start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
