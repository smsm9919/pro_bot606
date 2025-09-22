import os, time
import pandas as pd
from flask import Flask, jsonify
from threading import Thread
from datetime import datetime, timezone
from dotenv import load_dotenv

import bot_core as CORE
from trading_logger import get_logger

load_dotenv()

SYMBOL     = os.getenv("SYMBOL", "DOGE/USDT:USDT")
INTERVAL   = os.getenv("INTERVAL", "15m")
LEVERAGE   = int(float(os.getenv("LEVERAGE", "10")))
RISK_ALLOC = float(os.getenv("RISK_ALLOC", "0.6"))
TRADE_MODE = os.getenv("TRADE_MODE", "live")  # 'live' or 'paper'

app = Flask(__name__)

state = {
    "open": False, "side": None, "entry": None, "tp": None, "sl": None,
    "qty": None, "pnl": 0.0, "compound_profit": 0.0, "total_trades": 0
}

def build_metrics_dict():
    price = CORE.fetch_price(SYMBOL)
    bal   = CORE.fetch_balance()
    return {
        "symbol": SYMBOL, "interval": INTERVAL, "mode": TRADE_MODE,
        "balance": bal, "price": price,
        "position": {k: state[k] for k in ["open","side","entry","tp","sl","qty","pnl"]},
        "compound_profit": state["compound_profit"], "total_trades": state["total_trades"],
        "update_time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    }

logger = get_logger(metrics_fn=build_metrics_dict)

@app.route("/")
def index():
    return "<h3>✅ BingX Futures Live Bot — Running</h3>"

@app.route("/metrics")
def metrics():
    return jsonify(build_metrics_dict())

def indicators(df: pd.DataFrame):
    # RSI/ADX/ATR/EMA20/50/200 (تبسيط سريع بدون ta-lib)
    returns = df["close"].pct_change()
    rsi = (returns.clip(lower=0).rolling(14).mean() / (returns.abs().rolling(14).mean()+1e-9) * 100).fillna(0).iloc[-1]
    adx = returns.abs().rolling(14).mean().fillna(0).iloc[-1] * 100
    atr = (df["high"]-df["low"]).rolling(14).mean().fillna(0).iloc[-1]
    ema20  = df["close"].ewm(span=20).mean().iloc[-1]
    ema50  = df["close"].ewm(span=50).mean().iloc[-1]
    ema200 = df["close"].ewm(span=200).mean().iloc[-1]
    return float(rsi), float(adx), float(atr), float(ema20), float(ema50), float(ema200)

def position_size(balance, price):
    # قيمة العقد = balance * risk_alloc * leverage / price  => amount in base currency
    if balance is None or price is None or price <= 0: return 0
    amount = (balance * RISK_ALLOC * LEVERAGE) / price
    return CORE.round_amount(SYMBOL, amount)

def loop():
    while True:
        try:
            ohlcv = CORE.fetch_ohlcv(SYMBOL, INTERVAL, limit=300)
            df = pd.DataFrame(ohlcv, columns=["time","open","high","low","close","volume"])
            price = CORE.fetch_price(SYMBOL)
            bal   = CORE.fetch_balance()
            rsi, adx, atr, ema20, ema50, ema200 = indicators(df)

            # طباعة لقطة
            logger.snapshot(balance=bal, price=price, rsi=rsi, adx=adx, atr=atr,
                            ema20=ema20, ema50=ema50, ema200=ema200,
                            position=state if state["open"] else None,
                            compound_profit=state["compound_profit"], total_trades=state["total_trades"],
                            update_time=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"))

            # إشارة الاستراتيجية (مثال بسيط):
            reason = None; side = None
            if ema20 > ema50 and rsi > 52 and adx > 5:
                side = "buy"   # long
                reason = f"EMA20>EMA50 & RSI>52 & ADX>5"
            elif ema20 < ema50 and rsi < 48 and adx > 5:
                side = "sell"  # short
                reason = f"EMA20<EMA50 & RSI<48 & ADX>5"

            if side and not state["open"]:
                logger.signal(side=("long" if side=='buy' else 'short'), reason=reason, price=price, rsi=rsi, adx=adx, atr=atr,
                              ema20=ema20, ema50=ema50, ema200=ema200)

                qty = position_size(bal, price)
                if qty and qty > 0:
                    # حسبة SL/TP ديناميكية على ATR
                    sl = price - 1.2*atr if side=="buy" else price + 1.2*atr
                    tp = price + 1.8*atr if side=="buy" else price - 1.8*atr

                    if TRADE_MODE == "live":
                        try:
                            order = CORE.place_market_order(SYMBOL, side=side, amount=qty, leverage=LEVERAGE, sl=sl, tp=tp)
                        except Exception as e:
                            logger.signal(side=("long" if side=='buy' else 'short'), reason=f"order_error: {e}", price=price)
                            order = None
                    else:
                        order = {"id":"paper", "side": side, "amount": qty}

                    state.update({"open": True, "side": "long" if side=="buy" else "short", "entry": price,
                                  "tp": tp, "sl": sl, "qty": qty, "pnl": 0.0})
                    state["total_trades"] += 1
                    logger.entry(side=state["side"], entry=price, tp=tp, sl=sl, qty=qty)

            # متابعة وإغلاق عند TP/SL
            if state["open"] and price is not None:
                if state["side"] == "long":
                    state["pnl"] = price - state["entry"]
                    hit_tp = price >= state["tp"]
                    hit_sl = price <= state["sl"]
                else:
                    state["pnl"] = state["entry"] - price
                    hit_tp = price <= state["tp"]
                    hit_sl = price >= state["sl"]

                if hit_tp or hit_sl:
                    reason = "tp" if hit_tp else "sl"
                    state["compound_profit"] += state["pnl"]
                    logger.exit(reason=reason, pnl=state["pnl"], price=price, tp=state["tp"], sl=state["sl"])
                    # في اللايف BingX، ال TP/SL أوامر reduceOnly بالفعل؛ هنا بنقفل الحالة الداخلية:
                    state.update({"open": False, "side": None, "entry": None, "tp": None, "sl": None, "qty": None, "pnl": 0.0})

        except Exception as e:
            logger.signal(side="n/a", reason=f"loop_error: {e}", price=None)
        time.sleep(10)

from threading import Thread
Thread(target=loop, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
