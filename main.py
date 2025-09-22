# -*- coding: utf-8 -*-
"""
BingX Futures Bot — Balanced Live Trading
يدخل صفقات حقيقية مع SL/TP — يطبع كل شيء ملون باللوجز
"""

import os, time
import pandas as pd
import ccxt
from flask import Flask, jsonify
from threading import Thread
from datetime import datetime

# ===== إعدادات عامة =====
SYMBOL      = os.getenv("SYMBOL", "DOGE/USDT:USDT")
INTERVAL    = os.getenv("INTERVAL", "15m")
LEVERAGE    = int(float(os.getenv("LEVERAGE", "10")))
RISK_ALLOC  = float(os.getenv("RISK_ALLOC", "0.60"))
TRADE_MODE  = os.getenv("TRADE_MODE", "live")   # live / paper

API_KEY     = os.getenv("BINGX_API_KEY", "")
API_SECRET  = os.getenv("BINGX_API_SECRET", "")

# ===== Icons & Colors =====
try:
    from termcolor import colored
except:
    def colored(t, *_a, **_k): return t

IC_OK  = "✅"; IC_BAD="❌"; IC_BAL="💰"; IC_PRC="💲"
IC_TRD = "🟢"; IC_CLS="🔴"; IC_MTR="📊"
SEP = colored("—"*70,"cyan")

def fmt(v,d=2,na="N/A"):
    try: return f"{float(v):.{d}f}" if v is not None else na
    except: return na

# ===== BingX Exchange =====
def make_exchange():
    return ccxt.bingx({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "options": {"defaultType": "swap","defaultMarginMode":"isolated"},
    })
ex = make_exchange()

def safe_symbol(s): return s if s.endswith(":USDT") else s+":USDT"

# ===== بيانات الحساب =====
def balance_usdt():
    try:
        b=ex.fetch_balance(params={"type":"swap"})
        return b.get("total",{}).get("USDT",0)
    except: return 0

def price_now():
    try:
        t=ex.fetch_ticker(safe_symbol(SYMBOL))
        return t.get("last")
    except: return None

# ===== مؤشرات أساسية =====
def fetch_ohlcv(limit=200):
    rows = ex.fetch_ohlcv(safe_symbol(SYMBOL), timeframe=INTERVAL, limit=limit, params={"type":"swap"})
    df=pd.DataFrame(rows,columns=["time","open","high","low","close","volume"])
    return df

def indicators(df):
    ema20=df["close"].ewm(span=20).mean().iloc[-1]
    ema50=df["close"].ewm(span=50).mean().iloc[-1]
    ema200=df["close"].ewm(span=200).mean().iloc[-1]
    ret=df["close"].pct_change()
    rsi=100-(100/(1+(ret.clip(lower=0).rolling(14).mean()/( -ret.clip(upper=0)).rolling(14).mean().replace(0,1))))
    rsi=float(rsi.fillna(50).iloc[-1])
    return dict(ema20=float(ema20),ema50=float(ema50),ema200=float(ema200),rsi=rsi)

# ===== تنفيذ صفقات =====
compound_profit=0.0
current_pos=None

def trade_signal():
    """Balanced: Cross EMA20/EMA50 + RSI >60 شراء / <40 بيع"""
    df=fetch_ohlcv()
    ind=indicators(df)
    sig=None
    if ind["ema20"]>ind["ema50"] and ind["rsi"]>60: sig="long"
    elif ind["ema20"]<ind["ema50"] and ind["rsi"]<40: sig="short"
    return sig,ind

def open_trade(side,amount,sl,tp):
    global current_pos
    if TRADE_MODE!="live":
        print(colored(f"{IC_TRD} [PAPER] Open {side} {amount} SL:{sl} TP:{tp}","yellow"))
        current_pos=dict(side=side,amount=amount,entry=price_now(),sl=sl,tp=tp)
        return
    try:
        ex.set_leverage(LEVERAGE,safe_symbol(SYMBOL))
        order=ex.create_order(safe_symbol(SYMBOL),"market",side,amount)
        entry=order["average"]
        current_pos=dict(side=side,amount=amount,entry=entry,sl=sl,tp=tp)
        print(colored(f"{IC_TRD} Open {side} {amount}@{fmt(entry,6)} SL:{sl} TP:{tp}","green"))
    except Exception as e:
        print(colored(f"{IC_BAD} open error {e}","red"))

def close_trade():
    global current_pos,compound_profit
    if not current_pos: return
    if TRADE_MODE!="live":
        pnl=(price_now()-current_pos["entry"])*current_pos["amount"]*(1 if current_pos["side"]=="long" else -1)
        compound_profit+=pnl
        print(colored(f"{IC_CLS} [PAPER] Close {current_pos['side']} PnL:{fmt(pnl,4)} Total:{fmt(compound_profit,4)}","magenta"))
        current_pos=None; return
    try:
        side="sell" if current_pos["side"]=="long" else "buy"
        ex.create_order(safe_symbol(SYMBOL),"market",side,current_pos["amount"])
        pnl=(price_now()-current_pos["entry"])*current_pos["amount"]*(1 if current_pos["side"]=="long" else -1)
        compound_profit+=pnl
        print(colored(f"{IC_CLS} Close {current_pos['side']} PnL:{fmt(pnl,4)} Total:{fmt(compound_profit,4)}","magenta"))
        current_pos=None
    except Exception as e:
        print(colored(f"{IC_BAD} close error {e}","red"))

# ===== Loop =====
def loop():
    global current_pos
    while True:
        try:
            bal=balance_usdt()
            px=price_now()
            sig,ind=trade_signal()
            print(SEP)
            print(colored(f"{IC_MTR} {datetime.utcnow()} Bal:{fmt(bal,2)} Px:{fmt(px,6)} Profit:{fmt(compound_profit,4)}","cyan"))
            print(colored(f"EMA20:{fmt(ind['ema20'],6)} EMA50:{fmt(ind['ema50'],6)} EMA200:{fmt(ind['ema200'],6)} RSI:{fmt(ind['rsi'],2)}","yellow"))
            if not current_pos and sig:
                qty=(bal*RISK_ALLOC*LEVERAGE)/px
                sl=px*(0.985 if sig=="long" else 1.015)
                tp=px*(1.015 if sig=="long" else 0.985)
                open_trade(sig,qty,sl,tp)
            elif current_pos:
                if (current_pos["side"]=="long" and (px<=current_pos["sl"] or px>=current_pos["tp"])) or \
                   (current_pos["side"]=="short" and (px>=current_pos["sl"] or px<=current_pos["tp"])):
                    close_trade()
        except Exception as e:
            print(colored(f"{IC_BAD} loop error {e}","red"))
        time.sleep(60)

# ===== Flask =====
app=Flask(__name__)
@app.route("/")
def home(): return "✅ Bot Running — Live Trading"
@app.route("/metrics")
def metrics():
    return jsonify(dict(balance=balance_usdt(),price=price_now(),profit=compound_profit,pos=current_pos))

Thread(target=loop,daemon=True).start()
if __name__=="__main__":
    app.run(host="0.0.0.0",port=int(os.getenv("PORT",5000)))
