# -*- coding: utf-8 -*-
"""
MindBay — BingX Futures Live (15m • Balanced)
- Real trading via ccxt.bingx (swap, isolated)
- Dynamic SL/TP + guards
- Colored & iconized logs
- Self-ping to keep Render alive
"""

import os, time, math, threading, requests
import pandas as pd
import ccxt
from flask import Flask, jsonify
from datetime import datetime

# ===================== Config =====================
SYMBOL      = os.getenv("SYMBOL", "DOGE/USDT:USDT")
INTERVAL    = os.getenv("INTERVAL", "15m")
LEVERAGE    = int(float(os.getenv("LEVERAGE", "10")))
RISK_ALLOC  = float(os.getenv("RISK_ALLOC", "0.60"))   # 60% من الرصيد
TRADE_MODE  = os.getenv("TRADE_MODE", "live")          # live / paper

API_KEY     = os.getenv("BINGX_API_KEY", "")
API_SECRET  = os.getenv("BINGX_API_SECRET", "")

SELF_URL    = os.getenv("RENDER_EXTERNAL_URL", "") or os.getenv("SELF_URL", "")

# ================= Icons & Colors =================
try:
    from termcolor import colored
except Exception:
    def colored(t,*a,**k): return t

IC_OK="✅"; IC_BAD="❌"; IC_BAL="💰"; IC_PRC="💲"; IC_TRD="🟢"; IC_CLS="🔴"; IC_MTR="📊"; IC_SHD="🛡️"
SEP = colored("—"*74, "cyan")

def fmt(v,d=2,na="N/A"):
    try:
        if v is None: return na
        return f"{float(v):.{d}f}"
    except Exception:
        return na

def safe_symbol(s: str) -> str:
    if s.endswith(":USDT"): return s
    if s.endswith("/USDT"): return s + ":USDT"
    return s

# ================= Exchange (ccxt) =================
def make_exchange():
    return ccxt.bingx({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {
            "defaultType": "swap",
            "defaultMarginMode": "isolated",
        }
    })

ex = make_exchange()

# ================= Account & Market =================
def balance_usdt():
    try:
        b = ex.fetch_balance(params={"type":"swap"})
        return b.get("total", {}).get("USDT")
    except Exception as e:
        log(f"{IC_BAD} balance error: {e}", "red")
        return None

def price_now():
    try:
        t = ex.fetch_ticker(safe_symbol(SYMBOL))
        return t.get("last") or t.get("close")
    except Exception as e:
        log(f"{IC_BAD} ticker error: {e}", "red")
        return None

def market_amount(amount):
    # يراعي precision والحد الأدنى
    try:
        m = ex.market(safe_symbol(SYMBOL))
        prec = int(m.get("precision",{}).get("amount", 3))
        min_amt = m.get("limits",{}).get("amount",{}).get("min", 0.001)
        amt = float(f"{float(amount):.{prec}f}")
        return max(amt, float(min_amt or 0.001))
    except Exception:
        return float(amount)

# ================= Indicators =================
def fetch_ohlcv(limit=240):
    rows = ex.fetch_ohlcv(safe_symbol(SYMBOL), timeframe=INTERVAL, limit=limit, params={"type":"swap"})
    df = pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])
    return df

def indicators(df: pd.DataFrame):
    ema20  = df["close"].ewm(span=20).mean().iloc[-1]
    ema50  = df["close"].ewm(span=50).mean().iloc[-1]
    ema200 = df["close"].ewm(span=200).mean().iloc[-1]
    # RSI بديل خفيف
    ret = df["close"].pct_change()
    up  = ret.clip(lower=0).rolling(14).mean()
    down= (-ret.clip(upper=0)).rolling(14).mean().replace(0, 1e-9)
    rs  = up / (down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    rsi = float(rsi.fillna(50).iloc[-1])
    return dict(ema20=float(ema20), ema50=float(ema50), ema200=float(ema200), rsi=rsi)

# ================= Logging =================
def log(msg, color="white"):
    print(colored(msg, color), flush=True)

def snapshot(balance, price, ind, pos, total_pnl):
    print()
    log(SEP, "cyan")
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    log(f"{IC_MTR} SNAPSHOT @ {now}", "cyan")
    log(f"🔖 Symbol        : {SYMBOL}", "cyan")
    log(f"⏱️ Interval      : {INTERVAL}", "cyan")
    log(f"⚙️ Leverage      : {LEVERAGE}", "cyan")
    log(f"🛡️ Risk%         : {int(RISK_ALLOC*100)}", "cyan")
    log("—", "cyan")
    log(f"{IC_BAL} Balance (USDT): {fmt(balance,2)}", "yellow")
    log(f"{IC_PRC} Price          : {fmt(price,6)}", "green")
    if ind:
        log(f"📈 EMA20/50/200 : {fmt(ind['ema20'],6)} / {fmt(ind['ema50'],6)} / {fmt(ind['ema200'],6)}", "blue")
        log(f"📉 RSI           : {fmt(ind['rsi'],2)}", "magenta")
    if pos["open"]:
        side = pos["side"].upper()
        log(f"🧭 Position      : {side} | entry={fmt(pos['entry'],6)} qty={fmt(pos['qty'],4)}", "white")
        log(f"🎯 TP / 🛑 SL    : {fmt(pos['tp'],6)} / {fmt(pos['sl'],6)}", "white")
        log(f"💹 PnL curr      : {fmt(pos['pnl'],6)}", "white")
    log(f"📦 Compound PnL : {fmt(total_pnl,6)}", "yellow")
    log(SEP, "cyan")

# ================= Trade Engine =================
state = {
    "open": False, "side": None, "entry": None, "qty": None, "tp": None, "sl": None, "pnl": 0.0
}
compound_pnl = 0.0

def signal_balanced(ind):
    """Balanced: long EMA20>EMA50 و RSI>55 — short EMA20<EMA50 و RSI<45"""
    if ind["ema20"] > ind["ema50"] and ind["rsi"] > 55:
        return "buy"
    if ind["ema20"] < ind["ema50"] and ind["rsi"] < 45:
        return "sell"
    return None

def compute_size(balance, price):
    if not balance or not price: return 0
    raw = (balance * RISK_ALLOC * LEVERAGE) / price
    return market_amount(raw)

def place_protected_order(side, qty, entry_price, atr=None):
    """
    تنفيذ Market + محاولة ربط TP/SL (best-effort)
    لو البورصة رفضت أوامر الحماية، بنفعل حارس سوفت في اللوب.
    """
    global state
    # SL/TP ديناميكي — 1.2*ATR و 1.8*ATR (fallback: 1.5%)
    if atr is None or atr <= 0:
        sl = entry_price * (0.985 if side=="buy" else 1.015)
        tp = entry_price * (1.015 if side=="buy" else 0.985)
    else:
        sl = entry_price - 1.2*atr if side=="buy" else entry_price + 1.2*atr
        tp = entry_price + 1.8*atr if side=="buy" else entry_price - 1.8*atr

    if TRADE_MODE == "paper":
        state.update({"open": True, "side": "long" if side=="buy" else "short",
                      "entry": entry_price, "qty": qty, "tp": tp, "sl": sl, "pnl": 0.0})
        log(f"{IC_TRD} [PAPER] Open {side} qty={fmt(qty,4)} entry={fmt(entry_price,6)} TP={fmt(tp,6)} SL={fmt(sl,6)}", "green")
        return

    try:
        ex.set_leverage(LEVERAGE, safe_symbol(SYMBOL))
    except Exception as e:
        log(f"{IC_BAD} set_leverage: {e}", "red")

    try:
        ord = ex.create_order(safe_symbol(SYMBOL), "market", side, qty, params={"reduceOnly": False})
        entry = ord.get("average") or entry_price
        state.update({"open": True, "side": "long" if side=="buy" else "short",
                      "entry": entry, "qty": qty, "tp": tp, "sl": sl, "pnl": 0.0})
        log(f"{IC_TRD} Open {side} qty={fmt(qty,4)} entry={fmt(entry,6)} TP={fmt(tp,6)} SL={fmt(sl,6)}", "green")

        # Attach TP/SL best-effort
        opp_side = "sell" if side=="buy" else "buy"
        try:
            # BingX عبر ccxt بيختلف أحيانًا في أسماء الأنواع/البارامز، فنعمل محاولات متعددة
            ex.create_order(safe_symbol(SYMBOL), "take_profit_market", opp_side, qty,
                            params={"reduceOnly": True, "takeProfitPrice": tp})
            ex.create_order(safe_symbol(SYMBOL), "stop_market", opp_side, qty,
                            params={"reduceOnly": True, "stopLossPrice": sl})
        except Exception:
            try:
                ex.create_order(safe_symbol(SYMBOL), "limit", opp_side, qty, price=tp, params={"reduceOnly": True})
                ex.create_order(safe_symbol(SYMBOL), "stop",  opp_side, qty, params={"stopPrice": sl, "reduceOnly": True})
            except Exception as e2:
                log(f"{IC_SHD} soft-guard only (couldn't attach TP/SL): {e2}", "yellow")

    except Exception as e:
        log(f"{IC_BAD} open error: {e}", "red")

def close_position(reason):
    """إغلاق فوري + حساب PnL التراكمي"""
    global state, compound_pnl
    if not state["open"]: return
    side = "sell" if state["side"] == "long" else "buy"
    qty  = state["qty"]
    px   = price_now() or state["entry"]

    if TRADE_MODE == "paper":
        pnl = (px - state["entry"]) * qty * (1 if state["side"]=="long" else -1)
        compound_pnl += pnl
        log(f"{IC_CLS} [PAPER] Close {state['side']} reason={reason} pnl={fmt(pnl,6)} total={fmt(compound_pnl,6)}", "magenta")
        state.update({"open": False, "side": None, "entry": None, "qty": None, "tp": None, "sl": None, "pnl": 0.0})
        return

    try:
        ex.create_order(safe_symbol(SYMBOL), "market", side, qty, params={"reduceOnly": True})
    except Exception as e:
        log(f"{IC_BAD} close error: {e}", "red")

    pnl = (px - state["entry"]) * qty * (1 if state["side"]=="long" else -1)
    compound_pnl += pnl
    log(f"{IC_CLS} Close {state['side']} reason={reason} pnl={fmt(pnl,6)} total={fmt(compound_pnl,6)}", "magenta")
    state.update({"open": False, "side": None, "entry": None, "qty": None, "tp": None, "sl": None, "pnl": 0.0})

# ================= Loops =================
def trade_loop():
    global state
    while True:
        try:
            bal  = balance_usdt()
            px   = price_now()
            df   = fetch_ohlcv()
            ind  = indicators(df) if df is not None else None

            # ATR بسيط للحماية الديناميكية
            atr = (df["high"] - df["low"]).rolling(14).mean().iloc[-1] if df is not None else None

            # تحديث PnL الجاري
            if state["open"] and px:
                if state["side"] == "long":
                    state["pnl"] = (px - state["entry"]) * state["qty"]
                else:
                    state["pnl"] = (state["entry"] - px) * state["qty"]

            # Snapshot منظم
            snapshot(bal, px, ind, state.copy(), compound_pnl)

            # قرارات
            if ind is not None:
                sig = signal_balanced(ind)
            else:
                sig = None

            if not state["open"] and sig and px and bal:
                qty = compute_size(bal, px)
                if qty and qty > 0:
                    place_protected_order(sig, qty, px, atr=atr)

            # سوفت-حارس: اغلاق عند ضرب SL/TP لو أوامر الحماية مش متاحة
            if state["open"] and px:
                if state["side"]=="long" and (px <= state["sl"] or px >= state["tp"]):
                    close_position("tp" if px >= state["tp"] else "sl")
                elif state["side"]=="short" and (px >= state["sl"] or px <= state["tp"]):
                    close_position("tp" if px <= state["tp"] else "sl")

        except Exception as e:
            log(f"{IC_BAD} loop error: {e}", "red")
        time.sleep(60)  # لقطة وفحص كل دقيقة (فريم 15م كفاية)

def keepalive_loop():
    if not SELF_URL:
        return
    url = SELF_URL.rstrip("/")
    while True:
        try:
            requests.get(url, timeout=8)
        except Exception:
            pass
        time.sleep(50)  # كل 50 ثانية

# ================= Flask (status & metrics) =================
app = Flask(__name__)

@app.route("/")
def home():
    return "✅ Bot Running — Live Trading with SL/TP & Self-Ping"

@app.route("/metrics")
def metrics():
    return jsonify({
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "mode": TRADE_MODE,
        "leverage": LEVERAGE,
        "risk_alloc": RISK_ALLOC,
        "balance": balance_usdt(),
        "price": price_now(),
        "position": state,
        "compound_pnl": compound_pnl,
        "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    })

# ================= Start =================
threading.Thread(target=trade_loop, daemon=True).start()
threading.Thread(target=keepalive_loop, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
