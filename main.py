# -*- coding: utf-8 -*-
"""
BingX Futures — LIVE (15m • Balanced+ Filters • Hard SL/TP)
- Icons/colored logs (balance & indicators grouped vertically)
- Strict entry filters: EMA200 trend, EMA20/EMA50 confirmed cross, RSI, ADX, slippage guard
- Mandatory SL/TP with ATR; Soft-Guard closes on touch if attach fails
- Post-Trade Validation (confirm real position on BingX before updating state)
- Dynamic position sizing (risk_alloc * leverage), precision-safe
- Self-ping to keep Render awake
- /metrics endpoint
"""

import os, time, math, threading, requests
import pandas as pd
import ccxt
from flask import Flask, jsonify
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# ===================== Config =====================
SYMBOL       = os.getenv("SYMBOL", "DOGE/USDT:USDT")
INTERVAL     = os.getenv("INTERVAL", "15m")
LEVERAGE     = int(float(os.getenv("LEVERAGE", "10")))
RISK_ALLOC   = float(os.getenv("RISK_ALLOC", "0.60"))     # 60% من الرصيد
TRADE_MODE   = os.getenv("TRADE_MODE", "live").lower()    # live / paper
SELF_URL     = os.getenv("RENDER_EXTERNAL_URL", "") or os.getenv("SELF_URL", "")

# Strategy thresholds (قابلة للتعديل)
MIN_ADX          = float(os.getenv("MIN_ADX", "18"))
RSI_LONG_TH      = float(os.getenv("RSI_LONG_TH", "55"))
RSI_SHORT_TH     = float(os.getenv("RSI_SHORT_TH", "45"))
MAX_SLIPPAGE_PCT = float(os.getenv("MAX_SLIPPAGE_PCT", "0.004"))  # 0.4%
COOLDOWN_BARS    = int(os.getenv("COOLDOWN_BARS", "1"))
MAX_DRAWDOWN_PCT = float(os.getenv("MAX_DRAWDOWN_PCT", "0"))      # 0=تعطيل

API_KEY    = os.getenv("BINGX_API_KEY", "")
API_SECRET = os.getenv("BINGX_API_SECRET", "")

# ================= Icons / Colors =================
try:
    from termcolor import colored
except Exception:
    def colored(t,*a,**k): return t

IC_HDR="📊"; IC_BAL="💰"; IC_PRC="💲"; IC_EMA="📈"; IC_RSI="📉"; IC_ADX="📐"; IC_ATR="📏"
IC_POS="🧭"; IC_TP="🎯"; IC_SL="🛑"; IC_TRD="🟢"; IC_CLS="🔴"; IC_OK="✅"; IC_BAD="❌"; IC_SHD="🛡️"
SEP = colored("—"*78, "cyan")

def log(msg, color="white"):
    print(colored(msg, color), flush=True)

def fmt(v,d=2,na="N/A"):
    try:
        if v is None: return na
        return f"{float(v):.{d}f}"
    except Exception:
        return na

def safe_symbol(s: str) -> str:
    if s.endswith(":USDT") or s.endswith(":USDC"): return s
    if "/USDT" in s and not s.endswith(":USDT"): return s + ":USDT"
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
        log(f"{IC_BAD} balance error: {e}", "red"); return None

def price_now():
    try:
        t = ex.fetch_ticker(safe_symbol(SYMBOL))
        return t.get("last") or t.get("close")
    except Exception as e:
        log(f"{IC_BAD} ticker error: {e}", "red"); return None

def market_amount(amount):
    """التزام بحدود السوق (precision/min)."""
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

def adx_simplified(df):
    ret = df["close"].pct_change().abs()
    return float((ret.rolling(14).mean() * 100).fillna(0).iloc[-1])

def compute_atr(df, n=14):
    return float((df["high"] - df["low"]).rolling(n).mean().fillna(0).iloc[-1])

def indicators_strict(df):
    ema20  = df["close"].ewm(span=20).mean()
    ema50  = df["close"].ewm(span=50).mean()
    ema200 = df["close"].ewm(span=200).mean()

    ret = df["close"].pct_change()
    up = ret.clip(lower=0).rolling(14).mean()
    down = (-ret.clip(upper=0)).rolling(14).mean().replace(0,1e-9)
    rs  = up/(down+1e-9)
    rsi = (100 - (100/(1+rs))).fillna(50)

    # latest / previous
    e20_c, e20_p = float(ema20.iloc[-1]), float(ema20.iloc[-2])
    e50_c, e50_p = float(ema50.iloc[-1]), float(ema50.iloc[-2])
    e200_c       = float(ema200.iloc[-1])
    rsi_c        = float(rsi.iloc[-1])
    px_c         = float(df["close"].iloc[-1])
    adx          = adx_simplified(df)
    atr          = compute_atr(df)

    # cross confirmation + trend + thresholds
    cross_up   = (e20_p <= e50_p) and (e20_c > e50_c)
    cross_down = (e20_p >= e50_p) and (e20_c < e50_c)
    long_ok  = (px_c > e200_c)
    short_ok = (px_c < e200_c)

    side = None
    if cross_up and long_ok and rsi_c > RSI_LONG_TH and adx >= MIN_ADX:
        side = "buy"
    elif cross_down and short_ok and rsi_c < RSI_SHORT_TH and adx >= MIN_ADX:
        side = "sell"

    return side, dict(price=px_c, ema20=e20_c, ema50=e50_c, ema200=e200_c, rsi=rsi_c, adx=adx, atr=atr)

# ================= State & Snapshot =================
state = {"open": False, "side": None, "entry": None, "qty": None, "tp": None, "sl": None, "pnl": 0.0}
compound_pnl = 0.0
closed_bar_count = 0
start_balance = None

def snapshot(balance, price, ind, pos, total_pnl):
    print()
    log(SEP, "cyan")
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    log(f"{IC_HDR} SNAPSHOT • {now} • mode={TRADE_MODE.upper()} • {safe_symbol(SYMBOL)} • {INTERVAL}", "cyan")
    log("—", "cyan")
    # مجموعة الأيقونات رأسياً
    log(f"{IC_BAL} Balance (USDT): {fmt(balance,2)}", "yellow")
    log(f"{IC_PRC} Price          : {fmt(price,6)}", "green")
    if ind:
        log(f"{IC_EMA} EMA20/50/200  : {fmt(ind.get('ema20'),6)} / {fmt(ind.get('ema50'),6)} / {fmt(ind.get('ema200'),6)}", "blue")
        log(f"{IC_RSI} RSI           : {fmt(ind.get('rsi'),2)}", "magenta")
        log(f"{IC_ADX} ADX           : {fmt(ind.get('adx'),2)}", "magenta")
        log(f"{IC_ATR} ATR           : {fmt(ind.get('atr'),6)}", "magenta")
    if pos["open"]:
        side = pos["side"].upper()
        log(f"{IC_POS} Position      : {side} | entry={fmt(pos['entry'],6)} qty={fmt(pos['qty'],4)}", "white")
        log(f"{IC_TP}/{IC_SL} TP / SL       : {fmt(pos['tp'],6)} / {fmt(pos['sl'],6)}", "white")
        log(f"📈 PnL current  : {fmt(pos['pnl'],6)}", "white")
    log(f"📦 Compound PnL : {fmt(total_pnl,6)}", "yellow")
    log(SEP, "cyan")

# ================= Sizing & Helpers =================
def compute_size(balance, price):
    if not balance or not price: return 0
    raw = (balance * RISK_ALLOC * LEVERAGE) / price
    return market_amount(raw)

def attach_protection_orders(side_opp, qty, tp, sl):
    # محاولة ربط أوامر الحماية (أفضلية لأوامر السوق المشروطة)
    try:
        ex.create_order(safe_symbol(SYMBOL), "take_profit_market", side_opp, qty,
                        params={"reduceOnly": True, "takeProfitPrice": tp})
        ex.create_order(safe_symbol(SYMBOL), "stop_market", side_opp, qty,
                        params={"reduceOnly": True, "stopLossPrice": sl})
        return True
    except Exception:
        try:
            ex.create_order(safe_symbol(SYMBOL), "limit", side_opp, qty, price=tp, params={"reduceOnly": True})
            ex.create_order(safe_symbol(SYMBOL), "stop",  side_opp, qty, params={"stopPrice": sl, "reduceOnly": True})
            return True
        except Exception as e2:
            log(f"{IC_SHD} protection attach failed: {e2}", "yellow")
            return False

# ================= Open/Close =================
def place_protected_order(side, qty, ref_price, atr=None):
    """Market open + mandatory protection; validate real position before state update."""
    global state
    # SL/TP بالـ ATR (fallback ~1.5%)
    if not atr or atr <= 0:
        sl = ref_price * (0.985 if side=="buy" else 1.015)
        tp = ref_price * (1.015 if side=="buy" else 0.985)
    else:
        sl = ref_price - 1.2*atr if side=="buy" else ref_price + 1.2*atr
        tp = ref_price + 1.8*atr if side=="buy" else ref_price - 1.8*atr

    if TRADE_MODE == "paper":
        state.update({"open": True, "side": "long" if side=="buy" else "short",
                      "entry": ref_price, "qty": qty, "tp": tp, "sl": sl, "pnl": 0.0})
        log(f"{IC_TRD} [PAPER] Open {side} qty={fmt(qty,4)} entry={fmt(ref_price,6)} TP={fmt(tp,6)} SL={fmt(sl,6)}","green")
        return

    # BingX leverage requires side param
    try:
        ex.set_leverage(LEVERAGE, safe_symbol(SYMBOL), params={"side":"BOTH"})
    except Exception as e:
        log(f"{IC_BAD} set_leverage: {e}", "red")

    # نفّذ أمر ماركت
    try:
        ex.create_order(safe_symbol(SYMBOL), "market", side, qty, params={"reduceOnly": False})
        log(f"{IC_TRD} submit {side} qty={fmt(qty,4)}", "green")
    except Exception as e:
        log(f"{IC_BAD} open error: {e}", "red"); return

    # ✅ تحقق بعد التنفيذ (لا نحدّث الحالة إلا لو في Position فعلي)
    try:
        time.sleep(0.8)
        poss = ex.fetch_positions([safe_symbol(SYMBOL)], params={"type":"swap"})
        pos = None
        for p in poss:
            if (p.get("symbol") == safe_symbol(SYMBOL)) and abs(float(p.get("contracts") or 0)) > 0:
                pos = p; break
        if not pos:
            log(f"{IC_SHD} no real position detected; state not updated.", "yellow"); return

        entry = float(pos.get("entryPrice") or ref_price)
        size  = abs(float(pos.get("contracts") or qty))

        # ربط أوامر الحماية إلزامي
        side_opp = "sell" if side=="buy" else "buy"
        attached = attach_protection_orders(side_opp, size, tp, sl)
        if not attached:
            # لو فشل ربط الحماية، نقفل فورًا لحماية الحساب
            try:
                ex.create_order(safe_symbol(SYMBOL), "market", side_opp, size, params={"reduceOnly": True})
            except Exception as e3:
                log(f"{IC_BAD} emergency close after failed protection: {e3}", "red")
            log(f"{IC_SHD} aborted entry since SL/TP not attached.", "yellow")
            return

        state.update({"open": True, "side": "long" if side=="buy" else "short",
                      "entry": entry, "qty": size, "tp": tp, "sl": sl, "pnl": 0.0})
        log(f"{IC_TRD} Open {side} (confirmed) qty={fmt(size,4)} entry={fmt(entry,6)} TP={fmt(tp,6)} SL={fmt(sl,6)}","green")

    except Exception as e:
        log(f"{IC_BAD} post-trade validation error: {e}", "red")

def close_position(reason):
    global state, compound_pnl, closed_bar_count
    if not state["open"]: return
    px   = price_now() or state["entry"]
    qty  = state["qty"]
    side = "sell" if state["side"]=="long" else "buy"

    if TRADE_MODE == "paper":
        pnl = (px - state["entry"]) * qty * (1 if state["side"]=="long" else -1)
        compound_pnl += pnl
        log(f"{IC_CLS} [PAPER] Close {state['side']} reason={reason} pnl={fmt(pnl,6)} total={fmt(compound_pnl,6)}","magenta")
    else:
        try:
            ex.create_order(safe_symbol(SYMBOL), "market", side, qty, params={"reduceOnly": True})
        except Exception as e:
            log(f"{IC_BAD} close error: {e}", "red")
        pnl = (px - state["entry"]) * qty * (1 if state["side"]=="long" else -1)
        compound_pnl += pnl
        log(f"{IC_CLS} Close {state['side']} reason={reason} pnl={fmt(pnl,6)} total={fmt(compound_pnl,6)}","magenta")

    state.update({"open": False, "side": None, "entry": None, "qty": None, "tp": None, "sl": None, "pnl": 0.0})
    closed_bar_count = COOLDOWN_BARS

# ================= Trade Loop =================
def trade_loop():
    global state, compound_pnl, start_balance, closed_bar_count
    while True:
        try:
            bal = balance_usdt()
            if start_balance is None and bal is not None:
                start_balance = bal

            px  = price_now()
            df  = fetch_ohlcv()

            side=None; ind={}
            if df is not None and len(df) > 50:
                side, ind = indicators_strict(df)
            atr = ind.get("atr") if ind else None

            # تحديث PnL الجاري
            if state["open"] and px:
                state["pnl"] = (px - state["entry"]) * state["qty"] if state["side"]=="long" \
                               else (state["entry"] - px) * state["qty"]

            snapshot(bal, px, ind, state.copy(), compound_pnl)

            # Max DD (اختياري)
            if MAX_DRAWDOWN_PCT > 0 and start_balance and bal:
                if (start_balance - bal) / start_balance >= MAX_DRAWDOWN_PCT:
                    if state["open"]:
                        close_position("max_drawdown_hit")
                    log(f"{IC_SHD} Trading paused due to max drawdown", "yellow")
                    time.sleep(60); continue

            # Cooldown
            if closed_bar_count > 0:
                closed_bar_count -= 1
                log(f"{IC_SHD} cooldown bars left: {closed_bar_count}", "yellow")
                time.sleep(60); continue

            # دخول محمي
            if not state["open"] and side and px and bal and ind:
                # حماية انزلاق
                ref = ind["price"]
                if abs(px - ref)/ref > MAX_SLIPPAGE_PCT:
                    log(f"{IC_SHD} skip entry (slippage) px={fmt(px,6)} ref={fmt(ref,6)}","yellow")
                else:
                    qty = compute_size(bal, px)
                    if qty and qty > 0:
                        place_protected_order(side, qty, px, atr=atr)

            # Soft-Guard: اغلاق عند لمس TP/SL (زيادة أمان)
            if state["open"] and px:
                if state["side"]=="long" and (px <= state["sl"] or px >= state["tp"]):
                    close_position("tp" if px >= state["tp"] else "sl")
                elif state["side"]=="short" and (px >= state["sl"] or px <= state["tp"]):
                    close_position("tp" if px <= state["tp"] else "sl")

        except Exception as e:
            log(f"{IC_BAD} loop error: {e}", "red")

        time.sleep(60)  # فريم 15م — تحديث كل دقيقة

# ================= Keepalive =================
def keepalive_loop():
    if not SELF_URL:
        log("SELF_URL not set — keepalive disabled", "yellow"); return
    url = SELF_URL.rstrip("/")
    while True:
        try: requests.get(url, timeout=8)
        except Exception: pass
        time.sleep(50)

# ================= Flask =================
app = Flask(__name__)

@app.route("/")
def home():
    return f"{IC_OK} Bot Running — Dashboard Active"

@app.route("/metrics")
def metrics():
    return jsonify({
        "symbol": safe_symbol(SYMBOL),
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

# ================= Boot =================
threading.Thread(target=trade_loop, daemon=True).start()
threading.Thread(target=keepalive_loop, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
