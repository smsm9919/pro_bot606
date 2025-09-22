# LIVE BingX Futures adapter via CCXT
import os, math, time
import ccxt

API_KEY = os.getenv("BINGX_API_KEY", "")
API_SECRET = os.getenv("BINGX_API_SECRET", "")

def _make_exchange():
    ex = ccxt.bingx({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "options": {
            "defaultType": "swap",
            "defaultMarginMode": "isolated",
        },
        "timeout": 20000,
    })
    return ex

_EX = None
def exchange():
    global _EX
    if _EX is None:
        _EX = _make_exchange()
    return _EX

def safe_symbol(symbol: str) -> str:
    if symbol.endswith(":USDT"): return symbol
    if symbol.endswith("/USDT"): return symbol + ":USDT"
    return symbol

def fetch_ohlcv(symbol, interval, limit=300):
    ex = exchange()
    return ex.fetch_ohlcv(safe_symbol(symbol), timeframe=interval, limit=limit, params={"type":"swap"})

def fetch_price(symbol):
    ex = exchange()
    t = ex.fetch_ticker(safe_symbol(symbol))
    last = t.get("last") or t.get("close") or t.get("bid") or t.get("ask")
    return float(last) if last is not None else None

def fetch_balance():
    if not (API_KEY and API_SECRET): return None
    ex = exchange()
    b = ex.fetch_balance(params={"type":"swap"})
    total = None
    if isinstance(b, dict): total = b.get("total", {}).get("USDT")
    return float(total) if total is not None else None

def market_info(symbol):
    ex = exchange()
    m = ex.market(safe_symbol(symbol))
    amount_step = m.get("precision",{}).get("amount", 3)
    min_amt = m.get("limits",{}).get("amount",{}).get("min", 1e-3)
    return {"precision": amount_step, "min": min_amt}

def round_amount(symbol, amount):
    info = market_info(symbol)
    step = info["precision"]
    prec = int(step) if isinstance(step, int) else 3
    amt = float(f"{amount:.{prec}f}")
    if amt < info["min"]: amt = info["min"]
    return amt

def set_leverage(symbol, lev=10):
    ex = exchange()
    try: ex.set_leverage(int(lev), safe_symbol(symbol))
    except Exception: pass

def place_market_order(symbol, side, amount, leverage=10, sl=None, tp=None):
    ex = exchange()
    sym = safe_symbol(symbol)
    set_leverage(sym, leverage)
    amount = round_amount(sym, amount)

    # 1) market entry
    order = ex.create_order(sym, type="market", side=side, amount=amount, params={"reduceOnly": False})

    # 2) attach TP/SL (best-effort across ccxt variants)
    params = {"reduceOnly": True}
    try:
        if tp is not None:
            params_tp = dict(params); params_tp.update({"takeProfitPrice": tp})
            ex.create_order(sym, type="take_profit_market", side=("sell" if side=="buy" else "buy"), amount=amount, params=params_tp)
        if sl is not None:
            params_sl = dict(params); params_sl.update({"stopLossPrice": sl})
            ex.create_order(sym, type="stop_market", side=("sell" if side=="buy" else "buy"), amount=amount, params=params_sl)
    except Exception as e:
        # fallback generic stop/limit
        try:
            if tp is not None:
                ex.create_order(sym, type="limit", side=("sell" if side=="buy" else "buy"), amount=amount, price=tp, params={"reduceOnly":True})
            if sl is not None:
                ex.create_order(sym, type="stop", side=("sell" if side=="buy" else "buy"), amount=amount, params={"stopPrice": sl, "reduceOnly":True})
        except Exception:
            pass
    return order
