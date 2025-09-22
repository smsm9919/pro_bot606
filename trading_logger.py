import sys, json, time, threading
from datetime import datetime
try:
    from colorama import init as _cinit, Fore, Style
    _cinit(autoreset=True)
    GREEN, RED, CYAN, YELLOW, DIM, RST = Fore.GREEN, Fore.RED, Fore.CYAN, Fore.YELLOW, Style.DIM, Style.RESET_ALL
except Exception:
    GREEN = RED = CYAN = YELLOW = DIM = RST = ""

_lock = threading.Lock()

class TradeLogger:
    def __init__(self, metrics_fn=None, jsonl_path="audit_log.jsonl", enable_jsonl=True):
        self.metrics_fn   = metrics_fn
        self.jsonl_path   = jsonl_path
        self.enable_jsonl = enable_jsonl

    def _now(self): return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    def _print(self, txt=""): print(txt, flush=True, file=sys.stdout)
    def _jsonl(self, evt, data):
        if not self.enable_jsonl: return
        rec = {"ts": self._now(), "event": evt, "data": data}
        line = json.dumps(rec, ensure_ascii=False)
        with _lock:
            with open(self.jsonl_path, "a", encoding="utf-8") as f: f.write(line+"\n")

    def snapshot(self, **d):
        self._print(f"{DIM}================= METRICS SNAPSHOT @ {self._now()} ================={RST}")
        bal, price = d.get("balance"), d.get("price")
        rsi, adx, atr = d.get("rsi"), d.get("adx"), d.get("atr")
        ema20, ema50, ema200 = d.get("ema20"), d.get("ema50"), d.get("ema200")
        st = d.get("supertrend"); pos = d.get("position"); cprofit = d.get("compound_profit")
        t = d.get("total_trades"); last = d.get("update_time")
        self._print(f"{CYAN}Balance{RST}: {bal}   {CYAN}Price{RST}: {price}")
        self._print(f"{CYAN}RSI/ADX{RST}: {rsi:.2f if rsi is not None else 0.0} / {adx:.2f if adx is not None else 0.0}")
        self._print(f"{CYAN}ATR{RST}: {atr}")
        self._print(f"{CYAN}EMA20/50/200{RST}: {ema20} / {ema50} / {ema200}")
        if st is not None:
            self._print(f"{CYAN}Supertrend{RST}: {'BULLISH' if st==1 else 'BEARISH'}")
        if cprofit is not None or t is not None:
            self._print(f"{CYAN}Compound Profit{RST}: {cprofit}   {CYAN}Trades{RST}: {t}")
        if pos:
            col = GREEN if pos.get('side')=='long' else RED
            self._print(f"{CYAN}Position{RST}: side={col}{pos.get('side')}{RST} entry={pos.get('entry')} TP={pos.get('tp')} SL={pos.get('sl')} pnl={pos.get('pnl')}")
        if last:
            self._print(f"{CYAN}Last update{RST}: {last}")
        self._print(f"{DIM}====================================================================={RST}")
        self._jsonl("snapshot", d)

    def signal(self, **d):
        side = d.get("side"); col = GREEN if side=='long' else RED
        self._print(f"[SIGNAL] side={col}{side}{RST} reason={YELLOW}{d.get('reason')}{RST} price={d.get('price')} extra={d}")
        self._jsonl("signal", d)

    def entry(self, **d):
        side = d.get("side"); col = GREEN if side=='long' else RED
        self._print(f"[ENTRY ] side={col}{side}{RST} entry={d.get('entry')} tp={d.get('tp')} sl={d.get('sl')} qty={d.get('qty')}")
        self._jsonl("entry", d)

    def exit(self, **d):
        reason = d.get("reason"); col = GREEN if reason=='tp' else RED if reason=='sl' else CYAN
        self._print(f"[EXIT  ] reason={col}{reason}{RST} pnl={d.get('pnl')} price={d.get('price')} tp={d.get('tp')} sl={d.get('sl')}")
        self._jsonl("exit", d)

    def pnl(self, pnl):
        col = GREEN if (pnl or 0) >= 0 else RED
        self._print(f"[PNL   ] {col}{pnl}{RST}")
        self._jsonl("pnl", {"pnl": pnl})

_default = None
def get_logger(metrics_fn=None):
    global _default
    if _default is None: _default = TradeLogger(metrics_fn=metrics_fn)
    return _default
