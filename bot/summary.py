# bot/summary.py
import math
import time
import json
import pathlib
from typing import List, Tuple, Optional

import requests
import yaml

from .utils import env

# ---------- Telegram ----------
def _tg_enabled() -> bool:
    return bool(env("TELEGRAM_BOT_TOKEN") and env("TELEGRAM_CHAT_ID"))

def _tg_send(text: str):
    token = env("TELEGRAM_BOT_TOKEN")
    chat_id = env("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(url, data={"chat_id": chat_id, "text": text})
    except Exception:
        pass  # never fail the job because of Telegram

# ---------- Helpers ----------
def _load_cfg_and_state() -> Tuple[dict, dict, pathlib.Path]:
    cfg_path = pathlib.Path("config.yml")
    if not cfg_path.exists():
        raise FileNotFoundError("config.yml not found.")
    cfg = yaml.safe_load(cfg_path.read_text())

    state_path = pathlib.Path((cfg.get("state_file") or "state/state.json"))
    if not state_path.exists():
        return cfg, {"equity_history": [], "last_plan": None}, state_path

    try:
        state = json.loads(state_path.read_text())
    except Exception:
        state = {"equity_history": [], "last_plan": None}
    return cfg, state, state_path

def _closest_before(points: List[Tuple[int, float]], ts_cutoff: int) -> Optional[Tuple[int, float]]:
    """
    points: list of [ts, equity] sorted ascending or any order.
    Return the *last* point with ts <= ts_cutoff, or None if none exist.
    """
    best = None
    for ts, eq in points:
        if ts <= ts_cutoff and (best is None or ts > best[0]):
            best = (ts, float(eq))
    return best

def _fmt_money(x: float) -> str:
    sign = "+" if x >= 0 else "-"
    return f"{sign}${abs(x):,.2f}"

def _fmt_pct(x: float) -> str:
    sign = "+" if x >= 0 else "-"
    return f"{sign}{abs(x)*100:.2f}%"

def _build_plan_line(last_plan: dict) -> str:
    if not last_plan:
        return "Plan: (none)"
    w = last_plan.get("weights", {}) or {}
    # Estimate cash as 1 - sum(weights)
    cash = max(0.0, 1.0 - sum(float(v) for v in w.values()))
    parts = [f"{k} {float(v):.0%}" for k, v in w.items() if float(v) > 0]
    return "Plan: " + (", ".join(parts) if parts else "(none)") + f" | cash ~{cash:.0%}"

def main():
    cfg, state, _ = _load_cfg_and_state()
    base = (cfg.get("trading") or {}).get("base_ccy", "USD")

    eh = state.get("equity_history") or []
    if len(eh) == 0:
        if _tg_enabled():
            _tg_send(f"📊 PnL Summary ({base})\nNo equity data yet.")
        return

    # Ensure numeric types
    hist = [(int(ts), float(eq)) for ts, eq in eh]
    hist.sort(key=lambda x: x[0])

    ts_now = int(time.time())
    now_eq = float(hist[-1][1])

    # Cutoffs
    cutoff_24h = ts_now - 24*3600
    cutoff_7d  = ts_now - 7*24*3600

    p24 = _closest_before(hist, cutoff_24h)
    p7  = _closest_before(hist, cutoff_7d)

    def pnl(from_point):
        if not from_point:
            return None, None
        ts_from, eq_from = from_point
        abs_chg = now_eq - eq_from
        pct_chg = (abs_chg / eq_from) if eq_from != 0 else float("nan")
        return abs_chg, pct_chg

    abs24, pct24 = pnl(p24)
    abs7, pct7   = pnl(p7)

    lines = [f"📊 PnL Summary ({base})"]
    if abs24 is None:
        lines.append("24h: (insufficient data)")
    else:
        lines.append(f"24h: {_fmt_money(abs24)} ({_fmt_pct(pct24)})")

    if abs7 is None:
        lines.append("7d : (insufficient data)")
    else:
        lines.append(f"7d : {_fmt_money(abs7)} ({_fmt_pct(pct7)})")

    lines.append(f"Equity: ${now_eq:,.2f}")

    # Add plan context
    lines.append(_build_plan_line(state.get("last_plan")))

    msg = "\n".join(lines)
    if _tg_enabled():
        _tg_send(msg)

if __name__ == "__main__":
    main()
