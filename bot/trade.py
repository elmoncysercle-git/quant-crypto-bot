import time
from typing import Dict, List
import requests
from .utils import load_config, setup_logging, load_state, save_state, env
from .exchange import (
    make_client, price, balance_of, market_buy, market_sell,
    load_markets, min_trade_constraints, amount_to_precision
)
from .data import stack_closes
from .quantum_alloc import select_assets
from .strategy import equal_weights

INITIAL_EQUITY = 1000.0       # seed the first chart point if empty
USER_MIN_NOTIONAL = 1.0       # your threshold; exchange min will override if higher

# --------- Telegram helpers ---------
def _tg_enabled() -> bool:
    return bool(env("TELEGRAM_BOT_TOKEN") and env("TELEGRAM_CHAT_ID"))

def _tg_send(text: str):
    """Fire-and-forget Telegram message; no-op if not configured."""
    token = env("TELEGRAM_BOT_TOKEN")
    chat_id = env("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(url, data={"chat_id": chat_id, "text": text})
    except Exception:
        pass  # never crash trading on alert errors

# --------- Equity bookkeeping ---------
def _ensure_equity_history(state: dict):
    if "equity_history" not in state or not isinstance(state["equity_history"], list):
        state["equity_history"] = []
    if not state["equity_history"]:
        state["equity_history"].append([int(time.time()), float(INITIAL_EQUITY)])

def _record_live_equity(state: dict, client, symbols: List[str], base_ccy: str) -> float:
    """Returns equity after recording it."""
    _, free_base, used_base = balance_of(client, base_ccy)
    equity = free_base + used_base
    for s in symbols:
        asset = s.split("/")[0]
        total_asset, _, _ = balance_of(client, asset)
        if total_asset and total_asset > 0:
            px = price(client, s)
            if px:
                equity += total_asset * px
    state["equity_history"].append([int(time.time()), float(equity)])
    return float(equity)

def _record_paper_equity(state: dict, closes, weights: Dict[str, float]):
    if closes.shape[0] < 2:
        last_eq = float(state["equity_history"][-1][1])
        state["equity_history"].append([int(time.time()), last_eq])
        return
    prev = closes.iloc[-2]; curr = closes.iloc[-1]
    port_ret = 0.0
    if sum(weights.values()) > 0:
        for s, w in weights.items():
            if w > 0 and s in prev.index and s in curr.index:
                r = (float(curr[s]) - float(prev[s])) / float(prev[s])
                port_ret += w * r
    last_eq = float(state["equity_history"][-1][1])
    equity = last_eq * (1.0 + port_ret)
    state["equity_history"].append([int(time.time()), float(equity)])

def main():
    log = setup_logging("INFO")
    try:
        cfg = load_config("config.yml")
    except Exception as e:
        log.error(f"Failed to load config.yml: {e}")
        return

    # ---- Safe defaults if keys are missing ----
    ex_name = (cfg.get("exchange") or {}).get("name", "kraken")
    trading = cfg.get("trading") or {}
    base = trading.get("base_ccy", "USD")
    symbols = trading.get("symbols") or ["BTC/USD", "ETH/USD", "SOL/USD"]
    lookback = int(trading.get("lookback_days", 90))
    max_positions = int(trading.get("max_positions", 3))
    lam = float(trading.get("risk_aversion", 0.5))
    min_w = float(trading.get("min_weight", 0.05))
    max_w = float(trading.get("max_weight", 0.6))
    cash_buf = float(trading.get("cash_buffer_pct", 0.15))
    mode = cfg.get("mode", "paper")

    if "exchange" not in cfg:
        log.warning("config.yml is missing 'exchange:' block. Defaulting to exchange: kraken")

    state_path = cfg.get("state_file", "state/state.json")
    state = load_state(state_path)
    _ensure_equity_history(state)

    client = make_client(ex_name)
    load_markets(client)  # cache market metadata

    closes = stack_closes(client, symbols, timeframe="1d", lookback_days=lookback)

    # Allocation via QUBO (or fallback) + weight shaping
    chosen = select_assets(closes, lam=lam, max_positions=max_positions)
    weights = equal_weights(chosen, symbols, min_w=min_w, max_w=max_w, cash_buffer=cash_buf)

    # Plan summary alert
    cash_pct = f"{cash_buf:.0%}"
    plan_lines = [f"{s} {weights.get(s,0):.0%}" for s in symbols if weights.get(s,0) > 0]
    _tg_send(f"📣 Plan ({mode.upper()}): " + (", ".join(plan_lines) if plan_lines else "no positions")
             + f" | cash {cash_pct}")

    log.info(f"Selected: {chosen}")
    log.info(f"Target weights: {weights} (cash buffer {cash_buf})")
    state["last_plan"] = {"chosen": chosen, "weights": weights, "ts": int(time.time())}

    if mode == "paper":
        _record_paper_equity(state, closes, weights)
        log.info("PAPER mode: no orders will be placed.")
        save_state(state_path, state)
        return

    # LIVE: pre-trade equity (and alert)
    try:
        eq_pre = _record_live_equity(state, client, symbols, base)
        _tg_send(f"💼 Pre-trade equity: ${eq_pre:,.2f}")
    except Exception as e:
        log.warning(f"Could not record pre-trade live equity: {e}")

    # Compute total equity (base + coins) for sizing
    _, free_base, used_base = balance_of(client, base)
    equity = free_base + used_base
    for s in symbols:
        asset = s.split("/")[0]
        tot, _, _ = balance_of(client, asset)
        if tot and tot > 0:
            try:
                px = price(client, s)
                if px:
                    equity += tot * px
            except Exception:
                pass

    # Targets in asset units
    targets = {}
    for s in symbols:
        p = price(client, s)
        targets[s] = (equity * weights.get(s, 0.0)) / p if p else 0.0

    # Drift-correct with market orders respecting exchange minimums
    for s in symbols:
        p = price(client, s)
        if not p:
            log.warning(f"No price for {s}; skipping.")
            _tg_send(f"⚠️ Skip {s}: no price.")
            continue

        cons = min_trade_constraints(client, s, p)
        min_cost_ex = cons["min_cost"]          # exchange min notional in quote (USD)
        min_amt_ex = cons["min_amount"]         # exchange min amount in base
        min_notional = max(USER_MIN_NOTIONAL, min_cost_ex)

        asset = s.split("/")[0]
        cur_total, _, _ = balance_of(client, asset)
        diff = targets[s] - cur_total
        notional = abs(diff) * p

        if notional < min_notional:
            msg = f"⏭️ Skip {s}: notional ${notional:.2f} < min ${min_notional:.2f}"
            log.info(msg)
            _tg_send(msg)
            continue

        # ensure at least exchange min amount
        order_amt = abs(diff)
        if min_amt_ex and order_amt < min_amt_ex:
            order_amt = min_amt_ex

        order_amt = amount_to_precision(client, s, order_amt)
        if order_amt <= 0:
            log.info(f"Skip {s}: rounded amount too small.")
            _tg_send(f"⏭️ Skip {s}: rounded amount too small.")
            continue

        try:
            if diff > 0:
                log.info(f"BUY {s} amount={order_amt:.10f} (min_cost≈{min_cost_ex:.2f}, min_amt≈{min_amt_ex})")
                market_buy(client, s, order_amt)
                _tg_send(f"✅ BUY {s} {order_amt:.10f}")
            else:
                log.info(f"SELL {s} amount={order_amt:.10f} (min_cost≈{min_cost_ex:.2f}, min_amt≈{min_amt_ex})")
                market_sell(client, s, order_amt)
                _tg_send(f"✅ SELL {s} {order_amt:.10f}")
        except Exception as e:
            log.exception(f"Order error for {s}: {e}")
            _tg_send(f"❌ Order error {s}: {e}")

    # Record post-trade equity (and alert)
    try:
        eq_post = _record_live_equity(state, client, symbols, base)
        _tg_send(f"ℹ️ Post-trade equity: ${eq_post:,.2f}")
    except Exception as e:
        log.warning(f"Could not record post-trade live equity: {e}")

    save_state(state_path, state)

if __name__ == "__main__":
    main()
