import time
from typing import Dict, List
from .utils import load_config, setup_logging, load_state, save_state
from .exchange import make_client, price, balance_of, market_buy, market_sell
from .data import stack_closes
from .quantum_alloc import select_assets
from .strategy import equal_weights

INITIAL_EQUITY = 1000.0  # first point for the chart

def _ensure_equity_history(state: dict):
    if "equity_history" not in state or not isinstance(state["equity_history"], list):
        state["equity_history"] = []
    if not state["equity_history"]:
        state["equity_history"].append([int(time.time()), float(INITIAL_EQUITY)])

def _record_live_equity(state: dict, client, symbols: List[str], base_ccy: str):
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

    closes = stack_closes(client, symbols, timeframe="1d", lookback_days=lookback)

    chosen = select_assets(closes, lam=lam, max_positions=max_positions)
    weights = equal_weights(chosen, symbols, min_w=min_w, max_w=max_w, cash_buffer=cash_buf)

    log.info(f"Selected: {chosen}")
    log.info(f"Target weights: {weights} (cash buffer {cash_buf})")
    state["last_plan"] = {"chosen": chosen, "weights": weights, "ts": int(time.time())}

    if mode == "paper":
        _record_paper_equity(state, closes, weights)
        log.info("PAPER mode: no orders will be placed.")
        save_state(state_path, state)
        return

    # LIVE: record pre-trade equity
    try:
        _record_live_equity(state, client, symbols, base)
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

    # Drift-correct with simplistic market orders (skip <$1 notionals)
    for s in symbols:
        p = price(client, s)
        if not p:
            log.warning(f"No price for {s}; skipping.")
            continue
        asset = s.split("/")[0]
        cur_total, _, _ = balance_of(client, asset)
        diff = targets[s] - cur_total
        if abs(diff) * p < 1:   # lowered to $1 threshold
            continue
        try:
            if diff > 0:
                log.info(f"BUY {s} amount={diff:.6f}")
                market_buy(client, s, round(diff, 6))
            else:
                log.info(f"SELL {s} amount={-diff:.6f}")
                market_sell(client, s, round(-diff, 6))
        except Exception as e:
            log.exception(f"Order error for {s}: {e}")

    try:
        _record_live_equity(state, client, symbols, base)
    except Exception as e:
        log.warning(f"Could not record post-trade live equity: {e}")

    save_state(state_path, state)

if __name__ == "__main__":
    main()
