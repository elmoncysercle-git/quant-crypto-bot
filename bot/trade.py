import time
from typing import Dict, List
from .utils import load_config, setup_logging, load_state, save_state
from .exchange import make_client, price, balance_of, market_buy, market_sell
from .data import stack_closes
from .quantum_alloc import select_assets
from .strategy import equal_weights

def _init_equity_history(state: dict):
    if "equity_history" not in state or not isinstance(state["equity_history"], list):
        state["equity_history"] = []

def _record_live_equity(state: dict, client, symbols: List[str], base_ccy: str):
    """
    Reads balances of base and all assets, values them in base_ccy using current prices,
    and appends a (ts, equity) point to equity_history.
    """
    _, free_base, used_base = balance_of(client, base_ccy)
    equity = free_base + used_base
    for s in symbols:
        asset = s.split("/")[0]
        total_asset, _, _ = balance_of(client, asset)
        if total_asset and total_asset > 0:
            px = price(client, s)  # asset/base
            if px:
                equity += total_asset * px
    state["equity_history"].append([int(time.time()), float(equity)])

def _record_paper_equity(state: dict, closes, weights: Dict[str, float], cash_buffer: float):
    """
    Simulates equity using the last close-to-close return of the selected portfolio.
    If no previous equity point exists, start at 1000. Skips if <2 close bars.
    """
    if closes.shape[0] < 2:
        return
    prev = closes.iloc[-2]; curr = closes.iloc[-1]
    port_ret = 0.0
    if sum(weights.values()) > 0:
        for s, w in weights.items():
            if w > 0 and s in prev.index and s in curr.index:
                r = (float(curr[s]) - float(prev[s])) / float(prev[s])
                port_ret += w * r
    equity = 1000.0 if not state["equity_history"] else float(state["equity_history"][-1][1])
    equity *= (1.0 + port_ret)
    state["equity_history"].append([int(time.time()), float(equity)])

def main():
    log = setup_logging("INFO")
    cfg = load_config("config.yml")
    state = load_state(cfg["state_file"])
    _init_equity_history(state)

    client = make_client(cfg["exchange"]["name"])
    symbols = cfg["trading"]["symbols"]
    closes = stack_closes(client, symbols, timeframe="1d",
                          lookback_days=cfg["trading"]["lookback_days"])

    # Allocation via QUBO (or fallback) + weight shaping
    chosen = select_assets(closes,
                           lam=cfg["trading"]["risk_aversion"],
                           max_positions=cfg["trading"]["max_positions"])
    weights = equal_weights(
        chosen, symbols,
        min_w=cfg["trading"]["min_weight"],
        max_w=cfg["trading"]["max_weight"],
        cash_buffer=cfg["trading"]["cash_buffer_pct"]
    )

    log.info(f"Selected: {chosen}")
    log.info(f"Target weights: {weights} (cash buffer {cfg['trading']['cash_buffer_pct']})")
    state["last_plan"] = {"chosen": chosen, "weights": weights, "ts": int(time.time())}

    if cfg["mode"] == "paper":
        _record_paper_equity(state, closes, weights, cfg["trading"]["cash_buffer_pct"])
        log.info("PAPER mode: no orders will be placed.")
        save_state(cfg["state_file"], state)
        return

    # LIVE: record pre-trade equity
    base = cfg["trading"]["base_ccy"]
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
        w = weights.get(s, 0.0)
        p = price(client, s)
        targets[s] = (equity * w) / p if p else 0.0

    # Drift-correct with simplistic market orders (skip <$10 notionals)
    for s in symbols:
        p = price(client, s)
        if not p:
            continue
        asset = s.split("/")[0]
        cur_total, _, _ = balance_of(client, asset)
        diff = targets[s] - cur_total
        if abs(diff) * p < 10:
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

    # Record post-trade equity
    try:
        _record_live_equity(state, client, symbols, base)
    except Exception as e:
        log.warning(f"Could not record post-trade live equity: {e}")

    save_state(cfg["state_file"], state)

if __name__ == "__main__":
    main()
