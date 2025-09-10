# bot/trade.py
import time
from typing import Dict, List
from .utils import load_config, setup_logging, load_state, save_state
from .exchange import (
    make_client, price, balance_of, market_buy, market_sell,
    load_markets, min_trade_constraints, amount_to_precision
)
from .data import stack_closes
from .quantum_alloc import select_assets
from .strategy import vol_target_weights
from .regime import market_regime

INITIAL_EQUITY = 1000.0
USER_MIN_NOTIONAL = 1.0  # you can raise to 10–25 later

def _ensure_equity_history(state: dict):
    if "equity_history" not in state or not isinstance(state["equity_history"], list):
        state["equity_history"] = []
    if not state["equity_history"]:
        state["equity_history"].append([int(time.time()), float(INITIAL_EQUITY)])

def _record_live_equity(state: dict, client, symbols: List[str], base_ccy: str) -> float:
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
    cfg = load_config("config.yml")
    trading = cfg.get("trading") or {}

    ex_name = (cfg.get("exchange") or {}).get("name", "kraken")
    base = trading.get("base_ccy", "USD")
    symbols = trading.get("symbols") or ["BTC/USD", "ETH/USD", "SOL/USD"]
    lookback = int(trading.get("lookback_days", 90))
    min_w = float(trading.get("min_weight", 0.05))
    max_w = float(trading.get("max_weight", 0.6))
    mode = cfg.get("mode", "paper")

    state_path = cfg.get("state_file", "state/state.json")
    state = load_state(state_path)
    _ensure_equity_history(state)

    client = make_client(ex_name)
    load_markets(client)

    closes = stack_closes(client, symbols, timeframe="1d", lookback_days=lookback)

    # --- Regime & dynamic parameters ---
    regime = market_regime(closes, benchmark="BTC/USD")
    if regime == "bull":
        dyn_cash = 0.15
        dyn_maxpos = max(3, int(trading.get("max_positions", 3)))
        lam = 0.4
    elif regime == "bear":
        dyn_cash = 0.55
        dyn_maxpos = 1
        lam = 0.8
    else:  # chop
        dyn_cash = 0.35
        dyn_maxpos = 2
        lam = 0.6

    log.info(f"Regime: {regime} | dyn_cash={dyn_cash}, dyn_maxpos={dyn_maxpos}, lam={lam}")

    # --- Selection (quantum or greedy) ---
    chosen = select_assets(closes, lam=lam, max_positions=dyn_maxpos)

    # --- Weights: inverse-vol with bounds, cash, and turnover cap ---
    prev_weights = (state.get("last_plan") or {}).get("weights", {})
    weights = vol_target_weights(
        closes, selected=chosen, all_symbols=symbols,
        min_w=min_w, max_w=max_w, cash_buffer=dyn_cash,
        turnover_cap=0.10, prev_weights=prev_weights
    )

    log.info(f"Selected: {chosen}")
    log.info(f"Target weights: {weights} (cash buffer {dyn_cash}) regime={regime}")

    state["last_plan"] = {
        "chosen": chosen,
        "weights": weights,
        "regime": regime,
        "ts": int(time.time())
    }

    if mode == "paper":
        _record_paper_equity(state, closes, weights)
        log.info("PAPER mode: no orders will be placed.")
        save_state(state_path, state)
        return

    # LIVE: record equity
    eq_pre = _record_live_equity(state, client, symbols, base)
    log.info(f"Pre-trade equity (USD): {eq_pre:.2f}")

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

    # Execute respecting exchange minimums
    for s in symbols:
        p = price(client, s)
        if not p:
            log.info(f"Skip {s}: no price.")
            continue

        cons = min_trade_constraints(client, s, p)
        min_cost_ex = cons["min_cost"]
        min_amt_ex = cons["min_amount"]
        min_notional = max(USER_MIN_NOTIONAL, min_cost_ex)

        asset = s.split("/")[0]
        cur_total, _, _ = balance_of(client, asset)
        diff = targets[s] - cur_total
        notional = abs(diff) * p

        if notional < min_notional:
            log.info(f"Skip {s}: notional ${notional:.2f} < min ${min_notional:.2f}")
            continue

        order_amt = abs(diff)
        if min_amt_ex and order_amt < min_amt_ex:
            order_amt = min_amt_ex
        order_amt = amount_to_precision(client, s, order_amt)
        if order_amt <= 0:
            log.info(f"Skip {s}: rounded amount too small.")
            continue

        try:
            if diff > 0:
                log.info(f"BUY {s} amount={order_amt:.10f} (min_cost≈{min_cost_ex:.2f})")
                market_buy(client, s, order_amt)
            else:
                log.info(f"SELL {s} amount={order_amt:.10f} (min_cost≈{min_cost_ex:.2f})")
                market_sell(client, s, order_amt)
        except Exception as e:
            log.exception(f"Order error for {s}: {e}")

    eq_post = _record_live_equity(state, client, symbols, base)
    log.info(f"Post-trade equity (USD): {eq_post:.2f}")
    save_state(state_path, state)

if __name__ == "__main__":
    main()
