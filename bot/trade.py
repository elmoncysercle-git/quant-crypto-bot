import os, time, json
import pandas as pd
from .utils import load_config, setup_logging, load_state, save_state
from .exchange import make_client, price, balance_of, market_buy, market_sell
from .data import stack_closes
from .quantum_alloc import select_assets
from .strategy import equal_weights

def main():
    log = setup_logging("INFO")
    cfg = load_config("config.yml")
    state = load_state(cfg["state_file"])

    client = make_client(cfg["exchange"]["name"])
    symbols = cfg["trading"]["symbols"]
    lookback = int(cfg["trading"]["lookback_days"])
    rebalance_days = int(cfg["trading"]["rebalance_days"])
    cash_buffer = float(cfg["trading"]["cash_buffer_pct"])
    max_positions = int(cfg["trading"]["max_positions"])
    lam = float(cfg["trading"]["risk_aversion"])
    min_w = float(cfg["trading"]["min_weight"])
    max_w = float(cfg["trading"]["max_weight"])
    base = cfg["trading"]["base_ccy"]

    # fetch prices
    closes = stack_closes(client, symbols, timeframe="1d", lookback_days=lookback)
    # select assets via QUBO
    chosen = select_assets(closes, lam=lam, max_positions=max_positions)
    weights = equal_weights(chosen, symbols, min_w=min_w, max_w=max_w, cash_buffer=cash_buffer)

    log.info(f"Selected: {chosen}")
    log.info(f"Target weights (ex-cash): {weights} with cash buffer {cash_buffer:.2f}")

    # Persist selection regardless of mode
    state["last_plan"] = {"chosen": chosen, "weights": weights, "ts": int(time.time())}

    if cfg["mode"] == "paper":
        log.info("PAPER mode: no orders will be placed.")
        save_state(cfg["state_file"], state)
        return

    # LIVE mode: place orders towards target weights
    total_bal, free_bal, used_bal = balance_of(client, base)
    log.info(f"Balance {base}: total={total_bal} free={free_bal} used={used_bal}")
    equity = free_bal + used_bal

    # Compute target dollar amounts
    targets = {}
    for s in symbols:
        w = weights.get(s, 0.0)
        p = price(client, s)
        targets[s] = (equity * w) / p  # target amount in base units of the asset

    # Place simplistic market orders to approximate targets (delta-based)
    for s in symbols:
        p = price(client, s)
        # fetch current position by checking free balance of asset (simplified)
        asset = s.split("/")[0]
        cur_total, cur_free, cur_used = balance_of(client, asset)
        cur_amt = cur_total
        tgt_amt = targets[s]
        diff = tgt_amt - cur_amt
        if abs(diff) * p < 10:  # skip very small trades (<$10)
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

    save_state(cfg["state_file"], state)

if __name__ == "__main__":
    main()
