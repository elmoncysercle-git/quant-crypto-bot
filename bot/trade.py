import time
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
    closes = stack_closes(client, symbols, timeframe="1d",
                          lookback_days=cfg["trading"]["lookback_days"])

    chosen = select_assets(closes, lam=cfg["trading"]["risk_aversion"],
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
        log.info("PAPER mode: no orders will be placed.")
        save_state(cfg["state_file"], state)
        return

    base = cfg["trading"]["base_ccy"]
    total_bal, free_bal, used_bal = balance_of(client, base)
    equity = free_bal + used_bal

    targets = {}
    for s in symbols:
        w = weights.get(s, 0.0)
        p = price(client, s)
        targets[s] = (equity * w) / p

    for s in symbols:
        p = price(client, s)
        asset = s.split("/")[0]
        cur_total, _, _ = balance_of(client, asset)
        diff = targets[s] - cur_total
        if abs(diff) * p < 10:
            continue
        try:
            if diff > 0:
                log.info(f"BUY {s} amount={diff:.6f}")
                market_buy(client, s, round(diff,6))
            else:
                log.info(f"SELL {s} amount={-diff:.6f}")
                market_sell(client, s, round(-diff,6))
        except Exception as e:
            log.exception(f"Order error for {s}: {e}")

    save_state(cfg["state_file"], state)

if __name__ == "__main__":
    main()
