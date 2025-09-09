# Simple backward-looking evaluation of the rotation logic (paper only).
import pandas as pd, numpy as np, json, time
from .utils import load_config, setup_logging
from .exchange import make_client
from .data import stack_closes
from .quantum_alloc import select_assets
from .strategy import equal_weights

def run_backtest():
    log = setup_logging("INFO")
    cfg = load_config("config.yml")
    client = make_client(cfg["exchange"]["name"])
    symbols = cfg["trading"]["symbols"]
    closes = stack_closes(client, symbols, timeframe="1d", lookback_days=max(200, cfg["trading"]["lookback_days"]))
    dates = closes.index

    equity = 1000.0
    cash_buffer = cfg["trading"]["cash_buffer_pct"]
    max_positions = cfg["trading"]["max_positions"]
    lam = cfg["trading"]["risk_aversion"]
    min_w = cfg["trading"]["min_weight"]
    max_w = cfg["trading"]["max_weight"]
    rebalance_days = cfg["trading"]["rebalance_days"]

    last_reb = None
    weights = {s:0.0 for s in symbols}

    history = []
    for i in range(len(dates)):
        date = dates[i]
        # Rebalance periodically
        if last_reb is None or (date - last_reb).days >= rebalance_days:
            sub = closes.iloc[:i+1]
            if len(sub) > 30:
                chosen = select_assets(sub, lam=lam, max_positions=max_positions)
                weights = equal_weights(chosen, symbols, min_w=min_w, max_w=max_w, cash_buffer=cash_buffer)
                last_reb = date
                log.info(f"{date.date()} Rebalance -> {chosen}")

        if i == 0: 
            history.append((date, equity))
            continue

        # Apply daily return
        prev = closes.iloc[i-1]
        today = closes.iloc[i]
        # portfolio return (excluding cash buffer)
        port_ret = 0.0
        alloc_sum = sum(weights.values())
        if alloc_sum > 0:
            for s, w in weights.items():
                if w > 0:
                    r = (today[s] - prev[s]) / prev[s]
                    port_ret += w * r
        equity *= (1 + port_ret)
        history.append((date, equity))

    df = pd.DataFrame(history, columns=["date","equity"]).set_index("date")
    stats = {
        "final_equity": float(df["equity"].iloc[-1]),
        "return_pct": float((df["equity"].iloc[-1]/df["equity"].iloc[0]-1)*100.0),
        "max_drawdown_pct": float(((df["equity"]/df["equity"].cummax()).min()-1)*100.0),
    }
    print(stats)
    return df, stats

if __name__ == "__main__":
    run_backtest()
