import pandas as pd
import matplotlib.pyplot as plt
from .utils import setup_logging
from .exchange import make_client, fetch_ohlcv
from .quantum_alloc import select_assets
from .strategy import equal_weights
from .data import stack_closes

def run_backtest(start="2023-01-01", end="2025-01-01", initial_equity=1000.0):
    log = setup_logging("INFO")
    client = make_client("kraken")
    symbols = ["BTC/USD", "ETH/USD", "SOL/USD", "LINK/USD", "AVAX/USD"]

    # fetch OHLCV daily data for all symbols
    closes = stack_closes(client, symbols, timeframe="1d", lookback_days=800)
    closes = closes.loc[start:end]

    equity = initial_equity
    equity_curve = []
    cash_buffer = 0.25
    max_positions = 3
    lam = 0.5

    # weekly rebalancing
    last_rebalance = None
    weights = {s: 0.0 for s in symbols}

    for date, row in closes.iterrows():
        if last_rebalance is None or (date - last_rebalance).days >= 7:
            chosen = select_assets(closes.loc[:date], lam=lam, max_positions=max_positions)
            weights = equal_weights(
                chosen, symbols,
                min_w=0.05, max_w=0.6, cash_buffer=cash_buffer
            )
            last_rebalance = date
            log.info(f"Rebalanced {date.date()} → {chosen}, weights={weights}")

        # daily return
        daily_ret = 0.0
        for s, w in weights.items():
            if w > 0 and s in closes.columns:
                prev = closes[s].shift(1).loc[date]
                curr = row[s]
                if prev and prev > 0:
                    r = (curr - prev) / prev
                    daily_ret += w * r

        equity *= (1 + daily_ret)
        equity_curve.append({"date": date, "equity": equity})

    df = pd.DataFrame(equity_curve).set_index("date")
    return df

if __name__ == "__main__":
    df = run_backtest()
    df.to_csv("backtest_equity.csv")
    df.plot(title="Backtest Equity Curve")
    plt.ylabel("Equity ($)")
    plt.show()
