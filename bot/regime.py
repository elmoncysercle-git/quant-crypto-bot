# bot/regime.py
import numpy as np
import pandas as pd

def realized_vol(series: pd.Series, window=20):
    rets = np.log(series).diff()
    vol = rets.rolling(window).std() * np.sqrt(365)  # annualized approx (daily data)
    return vol

def ma_slope(series: pd.Series, window=100):
    ma = series.rolling(window).mean()
    # slope over last 10 days
    slope = (ma - ma.shift(10)) / 10.0
    return slope

def market_regime(closes: pd.DataFrame, benchmark: str = "BTC/USD") -> str:
    """
    Very simple regime classifier:
    - bull: positive MA slope & low/mid vol
    - chop: small slope (near zero) or mixed; moderate/high vol
    - bear: negative slope & high vol
    """
    if benchmark not in closes.columns:
        # fallback: use the first column
        benchmark = closes.columns[0]

    px = closes[benchmark].dropna()
    if len(px) < 120:
        return "chop"

    slope = ma_slope(px, window=100).iloc[-1]
    vol20 = realized_vol(px, window=20).iloc[-1]

    # compare vol to its 1y percentile
    vol_hist = realized_vol(px, window=20).dropna()
    if len(vol_hist) < 60:
        return "chop"
    vol_pct = (vol_hist <= vol20).mean()  # percentile rank in [0,1]

    # thresholds (tunable)
    pos_slope = slope > 0
    neg_slope = slope < 0
    high_vol = vol_pct > 0.7
    low_vol  = vol_pct < 0.35

    if pos_slope and (low_vol or not high_vol):
        return "bull"
    if neg_slope and high_vol:
        return "bear"
    return "chop"
