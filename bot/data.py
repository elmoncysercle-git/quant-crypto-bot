import time, math
import pandas as pd
from .exchange import fetch_ohlcv

def ohlcv_df(client, symbol: str, timeframe="1d", lookback_days=90):
    # ccxt times are in ms
    now = int(time.time()*1000)
    since = now - lookback_days*24*60*60*1000
    raw = fetch_ohlcv(client, symbol, timeframe=timeframe, since=since, limit=lookback_days+10)
    if not raw:
        raise RuntimeError(f"No OHLCV for {symbol}")
    cols = ["timestamp","open","high","low","close","volume"]
    df = pd.DataFrame(raw, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df = df[~df.index.duplicated(keep="last")]
    return df

def stack_closes(client, symbols, timeframe="1d", lookback_days=90):
    frames = []
    for s in symbols:
        df = ohlcv_df(client, s, timeframe=timeframe, lookback_days=lookback_days)
        frames.append(df["close"].rename(s))
    closes = pd.concat(frames, axis=1).dropna(how="any")
    return closes
