import ccxt
from .utils import env

def make_client(name: str):
    name = name.lower()
    ex_cls = getattr(ccxt, name)
    key = env("EXCHANGE_KEY")
    secret = env("EXCHANGE_SECRET")
    password = env("EXCHANGE_PASSWORD", "")
    client = ex_cls({
        "apiKey": key,
        "secret": secret,
        "password": password,
        "enableRateLimit": True,
        "options": {"adjustForTimeDifference": True}
    })
    return client

def fetch_ohlcv(client, symbol: str, timeframe="1d", since=None, limit=200):
    return client.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)

def balance_of(client, code: str):
    bal = client.fetch_balance()
    total = bal.get("total", {}).get(code, 0.0)
    free = bal.get("free", {}).get(code, 0.0)
    used = bal.get("used", {}).get(code, 0.0)
    return total, free, used

def market_buy(client, symbol: str, amount: float):
    return client.create_order(symbol, "market", "buy", amount)

def market_sell(client, symbol: str, amount: float):
    return client.create_order(symbol, "market", "sell", amount)

def price(client, symbol: str) -> float:
    t = client.fetch_ticker(symbol)
    return t["last"] or t["close"]
