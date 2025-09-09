import ccxt
from .utils import env

def make_client(name: str):
    ex_cls = getattr(ccxt, name.lower())
    return ex_cls({
        "apiKey": env("EXCHANGE_KEY"),
        "secret": env("EXCHANGE_SECRET"),
        "password": env("EXCHANGE_PASSWORD",""),
        "enableRateLimit": True,
        "options": {"adjustForTimeDifference": True}
    })

def fetch_ohlcv(client, symbol, timeframe="1d", since=None, limit=200):
    return client.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)

def balance_of(client, code: str):
    bal = client.fetch_balance()
    return bal.get("total",{}).get(code,0.0), bal.get("free",{}).get(code,0.0), bal.get("used",{}).get(code,0.0)

def market_buy(client, symbol, amount):
    return client.create_order(symbol, "market", "buy", amount)

def market_sell(client, symbol, amount):
    return client.create_order(symbol, "market", "sell", amount)

def price(client, symbol):
    t = client.fetch_ticker(symbol)
    return t.get("last") or t.get("close")
