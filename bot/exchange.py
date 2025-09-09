import ccxt
from .utils import env

def make_client(name: str):
    ex_cls = getattr(ccxt, name.lower())
    client = ex_cls({
        "apiKey": env("EXCHANGE_KEY"),
        "secret": env("EXCHANGE_SECRET"),
        "password": env("EXCHANGE_PASSWORD",""),
        "enableRateLimit": True,
        "options": {"adjustForTimeDifference": True}
    })
    return client

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

# ---------- Market metadata & constraints ----------

def load_markets(client):
    """Load markets once per run (ccxt caches on the client)."""
    try:
        return client.load_markets()
    except Exception:
        return {}

def market_info(client, symbol: str):
    """Return ccxt market dict for symbol (after load_markets)."""
    load_markets(client)
    try:
        return client.market(symbol)
    except Exception:
        return {}

def min_trade_constraints(client, symbol: str, px: float):
    """
    Returns a dict with Kraken/ccxt min trade constraints for this symbol:
      - min_cost: minimum notional in quote currency (e.g., USD)
      - min_amount: minimum amount of base asset
      - amount_precision: for rounding order sizes
    Falls back to reasonable defaults if not provided by the exchange.
    """
    m = market_info(client, symbol) or {}
    limits = m.get("limits", {}) or {}

    # cost (notional) min in quote (USD)
    min_cost = None
    if "cost" in limits and isinstance(limits["cost"], dict):
        min_cost = limits["cost"].get("min")

    # amount (base units) min
    min_amount = None
    if "amount" in limits and isinstance(limits["amount"], dict):
        min_amount = limits["amount"].get("min")

    # derive min_cost from min_amount * px if needed
    if (min_cost is None or min_cost == 0) and min_amount and px:
        min_cost = float(min_amount) * float(px)

    # sensible fallbacks if the exchange didn't provide values
    if min_cost is None or min_cost <= 0:
        min_cost = 10.0  # Kraken typically around $5–$10; we default to $10
    if min_amount is None or min_amount <= 0:
        min_amount = 0.0

    amount_precision = None
    precision = m.get("precision") or {}
    if "amount" in precision:
        amount_precision = precision["amount"]

    return {
        "min_cost": float(min_cost),
        "min_amount": float(min_amount),
        "amount_precision": amount_precision
    }

def amount_to_precision(client, symbol: str, amount: float) -> float:
    """Round amount to the exchange precision for this market."""
    try:
        return float(client.amount_to_precision(symbol, amount))
    except Exception:
        return float(amount)

