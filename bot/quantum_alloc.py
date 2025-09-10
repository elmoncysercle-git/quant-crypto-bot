# bot/quantum_alloc.py
import numpy as np
import pandas as pd

# Try quantum deps; fallback if missing
try:
    import dimod
    from dwave_neal.sampler import SimulatedAnnealingSampler
    _HAS_QUANTUM = True
except Exception:
    _HAS_QUANTUM = False

# ---------- Helpers for risk-adjusted path ----------

def _ewma_cov(returns: np.ndarray, alpha=0.94):
    T, N = returns.shape
    mu = returns.mean(axis=0, keepdims=True)
    X = returns - mu
    cov = np.zeros((N, N))
    denom = 0.0
    for t in range(T - 1, -1, -1):
        w = alpha ** (T - 1 - t)
        cov += w * np.outer(X[t], X[t])
        denom += w
    cov /= max(denom, 1e-9)
    return cov

def mean_variance_params(closes: pd.DataFrame):
    arr = closes.values
    rets = np.diff(np.log(arr), axis=0)
    mu = rets.mean(axis=0)
    Sigma = _ewma_cov(rets, alpha=0.94)
    return mu, Sigma

def _greedy_select(mu, Sigma, k=3, lam=0.5):
    scores = mu - lam * np.diag(Sigma)
    idx = np.argsort(scores)[::-1][:max(1, k)]
    return idx

def qubo_from_mean_variance(mu, Sigma, lam=0.5, k=3, penalty=2.0):
    n = len(mu); Q = {}
    for i in range(n):
        Q[(i, i)] = -mu[i] + lam * Sigma[i, i]
        for j in range(i + 1, n):
            Q[(i, j)] = lam * Sigma[i, j] + 2 * penalty
    # soft cardinality penalty toward k
    for i in range(n):
        Q[(i, i)] += penalty * ((-2 * k) + 1)
    return Q

def _solve_qubo(Q, num_reads=600):
    sampler = SimulatedAnnealingSampler()
    ss = sampler.sample_qubo(Q, num_reads=num_reads)
    x = ss.first.sample
    return np.array([x[i] for i in range(len(x))], dtype=int)

# ---------- Expected-return selector ----------

def _expected_return_scores(closes: pd.DataFrame, estimator="ema", window=30, penalize_vol=0.0):
    """
    Returns a np.array score per column, higher = better.
    - estimator: "ema" or "sma"
    - window: lookback in days
    - penalize_vol: subtract penalize_vol * vol (annualized) from the return estimate
    """
    px = closes.dropna()
    rets = np.log(px).diff()

    if estimator.lower() == "ema":
        mu = rets.ewm(span=window, adjust=False).mean().iloc[-1]
    else:  # sma
        mu = rets.tail(window).mean()

    if penalize_vol and penalize_vol > 0:
        vol = rets.tail(window).std() * np.sqrt(365)
        score = mu - penalize_vol * vol
    else:
        score = mu

    # Fill NaNs with very small value to avoid selecting dead series
    score = score.fillna(score.min() if score.notna().any() else 0.0)
    return score.values

# ---------- Public API ----------

def select_assets(closes: pd.DataFrame,
                  max_positions: int = 3,
                  lam: float = 0.5,
                  selection_cfg: dict | None = None):
    """
    Select a list of symbols.
    Modes:
      - expected_return: rank by estimated return (optionally penalize volatility)
      - risk_adjusted  : mean-variance via QUBO or greedy fallback (uses lam)
    """
    selection_cfg = selection_cfg or {}
    mode = (selection_cfg.get("mode") or "risk_adjusted").lower()

    if mode == "expected_return":
        est = selection_cfg.get("estimator", "ema")
        window = int(selection_cfg.get("window", 30))
        penalize_vol = float(selection_cfg.get("penalize_vol", 0.0))
        scores = _expected_return_scores(closes, estimator=est, window=window, penalize_vol=penalize_vol)
        idx = np.argsort(scores)[::-1][:max(1, max_positions)]
        return list(closes.columns[idx])

    # risk_adjusted path
    mu, Sigma = mean_variance_params(closes)

    if _HAS_QUANTUM:
        try:
            Q = qubo_from_mean_variance(mu, Sigma, lam=lam, k=max_positions, penalty=2.0)
            x = _solve_qubo(Q)
            if x.sum() == 0:
                x[np.argmax(mu)] = 1
            idx = np.where(x == 1)[0]
        except Exception:
            idx = _greedy_select(mu, Sigma, k=max_positions, lam=lam)
    else:
        idx = _greedy_select(mu, Sigma, k=max_positions, lam=lam)

    return list(closes.columns[idx])
