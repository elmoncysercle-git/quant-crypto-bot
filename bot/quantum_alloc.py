# bot/quantum_alloc.py
import numpy as np

# Try quantum deps; fallback if missing (e.g., Streamlit/Colab)
try:
    import dimod
    from dwave_neal.sampler import SimulatedAnnealingSampler
    _HAS_QUANTUM = True
except Exception:
    _HAS_QUANTUM = False

def _ewma_cov(returns: np.ndarray, alpha=0.94):
    """
    Exponentially-weighted covariance (RiskMetrics-style).
    returns: T x N (rows = time)
    """
    T, N = returns.shape
    mu = returns.mean(axis=0, keepdims=True)
    X = returns - mu
    cov = np.zeros((N, N))
    w = 1.0
    denom = 0.0
    for t in range(T-1, -1, -1):
        cov += (alpha**(T-1-t)) * np.outer(X[t], X[t])
        denom += (alpha**(T-1-t))
    cov /= max(denom, 1e-9)
    return cov

def mean_variance_params(closes):
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
        Q[(i,i)] = -mu[i] + lam*Sigma[i,i]
        for j in range(i+1, n):
            Q[(i,j)] = lam*Sigma[i,j] + 2*penalty
    for i in range(n):
        Q[(i,i)] += penalty*((-2*k)+1)
    return Q

def _solve_qubo(Q, num_reads=600):
    sampler = SimulatedAnnealingSampler()
    ss = sampler.sample_qubo(Q, num_reads=num_reads)
    x = ss.first.sample
    return np.array([x[i] for i in range(len(x))], dtype=int)

def select_assets(closes, lam=0.5, max_positions=3):
    """
    Returns a list of selected symbols using QUBO or greedy, with EWMA cov.
    """
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
