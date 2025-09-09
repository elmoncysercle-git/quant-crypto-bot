import numpy as np
import dimod
from dwave_neal.sampler import SimulatedAnnealingSampler

def mean_variance_params(closes: np.ndarray):
    rets = np.diff(np.log(closes), axis=0)
    mu = rets.mean(axis=0)
    Sigma = np.cov(rets, rowvar=False)
    return mu, Sigma

def qubo_from_mean_variance(mu, Sigma, lam=0.5, k=3, penalty=2.0):
    n = len(mu); Q = {}
    # Objective: minimize -(mu^T x) + lam x^T Sigma x + penalty*(|x|-k)^2 (simplified)
    for i in range(n):
        Q[(i,i)] = -mu[i] + lam*Sigma[i,i]
        for j in range(i+1,n):
            Q[(i,j)] = lam*Sigma[i,j] + 2*penalty
    for i in range(n):
        Q[(i,i)] += penalty*((-2*k)+1)
    return Q

def solve_qubo(Q, num_reads=800):
    sampler = SimulatedAnnealingSampler()
    ss = sampler.sample_qubo(Q, num_reads=num_reads)
    x = ss.first.sample
    return np.array([x[i] for i in range(len(x))], dtype=int)

def select_assets(closes, lam=0.5, max_positions=3):
    mu, Sigma = mean_variance_params(closes.values)
    Q = qubo_from_mean_variance(mu, Sigma, lam=lam, k=max_positions, penalty=2.0)
    x = solve_qubo(Q)
    if x.sum()==0:
        x[np.argmax(mu)] = 1
    return list(closes.columns[np.where(x==1)[0]])
