import numpy as np

def equal_weights(selected, all_symbols, min_w=0.05, max_w=0.6, cash_buffer=0.15):
    n = max(1, len(selected))
    w = np.array([1.0/n if s in selected else 0.0 for s in all_symbols], dtype=float)
    w = np.clip(w, 0.0, max_w)
    if w.sum()>0:
        w *= (1.0 - cash_buffer)/w.sum()
    for i in range(len(all_symbols)):
        if 0 < w[i] < min_w:
            w[i] = min_w
    if w.sum()>0:
        w *= (1.0 - cash_buffer)/w.sum()
    return {s: float(x) for s,x in zip(all_symbols, w)}

