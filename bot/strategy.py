import numpy as np
import pandas as pd

def equal_weights(selected, all_symbols, min_w=0.05, max_w=0.6, cash_buffer=0.15):
    n = max(1, len(selected))
    raw_w = np.array([1.0/n if s in selected else 0.0 for s in all_symbols], dtype=float)
    # enforce min/max
    raw_w = np.clip(raw_w, 0.0, max_w)
    # renormalize to 1 - cash_buffer
    if raw_w.sum() > 0:
        raw_w *= (1.0 - cash_buffer)/raw_w.sum()
    # enforce min weight floor for non-zero
    for i, s in enumerate(all_symbols):
        if raw_w[i] > 0 and raw_w[i] < min_w:
            raw_w[i] = min_w
    # renormalize again if needed
    if raw_w.sum() > 0:
        raw_w *= (1.0 - cash_buffer)/raw_w.sum()
    return {s: float(w) for s, w in zip(all_symbols, raw_w)}
