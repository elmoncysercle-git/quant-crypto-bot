# bot/strategy.py
import numpy as np
import pandas as pd

def _inv_vol_weights(closes: pd.DataFrame, selected, floor=1e-9, window=20):
    sub = closes[selected].dropna()
    if len(sub) < window + 2:
        # fallback equal weights
        w = np.ones(len(selected)) / max(1, len(selected))
        return {s: float(x) for s, x in zip(selected, w)}
    rets = np.log(sub).diff().iloc[-window:]
    vol = rets.std() * np.sqrt(365)  # annualized-ish
    vol = vol.replace(0, floor).fillna(vol.median() or 1.0)
    inv = 1.0 / vol
    w = inv / inv.sum()
    return {s: float(w[s]) for s in selected}

def _apply_bounds_and_cash(weights_dict, all_symbols, min_w, max_w, cash_buffer):
    # start with zeros for everything
    w = {s: 0.0 for s in all_symbols}
    for s, val in weights_dict.items():
        w[s] = float(np.clip(val, 0.0, max_w))
    # normalize to (1 - cash_buffer)
    total = sum(w.values())
    if total > 0:
        for s in w:
            w[s] = w[s] / total * (1.0 - cash_buffer)
    # bump tiny non-zeros up to min_w
    changed = False
    for s in w:
        if 0.0 < w[s] < min_w:
            w[s] = min_w
            changed = True
    if changed:
        total = sum(w.values())
        if total > 0:
            for s in w:
                w[s] = w[s] / total * (1.0 - cash_buffer)
    return w

def _cap_turnover(new_w: dict, prev_w: dict, cap_per_asset=0.10):
    """
    Limit per-asset change to +/- cap_per_asset (e.g., 0.10 = 10 percentage points).
    """
    if not prev_w:
        return new_w
    capped = {}
    for s in new_w:
        old = float(prev_w.get(s, 0.0))
        delta = new_w[s] - old
        max_delta = cap_per_asset
        if delta > max_delta:
            capped[s] = old + max_delta
        elif delta < -max_delta:
            capped[s] = old - max_delta
        else:
            capped[s] = new_w[s]
    # renormalize to keep total <= 1 (we’ll keep same cash)
    tot = sum(max(0.0, x) for x in capped.values())
    if tot > 0:
        scale = sum(new_w.values()) / tot if tot > 0 else 1.0
        for s in capped:
            capped[s] = max(0.0, capped[s]) * scale
    return capped

def vol_target_weights(closes, selected, all_symbols, min_w=0.05, max_w=0.6,
                       cash_buffer=0.15, turnover_cap=0.10, prev_weights=None):
    base = _inv_vol_weights(closes, selected, window=20)
    bounded = _apply_bounds_and_cash(base, all_symbols, min_w, max_w, cash_buffer)
    if prev_weights:
        bounded = _cap_turnover(bounded, prev_weights, cap_per_asset=turnover_cap)
    return {s: float(bounded.get(s, 0.0)) for s in all_symbols}

