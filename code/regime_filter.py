import numpy as np 
import pandas as pd

# --- MARKET REGIME CLASSIFICATION ---
def r2(y: np.ndarray): 
    """signed pearson r of price against time (sign = trend direction, |r| = trend strength), bounded [-1, 1]"""
    y = y.reshape(1, -1) # each row is variable as convension
    x = np.arange(0, y.size)

    cov = np.cov(x, y, bias = True)[0, 1]
    varx_vary = np.var(x, ddof = 0) * np.var(y, ddof = 0)

    return float(np.clip(cov / np.sqrt(varx_vary), -1.0, 1.0)) if (varx_vary > 0.0) else 0.0 # return bounded value and guard against edge cases

def rolling_r2(close: pd.Series, window: int) -> pd.Series:
    """
    vectorised close.rolling(window).apply(r2, raw = True): identical values (same population cov / var), NaN for the first
    window - 1 bars and any window containing a NaN, ~100x faster than the per-window python call (backtest hot path)
    """
    y = close.to_numpy(dtype = float)
    out = np.full(y.shape, np.nan)
    if y.size < window:
        return pd.Series(out, index = close.index)

    windows = np.lib.stride_tricks.sliding_window_view(y, window) # (n - window + 1, window) view, no copy
    x = np.arange(window, dtype = float)
    x_c = x - x.mean()
    y_c = windows - windows.mean(axis = 1, keepdims = True)

    cov = (y_c @ x_c) / window
    varx_vary = np.var(x) * (y_c ** 2).mean(axis = 1)

    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        r = np.where(varx_vary > 0.0, np.clip(cov / np.sqrt(varx_vary), -1.0, 1.0), 0.0)
    r[np.isnan(windows).any(axis = 1)] = np.nan # rolling apply skips windows with NaN (min_periods = window)
    out[window - 1:] = r

    return pd.Series(out, index = close.index)

def categorise_regime(trend_strength: np.ndarray):
    """
    categorises signed r-squared into regime states 

    - x > 0.5 = trending positive
    - -0.5 <= x <= 0.5 = consolidation 
    - x < -0.5 = trending negative 
    """
    regime = np.full(trend_strength.shape, np.nan, dtype = object)
    regime[trend_strength > 0.5] = 'trend_pos'
    regime[np.abs(trend_strength) <= 0.5] = 'consolidation'
    regime[trend_strength < -0.5] = 'trend_neg'

    return regime