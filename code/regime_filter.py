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