import numpy as np 
import pandas as pd

# --- MARKET REGIME CLASSIFICATION ---
def r2(y: np.ndarray): 
    """
    the ordinary least squares estimator of the slope coefficient for a simple linear model

    model is intended to be used as a regime filter of market direction in the ensemble model
    """
    y = y.reshape(1, -1) # each row is variable as convension
    x = np.arange(0, y.size)
    
    return float(np.cov(x, y, bias = True)[0, 1] ** 2 / np.var(x, ddof = 0) * np.var(y, ddof = 0))

def categorise_regime(trend_strength: np.ndarray):
    """
    categorises recent price action into states, 
    - -0.5 <= x <= 0.5 = consolidation 
    - x < -0.5 = trending negative 
    - x > 0.5 = trending positive
    """
    regime = np.full(trend_strength.shape, np.nan, dtype = object)
    regime[np.abs(trend_strength) <= 0.5] = 'consolidation'
    regime[trend_strength < -0.5] = 'trend_neg'
    regime[trend_strength > 0.5] = 'trend_pos'

    return regime