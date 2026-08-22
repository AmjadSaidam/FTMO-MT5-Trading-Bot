import numpy as np 
import pandas as pd

def sharpe_ratio(returns: np.ndarray, crypto = False): 
    """
    sharpe ratio assuming daily returns 
    """
    ror = returns[-1]
    mu = returns.mean(axis = -1)
    std = returns.std(axis = -1)
    sharpe = (ror - mu) / std
    mult = 365 if crypto else 252 

    return float(sharpe) * mult
