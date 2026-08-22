import numpy as np 
import math

def prob_consec_win(p: float, n: int, k: int = 10, consec_loss: float = False): 
    """
    given strategy win rate, calculates the probability of relaising k loosing trades 
    
    let p = 1-p to calculate probability of consecutive losses 
    """
    if consec_loss: 
        return p**k
    
    return  math.comb(n, k) * p**(n - k) * (1 - p)**k 

def prob_until_breakdown(r: float = 0.01, max_frac_loss: float = 0.03): 
    """
    given percentage of account risked, r, calculates the number of loosing trades required to excced maximum 
    percentage loss 
    """

    return max_frac_loss / r 