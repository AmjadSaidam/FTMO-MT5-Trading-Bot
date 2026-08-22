from datetime import time
import numpy as np
import pandas as pd

def vwap(data: pd.DataFrame, 
         session_start: time = time(9, 0, 0)): # time(H, m, s)
    """
    calculates volume weighted average price

    default exchange open is 9.00am Europe/London timezone
    """
    volume = data['volume']

    session_start_mask = data.index.time == session_start
    session_id = session_start_mask.cumsum()

    ohlc4 = data.loc[:, ['open', 'high', 'low', 'close']].mean(axis = 1)
    price_qty = ohlc4 * volume
    
    cum_price_qty = price_qty.groupby(session_id).cumsum() 
    cum_qty = volume.groupby(session_id).cumsum()
    
    return cum_price_qty / cum_qty

def bollinger_bands(data: pd.DataFrame,
                    sliding_window: int = 20,
                    low_standard_deviation_multiplier: float = 2.0, 
                    high_standard_deviation_multiplier: float = 2.0):
    """
    """
    close = data['close']
    rolling_mean = close.rolling(sliding_window).mean()
    rolling_std = close.rolling(sliding_window).std()
    lower_bb = rolling_mean - rolling_std * low_standard_deviation_multiplier 
    upper_bb = rolling_mean + rolling_std * high_standard_deviation_multiplier
    
    return {
        'rolling_mean': rolling_mean, 
        'rolling_std': rolling_std, 
        'lower_bb': lower_bb,
        'higher_bb': upper_bb
    }

def sesssion_cumulative_volume_delta(data: pd.DataFrame, 
                                     session_start: time = time(9, 0, 0)):
    """
    """
    volume = data['volume']
    signed_volume = volume * np.sign(data['close'] - data['open']) # candle-direction proxy for buy/sell taker volume (no trade-side data in OHLCV bars)

    session_start_mask = data.index.time == session_start
    session_id = session_start_mask.cumsum()

    cvd = signed_volume.groupby(session_id).cumsum()

    return cvd

def z_score(price_series: pd.Series, 
            sliding_window: int = 20):
    """
    """
    mu = price_series.rolling(sliding_window).mean()
    std = price_series.rolling(sliding_window).std()

    return (price_series - mu) / std