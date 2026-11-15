from datetime import time
import numpy as np
import pandas as pd

def session_flags(index: pd.DatetimeIndex,
                  session_start: time,
                  session_tz: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    returns (session_open, in_session) boolean arrays
    - session_open: first bar at/after session_start on each calendar day (robust to missing bars, eg. no bar stamped exactly at the open)
    - in_session: bar time >= session_start on its day (False for pre-market bars)
    session_tz evaluates the clock in the exchange timezone (DST-safe, eg. 'America/New_York'); naive index treated as UTC
    """
    if session_tz is not None:
        index = (index.tz_localize('UTC') if index.tz is None else index).tz_convert(session_tz)
    
    in_session = pd.Series(index.time >= session_start)
    prior_in_session = in_session.groupby(index.date).shift(1, fill_value = False).astype(bool)
    session_open = in_session & ~prior_in_session
    
    return session_open.to_numpy(), in_session.to_numpy()

def vwap(data: pd.DataFrame,
         session_start: time = time(9, 0, 0), # time(H, m, s)
         session_tz: str | None = None):
    """
    calculates volume weighted average price

    default exchange open is 9.00am in the data's timezone (Europe/London in the backtest) unless session_tz is given
    """
    volume = data['volume']

    session_start_mask = session_flags(data.index, session_start, session_tz)[0]
    session_id = session_start_mask.cumsum() # group per day

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