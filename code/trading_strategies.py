"""
3 Symbol 3 strategy functions
"""
import numpy as np 
import pandas as pd
from datetime import time 
from strategy_indicators import vwap, bollinger_bands, sesssion_cumulative_volume_delta, z_score

# ----------------------------------------
# --- SESSION BRAKOUT TRADING STRATEGY ---
# ----------------------------------------
# --- TIME FRAME ---
def lower_to_higher_timeframe(data: pd.DataFrame, 
                              base_tf: str = '5m', 
                              higher_tf: str | None = None):
    """upscale data resolution function: takes in a dataframe and combines to higher timeframe"""
    # base variables
    ohlc_dict = {
        'open': 'first', 
        'high': 'max', 
        'low': 'min', 
        'close': 'last', 
        'volume': 'sum'
    }
    # logic
    if base_tf == higher_tf: # base case 
        return data
    
    data = data.resample(higher_tf).agg(ohlc_dict)
    # set resmapled timeframe at bar open -> close, then backward in merge_asof(), function gets prior timestamp 
    data.index = data.index + pd.tseries.frequencies.to_offset(higher_tf)
    if higher_tf != '5min': 
        data.rename(columns = {k: k + higher_tf for k in ohlc_dict.keys()}, inplace = True)
    return data

def concatenate_timeframe_data(min_res_data: pd.DataFrame, higher_tf_data: list[pd.DataFrame]):
    """
    """
    alligned_data = []

    for df in higher_tf_data:
        merged = pd.merge_asof(
            min_res_data.sort_index(),
            df.sort_index(),
            left_index = True, # left dataframe index as join key
            right_index = True,
            direction = 'backward'
        )
        # merge_asof carries min_res_data's columns along with df's - keep only the
        # higher-timeframe columns so they aren't duplicated in the final concat
        alligned_data.append(merged[df.columns])

    return pd.concat([min_res_data] + alligned_data, axis = 1)

def add_session_flags(data: pd.DataFrame, 
                      timezone_one: str = 'Europe/London',
                      timezone_two: str = 'America/New_York',
                      session_one: tuple[int, int] = (8, 17), 
                      session_two: tuple[int, int] = (8, 17), 
                      trading_time_zone: str = 'Europe/London'):
    """
    set data to UTC time 
    """
    # retreive timezone componenet from data and standerdise to UTC 
    utc_index = data.index.tz_localize('UTC') if data.index.tz is None else data.index.tz_convert('UTC') 

    # target sessions
    session_one_hour = utc_index.tz_convert(timezone_one).hour 
    session_two_hour = utc_index.tz_convert(timezone_two).hour

    # exchange column names
    ex_one = str.split(timezone_one, sep = '/')[1]
    ex_two = str.split(timezone_two, sep = '/')[1]

    # add flags
    data[ex_one] = (session_one_hour >= session_one[0]) & (session_one_hour < session_one[1]) # pandas type based indexing using (&, |)
    data[ex_two] = (session_two_hour >= session_two[0]) & (session_two_hour < session_two[1])

    # convert index to local clock time
    if str(utc_index.tz) != trading_time_zone:
        data.set_index(utc_index.tz_convert(trading_time_zone), inplace = True)
    else:
        data.set_index(utc_index, inplace = True)

    return data

def valid_weekday(data: pd.DataFrame) -> pd.Series:
    """
    validates if timestamp is weekend or not
    """
    return pd.Series(data.index.day_of_week < 5, index = data.index)

# --- signal logic ---
def validate_session_overlap_(data: pd.DataFrame, 
                              exchange_one: str = 'London', 
                              exchange_two: str = 'New_York'):
    """
    filters data and forward fills data that proceed 1 hour of the session overlap
    """
    overlap = data[exchange_one] & data[exchange_two]
    overlap_start = overlap & ~overlap.shift(1, fill_value = False) # check overlap is first candle 

    confirm_time = data.index.to_series().where(overlap_start) + pd.Timedelta(value = 1, unit = 'hours') # first candle in overlap offset by 1h, ie the first 1h bar in overlap time 
    confirm_mask = data.index.isin(confirm_time.dropna()) # array of strictley daily 1h overlaps and associated UTC time

    confirm = data[['open1h', 'high1h', 'low1h', 'close1h']].copy() # filter data
    confirm[~confirm_mask] = np.nan # set all non first 1h overlap values to nan
    confirm = confirm.ffill() # forward fill from first 1h overlap times

    return confirm, overlap

def candle_threshold_(overlap_1h_data: pd.DataFrame, 
                      candle_body_ratio_threshold: float = 0.1) -> tuple[pd.Series, bool]:
    """
    multi-timeframe strategy logic for long/short
    """
    body = overlap_1h_data['close1h'] - overlap_1h_data['open1h']
    candle_range = overlap_1h_data['high1h'] - overlap_1h_data['low1h']
    ratio = (body.abs() / candle_range) > candle_body_ratio_threshold
    
    return (body, ratio)

def confluence_(trade_conditions: list[pd.Series]) -> pd.Series:
    """
    elementwise series boolean conditions
    """
    return pd.concat(trade_conditions, axis = 1).all(axis = 1)

def session_breakout(data: pd.DataFrame, 
                     body_ratio_threshold: float = 0.5, 
                     weekend_trading: bool = True) -> tuple[pd.Series, np.ndarray]:
    """
    forex strategy 
    When NY and LND sessions are both overlapping (live)
    - LONG: 8h bar negative and current price > 8h low (unbroken); 4h bar positive and current price > 4h low (unbroken); first 1h bar of overlapping session is green and candle body to candle length ratio > 0.5
    - SHORT: 8h bar positive and current price < 8h high (unbroken); 4h bar negative and current price < 4h high (unbroken); first 1h bar of overlapping session is red and candle body to candle length ratio > 0.5
    """
    # --- sessions check ---
    data = add_session_flags(data)

    # --- time logic ---
    confirm, overlap = validate_session_overlap_(data) # confirm validates 1h bars

    # --- confluence ---
    candle_direction, candle_ratio = candle_threshold_(confirm, body_ratio_threshold)

    # weekened constraint
    weekday = pd.Series(True, index = data.index)
    if not weekend_trading:
        weekday = valid_weekday(data)

    # long states
    candle_pos = candle_direction > 0
    val_long = confluence_([candle_pos, candle_ratio, weekday])

    # short states
    val_short = confluence_([~candle_pos, candle_ratio, weekday])

    # --- signal CRYPTO ---
    long_sig = (
        overlap
        & (data['close'] < data['open8h']) # neg 8h bar
        & (data['close'] > data['open4h']) # pos 4h bar 
        & val_long # pos 1h bar + confluence
    )
    short_sig = (
        overlap
        & (data['close'] > data['open8h']) # pos 8h bar 
        & (data['close'] < data['open4h']) # neg 4h bar 
        & val_short # neg 1h bar + confluence 
    )

    data['signal'] = long_sig | short_sig
    signal_direction = np.where(long_sig, 1, np.where(short_sig, -1, 0)) # vectorised signal

    return (data['signal'], signal_direction)

# -----------------------------
# --- VWAP TRADING STRATEGY ---
# -----------------------------
def standerdise_join_data(base_data: pd.DataFrame, 
                          target_data: pd.DataFrame, 
                          target_data_freq = '5min'):  
    """
    alligns a target data frame with timezone naive index to base data, target data may be of lower time frame (and scaled to base)
    """
    if (not isinstance(base_data.index, pd.DatetimeIndex)) or (not isinstance(target_data.index, pd.DatetimeIndex)):
        print('dataframes must have date time index')
        return 
    if target_data.index.tz is None:
        target_data.set_index(target_data.index.tz_localize('UTC'), inplace = True)

    # check if data needs to be uspscaled to target time frame
    target_data = lower_to_higher_timeframe(target_data, target_data_freq, '5min') # upscale data if lower resolution

    drop_cols = ['open_', 'high_', 'low_', 'close_', 'volume_']
    target_data = base_data.join(target_data, how = 'left', lsuffix = '_').ffill().dropna().drop(columns = drop_cols) # convert target data index to tz naive
    target_data.set_index(target_data.index.tz_convert('Europe/London'), inplace = True) # data already timezone aware (UTC) be default so just use timzone convert
    
    return target_data

def vwap_breakout(data: pd.DataFrame,
                  session_start: time = time(9, 5, 0),
                  retest_distance: int = 1,
                  volume_window: int = 1,
                  weekend_trading: bool = False,
                  min_bars_after_open: int = 12) -> tuple[pd.Series, np.ndarray]:
    """
    equity strategy
    n bars have elapsed from new session
    - LONG: price > VWAP, prior price < VWAP and price high <= VWAP before price closes above
    - SHORT: price < VWAP, prior price > VWAP and price low >= VWAP before price closes below

    min_bars_after_open blocks entries in the first n bars of the session, where VWAP is still
    forming off few observations and prone to noisy breakout/retest signals (default 12 bars @ 5min = first 1h)
    """
    # indicator
    data['vwap'] = vwap(data, session_start)

    close = data['close']
    high = data['high']
    low = data['low']
    volume = data['volume']
    vwap_ = data['vwap']
    
    # signal 
    long_test_bar = (close < vwap_) & (high >= vwap_)
    long_confirm_bar = close > vwap_ 

    short_test_bar = (close > vwap_) & (low <= vwap_)
    short_confirm_bar = close < vwap_
    
    long_signal = (long_confirm_bar & long_test_bar.shift(retest_distance))
    short_signal = (short_confirm_bar & short_test_bar.shift(retest_distance))

    # confluence 
    high_volume = pd.Series(True, index = data.index)
    if volume_window > 1:
        avg_volume = volume.shift(1).rolling(window = volume_window).mean() # avg volume (not including current observation)
        high_volume = volume > avg_volume

    # weekened constraint
    weekday = pd.Series(True, index = data.index)
    if not weekend_trading:
        weekday = valid_weekday(data)

    # session warm-up: bars elapsed since session_start, same session grouping as vwap()
    session_id = (data.index.time == session_start).cumsum()
    bars_since_open = data.groupby(session_id).cumcount()
    warmed_up = bars_since_open >= min_bars_after_open

    val_long_signal = confluence_([long_signal, high_volume, weekday, warmed_up])
    val_short_signal = confluence_([short_signal, high_volume, weekday, warmed_up])

    # all combined signals
    data['signal'] = val_long_signal | val_short_signal
    signal_direction = np.where(val_long_signal, 1, np.where(val_short_signal, -1, 0))

    return (data['signal'], signal_direction)

# ----------------------------------------
# --- BOLLINGER BANDS TRADING STRATEGY ---
# ----------------------------------------
def bollinger_band_cdv(data: pd.DataFrame, 
                       high_scvd_threshold: float = 2.0, 
                       low_scvd_threshold: float = 2.0, 
                       weekend_trading: bool = True) -> tuple[pd.Series, np.ndarray]: 
    """
    crypto strategy (trend continuation)
    - LONG: close > higherbb and standerdised cumulative session volume > upper threshold 
    - SHORT: close < lowerbb and standerdised cumulative session volume < lower threshold
    """
    # indicators 
    bb = bollinger_bands(data)
    data['lower_bb'] = bb['lower_bb']
    data['higher_bb'] = bb['higher_bb']
    data['session_cumulative_volume_delta'] = sesssion_cumulative_volume_delta(data)

    close = data['close']
    lower_bb = bb['lower_bb']
    higher_bb = bb['higher_bb']
    scvd = data['session_cumulative_volume_delta']
    z_scvd = z_score(scvd)

    # signal (trend continuation)
    long_signal = (close > higher_bb) & (z_scvd > high_scvd_threshold)
    
    short_signal = (close < lower_bb) & (z_scvd < -low_scvd_threshold)

    # confirm signals
    weekday = pd.Series(True, index = data.index)
    if not weekend_trading:
        weekday = valid_weekday(data)
    val_long_signal = confluence_([long_signal, weekday])
    val_short_signal = confluence_([short_signal, weekday])

    # all combined signals 
    data['signal'] = val_long_signal | val_short_signal
    signal_direction = np.where(val_long_signal, 1 ,np.where(val_short_signal, -1, 0))

    return (data['signal'], signal_direction)