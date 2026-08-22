"""
"""
# general imports 
import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
# type casting 
from functools import partial 
from typing import Callable
# .py importsGix 
from risk_management import atr, consecutive_loss_threshold
from regime_filter import r2, categorise_regime

# --- POSITION LOGIC ---
def risk_scaling(fraction_risked: float = 0.01, signal_number: int = 1):
    """
    scales risk by number of signals ensuring max risk constraint not exceeded 
    """
    return fraction_risked / signal_number

def position_sizing(asset_price: float, 
                    stop_distance: float, 
                    account_balance: float, 
                    fraction_risk: float = 0.01):
    """
    calculates required trade size in $ to loose exactley risk% when stopped out
    """
    qty_risked = account_balance * fraction_risk # max loss in $
    position_size = qty_risked / stop_distance # per unit loss in $ 
    trade_value = position_size * asset_price # required $ value in order to loose at least risk% if stopped out (decreasing for larger sl distance)
    
    leverage = trade_value / account_balance # value traded as % of account balance (fraction_risk * asset_price / stop_distance)

    return trade_value, leverage

# --- EQUITY ---
def account_equity(value, asset_return): 
    return value * (1 + asset_return)

# --- TRADING STRATEGY PIPELINE ---
SignalFunc = Callable[..., tuple[pd.Series, np.ndarray]]

def backtest_strategy(data: pd.DataFrame,
                      signal_fc: SignalFunc,
                      signal_kwargs: dict | None = {},
                      forex: bool = False,          # logs lots (forex related fields)
                      account_balance: float = 1e4,
                      risk_per_trade: float = 0.01,
                      atr_multiplier: int = 2.0,
                      risk_reward: float = 2.0,
                      consecutive_loss_before_skip: int | None = None,
                      ensamble_signal_validation: RandomForestClassifier | None = None,
                      leverage_multiplier: int = 1, 
                      # margin_percentage: float = 0.30, 
                      slippage_dollar: float = 0.0,
                      spread_dollar: float = 0.0,
                      commission_pct: float = 0.00004, 
                      standard_lot: float = 1e5, 
                      dollar_commission_lot: float = 5.0, 
                      trading_time_zone: str = 'Europe/London') -> dict:
    """
    signal agnostic (signal as input not calculated in logic) atr backtest boilerplate function. 

    commission_pct - float: the default one-way percentage commission for trade entry, is commission type is set to 'variable' this a becomes one-way percentage/volume commission
    """
    # ensure timezone is local clock time 
    if data.index.tz is None:
        data.set_index(data.index.tz_localize('UTC').tz_convert(trading_time_zone), inplace = True) # if naive localize and set to local clock time 
    if str(data.index.tz) != trading_time_zone:
        data.set_index(data.index.tz_convert(trading_time_zone), inplace = True) # if aware convert to local clock time 

    tot_trades: int = 0
    skipped_trades: int = 0
    winning_trades: int = 0

    # --- sessions check ---
    signal, signal_direction = signal_fc(data, **(signal_kwargs or {})) # if signal_kwargs empty returns False/None default to {}

    # --- backtest logic (event based analysis) ---
    # vars (ohlc from base timeframe)
    n = len(data)
    close_ = data['close'].to_numpy()
    vol_atr = atr(data)
    signal = signal.to_numpy() # use signal_fc's return value directly rather than re-reading data['signal'], which only exists if signal_fc mutates data as a side effect
    day = data.index.normalize().to_numpy() # convert to midnight time

    # default data arrays 
    position = np.zeros(n, dtype = int) # $ traded
    adjusted_returns = np.zeros(n, dtype = float) # adjusted_returns =  returns - transaction costs
    trade_rr = np.zeros(n, dtype = float) # trade risk_reward
    equity = np.full(n, account_balance, dtype = float) # rolling equity

    # default state based vars
    in_trade = False
    exit_price = None
    trade_direction = 0 

    traded_today = False 
    trade_time = None
    current_day = day[0]

    consec_loss = 0
    skip_next_day = False # set when a losing streak breaches the threshold, consumed on the following day's boundary

    stop_price = target_price = leverage = entry_day = 0.0

    # model ensemble features 
    open = data['open']
    high = data['high']
    low = data['low']
    close = data['close']
    volume = data['volume']
    rolling_window = 30
    atr_pct = vol_atr / data['close'].to_numpy()
    r_sq = data['close'].rolling(rolling_window).apply(r2, raw = True) # raw transfers input data['close'] to numpy array type
    regime = categorise_regime(r_sq)
    breakout_conv = ((close - open) / (high - low)).to_numpy()
    rel_volume = volume.rolling(rolling_window).mean().to_numpy()

    # ensemble model default parameters
    label_trd_w = np.full(n, fill_value = None)
    ml_label_predict = np.full(n, fill_value = None)

    # logic
    for t, df in enumerate(data.itertuples()):
        if t == 0: # no prior bar to reference (avoids numpy negative-index wraparound on signal[-1]/close_[-1])
            continue

        # check skip flag is false so signals are valid for trading
        if day[t] != current_day:
            current_day = day[t]
            if skip_next_day: # losing streak breached threshold on a prior day - skip this day's signals
                traded_today = True
                skipped_trades += 1
                skip_next_day = False
            else:
                traded_today = False

        if in_trade:
            # clear prior trade metadata
            exit_price = None

            # long
            if trade_direction == 1: 
                hit_stop = df.low <= stop_price # sell at bid (must cross sl) 
                hit_target = df.high > target_price # sell at bid (must be > tp, less favourable)
            # short 
            else: 
                hit_stop = df.high >= stop_price
                hit_target = df.low < target_price
            # end of day exit 
            eod_exit = day[t] > entry_day

            # exit logic
            if hit_stop:
                exit_price = stop_price
                consec_loss += 1
                if consecutive_loss_before_skip is not None and consecutive_loss_threshold(consec_loss, consecutive_loss_before_skip):
                    consec_loss = 0 # after skip reset counter
                    skip_next_day = True
                label_trd_w[trade_time] = 0
            elif hit_target:
                exit_price = target_price
                consec_loss = 0
                label_trd_w[trade_time] = 1
            elif eod_exit:
                exit_price = df.open
                label_trd_w[trade_time] = 1 if trade_direction * (exit_price - entry_price) > 0 else 0
            
            # logs
            price_for_return = exit_price if (exit_price is not None) else df.close
            adjusted_returns[t] = leverage * trade_direction * (price_for_return - close_[t-1]) / close_[t-1]
            winning_trades += 1 if (exit_price is not None and trade_direction * (exit_price - entry_price) > 0) else 0
            position[t] = trade_direction
            trade_rr[t] = trade_direction * (exit_price - entry_price) / stop_distance if (exit_price is not None) else 0
            equity[t] = account_equity(equity[t-1], adjusted_returns[t])

            if exit_price is not None:
                in_trade = False
        
        # not in trade 
        else: 
            equity[t] = equity[t-1]

            # guard against signal at session boundary
            # True when bar t is on the same day as bar t-1 (signal and entry share a session);
            # checks current date is equal to prior date -> so signal could occor 1 bar before new session and entry is last bar in session, otherwise signal is last bar and entry is new session (invalid)
            new_session = current_day == day[t-1] 

            # enter trade if last bar was valid signal (must populate trade variables) 
            # limit trades to one per day (all other signals ignored), so tarded_today must be false
            if signal[t-1] and not np.isnan(vol_atr[t-1]) and vol_atr[t-1]>0 and new_session:

                # ensemble point prediction (used only in out of sample testing)
                # must be instentiated and trained on data otherwise pass
                if ensamble_signal_validation is not None:
                    if t >= 30: # check window large enough
                        point_features = pd.DataFrame({
                            'atr_pct': [atr_pct[t - 1]], 
                            'regime': [regime[t - 1]], 
                            'breakout_conv': [breakout_conv[t - 1]],
                            'rel_volume': [rel_volume[t - 1]] 
                        })
                        valid_trade = ensamble_signal_validation.predict(point_features)[0]
                        if not valid_trade: # model predicted trade unprofitable 
                            skipped_trades += 1
                            traded_today = True
                        
                        # log prediction (required for model evaluation)
                        ml_label_predict[t] = int(valid_trade)

                # traded_today must be False, true when new day or no signal generated
                if not traded_today: 
                    
                    trade_direction = int(signal_direction[t-1])

                    # entry
                    entry_price = df.open

                    # stop loss
                    stop_distance = vol_atr[t-1] * atr_multiplier
                    stop_price = entry_price - trade_direction * stop_distance

                    # take_profit 
                    target_price = entry_price + trade_direction * stop_distance * risk_reward

                    # position
                    trade_value, leverage = position_sizing(df.open, stop_distance, equity[t], risk_per_trade)
                    trade_value *= leverage_multiplier # scale trade_value for variable commission
                    leverage *= leverage_multiplier # scale leverage for for fixed (None) commission

                    # transaction costs
                    slippage_pct = slippage_dollar / df.close
                    spread_pct = spread_dollar / df.close
                    
                    # fixed fees for crypto/indicies/commodities and fx
                    if forex:
                        entry_cost = (slippage_pct + spread_pct) + (trade_value / df.open / standard_lot * dollar_commission_lot / equity[t]) * 2  # position size / standard lot * 5, already $/equity so no leverage scaling
                    else:
                        entry_cost = (slippage_pct + spread_pct + commission_pct) * 2 * leverage # *2 for round trip cosst

                    # --- immediately resolved trades --- 
                    # does current bar hit trade
                    # high/low range can still breach the stop or target before the next bar is seen)
                    if trade_direction == 1:
                        entry_hit_stop = df.low <= stop_price
                        entry_hit_target = df.high > target_price
                    else:
                        entry_hit_stop = df.high >= stop_price
                        entry_hit_target = df.low < target_price

                    if entry_hit_stop:
                        exit_price = stop_price
                        consec_loss += 1
                        if consecutive_loss_before_skip is not None and consecutive_loss_threshold(consec_loss, consecutive_loss_before_skip):
                            consec_loss = 0 # after skip reset counter
                            skip_next_day = True
                        label_trd_w[t] = 0 # ml label: lossing trade
                    elif entry_hit_target:
                        exit_price = target_price
                        consec_loss = 0
                        label_trd_w[t] = 1 # ml label: winning trade
                    else:
                        exit_price = None

                    # --- logs ---
                    # update default arrays
                    price_for_return = exit_price if (exit_price is not None) else df.close
                    adjusted_returns[t] = leverage * trade_direction * (price_for_return - entry_price)/entry_price - entry_cost # rt = leverage*trade_direction*return - leverage*cost (cost only on entry)
                    winning_trades += 1 if (exit_price is not None and trade_direction * (exit_price - entry_price) > 0) else 0
                    position[t] = trade_direction
                    equity[t] = account_equity(equity[t-1], adjusted_returns[t])
                    trade_rr[t] = trade_direction * (exit_price - entry_price) / stop_distance if (exit_price is not None) else 0

                    # update default vars
                    entry_day = day[t]
                    in_trade = exit_price is None

                    trade_time = t
                    tot_trades += 1

                    # ensure limit of one trade per day
                    traded_today = True
        
    # append default var arrays to dataframe
    data = data.assign(
        position = position, 
        adjusted_returns = adjusted_returns, 
        equity = equity, 
        trade_rr = trade_rr,
        atr_pct = atr_pct,                  # feature 1 
        regime = regime,                    # feature 2
        breakout_conv = breakout_conv,      # feature 3 
        rel_volume = rel_volume,            # feature 4 
        ml_label = label_trd_w, 
        ml_label_predict = ml_label_predict
    )

    return {
        'data': data, 
        'total_trades': tot_trades, 
        'skipped_trades': skipped_trades,
        'winning_trades': winning_trades
    }