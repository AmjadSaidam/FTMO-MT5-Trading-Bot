"""
Live MetaTrader5 trade execution logic 

polling function that request and send trade information to MetaTrader5 terminal 
"""
import os
import sys
from collections import defaultdict
import pandas as pd
import numpy as np
from datetime import datetime
import MetaTrader5 as mt5
import time
import logging
from dotenv import load_dotenv
load_dotenv()

# code/*.py modules cross-import each other by bare module name (eg. "from risk_management import atr"),
# so both the project root and code/ itself must be on sys.path (mirrors app/dashboard.py)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, 'code')):
    if p not in sys.path:
        sys.path.insert(0, p)
# .py
import mt5_connector as mt5_conn
import mt5_errors as mt5_err
import logging_config as log_cfg
from code.trading_strategies import (session_breakout, vwap_breakout, bollinger_band_cdv,
                                     lower_to_higher_timeframe, concatenate_timeframe_data)
from code.backtest_engine import SignalFunc, risk_scaling, position_sizing
from code.regime_filter import r2, categorise_regime
from code.risk_management import atr, consecutive_loss_threshold, RandomForestVol
import code.optimisation as opt

# strategies requiring 1h/4h/8h columns merged onto the base timeframe data, see app/dashboard.py
MULTI_TIMEFRAME_STRATS = {session_breakout}

def add_multi_timeframe_columns(strat: SignalFunc, 
                                data: pd.DataFrame, 
                                base_tf: str = '5min') -> pd.DataFrame:
    """builds and merges 1h/4h/8h OHLC columns onto base_tf data for strategies that require them (e.g. session_breakout)"""
    if strat not in MULTI_TIMEFRAME_STRATS:
        return data

    h1_data = lower_to_higher_timeframe(data, base_tf, '1h')
    h4_data = lower_to_higher_timeframe(data, base_tf, '4h')
    h8_data = lower_to_higher_timeframe(data, base_tf, '8h')

    return concatenate_timeframe_data(data, [h1_data, h4_data, h8_data]).dropna()

def _reconnect_if_disconnected():
    """MT5RatesError/PositionError/OrderError can mean a lost terminal connection or a genuine
    request error; only reconnect when terminal_info() confirms the IPC link is actually down"""
    if not mt5_conn.is_connected():
        logging.error('MT5 terminal connection lost, reconnecting')
        time.sleep(5)
        mt5_conn.connect_account()

def run_live_loop(symbols_strats: dict[str, SignalFunc],
                  strat_ids: dict[str, int],
                  symbols_strats_kwargs: dict[str, dict] | None = None,
                  strat_weights: list[float] = [1 / 3] * 3,
                  grid_search_atrs: np.ndarray = np.linspace(1.0, 3.0, 5), 
                  grid_search_rrs: np.ndarray = np.linspace(1.0, 3.0, 5)):
    """multi-strategy logic"""
    log_cfg.setup_logging() # call once (dont re-initialise)
    while True:
        try:
            mt5_conn.connect_account()
            break
        except mt5_err.MT5ConnectionError as e:
            logging.error(f'{e} \ninitial connection failed, retrying in 5s')
            time.sleep(5)
    logging.info('connected to MT5 account, starting live loop')

    # account defaults 
    initial_account_balance = os.environ.get('META_INITIAL_ACCOUNT_BALANCE')
    initial_strategy_allocation = np.array(strat_weights) * float(initial_account_balance)
    strategy_equity_dict = {symbol: initial_strategy_allocation[strat_no] for strat_no, symbol in enumerate(symbols_strats.keys())} # initial strategy allocation in $
    strategy_live_trade = {symbol: False for symbol in symbols_strats.keys()} # track signal per symbol (max 1 order per symbol per day )
    strategy_open_tickets = {symbol: [] for symbol in symbols_strats.keys()} # order history
    strategy_last_bar_time = {symbol: None for symbol in symbols_strats.keys()} # track most recent bar (only request data and run strategy logic once per new bar)

    # schedule times
    wfa_in_sample_length = int(os.environ.get('META_WFA_IN_SAMPLE_DAYS'))
    run_wfa = False
    time_last_wfa = None 
    prior_day_time = datetime.today().now()
    new_day = False

    # wfa 
    strategy_opt_atr = {}
    strategy_opt_rr = {}

    # ml risk management
    feature_window = int(os.environ.get('META_ML_FEATURE_WINDOW'))
    strategy_models = {}

    # performance
    strategy_skip_trade = {symbol: False for symbol in symbols_strats.keys()}
    strategy_consec_loss = {symbol: 0 for symbol in symbols_strats.keys()}

    # strategy polling
    while True:
        try:
            current_day_time = datetime.today().now()
            # new day 
            if (current_day_time - prior_day_time).days == 1:
                new_day = True
                prior_day_time = current_day_time
            else:
                new_day = False

            strat_equity = sum(strategy_equity_dict.values())
            
            # wfa 
            if time_last_wfa is None: # on first iteration
                run_wfa = True
            else:
                time_diff = (current_day_time - time_last_wfa).days 
                if time_diff > wfa_in_sample_length: 
                    run_wfa = True
            if run_wfa:
                for symbol, strat in symbols_strats.items():
                    # data, drop the still-forming last bar so training only sees closed bars
                    # copy_rates_from anchors on "now" (default date_from) and counts backward,
                    # so count alone covers the wfa_in_sample_length-day window
                    wfa_bar_count = wfa_in_sample_length * 24 * 60 // 5
                    wfa_data = mt5_conn.get_latest_bars_dates(symbol, count = wfa_bar_count).iloc[:-1]
                    wfa_data = add_multi_timeframe_columns(strat, wfa_data)
                    # run grid search
                    gd_params = opt.grid_search_params(data = wfa_data,
                                                       strat = strat,
                                                       strat_kwargs = (symbols_strats_kwargs or {}).get(symbol) or {},
                                                       account_balance = strat_equity,
                                                       atr_multilpiers = grid_search_atrs,
                                                       risk_rewards = grid_search_rrs)
                    gd_res = opt.run_grid_search(gd_params)
                    gd_metadata = opt.grid_search_statistics(gd_res, grid_search_atrs, grid_search_rrs)
                    
                    # grid search parameters
                    gd_opt_frame: pd.DataFrame = gd_metadata['opt_df']

                    # train ensemble model
                    ml_features_in_sample = gd_opt_frame.loc[:, ['atr_pct', 'regime', 'breakout_conv', 'rel_volume']]
                    ensemble = RandomForestVol(ml_features_in_sample, gd_opt_frame['ml_label'], 'regime')
                    ensemble.train_model(train_split = 1) # trained model appended to class attribute

                    # optimal values 
                    strategy_models[symbol] = ensemble
                    strategy_opt_atr[symbol] = gd_metadata['opt_atr']
                    strategy_opt_rr[symbol] = gd_metadata['opt_rr']

                    # update timeframe
                    time_last_wfa = current_day_time

            for symbol, strat in symbols_strats.items():
                # eod exit
                if new_day:
                    # enable trading
                    strategy_skip_trade[symbol] = False
                    # strategy ticket
                    ticket = strategy_open_tickets.get(symbol)
                    if ticket:
                        positions = mt5_conn.get_positions(ticket = ticket[-1])
                        # sl/tp hit
                        if not positions:
                            # check trade outcome
                            pnl = mt5_conn.get_deal_profit(ticket[-1])
                            if pnl > 0:
                                strategy_consec_loss[symbol] = 0
                            else:
                                strategy_consec_loss[symbol] += 1
                            log_cfg.log_event('position_closed', symbol = symbol, ticket = ticket[-1],
                                              reason = 'sl_tp_hit', pnl = pnl)
                            # symbol orders history to 0
                            strategy_open_tickets[symbol] = [] # positions returns (), clear rather than delete the key so future orders can still append
                        else:
                            # trade-live force close
                            mt5_conn.close_position(positions[-1]) # close most recent position
                            pnl = mt5_conn.get_deal_profit(ticket[-1])
                            # check trade outcome
                            if pnl > 0:
                                strategy_consec_loss[symbol] = 0
                            else:
                                strategy_consec_loss[symbol] += 1
                            log_cfg.log_event('position_closed', symbol = symbol, ticket = ticket[-1],
                                              reason = 'eod_force_close', pnl = pnl)
                            strategy_open_tickets[symbol] = [] 
                        strategy_live_trade[symbol] = False # flat going into the new day, allow re-entry

                live_trade = strategy_live_trade[symbol]

                # signal logic (ran once ber new bar)
                if not live_trade:
                    # skip until a new bar has closed, avoids re-fetching/re-evaluating every 1s poll
                    latest_bar_time = mt5_conn.get_latest_bar_time(symbol)
                    if latest_bar_time == strategy_last_bar_time[symbol]:
                        continue # will not run subsequent code 
                    strategy_last_bar_time[symbol] = latest_bar_time

                    # pull data. last row is the just-opened, still-forming bar - drop it so
                    # signal/features are evaluated on the last CLOSED bar, matching
                    # backtest_engine.py's timing (signal/features at t-1, entry at open of t)
                    # copy_rates_from anchors on "now" and counts backward, so count alone covers
                    # bars since midnight; floor at feature_window so rolling features aren't all-NaN early in the day
                    minutes_since_midnight = current_day_time.hour * 60 + current_day_time.minute
                    bars_elapsed_from_new_day = max(minutes_since_midnight // 5, feature_window + 1)
                    data = mt5_conn.get_latest_bars_dates(symbol, count = bars_elapsed_from_new_day)
                    tob = data.iloc[-1, :] # forming bar - its open is the entry reference price
                    closed_data = data.iloc[:-1] # drop forming bar (last entry - values at snapshot)

                    # ml model features
                    atr_pct = (atr(closed_data) / closed_data.close.to_numpy())[-1].item()
                    regime = categorise_regime(
                        closed_data.close.rolling(feature_window).apply(r2, raw = True)
                    )[-1]
                    breakout_conv = ((closed_data.close - closed_data.open) / (closed_data.high - closed_data.low)).iloc[-1]
                    rel_volume = closed_data.volume.rolling(feature_window).mean().iloc[-1]
                    point_features = {
                        'atr_pct': [atr_pct],
                        'regime': [regime],
                        'breakout_conv': [breakout_conv],
                        'rel_volume': [rel_volume],
                    }
                    # guard by validating signal
                    ensemble: RandomForestVol = strategy_models[symbol]
                    valid_trade = ensemble.model_predict(point_features)
                    if consecutive_loss_threshold(strategy_consec_loss[symbol], int(os.environ.get('META_ML_CONSEC_LOSS_BEFORE_TRADE_SKIP'))):
                        strategy_skip_trade[symbol] = True

                    # asset vol
                    symbol_atr = atr(closed_data)[-1].item()

                    if not strategy_skip_trade[symbol]:
                        # generate signal off the last closed bar
                        strat_data = add_multi_timeframe_columns(strat, closed_data)
                        signal_series, signal_direction_arr = strat(strat_data, **((symbols_strats_kwargs or {}).get(symbol) or {}))
                        signal, signal_direction = signal_series.iloc[-1], signal_direction_arr[-1]
                        if signal:
                            # signal log 
                            log_cfg.log_event('signal_generated', symbol = symbol, strategy = strat.__name__,
                                              direction = int(signal_direction), valid_trade = bool(valid_trade))
                        if signal and valid_trade:
                            # type
                            type = 'LONG' if (signal_direction == 1) else 'SHORT' if (signal_direction == -1) else 0
                            # magin id
                            trade_magic_id = strat_ids.get(symbol)
                            # stop and tp, priced off the new bar's open (the actual entry fill reference)
                            sl_dist = symbol_atr * strategy_opt_atr[symbol]
                            sl = tob.open - sl_dist * signal_direction
                            tp = tob.open + sl_dist * strategy_opt_rr[symbol] * signal_direction
                            # strat account balance
                            account_balance = sum(strategy_equity_dict.values()) # assumes all position closed (100% free margin)
                            strat_account_balance = strategy_equity_dict.get(symbol)
                            strat_weight = strat_account_balance / account_balance
                            fraction_risk = float(os.environ.get('META_TOTAL_RISK_ALLOCATION')) # per strategy
                            # volume
                            trade_value = position_sizing(tob.open, sl_dist, account_balance, fraction_risk)[0] # position notional in account currency
                            if mt5.symbol_info(symbol).currency_base is not None:
                                volume = trade_value / tob.open / 1e5 # units of base currency -> standard forex lots
                            else:
                                volume = trade_value / tob.open
                            # order to MT5
                            order = mt5_conn.send_order(trade_magic_id, symbol, volume, sl, tp, type = type)
                            strategy_live_trade[symbol] = True
                            # symbol order id
                            strategy_open_tickets[symbol].append(order.order)
                            # order sent/executed log
                            log_cfg.log_event('order_sent', symbol = symbol, strategy = strat.__name__,
                                              ticket = order.order, type = type, volume = volume,
                                              sl = sl, tp = tp, magic = trade_magic_id)

        # exceptions
        # error printed to console 
        # events logged to notepad using log_event()
        except mt5_err.MT5ConnectionError as e:
            logging.error(f'{e} \nconnection lost, reconnecting') 
            time.sleep(5)
            mt5_conn.connect_account()

        except mt5_err.MT5RatesError as e:
            logging.error(f'{e}: \nbatch fetch failed, will retry next poll')
            _reconnect_if_disconnected()

        except mt5_err.MT5PositionError as e:
            logging.error(f'{e}: \nget positions failed, position with ticket does not exist')
            _reconnect_if_disconnected()

        except mt5_err.MT5OrderError as e:
            logging.error(f'{e}: \norder failed, skipping this signal')
            log_cfg.log_event('order_rejected', error = str(e))
            _reconnect_if_disconnected()

        except Exception as e:
            logging.error(f'{e} :\nun tracked failure')
            logging.exception('unhandled exception in live loop')
            _reconnect_if_disconnected()

        # poll next request 
        time.sleep(1) # re-run logic every 1 second

if __name__ == '__main__':
    symbols_strats = {
        'GBPUSD': session_breakout,
        'KO': vwap_breakout,
        'BTCUSD': bollinger_band_cdv,
    }
    strat_ids = {'GBPUSD': 1, 'KO': 2, 'BTCUSD': 3}
    run_live_loop(symbols_strats, strat_ids)