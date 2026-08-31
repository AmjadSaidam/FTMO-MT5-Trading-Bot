"""
Live MetaTrader5 trade execution logic 

polling function that request and send trade information to MetaTrader5 terminal

Fixes 
- add time gating a out-of-sample live code flag
"""
import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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

# --- MULTITIMEFRAME ---
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

def run_live_loop(symbols_strats: dict[str, SignalFunc],
                  strat_ids: dict[str, int],
                  symbols_strats_kwargs: dict[str, dict] | None = None,
                  strat_weights: float | None = None,
                  grid_search_atrs: np.ndarray = np.linspace(2.0, 5.0, 5), 
                  grid_search_rrs: np.ndarray = np.linspace(2.0, 6.0, 5)):
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

    # env defaults
    frac_risked = float(os.environ.get('META_TOTAL_RISK_ALLOCATION'))
    min_vol_lots = float(os.environ.get('META_MIN_VOLUME'))
    max_fx_lev = float(os.environ.get('META_MAX_FOREX_LEVERAGE'))
    max_other_lev = float(os.environ.get('META_MAX_OTHER_LEVERAGE'))
    initial_account_balance = os.environ.get('META_INITIAL_ACCOUNT_BALANCE')

    # account default arrays
    if strat_weights is None: 
        n = len(symbols_strats)
        strat_weights = [1 / n] * n 

    strategy_weights = dict(zip(symbols_strats.keys(), strat_weights))
    initial_strategy_allocation = np.array(strat_weights) * float(initial_account_balance)
    strategy_equity_dict = {symbol: initial_strategy_allocation[strat_no] for strat_no, symbol in enumerate(symbols_strats.keys())} # initial strategy allocation in $
    strategy_live_trade = {symbol: False for symbol in symbols_strats.keys()} # track signal per symbol (max 1 order per symbol per day )
    strategy_open_tickets = {symbol: [] for symbol in symbols_strats.keys()} # order history
    strategy_last_bar_time = {symbol: None for symbol in symbols_strats.keys()} # track most recent bar (only request data and run strategy logic once per new bar)
    strategy_traded_today = {symbol: False for symbol in symbols_strats.keys()} # track trade limit per aasset 

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
    strategy_models = {} # save to states on restart

    # performance
    strategy_skip_trade = {symbol: False for symbol in symbols_strats.keys()}
    strategy_consec_loss = {symbol: 0 for symbol in symbols_strats.keys()}

    def _update_strat_weight(symbol):
        strategy_weights[symbol] = strategy_equity_dict[symbol] / sum(strategy_equity_dict.values())

    def _train_models(as_of_time):
        """(re)trains strategy_models/strategy_opt_atr/strategy_opt_rr for every symbol off the
        current IS window; does not touch time_last_wfa, so it's safe to call outside the WFA schedule"""
        strat_equity = sum(strategy_equity_dict.values())
        for symbol, strat in symbols_strats.items():

            # data, drop the still-forming last bar so training only sees closed bars
            # date-range fetch (not a bar count) so the window is exactly wfa_in_sample_length
            # calendar days regardless of the symbol's trading session length (24/7 crypto vs FX vs equity hours)
            wfa_data = mt5_conn.get_bars_range(symbol,
                                               date_from = as_of_time - timedelta(days = wfa_in_sample_length),
                                               date_to = as_of_time).iloc[:-1]
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
            if ensemble.model is None:
                logging.warning(f'{symbol}: not enough labelled in-sample trades to train model, trading skipped until next WFA cycle')

            # optimal values
            strategy_models[symbol] = ensemble
            strategy_opt_atr[symbol] = gd_metadata['opt_atr']
            strategy_opt_rr[symbol] = gd_metadata['opt_rr']

    # bookkeeping
    # restore equity/order bookkeeping from a previous run (crash/restart safe)
    acc_info = mt5.account_info()
    if acc_info is not None:
        initial_account_balance = acc_info.balance
    saved_state = _load_state()

    def update_state_dicts(reason: str = 'sl_tp_hit') -> bool:
        """reconciles tickets against actual MT5 position state; returns True if any symbol's state changed"""
        reconciled = False
        for symbol, tickets in strategy_open_tickets.items():
            if not tickets:
                continue
            # live ticket (position still open)
            if not mt5_conn.get_positions(ticket = tickets[-1]): # named tuple of order metadata
                pnl = mt5_conn.get_deal_profit(tickets[-1])
                strategy_consec_loss[symbol] = 0 if pnl > 0 else strategy_consec_loss[symbol] + 1
                strategy_equity_dict[symbol] += pnl
                strategy_open_tickets[symbol] = []
                strategy_live_trade[symbol] = False
                _update_strat_weight(symbol)
                log_cfg.log_event('position_closed', symbol = symbol, ticket = tickets[-1], reason = reason, pnl = pnl)
                reconciled = True
        return reconciled

    # run once on restart/crash 
    if saved_state is not None:
        # update the default dicts to match saved states 
        # ensures dict persistance on restarts
        strategy_weights.update(saved_state['strategy_weights'])
        strategy_equity_dict.update(saved_state['strategy_equity_dict'])
        strategy_open_tickets.update(saved_state['strategy_open_tickets'])
        strategy_live_trade.update(saved_state['strategy_live_trade'])
        strategy_consec_loss.update(saved_state['strategy_consec_loss'])
        strategy_skip_trade.update(saved_state['strategy_skip_trade'])
        strategy_traded_today.update(saved_state['strategy_traded_today'])
        if saved_state.get('time_last_wfa') is not None:
            # pull last wfa run time
            time_last_wfa = datetime.fromisoformat(saved_state['time_last_wfa']) # keeps IS/OOS window boundaries stable across a restart
            # strategy_models isn't persisted (holds trained sklearn objects, not JSON-serialisable) and
            # is always empty on a fresh process - rebuild it off the restored IS window now, without
            # touching time_last_wfa/the WFA schedule, so polling doesn't KeyError on strategy_models[symbol]
            # until the next scheduled WFA cycle (which may be days away)
            logging.info('rebuilding in-memory strategy models after restart (not persisted across runs)')
            try:
                _train_models(time_last_wfa)
            except Exception as e:
                # runs before the polling loop's own try/except, so a transient MT5 fetch error here
                # would otherwise crash startup entirely; force run_wfa on the next poll instead so the
                # retry goes through the loop's normal (retried, exception-handled) path
                logging.error(f'{e}: failed to rebuild strategy models after restart, will retry on next poll')
                time_last_wfa = None
        logging.info('restored strategy equity/order and wfa time state from previous run')

        # a restored ticket may have already closed (SL/TP) while this process was down
        update_state_dicts(reason = 'reconciled_on_restart')

    def _persist():
        _save_state({
            'strategy_weights': strategy_weights,
            'strategy_equity_dict': strategy_equity_dict,
            'strategy_open_tickets': strategy_open_tickets,
            'strategy_live_trade': strategy_live_trade,
            'strategy_consec_loss': strategy_consec_loss,
            'strategy_skip_trade': strategy_skip_trade,
            'time_last_wfa': time_last_wfa.isoformat() if (time_last_wfa is not None) else None,
            'strategy_traded_today': strategy_traded_today
        })

    if saved_state is not None:
        _persist() # flush reconciliation immediately

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

            # wfa
            if time_last_wfa is None: # on first iteration
                run_wfa = True
            else:
                time_diff = (current_day_time - time_last_wfa).days
                run_wfa = time_diff > wfa_in_sample_length
            if run_wfa:
                _train_models(current_day_time)
                time_last_wfa = current_day_time
                _persist()

            # get symbol info
            for symbol, strat in symbols_strats.items():
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is None:
                    logging.warning(f'{symbol}: symbol_info() unavailable, skipping this poll') 
                    continue
                symbol_path = symbol_info.path.split('\\')[0]

                # strategy ticket
                ticket = strategy_open_tickets.get(symbol)
                # eod exit
                if new_day:
                    # enable live trading
                    strategy_live_trade[symbol] = False # flat going into the new day, allow re-entry
                    # reset max trades guard 
                    strategy_traded_today[symbol] = False
                    # enable trading
                    strategy_skip_trade[symbol] = False
                    # check ticket open 
                    if ticket:
                        positions = mt5_conn.get_positions(ticket = ticket[-1])
                        if positions:
                            # trade-live force close
                            mt5_conn.close_position(positions[-1]) # close most recent position
                            pnl = mt5_conn.get_deal_profit(ticket[-1])
                            # check trade outcome
                            if pnl > 0:
                                strategy_consec_loss[symbol] = 0
                            else:
                                strategy_consec_loss[symbol] += 1
                            # update strat equity 
                            strategy_equity_dict[symbol] += pnl
                            # reset order history
                            strategy_open_tickets[symbol] = [] 
                            # update weights
                            _update_strat_weight(symbol)
                            # log
                            log_cfg.log_event('position_closed', symbol = symbol, ticket = ticket[-1],
                                              reason = 'eod_force_close', pnl = pnl)
                    # commit updates to state dict
                    _persist()

                # signal logic (ran once ber new bar)
                # guard against pyrammiding orders and re-entry after an earlier close the same day
                live_trade = strategy_live_trade[symbol]
                if not live_trade and not strategy_traded_today[symbol]:
                    # skip until a new bar has closed, avoids re-fetching/re-evaluating every 1s poll
                    latest_bar_time = mt5_conn.get_latest_bar_time(symbol)
                    if latest_bar_time == strategy_last_bar_time[symbol]:
                        continue # will not run subsequent code 
                    strategy_last_bar_time[symbol] = latest_bar_time

                    # pull data, time range determined by asset class 
                    current_day_time = pd.to_datetime(current_day_time, unit = 's')
                    data_from = current_day_time - pd.Timedelta(value = 1, unit = 'D')
                    # pull only current day data for equities, anchord day vwap only requires current day data
                    if symbol_path == 'Equities I CFD':
                        data_from = current_day_time.normalize()
                    data = mt5_conn.get_bars_range(symbol,
                                                   date_from = data_from,
                                                   date_to = current_day_time)
                    # check data not-empty, eg market closed on weekend and not quotes for day
                    if data.empty:
                        logging.warning(f'{symbol}: empty dataframe, market closed, skipping this poll')
                        continue # skip and poll

                    # seperate current data and historical data
                    tob = data.iloc[-1, :] # forming bar - its open is the entry reference price
                    closed_data = data.iloc[:-1] # drop forming bar (last entry - values at snapshot)

                    # ml model features
                    atr_pct = (atr(closed_data) / closed_data.close.to_numpy())[-1].item()
                    regime = categorise_regime(
                        closed_data.close.rolling(feature_window).apply(r2, raw = True)
                    )[-1]
                    if pd.isna(regime):
                        # not enough closed bars yet for the rolling regime window (eg. early in the day) - nothing to classify
                        logging.warning(f'{symbol}: not enough bars to classify regime yet, skipping this poll')
                        continue
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
                    valid_trade = ensemble.model_predict(point_features) # returns False - trade skipped 
                    if consecutive_loss_threshold(strategy_consec_loss[symbol], int(os.environ.get('META_ML_CONSEC_LOSS_BEFORE_TRADE_SKIP'))):
                        strategy_skip_trade[symbol] = True
                        _persist()

                    # asset vol
                    symbol_atr = atr(closed_data)[-1].item()

                    if not strategy_skip_trade[symbol]:
                        # generate signal off the last closed bar
                        strat_data = add_multi_timeframe_columns(strat, closed_data)
                        # signal is bolean array and signal_direction is numeric encoded long/short array
                        signal_series, signal_direction_arr = strat(strat_data, **((symbols_strats_kwargs or {}).get(symbol) or {}))
                        if (signal_series.shape[0] > 0) and (signal_direction_arr.shape[0] > 0):
                            signal, signal_direction = signal_series.iloc[-1], signal_direction_arr[-1]
                        else: 
                            # write error to /live_trading.log
                            logging.warning('zero_length_signal', symbol = symbol, strategy = strat.__name__)
                            continue
                        if signal:
                            # write update to /trades.log
                            log_cfg.log_event('signal_generated', symbol = symbol, strategy = strat.__name__, direction = int(signal_direction), valid_trade = bool(valid_trade))
                        
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
                            fraction_risk = strategy_weights[symbol] * frac_risked # per strategy
                            # volume
                            trade_value = position_sizing(tob.open, sl_dist, account_balance, fraction_risk)[0] # position notional in account currency
                            volume = trade_value / tob.open
                            max_volume_lots = account_balance / tob.open
                            # volume lots flag
                            if symbol_path == 'Forex':
                                volume /= 1e5 # units of base currency -> standard forex lots
                                max_volume_lots = max_volume_lots / 1e5 * max_fx_lev
                            else:
                                max_volume_lots *= max_other_lev
                            # snap to the broker's tradable lot increment (must be multiple of symbol volume increment)
                            volume = round(volume / symbol_info.volume_step) * symbol_info.volume_step
                            # validate position size
                            if min_vol_lots <= volume <= max_volume_lots:
                                # order to MT5
                                order = mt5_conn.send_order(trade_magic_id, symbol, volume, sl, tp, type = type)
                                strategy_live_trade[symbol] = True
                                # symbol order id
                                strategy_open_tickets[symbol].append(order.order) # ticket is named tuple field
                                # traded today 
                                strategy_traded_today[symbol] = True
                                # order sent/executed log
                                log_cfg.log_event('order_sent', symbol = symbol, strategy = strat.__name__,
                                                ticket = order.order, type = type, volume = volume,
                                                opt_atr = strategy_opt_atr[symbol], opt_rr = strategy_opt_rr[symbol],
                                                sl = sl, tp = tp, magic = trade_magic_id)
                                # commit changes to state dict
                                _persist()
                            else:
                                log_cfg.log_event('order_rejected', symbol = symbol, strategy = strat.__name__,
                                                  reason = 'invalid position size', volume = volume,
                                                  opt_atr = strategy_opt_atr[symbol], opt_rr = strategy_opt_rr[symbol],
                                                  min_volume = min_vol_lots, max_volume = max_volume_lots)

        # exceptions
        # error printed to console 
        # events logged to notepad using log_event()
        # IPC link may still hold despite execption being raised 
        # error raised by connect_account()
        except mt5_err.MT5ConnectionError as e:
            logging.error(f'{e}: connection lost, reconnecting') 
            time.sleep(5)
            mt5_conn.connect_account()

        # raised by data pulling functions
        except mt5_err.MT5RatesError as e:
            logging.error(f'{e}: batch fetch failed, will retry next poll')
            _reconnect_if_disconnected()

        # error raised by position_get()
        except mt5_err.MT5PositionError as e:
            logging.error(f'{e}: get positions failed, position with ticket does not exist')
            _reconnect_if_disconnected()

        # error riased by order_send()
        except mt5_err.MT5OrderError as e:
            logging.error(f'{e}: order failed, skipping this signal')
            log_cfg.log_event('order_rejected', symbol = symbol, strategy = strat.__name__, reason = 'invalid position size', volume = volume, 
                              opt_atr = strategy_opt_atr[symbol], opt_rr = strategy_opt_rr[symbol], min_volume = min_vol_lots, max_volume = max_volume_lots, error = str(e))
            _reconnect_if_disconnected()

        # untraced errors
        except Exception as e:
            logging.error(f'{e}: untracked failure')
            logging.exception('unhandled exception in live loop')
            _reconnect_if_disconnected()

        # update states dicts
        # fires only if position ticket was closed 
        if update_state_dicts():
            _persist()
            
        # poll next request 
        time.sleep(1) # re-run logic every 1 second

# --- HELPERS ---
# --- VARIABLE BOOKKEEPING ---
STATE_PATH = os.path.join(ROOT, 'logs', 'live_state.json')

def _load_state() -> dict | None:
    """restores strategy equity/order bookkeeping saved by a previous run, if any"""
    if not os.path.exists(STATE_PATH):
        return None
    with open(STATE_PATH) as f:
        return json.load(f)

def _save_state(state: dict):
    """write-then-rename so a crash mid-write can't leave a corrupt/partial state file"""
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok = True)
    tmp_path = STATE_PATH + '.tmp'
    with open(tmp_path, 'w') as f: # write file
        json.dump(state, f, default = float) # casts numpy floats (equity values) to plain float
    os.replace(tmp_path, STATE_PATH)

# --- CONNECTION LOGIC ---
def _reconnect_if_disconnected():
    """MT5RatesError/PositionError/OrderError can mean a lost terminal connection or a genuine
    request error; only reconnect when terminal_info() confirms the IPC link is actually down"""
    if not mt5_conn.is_connected():
        logging.error('MT5 terminal connection lost, reconnecting')
        time.sleep(5)
        mt5_conn.connect_account() # will not print connection validation


# --- RUN SCRIPT ---
if __name__ == '__main__':
    symbols_strats = {
        'GBPUSD': session_breakout,
        'KO': vwap_breakout,
        'BTCUSD': bollinger_band_cdv,
    }
    strat_ids = {'GBPUSD': 1, 'KO': 2, 'BTCUSD': 3}
    run_live_loop(symbols_strats, strat_ids)