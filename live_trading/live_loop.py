"""
Live MetaTrader5 trade execution logic

polling function that request and send trade information to MetaTrader5 terminal

mirrors the backtest (notebooks/backtest.ipynb run_wfa + code/backtest_engine.backtest_strategy):
- anchored (expanding) walk forward: in sample = wfa anchor -> fold end, refit every oos window, in sample split
  (1 - is_eval_fraction) / is_eval_fraction by bars
- one ml model per (atr, rr), trained on the trade labels known before the split, every (atr, rr) + model scored on
  the in-sample eval by the prop firm strategy_score, the best one is traded out of sample (live) until the next refit
- bars converted to the trading time zone (Europe/London) before any strategy / feature / day logic
- trading day boundary = first bar of a new trading time zone day (per symbol): skip / traded-today reset, eod close
- entry at the forming bar's open off the last closed bar's signal, one trade per symbol per day
"""
import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import (datetime, timedelta, timezone)
from itertools import product
from joblib import (Parallel, delayed)
import MetaTrader5 as mt5
import time
import logging
from dotenv import load_dotenv; load_dotenv()

# code/*.py and mt5/*.py cross-import each other by bare module name (eg. "from risk_management import atr"),
# so their folders go on sys.path and are imported bare (mirrors notebooks/backtest.ipynb). The project root is
# not added: the code/ package shadows the stdlib `code` module, and importing a module both as code.x and x
# creates two copies (eg. two MT5 exception classes, so except clauses miss the one actually raised)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, 'code'), os.path.join(ROOT, 'mt5')):
    if p not in sys.path:
        sys.path.insert(0, p)
# .py
import mt5_connector as mt5_conn
import mt5_errors as mt5_err
import logging_config as log_cfg
import trading_strategies as ts
import challenge_validation as cv
from backtest_engine import (SignalFunc, backtest_strategy, position_sizing)
from regime_filter import (rolling_r2, categorise_regime)
from risk_management import (atr, consecutive_loss_threshold, RandomForestVol)

TRADING_TIME_ZONE = 'Europe/London' # backtest_strategy() default, day boundaries / eod exits are evaluated on this clock
ML_FEATURES = ['atr_pct', 'regime', 'breakout_conv', 'rel_volume'] # identical names / order to backtest_strategy() output
FOREX_LOT_SIZE = 1e5 # backtest_strategy() standard_lot

# --- STRATEGY DATA ---
# strategies requiring 1h/4h/8h columns + LDN / NY session flags (ts.session_breakout_data) on the base timeframe data
MULTI_TIMEFRAME_STRATS = {ts.session_breakout}

def prepare_strategy_data(strat: SignalFunc,
                          data: pd.DataFrame,
                          trading_time_zone: str = TRADING_TIME_ZONE) -> pd.DataFrame:
    """
    raw mt5 bars -> strategy input, identical to the backtest's mt5 data path:
    ohlcv only, index localised as UTC and converted to the trading time zone, then session_breakout_data()
    for multi-timeframe strategies (the time zone conversion must come first, it sets the 4h / 8h bin edges)
    """
    data = data.loc[:, ['open', 'high', 'low', 'close', 'volume']].copy()
    index = data.index if data.index.tz is not None else data.index.tz_localize('UTC')
    data.index = index.tz_convert(trading_time_zone)

    if strat in MULTI_TIMEFRAME_STRATS:
        data = ts.session_breakout_data(data)

    return data

def signal_bar_features(data: pd.DataFrame, feature_window: int) -> tuple[dict[str, list], float]:
    """ml features + atr of the last bar in data, computed exactly as backtest_strategy() does over the whole frame"""
    vol_atr = atr(data)
    close = data['close']
    atr_pct = vol_atr / close.to_numpy()
    regime = categorise_regime(rolling_r2(close, feature_window))
    breakout_conv = ((close - data['open']) / (data['high'] - data['low'])).to_numpy()
    rel_volume = data['volume'].rolling(feature_window).mean().to_numpy()

    features = {
        'atr_pct': [atr_pct[-1]],
        'regime': [regime[-1]],
        'breakout_conv': [breakout_conv[-1]],
        'rel_volume': [rel_volume[-1]],
    }
    return features, float(vol_atr[-1])

# --- WALK FORWARD MODEL SELECTION ---
def wfa_fold_end(anchor: pd.Timestamp,
                 now: pd.Timestamp,
                 initial_is: pd.DateOffset,
                 oos: pd.DateOffset) -> pd.Timestamp:
    """
    end of the in-sample window currently being traded: anchor + initial_is + k * oos, the latest one <= now
    (same fold grid as walk_forward_analysis(): first in sample spans initial_is, then steps by one oos window)
    """
    fold_end = anchor + initial_is
    while fold_end + oos <= now:
        fold_end = fold_end + oos
    return fold_end

def select_wfa_model(is_data: pd.DataFrame,
                     strat: SignalFunc,
                     account_balance: float,
                     strat_kwargs: dict | None = None,
                     forex: bool = False,
                     model_ensemble: bool = True,
                     spread_dollars: float = 0.0,
                     commission_value: float = 0.0,
                     slippage_dollars: float = 0.0,
                     fraction_risked: float = 0.01,
                     consec_loss_guard: int | None = None,
                     feature_window: int = 30,
                     atr_multipliers: np.ndarray = np.linspace(2.0, 5.0, 5),
                     risk_rewards: np.ndarray = np.linspace(3.0, 6.0, 5),
                     is_eval_fraction: float = 0.3,
                     warmup_lookback_bars: int = 100,
                     n_jobs: int = -1) -> dict | None:
    """
    one walk forward fold of the backtest's run_wfa() on the in-sample window is_data (wfa anchor -> fold end):
    1. grid search: every (atr, rr) is backtested over the ml training history (first 1 - is_eval_fraction of the bars)
    2. one ml model per (atr, rr), trained on that history's trade labels (labels known before the split only)
    3. every (atr, rr) + its model is backtested on the in-sample eval (last is_eval_fraction), the model filtering signals
    4. the (atr, rr) + model with the highest prop firm strategy_score is returned (first best on ties)
    without model_ensemble steps 2 / 3 run unfiltered, so only (atr, rr) is selected on the in-sample eval

    returns None if the window is too short to split
    """
    n = len(is_data)
    split = int(np.floor(n * (1 - is_eval_fraction)))
    if not 0 < split < n:
        return None
    index = is_data.index

    backtest_kwargs = {
        'signal_fc': strat,
        'signal_kwargs': strat_kwargs or {},
        'forex': forex,
        'account_balance': account_balance,
        'risk_per_trade': fraction_risked,
        'consecutive_loss_before_skip': consec_loss_guard,
        'feature_window': feature_window,
        'slippage_dollar': slippage_dollars,
        'spread_dollar': spread_dollars,
        'commission_pct': commission_value,
    }
    grid = list(product(atr_multipliers, risk_rewards))

    # ml training history labels: backtest_strategy() is causal, so trades before the split are identical to a backtest over the
    # whole window; signals on the last trading day before the split are dropped, their trade can close after the split
    # (eod exits fill on the next day's first bar), so no label is learnt from in-sample eval bars
    last_day = index[split - 1].normalize()

    def grid_history(atr_m: float, rr: float) -> pd.DataFrame:
        hist = backtest_strategy(data = is_data.iloc[: split].copy(), atr_multiplier = atr_m, risk_reward = rr, **backtest_kwargs)['data']
        hist = hist.loc[hist['ml_label'].notna(), ML_FEATURES + ['ml_label']]
        return hist[hist.index.normalize() < last_day]

    histories = {}
    if model_ensemble:
        histories = dict(zip(grid, Parallel(n_jobs = n_jobs)(delayed(grid_history)(atr_m, rr) for atr_m, rr in grid)))

    # in-sample eval + preceding history: >= warmup_lookback_bars and back to the prior day's midnight (session-anchored
    # vwap / bars_since_open restart from a real session open), only the eval bars are traded / scored
    day_start = int(index.searchsorted(index[split].normalize() - pd.Timedelta(days = 1)))
    hist_start = max(min(split - warmup_lookback_bars, day_start), 0)
    eval_warmup = split - hist_start

    def score_candidate(atr_m: float, rr: float):
        ensemble = None
        if model_ensemble:
            labels = histories[(atr_m, rr)]
            ensemble = RandomForestVol(labels[ML_FEATURES], labels['ml_label'], categorical_col = 'regime')
            ensemble.train_model(train_split = 1) # whole training history, no holdout (scored on the in-sample eval below)

        is_eval = backtest_strategy(data = is_data.iloc[hist_start:].copy(),
                                    atr_multiplier = atr_m,
                                    risk_reward = rr,
                                    ensamble_signal_validation = ensemble.model if ensemble is not None else None,
                                    warmup_bars = eval_warmup,
                                    **backtest_kwargs)['data']
        chall_val = cv.PropFirmValidation(is_eval['adjusted_returns'], is_eval['equity'], starting_balance = account_balance)
        chall_val.strategy_challenge_summary()

        return chall_val.strategy_score, ensemble

    candidates = Parallel(n_jobs = n_jobs)(delayed(score_candidate)(atr_m, rr) for (atr_m, rr) in grid)
    scores = np.array([score for score, _ in candidates], dtype = float)
    opt_idx = int(np.argmax(scores))
    opt_atr, opt_rr = grid[opt_idx]

    return {
        'opt_atr': float(opt_atr),
        'opt_rr': float(opt_rr),
        'ensemble': candidates[opt_idx][1],
        'is_eval_score': float(scores[opt_idx]),
        'split_time': index[split],
    }

# --- LIVE LOOP ---
def run_live_loop(symbols_strats: dict[str, SignalFunc],
                  strat_ids: dict[str, int],
                  symbols_strats_kwargs: dict[str, dict] | None = None,
                  symbols_config: dict[str, dict] | None = None,
                  strat_weights: list[float] | None = None,
                  grid_search_atrs: np.ndarray = np.linspace(2.0, 5.0, 5),
                  grid_search_rrs: np.ndarray = np.linspace(3.0, 6.0, 5)):
    """
    multi-strategy logic

    symbols_config: per symbol backtest settings used by the walk forward model selection (same keys as the backtest's
    asset_config): forex, model_ensemble, spread (single trip, $ of price), commission (% of volume traded), slippage ($)
    """
    log_cfg.setup_logging() # call once (dont re-initialise)
    while True:
        try:
            mt5_conn.connect_account()
            break
        except mt5_err.MT5ConnectionError as e:
            logging.error(f'{e} \ninitial connection failed, retrying in 5s')
            time.sleep(5)
    logging.info('connected to MT5 account, starting live loop')

    symbols = list(symbols_strats.keys())
    symbols_strats_kwargs = symbols_strats_kwargs or {}
    symbols_config = symbols_config or {}
    strat_kwargs_of = lambda symbol: symbols_strats_kwargs.get(symbol) or {}

    # env defaults
    frac_risked = float(os.environ.get('META_TOTAL_RISK_ALLOCATION'))      # fraction of strategy equity risked per trade (backtest risk_per_trade)
    min_vol_lots = float(os.environ.get('META_MIN_VOLUME'))
    max_fx_lev = float(os.environ.get('META_MAX_FOREX_LEVERAGE'))
    max_other_lev = float(os.environ.get('META_MAX_OTHER_LEVERAGE'))
    initial_account_balance = float(os.environ.get('META_INITIAL_ACCOUNT_BALANCE'))

    # walk forward schedule (defaults = the backtest's initial_is / oos / is_eval_fraction)
    initial_is = pd.DateOffset(months = int(os.environ.get('META_WFA_INITIAL_IS_MONTHS', 6)))
    oos = pd.DateOffset(months = int(os.environ.get('META_WFA_OOS_MONTHS', 1)))
    is_eval_fraction = float(os.environ.get('META_WFA_IS_EVAL_FRACTION', 0.3))
    wfa_n_jobs = int(os.environ.get('META_WFA_N_JOBS', -1))

    # ml risk management, identical in-sample and live
    feature_window = int(os.environ.get('META_ML_FEATURE_WINDOW'))
    consec_loss_env = (os.environ.get('META_ML_CONSEC_LOSS_BEFORE_TRADE_SKIP') or '').strip()
    consec_loss_guard = int(consec_loss_env) if consec_loss_env.lower() not in ('', 'none') else None # None disables the losing streak skip
    signal_lookback_days = 4 # live signal pull: covers weekends / holidays so indicators and features are warm on the last closed bar

    # account default arrays
    if strat_weights is None:
        strat_weights = [1 / len(symbols)] * len(symbols)
    strategy_weights = dict(zip(symbols, strat_weights))
    strategy_initial_allocation = {symbol: w * initial_account_balance for symbol, w in strategy_weights.items()} # backtest account_balance (max leverage reference)
    strategy_equity_dict = dict(strategy_initial_allocation) # strategy equity in $, backtest equity[t]
    strategy_live_trade = {symbol: False for symbol in symbols} # position open
    strategy_open_tickets = {symbol: [] for symbol in symbols} # order history
    strategy_last_bar_time = {symbol: None for symbol in symbols} # most recent bar (only request data and run strategy logic once per new bar)
    strategy_traded_today = {symbol: False for symbol in symbols} # one trade (or rejected signal) per symbol per day
    strategy_current_day = {symbol: None for symbol in symbols} # trading time zone day of the symbol's latest bar
    strategy_skip_next_day = {symbol: False for symbol in symbols} # losing streak breached -> next trading day skipped
    strategy_consec_loss = {symbol: 0 for symbol in symbols}

    # walk forward state
    wfa_anchor = None # in-sample start, fixed across folds (expanding window)
    time_last_wfa = None # end of the in-sample window the current models were selected on
    strategy_models = {} # selected RandomForestVol per symbol (None = unfiltered), not persisted -> rebuilt on restart
    strategy_opt_atr = {}
    strategy_opt_rr = {}
    wfa_retry_delay = pd.Timedelta(minutes = 15)
    wfa_retry_time = pd.Timestamp.min.tz_localize('UTC')

    def _update_strat_weight(symbol):
        strategy_weights[symbol] = strategy_equity_dict[symbol] / sum(strategy_equity_dict.values())

    def _train_models(fold_end: pd.Timestamp, fit_symbols: list[str]):
        """
        selects (atr, rr) + ml model per symbol on the in-sample window wfa_anchor -> fold_end (one backtest wfa fold)
        a symbol that fails stays unfitted (not traded, retried later) without blocking the others
        """
        for symbol in fit_symbols:
            strat = symbols_strats[symbol]
            cfg = symbols_config.get(symbol, {})
            try:
                # drop the last bar so selection only sees closed bars
                raw = mt5_conn.get_bars_range(symbol,
                                              date_from = wfa_anchor.to_pydatetime(),
                                              date_to = fold_end.to_pydatetime()).iloc[:-1]
                is_data = prepare_strategy_data(strat, raw) if not raw.empty else raw
                selection = select_wfa_model(is_data,
                                             strat,
                                             account_balance = strategy_equity_dict[symbol],
                                             strat_kwargs = strat_kwargs_of(symbol),
                                             forex = cfg.get('forex', False),
                                             model_ensemble = cfg.get('model_ensemble', True),
                                             spread_dollars = cfg.get('spread', 0.0),
                                             commission_value = cfg.get('commission', 0.0),
                                             slippage_dollars = cfg.get('slippage', 0.0),
                                             fraction_risked = frac_risked,
                                             consec_loss_guard = consec_loss_guard,
                                             feature_window = feature_window,
                                             atr_multipliers = grid_search_atrs,
                                             risk_rewards = grid_search_rrs,
                                             is_eval_fraction = is_eval_fraction,
                                             n_jobs = wfa_n_jobs) if len(is_data) else None
                if selection is None:
                    raise mt5_err.MT5RatesError(f'not enough in-sample bars ({len(is_data)}) between {wfa_anchor} and {fold_end} to split the walk forward window')
            except Exception as e:
                logging.error(f'{symbol}: {e}: walk forward selection failed, symbol not traded until it succeeds')
                if not mt5_conn.is_connected():
                    raise mt5_err.MT5ConnectionError('terminal link lost during walk forward selection')
                continue

            ensemble = selection['ensemble']
            if ensemble is not None and ensemble.model is None:
                # backtest passes ensemble.model = None -> signals trade unfiltered
                logging.warning(f'{symbol}: not enough labelled in-sample trades to train the model, trading unfiltered until the next wfa fold')
            strategy_models[symbol] = ensemble
            strategy_opt_atr[symbol] = selection['opt_atr']
            strategy_opt_rr[symbol] = selection['opt_rr']
            log_cfg.log_event('wfa_selected', symbol = symbol, strategy = strat.__name__, is_start = wfa_anchor, is_end = fold_end,
                              split_time = selection['split_time'], opt_atr = selection['opt_atr'], opt_rr = selection['opt_rr'],
                              is_eval_score = selection['is_eval_score'],
                              model_trained = bool(ensemble is not None and ensemble.model is not None))

    def _register_close(symbol: str, ticket: int, pnl: float, reason: str):
        """trade outcome bookkeeping shared by every close path"""
        strategy_consec_loss[symbol] = 0 if pnl > 0 else strategy_consec_loss[symbol] + 1
        strategy_equity_dict[symbol] += pnl
        strategy_open_tickets[symbol] = []
        strategy_live_trade[symbol] = False
        _update_strat_weight(symbol)
        log_cfg.log_event('position_closed', symbol = symbol, ticket = ticket, reason = reason, pnl = pnl)

    def _losing_streak_breached(symbol: str) -> bool:
        """backtest: counter resets once the threshold is hit"""
        if consec_loss_guard is not None and consecutive_loss_threshold(strategy_consec_loss[symbol], consec_loss_guard):
            strategy_consec_loss[symbol] = 0
            return True
        return False

    def update_state_dicts(reason: str = 'sl_tp_hit_or_manual_close') -> bool:
        """reconciles tickets against actual MT5 position state; returns True if any symbol's state changed"""
        reconciled = False
        for symbol, tickets in strategy_open_tickets.items():
            if not tickets:
                continue
            if not mt5_conn.get_positions(ticket = tickets[-1]): # position no longer open (sl / tp / manual)
                _register_close(symbol, tickets[-1], mt5_conn.get_deal_profit(tickets[-1]), reason)
                # backtest: intraday stop out breaching the streak skips the next trading day
                if _losing_streak_breached(symbol):
                    strategy_skip_next_day[symbol] = True
                reconciled = True
        return reconciled

    def _roll_day(symbol: str, bar_day: pd.Timestamp):
        """first bar of a new trading day (backtest day boundary): consume the skip flag / reset the trade limit, then eod close"""
        strategy_current_day[symbol] = bar_day
        if strategy_skip_next_day[symbol]:
            strategy_traded_today[symbol] = True
            strategy_skip_next_day[symbol] = False
            log_cfg.log_event('day_skipped', symbol = symbol, reason = 'consecutive_loss_threshold', day = bar_day)
        else:
            strategy_traded_today[symbol] = False

        ticket = strategy_open_tickets[symbol]
        if ticket:
            positions = mt5_conn.get_positions(ticket = ticket[-1])
            if positions:
                mt5_conn.close_position(positions[-1]) # backtest exits at the new day's first bar open
                _register_close(symbol, ticket[-1], mt5_conn.get_deal_profit(ticket[-1]), 'eod_force_close')
                # backtest: the eod close happens on the new day, so a breach skips this day
                if _losing_streak_breached(symbol):
                    strategy_traded_today[symbol] = True
                    log_cfg.log_event('day_skipped', symbol = symbol, reason = 'consecutive_loss_threshold', day = bar_day)

    def _persist():
        _save_state({
            'strategy_weights': strategy_weights,
            'strategy_equity_dict': strategy_equity_dict,
            'strategy_open_tickets': strategy_open_tickets,
            'strategy_live_trade': strategy_live_trade,
            'strategy_consec_loss': strategy_consec_loss,
            'strategy_skip_next_day': strategy_skip_next_day,
            'strategy_traded_today': strategy_traded_today,
            'strategy_current_day': {s: (d.isoformat() if d is not None else None) for s, d in strategy_current_day.items()},
            'wfa_anchor': wfa_anchor.isoformat() if (wfa_anchor is not None) else None,
            'time_last_wfa': time_last_wfa.isoformat() if (time_last_wfa is not None) else None,
        })

    # bookkeeping
    # restore equity/order bookkeeping from a previous run (crash/restart safe)
    saved_state = _load_state()
    if saved_state is not None:
        strategy_weights.update(saved_state.get('strategy_weights', {}))
        strategy_equity_dict.update(saved_state.get('strategy_equity_dict', {}))
        strategy_open_tickets.update(saved_state.get('strategy_open_tickets', {}))
        strategy_live_trade.update(saved_state.get('strategy_live_trade', {}))
        strategy_consec_loss.update(saved_state.get('strategy_consec_loss', {}))
        strategy_skip_next_day.update(saved_state.get('strategy_skip_next_day', {}))
        strategy_traded_today.update(saved_state.get('strategy_traded_today', {}))
        strategy_current_day.update({s: pd.Timestamp(d) for s, d in (saved_state.get('strategy_current_day') or {}).items()
                                     if d is not None and s in strategy_current_day})
        if saved_state.get('wfa_anchor') is not None:
            wfa_anchor = pd.Timestamp(saved_state['wfa_anchor']) # keeps the expanding in-sample start / fold grid stable across restarts
        # strategy_models isn't persisted (trained sklearn objects, not JSON-serialisable): the first poll re-selects them
        # on the fold grid anchored at wfa_anchor (same in-sample window as before the restart unless a new fold has started)
        logging.info('restored strategy equity/order and wfa state from previous run')

        # a restored ticket may have already closed (SL/TP) while this process was down
        try:
            update_state_dicts(reason = 'reconciled_on_restart')
        except mt5_err.MT5PositionError as e:
            logging.error(f'{e}: restart reconciliation failed, will retry on next poll')
        _persist() # flush reconciliation immediately

    # strategy polling
    while True:
        symbol, strat = None, None # exception handler context
        try:
            now = _utc_now()

            # position closes (sl / tp / manual) are booked before any day roll, so a stop out on the prior day
            # sets the skip flag that day's roll consumes (backtest order)
            if update_state_dicts():
                _persist()

            # wfa: refit when a new fold starts (anchor + initial_is + k * oos)
            if wfa_anchor is None: # first ever run: in sample spans initial_is back from today
                wfa_anchor = (now - initial_is).normalize()

            fold_end = wfa_fold_end(wfa_anchor, now, initial_is, oos)
            
            if time_last_wfa != fold_end: # new fold: every symbol re-selected (prior fold's models are not traded past their oos window)
                strategy_models.clear(); strategy_opt_atr.clear(); strategy_opt_rr.clear()
                _train_models(fold_end, symbols)
                time_last_wfa, wfa_retry_time = fold_end, now + wfa_retry_delay
                _persist()
            elif now >= wfa_retry_time and (unfitted := [s for s in symbols if s not in strategy_opt_atr]):
                _train_models(fold_end, unfitted) # throttled retry of symbols whose selection failed
                wfa_retry_time = now + wfa_retry_delay

            for symbol, strat in symbols_strats.items():
                symbol_info = mt5.symbol_info(symbol)
            
                # closed terminal64.exe raises warning, re-establish connection
                if symbol_info is None:
                    logging.warning(f'{symbol}: symbol_info() unavailable, skipping this poll')
                    _reconnect_if_disconnected()
                    continue
                
                # lot convention must match the in-sample backtest's forex flag, broker path as fallback
                forex = symbols_config.get(symbol, {}).get('forex', symbol_info.path.split('\\')[0] == 'Forex')

                # skip until a new bar has opened, avoids re-fetching/re-evaluating every 1s poll
                latest_bar_time = mt5_conn.get_latest_bar_time(symbol)
                if latest_bar_time == strategy_last_bar_time[symbol]:
                    continue
            
                strategy_last_bar_time[symbol] = latest_bar_time

                # day boundary on the symbol's own bars (equities: next session open, not midnight)
                bar_day = _bar_day(latest_bar_time)

                if strategy_current_day[symbol] is None:
                    strategy_current_day[symbol] = bar_day
                    _persist()
                elif bar_day != strategy_current_day[symbol]:
                    _roll_day(symbol, bar_day)
                    _persist()

                # guard against pyramiding orders and re-entry after an earlier close / rejection the same day
                if strategy_live_trade[symbol] or strategy_traded_today[symbol]:
                    continue
                # no walk forward selection for this fold yet (failed fit, retried) -> nothing to trade with
                if symbol not in strategy_opt_atr:
                    continue

                # pull data
                raw = mt5_conn.get_bars_range(symbol,
                                              date_from = (now - pd.Timedelta(days = signal_lookback_days)).to_pydatetime(),
                                              date_to = now.to_pydatetime())
                # market closed (eg. weekend, no quotes for the day) / not enough history for the signal bar
                if len(raw) < 2:
                    logging.warning(f'{symbol}: not enough bars returned, market closed, skipping this poll')
                    continue

                # forming bar (entry reference price, backtest df.open at bar t) and closed bars (signal bar t-1)
                tob = raw.iloc[-1]
                closed_data = prepare_strategy_data(strat, raw.iloc[:-1])
                if closed_data.empty:
                    logging.warning(f'{symbol}: no closed bars after strategy data preparation, skipping this poll')
                    continue

                # signal off the last closed bar
                signal_series, signal_direction_arr = strat(closed_data.copy(), **strat_kwargs_of(symbol))
                if (signal_series.shape[0] == 0) or (signal_direction_arr.shape[0] == 0):
                    logging.warning(f'{symbol}: zero length signal from {strat.__name__}')
                    continue
                if not bool(signal_series.iloc[-1]):
                    continue
                signal_direction = int(signal_direction_arr[-1])

                # backtest entry conditions: valid atr on the signal bar, signal and entry bars in the same trading day
                point_features, symbol_atr = signal_bar_features(closed_data, feature_window)
                entry_day = _bar_day(raw.index[-1])
                if np.isnan(symbol_atr) or symbol_atr <= 0 or entry_day != closed_data.index[-1].normalize():
                    continue

                # model filter (None / untrained model -> unfiltered, as in the backtest)
                ensemble: RandomForestVol | None = strategy_models.get(symbol)
                valid_trade = True
                if ensemble is not None and ensemble.model is not None:
                    valid_trade = bool(np.asarray(ensemble.model_predict(point_features)).ravel()[0])
                log_cfg.log_event('signal_generated', symbol = symbol, strategy = strat.__name__,
                                  direction = signal_direction, valid_trade = valid_trade)
                if not valid_trade: # model rejected the day's signal -> no further trades today
                    strategy_traded_today[symbol] = True
                    _persist()
                    continue

                # order
                trade_type = 'LONG' if (signal_direction == 1) else 'SHORT'
                trade_magic_id = strat_ids.get(symbol)
                entry_price = tob.open
                # round quotes to asset listed precision - otherwise order rejects
                sl_dist = symbol_atr * strategy_opt_atr[symbol]
                sl = round(entry_price - sl_dist * signal_direction, symbol_info.digits)
                tp = round(entry_price + sl_dist * strategy_opt_rr[symbol] * signal_direction, symbol_info.digits)

                # volume: lose exactly frac_risked of strategy equity at the stop (backtest position_sizing on equity[t])
                lot_size, max_leverage = (FOREX_LOT_SIZE, max_fx_lev) if forex else (1.0, max_other_lev)
                trade_value = position_sizing(entry_price, sl_dist, strategy_equity_dict[symbol], frac_risked)[0]
                volume = trade_value / entry_price / lot_size
                max_volume_lots = strategy_initial_allocation[symbol] / entry_price / lot_size * max_leverage
                # snap to the broker's tradable lot increment
                volume = round(round(volume / symbol_info.volume_step) * symbol_info.volume_step, 8)

                if min_vol_lots <= volume <= max_volume_lots:
                    order = mt5_conn.send_order(trade_magic_id, symbol, volume, sl, tp, type = trade_type)
                    strategy_live_trade[symbol] = True
                    strategy_open_tickets[symbol].append(order.order)
                    strategy_traded_today[symbol] = True
                    log_cfg.log_event('order_sent', symbol = symbol, strategy = strat.__name__,
                                      ticket = order.order, type = trade_type, volume = volume,
                                      opt_atr = strategy_opt_atr[symbol], opt_rr = strategy_opt_rr[symbol],
                                      sl = sl, tp = tp, magic = trade_magic_id)
                    _persist()
                else:
                    # backtest: an untradable size uses up the day's trade
                    strategy_traded_today[symbol] = True
                    log_cfg.log_event('order_rejected', symbol = symbol, strategy = strat.__name__,
                                      reason = 'invalid position size', volume = volume,
                                      opt_atr = strategy_opt_atr[symbol], opt_rr = strategy_opt_rr[symbol],
                                      min_volume = min_vol_lots, max_volume = max_volume_lots)
                    _persist()

        # exceptions
        # error printed to console
        # events logged to notepad using log_event()
        # IPC link may still hold despite execption being raised
        # error raised by connect_account()
        except mt5_err.MT5ConnectionError as e:
            logging.error(f'{e}: connection lost, reconnecting')
            time.sleep(5)
            try:
                mt5_conn.connect_account()
            except mt5_err.MT5ConnectionError as reconnect_error:
                logging.error(f'{reconnect_error}: reconnect failed, retrying next poll')

        # raised by data pulling functions
        except mt5_err.MT5RatesError as e:
            logging.error(f'{e}: batch fetch failed, will retry next poll')
            _reconnect_if_disconnected()

        # error raised by position_get() / history_deals_get()
        except mt5_err.MT5PositionError as e:
            logging.error(f'{e}: get positions failed, position with ticket does not exist')
            _reconnect_if_disconnected()

        # error raised by order_send() / close_position()
        except mt5_err.MT5OrderError as e:
            logging.error(f'{e}: order failed, skipping this signal')
            log_cfg.log_event('order_rejected', symbol = symbol, strategy = getattr(strat, '__name__', None),
                              reason = 'mt5_order_error', error = str(e))
            _reconnect_if_disconnected()

        # untraced errors
        except Exception as e:
            logging.error(f'{e}: untracked failure')
            logging.exception('unhandled exception in live loop')
            _reconnect_if_disconnected()

        # poll next request
        time.sleep(1) # re-run logic every 1 second

# --- HELPERS ---
def _utc_now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))

def _bar_day(bar_time, trading_time_zone: str = TRADING_TIME_ZONE) -> pd.Timestamp:
    """trading time zone day of a bar (epoch seconds or naive UTC timestamp, as mt5 returns them)"""
    ts_ = pd.Timestamp(bar_time, unit = 's') if isinstance(bar_time, (int, float, np.integer, np.floating)) else pd.Timestamp(bar_time)
    ts_ = ts_.tz_localize('UTC') if ts_.tz is None else ts_
    return ts_.tz_convert(trading_time_zone).normalize()

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
        try:
            mt5_conn.connect_account() # will not print connection validation
        except mt5_err.MT5ConnectionError as e:
            logging.error(f'{e}: reconnect failed, retrying next poll')


# --- RUN SCRIPT ---
if __name__ == '__main__':
    symbols_strats = {
        'GBPUSD': ts.session_breakout,
        'GOOG': ts.vwap_breakout,
        'AAPL': ts.vwap_breakout,
    }
    strat_ids = {'GBPUSD': 1, 'GOOG': 2, 'AAPL': 3}
    # per asset backtest configuration, costs are single trip: spread in $ of price, commission in % of volume traded
    # forex commission is charged per lot in the backtest (dollar_commission_lot), so commission is 0.0 here
    symbols_config = {
        'GBPUSD': {'forex': True, 'spread': 0.00006, 'commission': 0.0, 'model_ensemble': True},
        'GOOG': {'forex': False, 'spread': 0.0, 'commission': 0.00002, 'model_ensemble': True},
        'AAPL': {'forex': False, 'spread': 0.0, 'commission': 0.00002, 'model_ensemble': True},
    }
    run_live_loop(symbols_strats, strat_ids, symbols_config = symbols_config)
