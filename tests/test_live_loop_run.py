"""
Behavioural tests for run_live_loop() in live_trading/live_loop.py.

run_live_loop() is an infinite polling loop, so time.sleep is patched to raise a sentinel after a set
number of calls. The sleep(1) at the end of each pass sits outside the loop's try/except, so it unwinds
the loop deterministically. Every external dependency (mt5_conn, log_cfg, walk forward selection, ML model,
feature helpers, clock, state file) is patched onto the live_loop module: no terminal, disk or model training involved.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import live_loop


class _StopLoop(Exception):
    pass


NOW = pd.Timestamp('2026-01-01 03:30', tz = 'UTC')
DAY1 = int(pd.Timestamp('2026-01-01 03:25').timestamp()) # latest bar epoch seconds (naive UTC, as mt5 returns them)
DAY2 = int(pd.Timestamp('2026-01-02 08:00').timestamp())


def _bars(n = 40, close = 1.10, start = '2026-01-01 00:00'):
    idx = pd.date_range(start, periods = n, freq = '5min')
    return pd.DataFrame({'open': close, 'high': close + 0.001, 'low': close - 0.001, 'close': close, 'volume': 100.0}, index = idx)

def _no_signal(data, **kw):
    return pd.Series([False] * len(data), index = data.index), np.zeros(len(data), dtype = int)

def _long_signal(data, **kw):
    return pd.Series([True] * len(data), index = data.index), np.ones(len(data), dtype = int)

def _short_signal(data, **kw):
    return pd.Series([True] * len(data), index = data.index), -np.ones(len(data), dtype = int)


def _run(monkeypatch, kwargs, passes = 1):
    """run the loop for `passes` full polling passes, then stop; returns the list of sleep() durations"""
    calls = []

    def fake_sleep(seconds):
        calls.append(seconds)
        if len(calls) > passes - 1 + sum(1 for c in calls if c == 5):
            raise _StopLoop()

    monkeypatch.setattr(live_loop.time, 'sleep', fake_sleep)
    with pytest.raises(_StopLoop):
        live_loop.run_live_loop(**kwargs)
    return calls


@pytest.fixture(autouse = True)
def env(monkeypatch):
    for k, v in {'META_INITIAL_ACCOUNT_BALANCE': '10000', 'META_ML_FEATURE_WINDOW': '5',
                 'META_ML_CONSEC_LOSS_BEFORE_TRADE_SKIP': '3', 'META_TOTAL_RISK_ALLOCATION': '0.01',
                 'META_MIN_VOLUME': '0.01', 'META_MAX_FOREX_LEVERAGE': '30', 'META_MAX_OTHER_LEVERAGE': '5'}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(live_loop, '_utc_now', lambda: NOW)


@pytest.fixture
def conn(monkeypatch):
    fake = MagicMock(name = 'mt5_conn')
    fake.get_latest_bar_time.return_value = DAY1
    fake.get_bars_range.side_effect = lambda *a, **k: _bars()
    monkeypatch.setattr(live_loop, 'mt5_conn', fake)
    return fake


@pytest.fixture
def log(monkeypatch):
    fake = MagicMock(name = 'log_cfg')
    monkeypatch.setattr(live_loop, 'log_cfg', fake)
    return fake


@pytest.fixture
def state(monkeypatch):
    """no state file on disk; captures every _save_state() payload"""
    saved = MagicMock(name = '_save_state')
    monkeypatch.setattr(live_loop, '_load_state', lambda: None)
    monkeypatch.setattr(live_loop, '_save_state', saved)
    return saved


@pytest.fixture
def symbol_info(monkeypatch):
    info = SimpleNamespace(path = 'Forex\\Majors\\GBPUSD', digits = 5, volume_step = 0.01)
    monkeypatch.setattr(live_loop.mt5, 'symbol_info', MagicMock(return_value = info))
    return info


class FakeEnsemble:
    def __init__(self, knob, trained = True):
        self.knob = knob
        self.model = object() if trained else None
    def model_predict(self, features):
        return np.array([int(self.knob['valid'])])


@pytest.fixture
def model(monkeypatch):
    """walk forward selection + feature stubs; model['valid'] drives the trade filter, model['select'] records calls"""
    knob = {'valid': True, 'trained': True, 'select': MagicMock(name = 'select_wfa_model')}

    def fake_select(is_data, strat, account_balance, **kw):
        knob['select'](is_data, strat, account_balance, **kw)
        return {'opt_atr': 2.0, 'opt_rr': 2.0, 'ensemble': FakeEnsemble(knob, knob['trained']),
                'is_eval_score': 1.0, 'split_time': is_data.index[-1]}

    monkeypatch.setattr(live_loop, 'select_wfa_model', fake_select)
    monkeypatch.setattr(live_loop, 'signal_bar_features', lambda d, w: ({f: [0.0] for f in live_loop.ML_FEATURES}, 0.01))
    monkeypatch.setattr(live_loop, 'position_sizing', lambda *a, **k: (1100.0, 1.1))  # 1000 units at 1.10
    return knob


@pytest.fixture
def kw(conn, log, state, symbol_info, model):
    return dict(symbols_strats = {'GBPUSD': _no_signal}, strat_ids = {'GBPUSD': 1}, strat_weights = [1.0])


def _last_state(state):
    return state.call_args.args[0]

def _events(log, name):
    return [c for c in log.log_event.call_args_list if c.args[0] == name]


# --- signal -> order ---
class TestOrderPlacement:
    def test_no_signal_sends_no_order(self, monkeypatch, kw, conn):
        _run(monkeypatch, kw)
        conn.send_order.assert_not_called()

    @pytest.mark.parametrize('strat, side, sl, tp', [(_long_signal, 'LONG', 1.08, 1.14), (_short_signal, 'SHORT', 1.12, 1.06)])
    def test_sl_tp_are_placed_on_the_correct_side_of_entry(self, monkeypatch, kw, conn, strat, side, sl, tp):
        kw['symbols_strats'] = {'GBPUSD': strat}
        conn.send_order.return_value = SimpleNamespace(order = 555)
        _run(monkeypatch, kw)

        args, call_kw = conn.send_order.call_args
        assert (args[0], args[1]) == (1, 'GBPUSD')  # strategy magic id, symbol
        assert args[3] == pytest.approx(sl) and args[4] == pytest.approx(tp)
        assert call_kw['type'] == side

    def test_forex_volume_is_in_standard_lots(self, monkeypatch, kw, conn):
        kw['symbols_strats'] = {'GBPUSD': _long_signal}
        conn.send_order.return_value = SimpleNamespace(order = 1)
        _run(monkeypatch, kw)
        assert conn.send_order.call_args.args[2] == pytest.approx(0.01)  # 1000 units / 1e5

    def test_non_forex_volume_is_in_asset_units(self, monkeypatch, kw, conn, symbol_info):
        symbol_info.path = 'Stocks\\US\\GOOG'
        symbol_info.volume_step = 1
        kw['symbols_strats'] = {'GBPUSD': _long_signal}
        conn.send_order.return_value = SimpleNamespace(order = 1)
        _run(monkeypatch, kw)
        assert conn.send_order.call_args.args[2] == pytest.approx(1000)

    def test_symbols_config_forex_flag_overrides_broker_path(self, monkeypatch, kw, conn, symbol_info):
        symbol_info.path = 'Stocks\\US\\GOOG'
        kw['symbols_strats'] = {'GBPUSD': _long_signal}
        kw['symbols_config'] = {'GBPUSD': {'forex': True}}
        conn.send_order.return_value = SimpleNamespace(order = 1)
        _run(monkeypatch, kw)
        assert conn.send_order.call_args.args[2] == pytest.approx(0.01)

    def test_risk_is_a_fraction_of_strategy_equity(self, monkeypatch, kw, conn):
        seen = []
        monkeypatch.setattr(live_loop, 'position_sizing', lambda *a: seen.append(a) or (1100.0, 1.1))
        kw.update(symbols_strats = {'GBPUSD': _long_signal, 'GOOG': _no_signal}, strat_ids = {'GBPUSD': 1, 'GOOG': 2},
                  strat_weights = [0.25, 0.75])
        conn.send_order.return_value = SimpleNamespace(order = 1)
        _run(monkeypatch, kw)
        entry, sl_dist, equity, frac = seen[0]
        assert (entry, sl_dist, equity, frac) == pytest.approx((1.10, 0.02, 2500.0, 0.01))  # backtest: equity[t] * risk_per_trade

    def test_ticket_and_traded_today_are_persisted_after_fill(self, monkeypatch, kw, conn, state):
        kw['symbols_strats'] = {'GBPUSD': _long_signal}
        conn.send_order.return_value = SimpleNamespace(order = 777)
        _run(monkeypatch, kw)

        saved = _last_state(state)
        assert saved['strategy_open_tickets'] == {'GBPUSD': [777]}
        assert saved['strategy_live_trade'] == {'GBPUSD': True}
        assert saved['strategy_traded_today'] == {'GBPUSD': True}

    def test_only_one_order_per_symbol_across_polls(self, monkeypatch, kw, conn):
        kw['symbols_strats'] = {'GBPUSD': _long_signal}
        conn.send_order.return_value = SimpleNamespace(order = 1)
        conn.get_positions.return_value = [SimpleNamespace()]  # position still open
        conn.get_latest_bar_time.side_effect = [DAY1, DAY1 + 300, DAY1 + 600]  # a new bar every poll, same day
        _run(monkeypatch, kw, passes = 3)
        assert conn.send_order.call_count == 1


# --- trade filters / guards (backtest entry conditions) ---
class TestTradeGuards:
    def test_model_rejection_blocks_the_rest_of_the_day(self, monkeypatch, kw, conn, model, state):
        kw['symbols_strats'] = {'GBPUSD': _long_signal}
        model['valid'] = False
        _run(monkeypatch, kw)
        conn.send_order.assert_not_called()
        assert _last_state(state)['strategy_traded_today'] == {'GBPUSD': True}

    def test_untrained_model_trades_unfiltered_like_the_backtest(self, monkeypatch, kw, conn, model):
        kw['symbols_strats'] = {'GBPUSD': _long_signal}
        model['valid'], model['trained'] = False, False  # predict would reject, but an untrained model isn't consulted
        conn.send_order.return_value = SimpleNamespace(order = 1)
        _run(monkeypatch, kw)
        conn.send_order.assert_called_once()

    def test_signal_on_the_prior_days_last_bar_is_not_entered_on_a_new_day(self, monkeypatch, kw, conn):
        # last closed bar 23:55 London, forming bar 00:00 next day -> backtest new_session guard blocks the entry
        conn.get_bars_range.side_effect = lambda *a, **k: _bars(n = 40, start = '2026-01-01 20:45')
        conn.get_latest_bar_time.return_value = int(pd.Timestamp('2026-01-02 00:00').timestamp())
        kw['symbols_strats'] = {'GBPUSD': _long_signal}
        _run(monkeypatch, kw)
        conn.send_order.assert_not_called()

    def test_invalid_atr_on_signal_bar_skips_entry(self, monkeypatch, kw, conn):
        monkeypatch.setattr(live_loop, 'signal_bar_features', lambda d, w: ({}, float('nan')))
        kw['symbols_strats'] = {'GBPUSD': _long_signal}
        _run(monkeypatch, kw)
        conn.send_order.assert_not_called()

    @pytest.mark.parametrize('min_lots, order_sent', [('0.01', True), ('0.05', False)])
    def test_volume_below_broker_minimum_is_rejected_and_uses_up_the_day(self, monkeypatch, kw, conn, log, state, min_lots, order_sent):
        monkeypatch.setenv('META_MIN_VOLUME', min_lots)
        kw['symbols_strats'] = {'GBPUSD': _long_signal}
        conn.send_order.return_value = SimpleNamespace(order = 1)
        _run(monkeypatch, kw)

        assert conn.send_order.called is order_sent
        if not order_sent:
            assert log.log_event.call_args.args[0] == 'order_rejected'
            assert _last_state(state)['strategy_traded_today'] == {'GBPUSD': True}

    def test_volume_above_leverage_cap_is_rejected(self, monkeypatch, kw, conn, model):
        monkeypatch.setattr(live_loop, 'position_sizing', lambda *a, **k: (1e9, 1e5))
        kw['symbols_strats'] = {'GBPUSD': _long_signal}
        _run(monkeypatch, kw)
        conn.send_order.assert_not_called()

    def test_no_reentry_after_todays_trade_has_closed(self, monkeypatch, kw, conn):
        monkeypatch.setattr(live_loop, '_load_state', lambda: {'strategy_traded_today': {'GBPUSD': True},
                                                               'strategy_current_day': {'GBPUSD': '2026-01-01T00:00:00+00:00'}})
        kw['symbols_strats'] = {'GBPUSD': _long_signal}
        _run(monkeypatch, kw)
        conn.send_order.assert_not_called()

    def test_unchanged_bar_does_not_refetch_or_reevaluate(self, monkeypatch, kw, conn):
        _run(monkeypatch, kw, passes = 3)  # same bar time every poll
        assert conn.get_latest_bar_time.call_count == 3
        assert conn.get_bars_range.call_count == 2  # 1 WFA fetch + 1 signal fetch, none on the repeat polls

    def test_empty_market_data_skips_poll_without_error(self, monkeypatch, kw, conn):
        conn.get_bars_range.side_effect = [_bars(), pd.DataFrame()]  # WFA ok, then closed-market empty frame
        kw['symbols_strats'] = {'GBPUSD': _long_signal}
        _run(monkeypatch, kw)
        conn.send_order.assert_not_called()

    def test_unavailable_symbol_skips_and_checks_connection(self, monkeypatch, kw, conn):
        monkeypatch.setattr(live_loop.mt5, 'symbol_info', MagicMock(return_value = None))
        _run(monkeypatch, kw)
        conn.is_connected.assert_called()
        conn.send_order.assert_not_called()


# --- walk forward schedule ---
class TestWalkForward:
    def test_first_run_anchors_the_expanding_window_initial_is_back_and_persists_it(self, monkeypatch, kw, conn, model, state):
        _run(monkeypatch, kw)
        saved = _last_state(state)
        assert saved['wfa_anchor'] == pd.Timestamp('2025-07-01', tz = 'UTC').isoformat()
        assert saved['time_last_wfa'] == pd.Timestamp('2026-01-01', tz = 'UTC').isoformat()
        date_from, date_to = conn.get_bars_range.call_args_list[0].kwargs.values()
        assert (pd.Timestamp(date_from), pd.Timestamp(date_to)) == (pd.Timestamp('2025-07-01', tz = 'UTC'), pd.Timestamp('2026-01-01', tz = 'UTC'))

    def test_in_sample_data_is_prepared_in_the_trading_time_zone(self, monkeypatch, kw, model):
        _run(monkeypatch, kw)
        is_data = model['select'].call_args.args[0]
        assert str(is_data.index.tz) == 'Europe/London'
        assert len(is_data) == len(_bars()) - 1  # last (possibly forming) bar dropped

    def test_selection_uses_the_live_risk_settings(self, monkeypatch, kw, model):
        kw['symbols_config'] = {'GBPUSD': {'forex': True, 'spread': 0.00006, 'commission': 0.0, 'model_ensemble': False}}
        _run(monkeypatch, kw)
        call_kw = model['select'].call_args.kwargs
        assert call_kw['forex'] is True and call_kw['spread_dollars'] == 0.00006 and call_kw['model_ensemble'] is False
        assert call_kw['consec_loss_guard'] == 3 and call_kw['feature_window'] == 5 and call_kw['fraction_risked'] == 0.01
        assert model['select'].call_args.args[2] == pytest.approx(10_000.0)  # strategy equity

    def test_models_are_not_refit_within_a_fold(self, monkeypatch, kw, model):
        _run(monkeypatch, kw, passes = 3)
        assert model['select'].call_count == 1

    def test_restart_refits_on_the_persisted_fold_grid(self, monkeypatch, kw, conn, model):
        monkeypatch.setattr(live_loop, '_load_state', lambda: {'wfa_anchor': '2025-06-15T00:00:00+00:00'})
        _run(monkeypatch, kw)
        assert model['select'].call_count == 1  # models aren't persisted -> rebuilt
        date_from, date_to = conn.get_bars_range.call_args_list[0].kwargs.values()
        assert (pd.Timestamp(date_from), pd.Timestamp(date_to)) == (pd.Timestamp('2025-06-15', tz = 'UTC'), pd.Timestamp('2025-12-15', tz = 'UTC'))

    def test_a_symbol_whose_selection_fails_is_not_traded_but_others_are(self, monkeypatch, kw, conn, model):
        def fake_select(is_data, strat, account_balance, **k):
            if strat is _short_signal:
                raise ValueError('boom')
            return {'opt_atr': 2.0, 'opt_rr': 2.0, 'ensemble': None, 'is_eval_score': 0.0, 'split_time': is_data.index[-1]}
        monkeypatch.setattr(live_loop, 'select_wfa_model', fake_select)
        conn.is_connected.return_value = True
        kw.update(symbols_strats = {'GBPUSD': _long_signal, 'EURUSD': _short_signal}, strat_ids = {'GBPUSD': 1, 'EURUSD': 2},
                  strat_weights = [0.5, 0.5])
        conn.send_order.return_value = SimpleNamespace(order = 1)
        _run(monkeypatch, kw)
        assert [c.args[1] for c in conn.send_order.call_args_list] == ['GBPUSD']


# --- error handling (loop must survive each failure) ---
class TestErrorHandling:
    def test_rejected_order_is_logged_and_no_ticket_recorded(self, monkeypatch, kw, conn, log, state):
        kw['symbols_strats'] = {'GBPUSD': _long_signal}
        conn.send_order.side_effect = live_loop.mt5_err.MT5OrderError('rejected')
        _run(monkeypatch, kw)

        assert any(c.args[0] == 'order_rejected' and 'error' in c.kwargs for c in log.log_event.call_args_list)
        assert _last_state(state)['strategy_open_tickets'] == {'GBPUSD': []}

    def test_rates_error_is_caught_and_loop_continues(self, monkeypatch, kw, conn):
        conn.get_latest_bar_time.side_effect = live_loop.mt5_err.MT5RatesError('no data')
        _run(monkeypatch, kw, passes = 2)
        assert conn.get_latest_bar_time.call_count == 2

    def test_connection_error_triggers_reconnect_after_backoff(self, monkeypatch, kw, conn):
        conn.get_latest_bar_time.side_effect = live_loop.mt5_err.MT5ConnectionError('lost')
        calls = _run(monkeypatch, kw)
        assert 5 in calls
        assert conn.connect_account.call_count == 2  # initial + reconnect

    def test_failed_reconnect_does_not_crash_loop(self, monkeypatch, kw, conn):
        conn.get_latest_bar_time.side_effect = live_loop.mt5_err.MT5ConnectionError('lost')
        conn.connect_account.side_effect = [None, live_loop.mt5_err.MT5ConnectionError('still down'), None]
        calls = _run(monkeypatch, kw, passes = 2)
        assert calls.count(1) == 2

    def test_position_error_during_reconciliation_does_not_crash_loop(self, monkeypatch, kw, conn):
        monkeypatch.setattr(live_loop, '_load_state', lambda: {'strategy_open_tickets': {'GBPUSD': [900]}})
        conn.get_positions.side_effect = live_loop.mt5_err.MT5PositionError('down')
        _run(monkeypatch, kw, passes = 2)

    def test_unexpected_exception_does_not_crash_loop(self, monkeypatch, kw, conn):
        kw['symbols_strats'] = {'GBPUSD': lambda d, **k: 1 / 0}
        _run(monkeypatch, kw, passes = 2)
        conn.send_order.assert_not_called()

    def test_initial_connection_is_retried_until_it_succeeds(self, monkeypatch, kw, conn):
        conn.connect_account.side_effect = [live_loop.mt5_err.MT5ConnectionError('down'), None]
        calls = _run(monkeypatch, kw)
        assert calls[0] == 5 and conn.connect_account.call_count == 2


# --- restart / trading day boundary / losing streak (backtest consec loss rules) ---
class TestRestartAndDayBoundary:
    OPEN = {'strategy_open_tickets': {'GBPUSD': [900]}, 'strategy_live_trade': {'GBPUSD': True},
            'strategy_current_day': {'GBPUSD': '2026-01-01T00:00:00+00:00'}}

    def test_position_closed_while_down_is_reconciled_on_restart(self, monkeypatch, kw, conn, log, state):
        monkeypatch.setattr(live_loop, '_load_state', lambda: self.OPEN)
        conn.get_positions.return_value = ()  # SL/TP hit while the process was down
        conn.get_deal_profit.return_value = 120.0
        _run(monkeypatch, kw)

        saved = _last_state(state)
        assert saved['strategy_open_tickets'] == {'GBPUSD': []}
        assert saved['strategy_live_trade'] == {'GBPUSD': False}
        assert saved['strategy_equity_dict']['GBPUSD'] == pytest.approx(10_120.0)
        conn.close_position.assert_not_called()

    def test_losing_close_increments_consecutive_losses_and_a_win_resets_them(self, monkeypatch, kw, conn, state):
        monkeypatch.setattr(live_loop, '_load_state', lambda: {**self.OPEN, 'strategy_consec_loss': {'GBPUSD': 1}})
        conn.get_positions.return_value = ()
        conn.get_deal_profit.return_value = -50.0
        _run(monkeypatch, kw)
        assert _last_state(state)['strategy_consec_loss'] == {'GBPUSD': 2}

        conn.get_deal_profit.return_value = 50.0
        _run(monkeypatch, kw)
        assert _last_state(state)['strategy_consec_loss'] == {'GBPUSD': 0}

    def test_stop_out_breaching_the_streak_skips_the_next_day(self, monkeypatch, kw, conn, log, state):
        monkeypatch.setattr(live_loop, '_load_state', lambda: {**self.OPEN, 'strategy_consec_loss': {'GBPUSD': 2},
                                                               'strategy_traded_today': {'GBPUSD': True}})  # stopped trade was today's trade
        conn.get_positions.return_value = ()  # 3rd consecutive loss (META_ML_CONSEC_LOSS_BEFORE_TRADE_SKIP = 3)
        conn.get_deal_profit.return_value = -50.0
        kw['symbols_strats'] = {'GBPUSD': _long_signal}
        conn.get_latest_bar_time.side_effect = [DAY1, DAY2]  # pass 1 same day, pass 2 first bar of the next day
        _run(monkeypatch, kw, passes = 2)

        saved = _last_state(state)
        assert saved['strategy_consec_loss'] == {'GBPUSD': 0}  # reset once the threshold is hit
        assert saved['strategy_skip_next_day'] == {'GBPUSD': False}  # consumed by the day roll
        assert saved['strategy_traded_today'] == {'GBPUSD': True}  # next day skipped
        assert len(_events(log, 'day_skipped')) == 1
        conn.send_order.assert_not_called()

    def test_skip_flag_lasts_exactly_one_day(self, monkeypatch, kw, conn, state):
        day3 = int(pd.Timestamp('2026-01-03 08:00').timestamp())
        monkeypatch.setattr(live_loop, '_load_state', lambda: {'strategy_skip_next_day': {'GBPUSD': True},
                                                               'strategy_current_day': {'GBPUSD': '2026-01-01T00:00:00+00:00'}})
        conn.get_bars_range.side_effect = lambda *a, **k: _bars(start = '2026-01-03 05:00')
        kw['symbols_strats'] = {'GBPUSD': _long_signal}
        conn.send_order.return_value = SimpleNamespace(order = 1)
        conn.get_positions.return_value = [SimpleNamespace()]
        conn.get_latest_bar_time.side_effect = [DAY2, day3]
        _run(monkeypatch, kw, passes = 2)
        conn.send_order.assert_called_once()  # day 2 skipped, day 3 traded

    def test_open_position_is_force_closed_on_the_first_bar_of_a_new_day(self, monkeypatch, kw, conn, log, state):
        monkeypatch.setattr(live_loop, '_load_state', lambda: self.OPEN)
        position = SimpleNamespace(ticket = 900)
        conn.get_positions.return_value = [position]
        conn.get_deal_profit.return_value = -30.0
        conn.get_latest_bar_time.side_effect = [DAY1, DAY2]
        _run(monkeypatch, kw, passes = 2)

        conn.close_position.assert_called_once_with(position)
        assert _events(log, 'position_closed')[0].kwargs['reason'] == 'eod_force_close'
        saved = _last_state(state)
        assert saved['strategy_open_tickets'] == {'GBPUSD': []}
        assert saved['strategy_equity_dict']['GBPUSD'] == pytest.approx(9_970.0)
        assert saved['strategy_traded_today'] == {'GBPUSD': False}
        assert saved['strategy_current_day']['GBPUSD'] == pd.Timestamp('2026-01-02', tz = 'Europe/London').isoformat()

    def test_eod_close_breaching_the_streak_skips_the_same_day(self, monkeypatch, kw, conn, state):
        monkeypatch.setattr(live_loop, '_load_state', lambda: {**self.OPEN, 'strategy_consec_loss': {'GBPUSD': 2}})
        conn.get_positions.return_value = [SimpleNamespace(ticket = 900)]
        conn.get_deal_profit.return_value = -30.0
        conn.get_latest_bar_time.return_value = DAY2
        _run(monkeypatch, kw)
        saved = _last_state(state)
        assert saved['strategy_traded_today'] == {'GBPUSD': True}
        assert saved['strategy_consec_loss'] == {'GBPUSD': 0}

    def test_no_day_roll_without_a_new_days_bar(self, monkeypatch, kw, conn):
        monkeypatch.setattr(live_loop, '_load_state', lambda: self.OPEN)
        conn.get_positions.return_value = [SimpleNamespace(ticket = 900)]
        conn.get_latest_bar_time.side_effect = [DAY1, DAY1 + 300]
        _run(monkeypatch, kw, passes = 2)
        conn.close_position.assert_not_called()

    def test_losing_streak_guard_can_be_disabled(self, monkeypatch, kw, conn, state):
        monkeypatch.setenv('META_ML_CONSEC_LOSS_BEFORE_TRADE_SKIP', 'None')
        monkeypatch.setattr(live_loop, '_load_state', lambda: {**self.OPEN, 'strategy_consec_loss': {'GBPUSD': 5}})
        conn.get_positions.return_value = ()
        conn.get_deal_profit.return_value = -50.0
        _run(monkeypatch, kw)
        saved = _last_state(state)
        assert saved['strategy_consec_loss'] == {'GBPUSD': 6} and saved['strategy_skip_next_day'] == {'GBPUSD': False}
