"""
Behavioural tests for run_live_loop() in live_trading/live_loop.py.

run_live_loop() is an infinite polling loop, so time.sleep is patched to raise a sentinel after a set
number of calls. The sleep(1) at the end of each pass sits outside the loop's try/except, so it unwinds
the loop deterministically. Every external dependency (mt5_conn, log_cfg, optimiser, ML model, indicator
helpers, state file) is patched onto the live_loop module: no terminal, disk or model training involved.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import live_loop


class _StopLoop(Exception):
    pass


def _bars(n = 40, close = 1.10):
    idx = pd.date_range('2026-01-01', periods = n, freq = '5min')
    return pd.DataFrame({'open': close, 'high': close + 0.001, 'low': close - 0.001, 'close': close, 'volume': 100.0}, index = idx)

def _no_signal(data, **kw):
    return pd.Series([False] * len(data)), np.zeros(len(data), dtype = int)

def _long_signal(data, **kw):
    return pd.Series([True] * len(data)), np.ones(len(data), dtype = int)

def _short_signal(data, **kw):
    return pd.Series([True] * len(data)), -np.ones(len(data), dtype = int)


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
    for k, v in {'META_INITIAL_ACCOUNT_BALANCE': '10000', 'META_WFA_IN_SAMPLE_DAYS': '30', 'META_ML_FEATURE_WINDOW': '5',
                 'META_ML_CONSEC_LOSS_BEFORE_TRADE_SKIP': '3', 'META_TOTAL_RISK_ALLOCATION': '0.01',
                 'META_MIN_VOLUME': '0.01', 'META_MAX_FOREX_LEVERAGE': '30', 'META_MAX_OTHER_LEVERAGE': '5'}.items():
        monkeypatch.setenv(k, v)


@pytest.fixture
def conn(monkeypatch):
    fake = MagicMock(name = 'mt5_conn')
    fake.get_latest_bar_time.return_value = 1_700_000_000
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
    monkeypatch.setattr(live_loop.mt5, 'account_info', MagicMock(return_value = None))
    return info


@pytest.fixture
def model(monkeypatch):
    """optimiser + ML model stubs; model['valid'] drives the trade filter"""
    opt = MagicMock(name = 'opt')
    opt.run_grid_search.return_value = []
    opt.grid_search_statistics.return_value = {
        'opt_df': pd.DataFrame({'atr_pct': [0.01], 'regime': [0], 'breakout_conv': [0.1], 'rel_volume': [100.0], 'ml_label': [1]}),
        'opt_atr': 2.0, 'opt_rr': 2.0}
    monkeypatch.setattr(live_loop, 'opt', opt)

    knob = {'valid': True}

    class FakeEnsemble:
        model = object()
        def __init__(self, *a, **k): pass
        def train_model(self, *a, **k): pass
        def model_predict(self, *a, **k): return knob['valid']

    monkeypatch.setattr(live_loop, 'RandomForestVol', FakeEnsemble)
    monkeypatch.setattr(live_loop, 'atr', lambda d, *a, **k: np.full(len(d), 0.01))
    monkeypatch.setattr(live_loop, 'categorise_regime', lambda x: np.zeros(len(x)))
    monkeypatch.setattr(live_loop, 'r2', lambda y: 0.0)
    monkeypatch.setattr(live_loop, 'position_sizing', lambda *a, **k: (1100.0, 1.1))  # 1000 units at 1.10
    return knob


@pytest.fixture
def kw(conn, log, state, symbol_info, model):
    return dict(symbols_strats = {'GBPUSD': _no_signal}, strat_ids = {'GBPUSD': 1}, strat_weights = [1.0])


def _last_state(state):
    return state.call_args.args[0]


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
        symbol_info.path = 'Stocks\\US\\KO'
        symbol_info.volume_step = 1
        kw['symbols_strats'] = {'GBPUSD': _long_signal}
        conn.send_order.return_value = SimpleNamespace(order = 1)
        _run(monkeypatch, kw)
        assert conn.send_order.call_args.args[2] == pytest.approx(1000)

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
        conn.get_latest_bar_time.side_effect = [1, 2, 3]  # a new bar every poll
        _run(monkeypatch, kw, passes = 3)
        assert conn.send_order.call_count == 1


# --- trade filters / guards ---
class TestTradeGuards:
    def test_model_rejection_blocks_a_valid_signal(self, monkeypatch, kw, conn, model):
        kw['symbols_strats'] = {'GBPUSD': _long_signal}
        model['valid'] = False
        _run(monkeypatch, kw)
        conn.send_order.assert_not_called()

    @pytest.mark.parametrize('min_lots, order_sent', [('0.01', True), ('0.05', False)])
    def test_volume_below_broker_minimum_is_rejected_and_logged(self, monkeypatch, kw, conn, log, min_lots, order_sent):
        monkeypatch.setenv('META_MIN_VOLUME', min_lots)
        kw['symbols_strats'] = {'GBPUSD': _long_signal}
        conn.send_order.return_value = SimpleNamespace(order = 1)
        _run(monkeypatch, kw)

        assert conn.send_order.called is order_sent
        if not order_sent:
            assert log.log_event.call_args.args[0] == 'order_rejected'

    def test_volume_above_leverage_cap_is_rejected(self, monkeypatch, kw, conn, model):
        monkeypatch.setattr(live_loop, 'position_sizing', lambda *a, **k: (1e9, 1e5))
        kw['symbols_strats'] = {'GBPUSD': _long_signal}
        _run(monkeypatch, kw)
        conn.send_order.assert_not_called()

    def test_consecutive_loss_threshold_pauses_trading(self, monkeypatch, kw, conn, state):
        monkeypatch.setattr(live_loop, '_load_state', lambda: {'strategy_consec_loss': {'GBPUSD': 3}})
        kw['symbols_strats'] = {'GBPUSD': _long_signal}
        _run(monkeypatch, kw)
        conn.send_order.assert_not_called()

    def test_no_reentry_after_todays_trade_has_closed(self, monkeypatch, kw, conn):
        monkeypatch.setattr(live_loop, '_load_state', lambda: {'strategy_traded_today': {'GBPUSD': True}})
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

    def test_unexpected_exception_does_not_crash_loop(self, monkeypatch, kw, conn):
        kw['symbols_strats'] = {'GBPUSD': lambda d, **k: 1 / 0}
        _run(monkeypatch, kw, passes = 2)
        conn.send_order.assert_not_called()

    def test_initial_connection_is_retried_until_it_succeeds(self, monkeypatch, kw, conn):
        conn.connect_account.side_effect = [live_loop.mt5_err.MT5ConnectionError('down'), None]
        calls = _run(monkeypatch, kw)
        assert calls[0] == 5 and conn.connect_account.call_count == 2


# --- restart / end-of-day ---
class TestRestartAndEndOfDay:
    OPEN = {'strategy_open_tickets': {'GBPUSD': [900]}, 'strategy_live_trade': {'GBPUSD': True}}

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

    def test_loss_streak_breach_skips_trading_and_resets_counter(self, monkeypatch, kw, conn, state):
        monkeypatch.setattr(live_loop, '_load_state', lambda: {'strategy_consec_loss': {'GBPUSD': 3}})  # META_ML_CONSEC_LOSS_BEFORE_TRADE_SKIP = 3
        _run(monkeypatch, kw)
        saved = _last_state(state)
        assert saved['strategy_skip_trade'] == {'GBPUSD': True}
        assert saved['strategy_consec_loss'] == {'GBPUSD': 0}  # reset, so the skip lifts on the next day
        conn.send_order.assert_not_called()

    def test_open_position_is_force_closed_on_new_day(self, monkeypatch, kw, conn, state):
        monkeypatch.setattr(live_loop, '_load_state', lambda: self.OPEN)
        position = SimpleNamespace(ticket = 900)
        # reconcile at startup + end of pass 1: still open; new-day check in pass 2: still open, then closed
        conn.get_positions.side_effect = [[position], [position], [position], []]
        conn.get_deal_profit.return_value = -30.0

        clock = MagicMock(name = 'datetime')
        clock.today.return_value.now.side_effect = [pd.Timestamp('2026-01-01')] * 2 + [pd.Timestamp('2026-01-02')]
        monkeypatch.setattr(live_loop, 'datetime', clock)
        _run(monkeypatch, kw, passes = 2)

        conn.close_position.assert_called_once_with(position)
        saved = _last_state(state)
        assert saved['strategy_open_tickets'] == {'GBPUSD': []}
        assert saved['strategy_equity_dict']['GBPUSD'] == pytest.approx(9_970.0)
        assert saved['strategy_traded_today'] == {'GBPUSD': False}
