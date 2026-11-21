"""Unit tests for the standalone helpers in live_trading/live_loop.py (state persistence, reconnect, multi-timeframe)."""
import json
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import live_loop


# --- state persistence (crash/restart safety) ---
class TestStatePersistence:
    @pytest.fixture(autouse = True)
    def state_path(self, monkeypatch, tmp_path):
        path = tmp_path / 'logs' / 'live_state.json'
        monkeypatch.setattr(live_loop, 'STATE_PATH', str(path))
        return path

    def test_load_returns_none_when_no_previous_run(self):
        assert live_loop._load_state() is None

    def test_save_then_load_roundtrips_and_creates_directory(self):
        state = {'strategy_open_tickets': {'GBPUSD': [11]}, 'time_last_wfa': '2026-01-01T00:00:00'}
        live_loop._save_state(state)
        assert live_loop._load_state() == state

    def test_numpy_floats_are_serialised(self, state_path):
        live_loop._save_state({'strategy_equity_dict': {'GBPUSD': np.float64(10_050.5)}})
        assert json.loads(state_path.read_text())['strategy_equity_dict']['GBPUSD'] == 10_050.5

    def test_save_is_atomic_and_leaves_no_tmp_file(self, state_path):
        live_loop._save_state({'a': 1})
        live_loop._save_state({'a': 2})
        assert live_loop._load_state() == {'a': 2}
        assert not (state_path.parent / 'live_state.json.tmp').exists()


# --- reconnect ---
class TestReconnect:
    @pytest.fixture
    def conn(self, monkeypatch):
        fake = MagicMock(name = 'mt5_conn')
        monkeypatch.setattr(live_loop, 'mt5_conn', fake)
        monkeypatch.setattr(live_loop.time, 'sleep', lambda s: None)
        return fake

    def test_no_reconnect_while_terminal_link_is_alive(self, conn):
        conn.is_connected.return_value = True
        live_loop._reconnect_if_disconnected()
        conn.connect_account.assert_not_called()

    def test_reconnects_when_terminal_link_is_down(self, conn):
        conn.is_connected.return_value = False
        live_loop._reconnect_if_disconnected()
        conn.connect_account.assert_called_once()


# --- strategy data preparation (must match the backtest's mt5 data path) ---
def _raw_bars(n = 3, start = '2026-07-01 12:00'):
    idx = pd.date_range(start, periods = n, freq = '5min') # naive UTC, as mt5 returns them
    return pd.DataFrame({'open': 1.0, 'high': 1.1, 'low': 0.9, 'close': 1.0, 'volume': 10.0,
                         'spread': 1, 'real_volume': 0}, index = idx)

class TestPrepareStrategyData:
    def test_index_is_converted_from_utc_to_trading_time_zone_and_only_ohlcv_kept(self):
        out = live_loop.prepare_strategy_data(lambda d: None, _raw_bars())
        assert str(out.index.tz) == 'Europe/London'
        assert out.index[0] == pd.Timestamp('2026-07-01 13:00', tz = 'Europe/London') # BST = UTC + 1
        assert list(out.columns) == ['open', 'high', 'low', 'close', 'volume']

    def test_input_frame_is_not_mutated(self):
        raw = _raw_bars()
        live_loop.prepare_strategy_data(lambda d: None, raw)
        assert raw.index.tz is None and 'spread' in raw.columns

    def test_session_breakout_goes_through_session_breakout_data_after_tz_conversion(self, monkeypatch):
        seen = []
        monkeypatch.setattr(live_loop.ts, 'session_breakout_data', lambda d: seen.append(str(d.index.tz)) or d.assign(mtf = True))
        out = live_loop.prepare_strategy_data(live_loop.ts.session_breakout, _raw_bars())
        assert seen == ['Europe/London'] and out['mtf'].all()

    def test_vwap_breakout_is_not_multi_timeframe(self, monkeypatch):
        monkeypatch.setattr(live_loop.ts, 'session_breakout_data', lambda d: pytest.fail('should not be called'))
        live_loop.prepare_strategy_data(live_loop.ts.vwap_breakout, _raw_bars())


# --- walk forward fold grid / trading day ---
class TestWalkForwardSchedule:
    ANCHOR = pd.Timestamp('2026-01-15', tz = 'UTC')
    IS, OOS = pd.DateOffset(months = 6), pd.DateOffset(months = 1)

    @pytest.mark.parametrize('now, fold_end', [
        ('2026-07-15 00:00', '2026-07-15'),  # first fold starts exactly at anchor + initial_is
        ('2026-08-14 23:59', '2026-07-15'),  # still within the first oos window
        ('2026-08-15 00:00', '2026-08-15'),  # next fold
        ('2027-01-20 00:00', '2027-01-15'),  # several folds later (eg. after downtime)
    ])
    def test_fold_end_steps_by_oos_from_anchor(self, now, fold_end):
        out = live_loop.wfa_fold_end(self.ANCHOR, pd.Timestamp(now, tz = 'UTC'), self.IS, self.OOS)
        assert out == pd.Timestamp(fold_end, tz = 'UTC')

    def test_bar_day_is_the_trading_time_zone_day(self):
        # 23:30 UTC in summer is 00:30 next day in London
        assert live_loop._bar_day(pd.Timestamp('2026-07-01 23:30')) == pd.Timestamp('2026-07-02', tz = 'Europe/London')
        assert live_loop._bar_day(np.int64(pd.Timestamp('2026-01-01 23:30').timestamp())) == pd.Timestamp('2026-01-01', tz = 'Europe/London')


# --- one walk forward fold (mirrors notebooks/backtest.ipynb run_wfa) ---
class TestSelectWfaModel:
    @pytest.fixture
    def data(self):
        idx = pd.date_range('2026-01-05', periods = 10 * 288, freq = '5min', tz = 'Europe/London') # 10 days of 5min bars
        return pd.DataFrame({'open': 1.0, 'high': 1.1, 'low': 0.9, 'close': 1.0, 'volume': 1.0}, index = idx)

    @pytest.fixture
    def fake_backtest(self, monkeypatch):
        """records every backtest call; returns one labelled signal bar per day and an equity curve whose score is set by (atr, rr)"""
        calls = []

        def backtest(data, atr_multiplier, risk_reward, ensamble_signal_validation = None, warmup_bars = 0, **kw):
            calls.append({'start': data.index[0], 'end': data.index[-1], 'warmup': warmup_bars, 'atr': atr_multiplier,
                          'rr': risk_reward, 'model': ensamble_signal_validation, 'kw': kw})
            out = data.iloc[warmup_bars:].copy()
            for f in ['atr_pct', 'breakout_conv', 'rel_volume']:
                out[f] = 0.1
            out['regime'] = 'consolidation'
            out['ml_label'] = None
            out.loc[out.index.hour == 12, 'ml_label'] = 1 # one label per day (12:00 .. 12:55 -> keep first)
            out['adjusted_returns'] = 0.0
            out['equity'] = 1e4 * (1 + 0.001 * atr_multiplier * risk_reward) # higher atr * rr -> higher score
            return {'data': out}

        monkeypatch.setattr(live_loop, 'backtest_strategy', backtest)
        monkeypatch.setattr(live_loop.cv.PropFirmValidation, 'strategy_challenge_summary',
                            lambda self: setattr(self, 'strategy_score', float(self.eq.iloc[-1])))
        return calls

    def test_grid_history_only_sees_pre_split_bars_and_eval_is_warmed_up(self, data, fake_backtest):
        sel = live_loop.select_wfa_model(data, lambda d: None, 1e4, atr_multipliers = [2.0], risk_rewards = [3.0], n_jobs = 1)
        split = int(np.floor(len(data) * 0.7))
        history, is_eval = fake_backtest
        assert sel['split_time'] == data.index[split]
        assert history['end'] == data.index[split - 1] and history['warmup'] == 0
        assert is_eval['end'] == data.index[-1]
        assert is_eval['start'] <= data.index[split].normalize() - pd.Timedelta(days = 1) # back to the prior day's midnight
        assert is_eval['start'] + is_eval['warmup'] * pd.Timedelta('5min') == data.index[split]
        assert is_eval['model'] is sel['ensemble'].model and sel['ensemble'].model is not None

    def test_labels_from_the_last_day_before_the_split_are_dropped(self, data, fake_backtest):
        sel = live_loop.select_wfa_model(data, lambda d: None, 1e4, atr_multipliers = [2.0], risk_rewards = [3.0], n_jobs = 1)
        last_day = data.index[int(np.floor(len(data) * 0.7)) - 1].normalize()
        assert (sel['ensemble'].x.index.normalize() < last_day).all()

    def test_best_in_sample_eval_score_is_selected(self, data, fake_backtest):
        sel = live_loop.select_wfa_model(data, lambda d: None, 1e4, atr_multipliers = [2.0, 5.0], risk_rewards = [3.0, 6.0], n_jobs = 1)
        assert (sel['opt_atr'], sel['opt_rr']) == (5.0, 6.0)

    def test_without_model_ensemble_signals_are_unfiltered_and_no_history_is_run(self, data, fake_backtest):
        sel = live_loop.select_wfa_model(data, lambda d: None, 1e4, model_ensemble = False,
                                         atr_multipliers = [2.0], risk_rewards = [3.0], n_jobs = 1)
        assert sel['ensemble'] is None
        assert len(fake_backtest) == 1 and fake_backtest[0]['model'] is None

    def test_live_risk_settings_reach_every_backtest(self, data, fake_backtest):
        live_loop.select_wfa_model(data, lambda d: None, 1e4, forex = True, spread_dollars = 6e-5, consec_loss_guard = 3,
                                   feature_window = 30, fraction_risked = 0.01, atr_multipliers = [2.0], risk_rewards = [3.0], n_jobs = 1)
        for call in fake_backtest:
            assert call['kw']['forex'] is True and call['kw']['spread_dollar'] == 6e-5
            assert call['kw']['consecutive_loss_before_skip'] == 3 and call['kw']['feature_window'] == 30

    def test_window_too_short_to_split_returns_none(self, data):
        assert live_loop.select_wfa_model(data.iloc[:1], lambda d: None, 1e4) is None
