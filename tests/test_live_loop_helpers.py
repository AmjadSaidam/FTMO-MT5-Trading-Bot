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


# --- multi-timeframe ---
class TestMultiTimeframeColumns:
    def test_strategies_outside_the_set_get_data_unchanged(self):
        data = pd.DataFrame({'close': [1.0, 2.0]})
        out = live_loop.add_multi_timeframe_columns(lambda d: None, data)
        assert out is data

    def test_session_breakout_receives_higher_timeframe_columns(self, monkeypatch):
        data = pd.DataFrame({'close': [1.0, 2.0, 3.0]})
        seen = []
        monkeypatch.setattr(live_loop, 'lower_to_higher_timeframe', lambda d, base, tf: seen.append(tf) or d.add_suffix(f'_{tf}'))
        monkeypatch.setattr(live_loop, 'concatenate_timeframe_data', lambda d, frames: pd.concat([d, *frames], axis = 1))

        out = live_loop.add_multi_timeframe_columns(live_loop.session_breakout, data)

        assert seen == ['1h', '4h', '8h']
        assert {'close_1h', 'close_4h', 'close_8h'} <= set(out.columns)
