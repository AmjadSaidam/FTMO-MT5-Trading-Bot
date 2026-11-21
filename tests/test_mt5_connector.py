"""
Unit tests for mt5/mt5_connector.py.

MetaTrader5 calls are monkeypatched (see the `mt5` fixture in conftest.py): the connector's own logic
(error translation, order-type mapping, frame conversion, position filtering) is what's under test,
not the terminal.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import mt5_connector as conn
import mt5_errors


def _rates(n = 3, start = 1_700_000_000, step = 300):
    return pd.DataFrame({'time': start + np.arange(n) * step, 'open': 1.1, 'high': 1.2, 'low': 1.0,
                         'close': 1.1, 'tick_volume': 10, 'spread': 1, 'real_volume': 0}).to_records(index = False)


# --- connection ---
class TestConnectAccount:
    @pytest.fixture(autouse = True)
    def env(self, monkeypatch):
        monkeypatch.setenv('META_PATH', 'C:/terminal64.exe')
        monkeypatch.setenv('META_LOGIN', '12345')
        monkeypatch.setenv('META_PASSWORD', 'pw')
        monkeypatch.setenv('META_SERVER', 'FTMO-Demo')

    def test_login_is_cast_to_int_and_stale_session_closed_first(self, mt5):
        mt5.initialize.return_value = True
        conn.connect_account()

        mt5.shutdown.assert_called_once()
        assert mt5.initialize.call_args.kwargs['login'] == 12345

    def test_failed_initialize_raises_connection_error_with_mt5_code(self, mt5):
        mt5.initialize.return_value = False
        mt5.last_error.return_value = (-10005, 'IPC timeout')

        with pytest.raises(mt5_errors.MT5ConnectionError, match = '-10005'):
            conn.connect_account()

    def test_is_connected_false_when_terminal_info_is_none(self, mt5):
        mt5.terminal_info.return_value = None
        assert conn.is_connected() is False

        mt5.terminal_info.return_value = object()
        assert conn.is_connected() is True


# --- market data ---
class TestBars:
    def test_range_returns_time_indexed_frame_with_volume_column(self, mt5):
        mt5.copy_rates_range.return_value = _rates()
        df = conn.get_bars_range('EURUSD', pd.Timestamp('2025-01-01'), pd.Timestamp('2025-01-02'))

        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index[0] == pd.Timestamp(1_700_000_000, unit = 's')
        assert 'volume' in df.columns and 'tick_volume' not in df.columns
        assert len(df) == 3

    def test_range_none_raises_rates_error(self, mt5):
        mt5.copy_rates_range.return_value = None
        mt5.last_error.return_value = (-1, 'no data')

        with pytest.raises(mt5_errors.MT5RatesError):
            conn.get_bars_range('EURUSD', pd.Timestamp('2025-01-01'), pd.Timestamp('2025-01-02'))

    def test_latest_bars_none_raises_rates_error(self, mt5):
        mt5.copy_rates_from.return_value = None
        mt5.last_error.return_value = (-1, 'no data')

        with pytest.raises(mt5_errors.MT5RatesError):
            conn.get_latest_bars_dates('EURUSD')

    def test_latest_bar_time_is_open_time_of_last_bar(self, mt5):
        mt5.copy_rates_from.return_value = _rates(n = 2)
        assert conn.get_latest_bar_time('EURUSD') == 1_700_000_300

    def test_latest_bar_time_none_raises_rates_error(self, mt5):
        mt5.copy_rates_from.return_value = None
        mt5.last_error.return_value = (-1, 'no data')

        with pytest.raises(mt5_errors.MT5RatesError):
            conn.get_latest_bar_time('EURUSD')


# --- orders ---
class TestSendOrder:
    @pytest.mark.parametrize('side, mt5_type', [('LONG', 0), ('SHORT', 1)])
    def test_side_maps_to_mt5_order_type_and_request_fields(self, mt5, side, mt5_type):
        mt5.order_send.return_value = SimpleNamespace(retcode = mt5.TRADE_RETCODE_DONE, order = 1)
        conn.send_order(7, 'EURUSD', 0.1, 1.05, 1.20, type = side)

        request = mt5.order_send.call_args.args[0]
        assert request['type'] == mt5_type
        assert (request['magic'], request['symbol'], request['volume'], request['sl'], request['tp']) == (7, 'EURUSD', 0.1, 1.05, 1.20)

    def test_invalid_side_raises_before_contacting_terminal(self, mt5):
        with pytest.raises(mt5_errors.MT5OrderError):
            conn.send_order(7, 'EURUSD', 0.1, 1.05, 1.20, type = 'BUY')
        mt5.order_send.assert_not_called()

    def test_none_result_raises_order_error(self, mt5):
        mt5.order_send.return_value = None
        mt5.last_error.return_value = (-1, 'fail')

        with pytest.raises(mt5_errors.MT5OrderError):
            conn.send_order(7, 'EURUSD', 0.1, 1.05, 1.20)

    def test_rejected_retcode_raises_order_error(self, mt5):
        mt5.order_send.return_value = SimpleNamespace(retcode = 10013, comment = 'Invalid request')

        with pytest.raises(mt5_errors.MT5OrderError, match = '10013'):
            conn.send_order(7, 'EURUSD', 0.1, 1.05, 1.20)


# --- positions ---
class TestPositions:
    def test_none_raises_position_error(self, mt5):
        mt5.positions_get.return_value = None
        mt5.last_error.return_value = (-1, 'fail')

        with pytest.raises(mt5_errors.MT5PositionError):
            conn.get_positions()

    def test_magic_filter_keeps_only_strategy_positions(self, mt5):
        mt5.positions_get.return_value = (SimpleNamespace(magic = 1), SimpleNamespace(magic = 2), SimpleNamespace(magic = 1))
        assert len(conn.get_positions(magic = 1)) == 2

    def test_ticket_and_symbol_are_forwarded_to_terminal(self, mt5):
        mt5.positions_get.return_value = ()
        conn.get_positions(ticket = 9, symbol = 'KO')
        mt5.positions_get.assert_called_once_with(ticket = 9, symbol = 'KO')

    @pytest.mark.parametrize('pos_type, close_type, price', [(0, 1, 1.0990), (1, 0, 1.1010)])
    def test_close_takes_opposite_side_at_bid_for_long_and_ask_for_short(self, mt5, pos_type, close_type, price):
        mt5.symbol_info_tick.return_value = SimpleNamespace(bid = 1.0990, ask = 1.1010)
        mt5.order_send.return_value = SimpleNamespace(retcode = mt5.TRADE_RETCODE_DONE)
        position = SimpleNamespace(symbol = 'EURUSD', type = pos_type, ticket = 5, volume = 0.1, magic = 3)
        conn.close_position(position)

        request = mt5.order_send.call_args.args[0]
        assert (request['type'], request['price'], request['position']) == (close_type, price, 5)

    def test_close_rejected_raises_order_error(self, mt5):
        mt5.symbol_info_tick.return_value = SimpleNamespace(bid = 1.0, ask = 1.1)
        mt5.order_send.return_value = SimpleNamespace(retcode = 10006, comment = 'rejected')
        position = SimpleNamespace(symbol = 'EURUSD', type = 0, ticket = 5, volume = 0.1, magic = 3)

        with pytest.raises(mt5_errors.MT5OrderError):
            conn.close_position(position)


class TestDealProfit:
    def test_sums_profit_commission_and_swap_across_deals(self, mt5):
        mt5.history_deals_get.return_value = [SimpleNamespace(profit = 100.0, commission = -2.0, swap = -1.0),
                                              SimpleNamespace(profit = -20.0, commission = -2.0, swap = 0.0)]
        assert conn.get_deal_profit(5) == pytest.approx(75.0)

    def test_none_raises_position_error(self, mt5):
        mt5.history_deals_get.return_value = None
        mt5.last_error.return_value = (-1, 'fail')

        with pytest.raises(mt5_errors.MT5PositionError):
            conn.get_deal_profit(5)


# --- regression: live_loop must catch the exception classes mt5_connector actually raises (one mt5_errors module) ---
def test_live_loop_and_connector_share_the_same_error_classes():
    import live_loop
    assert live_loop.mt5_err.MT5RatesError is conn.mt5_errors.MT5RatesError
