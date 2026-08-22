"""Essential error-raising and edge-case tests for live_trading/mt5_connector.py"""
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import mt5_connector as conn
import mt5_errors


def _fake_rates(n=3, start_time=1_700_000_000, step=300, close=1.10):
    return np.array(
        [(start_time + i * step, close, close + 0.001, close - 0.001, close, 100) for i in range(n)],
        dtype=[
            ("time", "i8"),
            ("open", "f8"),
            ("high", "f8"),
            ("low", "f8"),
            ("close", "f8"),
            ("volume", "i8"),
        ],
    )


# ---------------------------------------------------------------------------
# connect_account
# ---------------------------------------------------------------------------
class TestConnectAccount:
    def test_shuts_down_any_stale_session_before_reconnecting(self, mt5, monkeypatch):
        for key in ("META_PATH", "META_LOGIN", "META_PASSWORD", "META_SERVER"):
            monkeypatch.setenv(key, "x")
        mt5.initialize.return_value = True

        conn.connect_account()

        mt5.shutdown.assert_called_once_with()

    def test_passes_env_credentials_to_initialize(self, mt5, monkeypatch):
        monkeypatch.setenv("META_PATH", "C:/mt5/terminal64.exe")
        monkeypatch.setenv("META_LOGIN", "12345")
        monkeypatch.setenv("META_PASSWORD", "secret")
        monkeypatch.setenv("META_SERVER", "FTMO-Demo")
        mt5.initialize.return_value = True

        conn.connect_account()

        mt5.initialize.assert_called_once_with(
            path="C:/mt5/terminal64.exe",
            login="12345",
            password="secret",
            server="FTMO-Demo",
        )

    def test_raises_connection_error_when_initialize_fails(self, mt5, monkeypatch):
        monkeypatch.delenv("META_PATH", raising=False)
        mt5.initialize.return_value = False
        mt5.last_error.return_value = (10014, "invalid credentials")

        with pytest.raises(mt5_errors.MT5ConnectionError, match="10014"):
            conn.connect_account()

    def test_no_error_raised_and_no_return_value_on_success(self, mt5):
        mt5.initialize.return_value = True

        assert conn.connect_account() is None

    def test_missing_env_vars_are_passed_through_as_none(self, mt5, monkeypatch):
        """edge case: unset credentials don't raise early, they fall through to mt5.initialize as None"""
        for key in ("META_PATH", "META_LOGIN", "META_PASSWORD", "META_SERVER"):
            monkeypatch.delenv(key, raising=False)
        mt5.initialize.return_value = True

        conn.connect_account()

        _, kwargs = mt5.initialize.call_args
        assert all(v is None for v in kwargs.values())


# ---------------------------------------------------------------------------
# get_latest_bar_time
# ---------------------------------------------------------------------------
class TestGetLatestBarTime:
    def test_raises_rates_error_when_copy_rates_returns_none(self, mt5):
        mt5.copy_rates_from.return_value = None
        mt5.last_error.return_value = (1, "no data")

        with pytest.raises(mt5_errors.MT5RatesError, match="1"):
            conn.get_latest_bar_time("GBPUSD")

    def test_returns_time_of_the_returned_bar(self, mt5):
        mt5.copy_rates_from.return_value = _fake_rates(n=1)

        assert conn.get_latest_bar_time("GBPUSD") == 1_700_000_000

    def test_requests_exactly_one_bar(self, mt5):
        """edge case: this is a cheap 1-bar poll, count must always be 1 regardless of caller args"""
        mt5.copy_rates_from.return_value = _fake_rates(n=1)

        conn.get_latest_bar_time("GBPUSD", timeframe=mt5.TIMEFRAME_M5)

        args, _ = mt5.copy_rates_from.call_args
        assert args[0] == "GBPUSD"
        assert args[1] == mt5.TIMEFRAME_M5
        assert args[3] == 1


# ---------------------------------------------------------------------------
# get_latest_bars_dates
# ---------------------------------------------------------------------------
class TestGetLatestBarsDates:
    def test_raises_rates_error_when_copy_rates_returns_none(self, mt5):
        mt5.copy_rates_from.return_value = None
        mt5.last_error.return_value = (2, "no history")

        with pytest.raises(mt5_errors.MT5RatesError, match="2"):
            conn.get_latest_bars_dates("GBPUSD")

    def test_defaults_date_from_to_today_when_not_given(self, mt5):
        mt5.copy_rates_from.return_value = _fake_rates()

        conn.get_latest_bars_dates("GBPUSD")

        args, _ = mt5.copy_rates_from.call_args
        assert hasattr(args[2], "year")  # a real datetime, not left as None

    def test_uses_given_date_from_instead_of_today(self, mt5):
        mt5.copy_rates_from.return_value = _fake_rates()
        custom_date = pd.Timestamp("2024-01-01")

        conn.get_latest_bars_dates("GBPUSD", date_from=custom_date)

        args, _ = mt5.copy_rates_from.call_args
        assert args[2] == custom_date

    def test_converts_time_column_to_datetime(self, mt5):
        mt5.copy_rates_from.return_value = _fake_rates()

        df = conn.get_latest_bars_dates("GBPUSD")

        assert pd.api.types.is_datetime64_any_dtype(df["time"])

    def test_default_count_is_8_hours_of_5_minute_bars(self, mt5):
        """edge case: default count = 60 / 5 * 8 = 96"""
        mt5.copy_rates_from.return_value = _fake_rates()

        conn.get_latest_bars_dates("GBPUSD")

        args, _ = mt5.copy_rates_from.call_args
        assert args[3] == 96

    def test_empty_rates_array_returns_empty_dataframe_without_raising(self, mt5):
        """edge case: an empty-but-not-None result must not be mistaken for the None/error case"""
        mt5.copy_rates_from.return_value = _fake_rates(n=0)

        df = conn.get_latest_bars_dates("GBPUSD")

        assert df.empty
        assert "time" in df.columns


# ---------------------------------------------------------------------------
# send_order
# ---------------------------------------------------------------------------
class TestSendOrder:
    def test_raises_order_error_for_invalid_type(self, mt5):
        with pytest.raises(mt5_errors.MT5OrderError):
            conn.send_order(1, "GBPUSD", 0.1, 1.0, 2.0, type="FLAT")
        mt5.order_send.assert_not_called()

    @pytest.mark.parametrize("bad_type", [0, None, "", "long", "buy", "Long"])
    def test_raises_order_error_for_near_miss_type_values(self, mt5, bad_type):
        """edge case: falsy/near-miss values for `type` must be rejected, not silently forwarded"""
        with pytest.raises(mt5_errors.MT5OrderError):
            conn.send_order(1, "GBPUSD", 0.1, 1.0, 2.0, type=bad_type)
        mt5.order_send.assert_not_called()

    def test_maps_long_to_buy_order_type(self, mt5):
        mt5.order_send.return_value = MagicMock(retcode=mt5.TRADE_RETCODE_DONE)

        conn.send_order(1, "GBPUSD", 0.1, 1.0, 2.0, type="LONG")

        request = mt5.order_send.call_args.args[0]
        assert request["type"] == mt5.ORDER_TYPE_BUY

    def test_maps_short_to_sell_order_type(self, mt5):
        mt5.order_send.return_value = MagicMock(retcode=mt5.TRADE_RETCODE_DONE)

        conn.send_order(1, "GBPUSD", 0.1, 1.0, 2.0, type="SHORT")

        request = mt5.order_send.call_args.args[0]
        assert request["type"] == mt5.ORDER_TYPE_SELL

    def test_request_carries_the_given_trade_fields(self, mt5):
        mt5.order_send.return_value = MagicMock(retcode=mt5.TRADE_RETCODE_DONE)

        conn.send_order(42, "GBPUSD", 0.5, 1.05, 1.15, type="LONG")

        request = mt5.order_send.call_args.args[0]
        assert request["magic"] == 42
        assert request["symbol"] == "GBPUSD"
        assert request["volume"] == 0.5
        assert request["sl"] == 1.05
        assert request["tp"] == 1.15

    def test_raises_order_error_when_order_send_returns_none(self, mt5):
        mt5.order_send.return_value = None
        mt5.last_error.return_value = (3, "no connection")

        with pytest.raises(mt5_errors.MT5OrderError, match="3"):
            conn.send_order(1, "GBPUSD", 0.1, 1.0, 2.0, type="LONG")

    def test_raises_order_error_when_retcode_is_not_done(self, mt5):
        mt5.order_send.return_value = MagicMock(retcode=99999, comment="rejected")

        with pytest.raises(mt5_errors.MT5OrderError, match="99999"):
            conn.send_order(1, "GBPUSD", 0.1, 1.0, 2.0, type="LONG")

    def test_returns_the_order_on_success(self, mt5):
        order = MagicMock(retcode=mt5.TRADE_RETCODE_DONE)
        mt5.order_send.return_value = order

        assert conn.send_order(1, "GBPUSD", 0.1, 1.0, 2.0, type="LONG") is order

    def test_kwargs_can_override_a_fixed_request_field(self, mt5):
        """
        edge case: **kwargs is spread into the request dict after the fixed keys, so a
        kwarg whose name matches a fixed key (but isn't itself a named parameter, e.g.
        `magic` vs. the `trade_magic_id` parameter) silently overrides it.
        """
        mt5.order_send.return_value = MagicMock(retcode=mt5.TRADE_RETCODE_DONE)

        conn.send_order(1, "GBPUSD", 0.1, 1.0, 2.0, type="LONG", magic=999)

        request = mt5.order_send.call_args.args[0]
        assert request["magic"] == 999


# ---------------------------------------------------------------------------
# get_positions
# ---------------------------------------------------------------------------
class TestGetPositions:
    def test_raises_position_error_when_positions_get_returns_none(self, mt5):
        mt5.positions_get.return_value = None
        mt5.last_error.return_value = (4, "terminal not connected")

        with pytest.raises(mt5_errors.MT5PositionError, match="4"):
            conn.get_positions()

    def test_calls_positions_get_with_no_filters_when_all_none(self, mt5):
        mt5.positions_get.return_value = ()

        conn.get_positions()

        mt5.positions_get.assert_called_once_with()

    def test_filters_by_ticket_and_symbol_when_given(self, mt5):
        mt5.positions_get.return_value = ()

        conn.get_positions(ticket=42, symbol="GBPUSD")

        mt5.positions_get.assert_called_once_with(ticket=42, symbol="GBPUSD")

    def test_filters_returned_positions_by_magic(self, mt5):
        p1, p2 = MagicMock(magic=1), MagicMock(magic=2)
        mt5.positions_get.return_value = (p1, p2)

        assert conn.get_positions(magic=2) == (p2,)

    def test_magic_filter_with_no_match_returns_empty_tuple(self, mt5):
        """edge case"""
        mt5.positions_get.return_value = (MagicMock(magic=1),)

        assert conn.get_positions(magic=999) == ()

    def test_empty_positions_result_is_not_treated_as_an_error(self, mt5):
        """edge case: () is falsy but not None, must not raise MT5PositionError"""
        mt5.positions_get.return_value = ()

        assert conn.get_positions() == ()


# ---------------------------------------------------------------------------
# close_position
# ---------------------------------------------------------------------------
class TestClosePosition:
    def _position(self, mt5, type_, ticket=1, symbol="GBPUSD", volume=0.1, magic=1):
        return MagicMock(type=type_, ticket=ticket, symbol=symbol, volume=volume, magic=magic)

    def test_closes_a_long_position_by_selling_at_bid(self, mt5):
        position = self._position(mt5, mt5.ORDER_TYPE_BUY)
        mt5.symbol_info_tick.return_value = MagicMock(bid=1.10, ask=1.11)
        mt5.order_send.return_value = MagicMock(retcode=mt5.TRADE_RETCODE_DONE)

        conn.close_position(position)

        request = mt5.order_send.call_args.args[0]
        assert request["type"] == mt5.ORDER_TYPE_SELL
        assert request["price"] == 1.10

    def test_closes_a_short_position_by_buying_at_ask(self, mt5):
        position = self._position(mt5, mt5.ORDER_TYPE_SELL)
        mt5.symbol_info_tick.return_value = MagicMock(bid=1.10, ask=1.11)
        mt5.order_send.return_value = MagicMock(retcode=mt5.TRADE_RETCODE_DONE)

        conn.close_position(position)

        request = mt5.order_send.call_args.args[0]
        assert request["type"] == mt5.ORDER_TYPE_BUY
        assert request["price"] == 1.11

    def test_closes_full_position_volume_at_market(self, mt5):
        position = self._position(mt5, mt5.ORDER_TYPE_BUY, ticket=7, symbol="KO", volume=1.5, magic=2)
        mt5.symbol_info_tick.return_value = MagicMock(bid=1.10, ask=1.11)
        mt5.order_send.return_value = MagicMock(retcode=mt5.TRADE_RETCODE_DONE)

        conn.close_position(position)

        request = mt5.order_send.call_args.args[0]
        assert request["position"] == 7
        assert request["symbol"] == "KO"
        assert request["volume"] == 1.5
        assert request["magic"] == 2

    def test_raises_order_error_when_order_send_returns_none(self, mt5):
        position = self._position(mt5, mt5.ORDER_TYPE_BUY)
        mt5.symbol_info_tick.return_value = MagicMock(bid=1.10, ask=1.11)
        mt5.order_send.return_value = None
        mt5.last_error.return_value = (5, "trade disabled")

        with pytest.raises(mt5_errors.MT5OrderError, match="5"):
            conn.close_position(position)

    def test_raises_order_error_when_retcode_is_not_done(self, mt5):
        position = self._position(mt5, mt5.ORDER_TYPE_BUY)
        mt5.symbol_info_tick.return_value = MagicMock(bid=1.10, ask=1.11)
        mt5.order_send.return_value = MagicMock(retcode=99999, comment="requote")

        with pytest.raises(mt5_errors.MT5OrderError, match="99999"):
            conn.close_position(position)


# ---------------------------------------------------------------------------
# get_deal_profit
# ---------------------------------------------------------------------------
class TestGetDealProfit:
    def test_raises_position_error_when_history_deals_get_returns_none(self, mt5):
        mt5.history_deals_get.return_value = None
        mt5.last_error.return_value = (6, "no history")

        with pytest.raises(mt5_errors.MT5PositionError, match="6"):
            conn.get_deal_profit(123)

    def test_sums_profit_commission_and_swap_across_deals(self, mt5):
        deals = (
            MagicMock(profit=10.0, commission=-1.0, swap=-0.5),
            MagicMock(profit=-5.0, commission=-1.0, swap=0.0),
        )
        mt5.history_deals_get.return_value = deals

        assert conn.get_deal_profit(123) == pytest.approx(2.5)

    def test_empty_deal_history_sums_to_zero(self, mt5):
        """edge case: () is not None, sum over no deals must be a valid 0, not an error"""
        mt5.history_deals_get.return_value = ()

        assert conn.get_deal_profit(123) == 0

    def test_queries_by_position_ticket(self, mt5):
        mt5.history_deals_get.return_value = ()

        conn.get_deal_profit(999)

        mt5.history_deals_get.assert_called_once_with(position=999)
