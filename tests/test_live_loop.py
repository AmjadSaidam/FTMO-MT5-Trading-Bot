"""
Tests for the exception-handling and edge-case branches of live_trading/live_loop.py.

run_live_loop() is an infinite polling loop, so every test here patches
live_loop.time.sleep to raise a sentinel exception after a controlled number of
calls. Because `time.sleep(1)` at the end of each pass sits *outside* the loop's
try/except, raising from it unwinds run_live_loop deterministically and lets the
test assert on everything that happened up to that point.

Every external dependency (mt5_connector, logging_config, code.optimisation, the ML
ensemble, and the feature/risk helper functions) is monkeypatched directly onto the
live_loop module, so no real MetaTrader5 terminal, disk logging, or model training
happens while these run.
"""
import logging
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import live_loop
import mt5_errors as mt5_err


class _StopLoop(Exception):
    """raised from the mocked time.sleep to end run_live_loop's `while True` deterministically"""


def _bars(n=40, start=1_700_000_000, step=300, close=1.10):
    idx = np.arange(n)
    return pd.DataFrame({
        "time": start + idx * step,
        "open": close,
        "high": close + 0.001,
        "low": close - 0.001,
        "close": close,
        "volume": 100,
    })


def _no_signal_strategy(data, **kwargs):
    return pd.Series([False] * len(data)), np.zeros(len(data), dtype=int)


def _long_signal_strategy(data, **kwargs):
    return pd.Series([True] * len(data)), np.ones(len(data), dtype=int)


def _short_signal_strategy(data, **kwargs):
    return pd.Series([True] * len(data)), -np.ones(len(data), dtype=int)


def _run_and_stop(monkeypatch, allowed_sleep_calls):
    """
    patches live_loop.time.sleep to succeed `allowed_sleep_calls` times, then raise
    _StopLoop on the next call. sleep is invoked both inside the MT5ConnectionError
    handler (sleep(5)) and once per finished iteration (sleep(1)), so pick the count
    to land the stop exactly where the test needs it, then run_live_loop() under
    `pytest.raises(_StopLoop)`.
    """
    calls = []

    def fake_sleep(seconds):
        calls.append(seconds)
        if len(calls) > allowed_sleep_calls:
            raise _StopLoop()

    monkeypatch.setattr(live_loop.time, "sleep", fake_sleep)
    return calls


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("META_INITIAL_ACCOUNT_BALANCE", "10000")
    monkeypatch.setenv("META_WFA_IN_SAMPLE_DAYS", "30")
    monkeypatch.setenv("META_ML_FEATURE_WINDOW", "5")
    monkeypatch.setenv("META_ML_CONSEC_LOSS_BEFORE_TRADE_SKIP", "3")
    monkeypatch.setenv("META_TOTAL_RISK_ALLOCATION", "0.01")


@pytest.fixture
def mt5_conn(monkeypatch):
    fake = MagicMock(name="mt5_conn")
    fake.connect_account.return_value = None
    fake.get_latest_bars_dates.return_value = _bars()
    fake.get_latest_bar_time.return_value = 1_700_000_000
    monkeypatch.setattr(live_loop, "mt5_conn", fake)
    return fake


@pytest.fixture
def log_cfg(monkeypatch):
    fake = MagicMock(name="log_cfg")
    monkeypatch.setattr(live_loop, "log_cfg", fake)
    return fake


@pytest.fixture
def opt(monkeypatch):
    fake = MagicMock(name="opt")
    fake.grid_search_params.return_value = [{}]
    fake.run_grid_search.return_value = [pd.DataFrame({"adjusted_returns": [0.0], "equity": [10000.0]})]
    fake.grid_search_statistics.return_value = {
        "opt_df": pd.DataFrame({
            "atr_pct": [0.01],
            "regime": [0],
            "breakout_conv": [0.1],
            "rel_volume": [100.0],
            "ml_label": [1],
        }),
        "opt_atr": 2.0,
        "opt_rr": 2.0,
    }
    monkeypatch.setattr(live_loop, "opt", fake)
    return fake


@pytest.fixture
def ensemble(monkeypatch):
    """stubs RandomForestVol so WFA doesn't actually train a model; state['valid_trade'] drives model_predict"""
    state = {"valid_trade": True}

    class FakeEnsemble:
        def __init__(self, *a, **k):
            pass

        def train_model(self, *a, **k):
            return None

        def model_predict(self, *a, **k):
            return np.array([int(state["valid_trade"])])

    monkeypatch.setattr(live_loop, "RandomForestVol", FakeEnsemble)
    return state


@pytest.fixture
def calcs(monkeypatch):
    """stubs the feature/risk helper functions so the loop body runs deterministically"""
    monkeypatch.setattr(live_loop, "atr", lambda data, *a, **k: np.full(len(data), 0.01))
    monkeypatch.setattr(live_loop, "categorise_regime", lambda x: np.zeros(len(x)))
    monkeypatch.setattr(live_loop, "r2", lambda y: 0.0)
    monkeypatch.setattr(live_loop, "consecutive_loss_threshold", lambda consec, threshold: consec >= threshold)
    # trade_value(=units*price)=1100.0 -> 1000 units at tob.open=1.10
    monkeypatch.setattr(live_loop, "position_sizing", lambda *a, **k: (1100.0, 1.1))


@pytest.fixture
def mt5_symbol_info(monkeypatch):
    """stubs live_loop.mt5.symbol_info(); default currency_base marks the symbol as forex"""
    fake = MagicMock(name="mt5.symbol_info", return_value=MagicMock(currency_base="GBP"))
    monkeypatch.setattr(live_loop.mt5, "symbol_info", fake)
    return fake


@pytest.fixture
def kwargs(mt5_conn, log_cfg, opt, ensemble, calcs, mt5_symbol_info):
    """default run_live_loop() call args: one symbol, no trading signal"""
    return dict(
        symbols_strats={"GBPUSD": _no_signal_strategy},
        strat_ids={"GBPUSD": 1},
        strat_weights=[1.0],
    )


def _fake_two_day_clock(monkeypatch):
    """1 call for the initial prior_day, then one per iteration: day0, day0, day1 (new_day on iteration 2)"""
    fake_datetime = MagicMock(name="datetime")
    fake_datetime.today.return_value.now.side_effect = [
        pd.Timestamp("2026-01-01"),
        pd.Timestamp("2026-01-01"),
        pd.Timestamp("2026-01-02"),
    ]
    monkeypatch.setattr(live_loop, "datetime", fake_datetime)


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------
class TestRunLiveLoopHappyPath:
    def test_completes_one_iteration_without_trading_when_no_signal(self, monkeypatch, kwargs, mt5_conn, log_cfg):
        _run_and_stop(monkeypatch, 0)

        with pytest.raises(_StopLoop):
            live_loop.run_live_loop(**kwargs)

        mt5_conn.connect_account.assert_called_once()
        mt5_conn.send_order.assert_not_called()

    def test_sends_a_long_order_with_correctly_derived_sl_tp_when_signal_is_valid(
        self, monkeypatch, kwargs, mt5_conn, log_cfg
    ):
        kwargs["symbols_strats"] = {"GBPUSD": _long_signal_strategy}
        mt5_conn.send_order.return_value = MagicMock(order=555)
        _run_and_stop(monkeypatch, 0)

        with pytest.raises(_StopLoop):
            live_loop.run_live_loop(**kwargs)

        mt5_conn.send_order.assert_called_once()
        args, call_kwargs = mt5_conn.send_order.call_args
        magic_id, symbol, volume, sl, tp = args
        assert magic_id == 1
        assert symbol == "GBPUSD"
        assert volume == pytest.approx(0.01)  # trade_value(1100)/tob.open(1.10)/1e5, forex lots
        assert sl == pytest.approx(1.08)  # tob.open(1.10) - atr(0.01)*opt_atr(2.0)*direction(1)
        assert tp == pytest.approx(1.14)  # tob.open(1.10) + atr(0.01)*opt_atr(2.0)*opt_rr(2.0)*direction(1)
        assert call_kwargs["type"] == "LONG"

    def test_sends_a_short_order_when_signal_direction_is_negative(self, monkeypatch, kwargs, mt5_conn, log_cfg):
        kwargs["symbols_strats"] = {"GBPUSD": _short_signal_strategy}
        mt5_conn.send_order.return_value = MagicMock(order=556)
        _run_and_stop(monkeypatch, 0)

        with pytest.raises(_StopLoop):
            live_loop.run_live_loop(**kwargs)

        args, call_kwargs = mt5_conn.send_order.call_args
        assert call_kwargs["type"] == "SHORT"
        assert args[3] == pytest.approx(1.12)  # sl = 1.10 - 0.02*(-1)
        assert args[4] == pytest.approx(1.06)  # tp = 1.10 + 0.04*(-1)

    def test_records_the_new_ticket_via_log_event_on_a_filled_order(self, monkeypatch, kwargs, mt5_conn, log_cfg):
        kwargs["symbols_strats"] = {"GBPUSD": _long_signal_strategy}
        mt5_conn.send_order.return_value = MagicMock(order=777)
        _run_and_stop(monkeypatch, 0)

        with pytest.raises(_StopLoop):
            live_loop.run_live_loop(**kwargs)

        log_cfg.log_event.assert_any_call(
            "order_sent",
            symbol="GBPUSD",
            strategy="_long_signal_strategy",
            ticket=777,
            type="LONG",
            volume=pytest.approx(0.01),
            sl=pytest.approx(1.08),
            tp=pytest.approx(1.14),
            magic=1,
        )


class TestVolumeCalculation:
    """
    position_sizing() returns (trade_value, leverage), where trade_value = units *
    asset_price (see code/backtest_engine.py:30-34). Forex volume must therefore be
    units / standard_lot, i.e. (trade_value / price) / 1e5 - not trade_value / 1e5.
    """

    def test_forex_symbol_converts_units_to_standard_lots(
        self, monkeypatch, kwargs, mt5_conn, log_cfg, mt5_symbol_info
    ):
        mt5_symbol_info.return_value = MagicMock(currency_base="GBP")
        kwargs["symbols_strats"] = {"GBPUSD": _long_signal_strategy}
        mt5_conn.send_order.return_value = MagicMock(order=1)
        _run_and_stop(monkeypatch, 0)

        with pytest.raises(_StopLoop):
            live_loop.run_live_loop(**kwargs)

        _, _, volume, _, _ = mt5_conn.send_order.call_args.args
        # trade_value=1100.0, tob.open=1.10 -> 1000 units -> 1000/1e5 standard lots
        assert volume == pytest.approx(0.01)

    def test_non_forex_symbol_uses_asset_units_directly(
        self, monkeypatch, kwargs, mt5_conn, log_cfg, mt5_symbol_info
    ):
        mt5_symbol_info.return_value = MagicMock(currency_base=None)
        kwargs["symbols_strats"] = {"GBPUSD": _long_signal_strategy}
        mt5_conn.send_order.return_value = MagicMock(order=1)
        _run_and_stop(monkeypatch, 0)

        with pytest.raises(_StopLoop):
            live_loop.run_live_loop(**kwargs)

        _, _, volume, _, _ = mt5_conn.send_order.call_args.args
        # trade_value=1100.0, tob.open=1.10 -> 1000 units, used as-is (no lot conversion)
        assert volume == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# exception handling
# ---------------------------------------------------------------------------
class TestConnectionErrorHandling:
    def test_reconnects_after_a_connection_error(self, monkeypatch, kwargs, mt5_conn, log_cfg):
        mt5_conn.get_latest_bar_time.side_effect = mt5_err.MT5ConnectionError("lost connection")
        calls = _run_and_stop(monkeypatch, 1)  # sleep(5) in the handler, then sleep(1) at loop end raises

        with pytest.raises(_StopLoop):
            live_loop.run_live_loop(**kwargs)

        assert calls == [5, 1]
        assert mt5_conn.connect_account.call_count == 2  # initial connect + reconnect after the error

    def test_connection_error_is_logged(self, monkeypatch, kwargs, mt5_conn, log_cfg, caplog):
        mt5_conn.get_latest_bar_time.side_effect = mt5_err.MT5ConnectionError("lost connection")
        _run_and_stop(monkeypatch, 1)

        with pytest.raises(_StopLoop), caplog.at_level(logging.ERROR):
            live_loop.run_live_loop(**kwargs)

        assert "lost connection" in caplog.text


class TestRatesErrorHandling:
    def test_rates_error_during_wfa_fetch_is_caught_and_logged_without_reconnecting(
        self, monkeypatch, kwargs, mt5_conn, log_cfg, caplog
    ):
        mt5_conn.get_latest_bars_dates.side_effect = mt5_err.MT5RatesError("history unavailable")
        calls = _run_and_stop(monkeypatch, 0)

        with pytest.raises(_StopLoop), caplog.at_level(logging.ERROR):
            live_loop.run_live_loop(**kwargs)

        assert "history unavailable" in caplog.text
        assert calls == [1]
        mt5_conn.connect_account.assert_called_once()  # no reconnect for a rates error


class TestPositionErrorHandling:
    def test_position_error_during_eod_ticket_check_is_caught_and_logged(
        self, monkeypatch, kwargs, mt5_conn, log_cfg, caplog
    ):
        kwargs["symbols_strats"] = {"GBPUSD": _long_signal_strategy}
        mt5_conn.send_order.return_value = MagicMock(order=42)
        _fake_two_day_clock(monkeypatch)
        mt5_conn.get_positions.side_effect = mt5_err.MT5PositionError("ticket not found")
        calls = _run_and_stop(monkeypatch, 1)  # iteration 1 opens the trade, iteration 2 hits the eod check

        with pytest.raises(_StopLoop), caplog.at_level(logging.ERROR):
            live_loop.run_live_loop(**kwargs)

        assert calls == [1, 1]
        assert "ticket not found" in caplog.text


class TestOrderErrorHandling:
    def test_order_error_from_send_order_is_caught_logged_and_recorded(
        self, monkeypatch, kwargs, mt5_conn, log_cfg, caplog
    ):
        kwargs["symbols_strats"] = {"GBPUSD": _long_signal_strategy}
        mt5_conn.send_order.side_effect = mt5_err.MT5OrderError("order rejected (retcode = 99999)")
        _run_and_stop(monkeypatch, 0)

        with pytest.raises(_StopLoop), caplog.at_level(logging.ERROR):
            live_loop.run_live_loop(**kwargs)

        assert "order rejected" in caplog.text
        log_cfg.log_event.assert_any_call("order_rejected", error="order rejected (retcode = 99999)")


class TestUnhandledExceptionHandling:
    def test_an_unexpected_exception_is_caught_and_does_not_crash_the_loop(
        self, monkeypatch, kwargs, mt5_conn, log_cfg
    ):
        def _broken_strategy(data, **kw):
            raise ValueError("strategy blew up")

        kwargs["symbols_strats"] = {"GBPUSD": _broken_strategy}
        calls = _run_and_stop(monkeypatch, 0)

        with pytest.raises(_StopLoop):
            live_loop.run_live_loop(**kwargs)

        # the loop survived the ValueError and reached the (single) end-of-iteration sleep
        assert calls == [1]


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------
class TestConsecutiveLossSkip:
    def test_skips_trade_generation_once_the_consecutive_loss_threshold_is_reached(
        self, monkeypatch, kwargs, mt5_conn, log_cfg
    ):
        strat = MagicMock(side_effect=_long_signal_strategy)
        strat.__name__ = "flagged_strategy"
        kwargs["symbols_strats"] = {"GBPUSD": strat}
        monkeypatch.setattr(live_loop, "consecutive_loss_threshold", lambda *a, **k: True)
        _run_and_stop(monkeypatch, 0)

        with pytest.raises(_StopLoop):
            live_loop.run_live_loop(**kwargs)

        strat.assert_not_called()
        mt5_conn.send_order.assert_not_called()


class TestSignalRejectedByModel:
    def test_no_order_sent_when_the_ml_model_rejects_a_valid_signal(
        self, monkeypatch, kwargs, mt5_conn, log_cfg, ensemble
    ):
        ensemble["valid_trade"] = False
        kwargs["symbols_strats"] = {"GBPUSD": _long_signal_strategy}
        _run_and_stop(monkeypatch, 0)

        with pytest.raises(_StopLoop):
            live_loop.run_live_loop(**kwargs)

        mt5_conn.send_order.assert_not_called()
        # the signal is still logged, just without leading to an order
        log_cfg.log_event.assert_any_call(
            "signal_generated", symbol="GBPUSD", strategy="_long_signal_strategy",
            direction=1, valid_trade=False,
        )


class TestNewBarGate:
    def test_skips_refetching_bars_when_the_latest_bar_time_is_unchanged(
        self, monkeypatch, kwargs, mt5_conn, log_cfg
    ):
        """
        edge case: get_latest_bar_time returning the same value on consecutive polls
        means no new bar has closed - the signal path must `continue` rather than
        re-fetching/re-evaluating every 1s poll.
        """
        calls = _run_and_stop(monkeypatch, 1)  # 2 iterations, same bar time both times

        with pytest.raises(_StopLoop):
            live_loop.run_live_loop(**kwargs)

        assert calls == [1, 1]
        # iteration 1: one WFA fetch + one signal-path fetch. iteration 2: bar unchanged -> no extra fetch
        assert mt5_conn.get_latest_bars_dates.call_count == 2


class TestEodExit:
    def _run_two_days_with_open_trade(self, monkeypatch, kwargs, mt5_conn, log_cfg):
        kwargs["symbols_strats"] = {"GBPUSD": _long_signal_strategy}
        mt5_conn.send_order.return_value = MagicMock(order=888)
        _fake_two_day_clock(monkeypatch)
        _run_and_stop(monkeypatch, 1)

        with pytest.raises(_StopLoop):
            live_loop.run_live_loop(**kwargs)

    def test_missing_position_on_new_day_is_treated_as_sl_tp_hit(self, monkeypatch, kwargs, mt5_conn, log_cfg):
        mt5_conn.get_positions.return_value = ()
        mt5_conn.get_deal_profit.return_value = -50.0

        self._run_two_days_with_open_trade(monkeypatch, kwargs, mt5_conn, log_cfg)

        mt5_conn.close_position.assert_not_called()
        log_cfg.log_event.assert_any_call(
            "position_closed", symbol="GBPUSD", ticket=888, reason="sl_tp_hit", pnl=-50.0,
        )

    def test_still_open_position_on_new_day_is_force_closed(self, monkeypatch, kwargs, mt5_conn, log_cfg):
        open_position = MagicMock()
        mt5_conn.get_positions.return_value = (open_position,)
        mt5_conn.get_deal_profit.return_value = 25.0

        self._run_two_days_with_open_trade(monkeypatch, kwargs, mt5_conn, log_cfg)

        mt5_conn.close_position.assert_called_once_with(open_position)
        log_cfg.log_event.assert_any_call(
            "position_closed", symbol="GBPUSD", ticket=888, reason="eod_force_close", pnl=25.0,
        )
