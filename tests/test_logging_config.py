"""Tests for live_trading/logging_config.py"""
import json
import logging

import pytest

import logging_config as log_cfg


@pytest.fixture(autouse=True)
def _restore_logging_state():
    """setup_logging() mutates the (process-global) root and 'trades' loggers - restore after each test"""
    root = logging.getLogger()
    prior_root_handlers = root.handlers[:]
    prior_root_level = root.level

    trades_logger = logging.getLogger("trades")
    prior_trades_handlers = trades_logger.handlers[:]
    prior_trades_propagate = trades_logger.propagate

    yield

    root.handlers[:] = prior_root_handlers
    root.setLevel(prior_root_level)
    trades_logger.handlers[:] = prior_trades_handlers
    trades_logger.propagate = prior_trades_propagate


def test_setup_logging_creates_the_log_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(log_cfg, "LOG_DIR", tmp_path / "logs")

    log_cfg.setup_logging()

    assert (tmp_path / "logs").is_dir()


def test_setup_logging_attaches_a_jsonl_handler_to_the_trades_logger(tmp_path, monkeypatch):
    monkeypatch.setattr(log_cfg, "LOG_DIR", tmp_path / "logs")

    log_cfg.setup_logging()

    trades_logger = logging.getLogger("trades")
    assert any(isinstance(h, log_cfg.JsonlHandler) for h in trades_logger.handlers)
    assert trades_logger.propagate is False


def test_log_event_writes_one_json_line_with_the_given_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(log_cfg, "LOG_DIR", tmp_path / "logs")
    log_cfg.setup_logging()

    log_cfg.log_event("order_sent", symbol="GBPUSD", ticket=123, volume=0.1)

    lines = (tmp_path / "logs" / "trades.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["event"] == "order_sent"
    assert record["symbol"] == "GBPUSD"
    assert record["ticket"] == 123
    assert "timestamp" in record


def test_log_event_appends_across_multiple_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(log_cfg, "LOG_DIR", tmp_path / "logs")
    log_cfg.setup_logging()

    log_cfg.log_event("signal_generated", symbol="KO")
    log_cfg.log_event("order_sent", symbol="KO")

    lines = (tmp_path / "logs" / "trades.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["event"] == "signal_generated"
    assert json.loads(lines[1])["event"] == "order_sent"


def test_log_event_serialises_non_json_native_values(tmp_path, monkeypatch):
    """edge case: non-JSON-serialisable values (e.g. numpy scalars, exceptions) fall back to str() via default=str"""
    monkeypatch.setattr(log_cfg, "LOG_DIR", tmp_path / "logs")
    log_cfg.setup_logging()

    log_cfg.log_event("order_rejected", error=ValueError("bad volume"))

    lines = (tmp_path / "logs" / "trades.jsonl").read_text().strip().splitlines()
    record = json.loads(lines[0])
    assert record["error"] == "bad volume"


def test_jsonl_handler_ignores_records_without_an_event_payload(tmp_path):
    """edge case: a plain (non log_event) record on the same logger must not write an empty/invalid line"""
    path = tmp_path / "trades.jsonl"
    handler = log_cfg.JsonlHandler(path)
    trades_logger = logging.getLogger("trades.no_event_test")
    trades_logger.addHandler(handler)
    trades_logger.setLevel(logging.INFO)
    trades_logger.propagate = False

    trades_logger.info("plain message with no 'event' extra")

    assert not path.exists()