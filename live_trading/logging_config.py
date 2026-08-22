"""
Logging setup for the live trading loop.

Two channels:
  logs/live_trading.log  - rotating human-readable log (connections, errors, general flow)
  logs/trades.jsonl       - one JSON record per signal/order/close event, for later analysis
                            with code.analytics / code.performance_metrics
"""
import json
import logging
import logging.handlers
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / 'logs'


class JsonlHandler(logging.Handler):
    """appends the record's 'event' payload as one JSON line per emit"""
    def __init__(self, path: Path):
        super().__init__()
        self.path = path

    def emit(self, record):
        payload = record.__dict__.get('event') # get event payload from log_event() 
        if payload is None:
            return
        with open(self.path, 'a') as f:
            f.write(json.dumps(payload, default = str) + '\n') # write to .json


def setup_logging(level = logging.INFO):
    """configure the root logger (rotating file + console) and the trades logger"""
    LOG_DIR.mkdir(exist_ok = True)

    # debug logging
    root = logging.getLogger() 
    root.setLevel(level) # by defaults prints at most info logs to consol
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s') # log event formating

    file_handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / 'live_trading.log', maxBytes = 10_000_000, backupCount = 5
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler) # root logging at /logs/live_trading.log

    console_handler = logging.StreamHandler() # print to console
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    # trade logging
    trades_logger = logging.getLogger('trades')
    trades_logger.setLevel(logging.INFO)
    trades_logger.addHandler(JsonlHandler(LOG_DIR / 'trades.jsonl')) # trade logging at /logs/trades.jsonl
    trades_logger.propagate = False


def log_event(event_type: str, **fields):
    """emit one structured record to logs/trades.jsonl"""
    payload = {'timestamp': datetime.now(timezone.utc).isoformat(), 'event': event_type, **fields}
    logging.getLogger('trades').info(event_type, extra = {'event': payload})