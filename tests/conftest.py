"""
Shared pytest setup for the live trading test suite.

Tests never touch a real MetaTrader5 terminal: the MetaTrader5 package is replaced by a
stub module (it only installs on Windows), and each test patches the terminal calls it needs.

Path handling:
- mt5/*.py and code/*.py use bare imports (`import mt5_errors`), so mt5/, code/ and live_trading/ go on sys.path
- the project's `code/` package shares its name with the stdlib `code` module (used by pdb),
  so the project root is only on sys.path while live_loop is first imported, then removed
"""
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent

# --- MetaTrader5 stub (the real package is windows-only) ---
try:
    import MetaTrader5  # noqa: F401
except ImportError:
    stub = types.ModuleType("MetaTrader5")
    stub.TIMEFRAME_M5 = 5
    stub.ORDER_TYPE_BUY, stub.ORDER_TYPE_SELL = 0, 1
    stub.TRADE_ACTION_DEAL = 1
    stub.ORDER_FILLING_FOK = 0
    stub.TRADE_RETCODE_DONE = 10009
    for name in ("initialize", "shutdown", "terminal_info", "account_info", "symbol_info", "symbol_info_tick",
                 "copy_rates_from", "copy_rates_range", "order_send", "positions_get", "history_deals_get",
                 "last_error"):
        setattr(stub, name, MagicMock(name = name))
    sys.modules["MetaTrader5"] = stub

# --- import paths ---
for p in (ROOT / "live_trading", ROOT / "mt5", ROOT / "code"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

_prior_code = sys.modules.get("code")
sys.path.insert(0, str(ROOT))
try:
    import live_loop  # noqa: F401  (primes live_loop + code.* in sys.modules)
finally:
    sys.path.remove(str(ROOT))
    if _prior_code is not None:
        sys.modules["code"] = _prior_code
    else:
        sys.modules.pop("code", None)


@pytest.fixture
def mt5(monkeypatch):
    """the MetaTrader5 module as seen by mt5_connector, all terminal calls replaced by fresh MagicMocks"""
    import mt5_connector
    for fn in ("initialize", "shutdown", "terminal_info", "copy_rates_from", "copy_rates_range",
               "order_send", "positions_get", "symbol_info_tick", "history_deals_get", "last_error"):
        monkeypatch.setattr(mt5_connector.mt5, fn, MagicMock(name = fn))
    return mt5_connector.mt5
