"""
Shared pytest fixtures for the live_trading test suite.

live_trading/*.py and code/*.py use bare, non-package imports (e.g. `import
mt5_errors`, `from risk_management import atr`), so their directories are added to
sys.path here so `import mt5_connector`, `import live_loop`, etc. resolve when
pytest is run from the project root.

Requires the real MetaTrader5 package (Windows, Python < 3.12 - see the module
docstring in live_trading/mt5_connector.py). Tests never call the live terminal;
they monkeypatch the specific MetaTrader5 functions they exercise.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = PROJECT_ROOT / "code"
LIVE_TRADING_DIR = PROJECT_ROOT / "live_trading"

# live_trading/*.py and code/*.py use bare imports (e.g. `import mt5_errors`, `from
# risk_management import atr`) - neither directory's module names collide with the
# stdlib, so these stay on sys.path for the whole session.
for _path in (LIVE_TRADING_DIR, CODE_DIR):
    _p = str(_path)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# live_loop.py also does `from code.trading_strategies import ...`, which needs the
# project root on sys.path so `code` resolves as a package - but this project's
# `code/` package collides with the stdlib `code` module (used internally by pdb,
# among others). Leaving the project root on sys.path would silently break every
# later `import code` for the rest of the process (pytest itself hits this within
# its own startup). So: prime the import once with the project root on sys.path,
# then take it back off and restore sys.modules['code'] to whatever it was before.
# live_loop, mt5_connector, and every code.* submodule stay cached under their own
# sys.modules keys, so no test needs the project root on sys.path again afterwards.
_prior_code_module = sys.modules.get("code")
_project_root_str = str(PROJECT_ROOT)
sys.path.insert(0, _project_root_str)
try:
    import live_loop  # noqa: F401
except ImportError:
    # e.g. MetaTrader5 isn't installed on this platform - live_loop/mt5_connector
    # test modules will fail on their own `import` line with a clear error; other
    # test modules (logging_config, mt5_errors's non-mt5 bits, etc.) still run.
    pass
finally:
    sys.path.remove(_project_root_str)
    if _prior_code_module is not None:
        sys.modules["code"] = _prior_code_module
    else:
        sys.modules.pop("code", None)

MT5_FUNCTIONS = [
    "initialize",
    "shutdown",
    "copy_rates_from",
    "order_send",
    "positions_get",
    "symbol_info_tick",
    "history_deals_get",
    "last_error",
]


@pytest.fixture
def mt5(monkeypatch):
    """
    the real MetaTrader5 module (as imported by mt5_connector.py), with its
    terminal-facing functions swapped for MagicMocks so tests never touch a live
    terminal. Each test configures return_value/side_effect on the functions it needs;
    monkeypatch restores the originals automatically after the test.
    """
    import mt5_connector

    for fn_name in MT5_FUNCTIONS:
        monkeypatch.setattr(mt5_connector.mt5, fn_name, MagicMock(name=fn_name))
    return mt5_connector.mt5