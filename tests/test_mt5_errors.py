"""Tests for live_trading/mt5_errors.py"""
import pytest

import mt5_errors

ALL_ERROR_TYPES = [
    mt5_errors.MT5ConnectionError,
    mt5_errors.MT5RatesError,
    mt5_errors.MT5PositionError,
    mt5_errors.MT5OrderError,
]


class TestMT5error:
    def test_passes_through_mt5_last_error(self, mt5):
        mt5.last_error.return_value = (10014, "invalid credentials")

        assert mt5_errors.MT5error() == (10014, "invalid credentials")
        mt5.last_error.assert_called_once_with()


@pytest.mark.parametrize("exc_cls", ALL_ERROR_TYPES)
class TestCustomExceptionTypes:
    def test_is_an_exception_subclass(self, exc_cls):
        assert issubclass(exc_cls, Exception)

    def test_raises_and_carries_its_message(self, exc_cls):
        with pytest.raises(exc_cls, match="boom"):
            raise exc_cls("boom")

    def test_not_conflated_with_sibling_error_types(self, exc_cls):
        """
        edge case: live_loop.py routes each MT5*Error to a different except clause
        (reconnect vs skip vs log-only), so these types must stay distinct - none may
        be a subclass of another.
        """
        for sibling in ALL_ERROR_TYPES:
            if sibling is exc_cls:
                continue
            assert not issubclass(exc_cls, sibling)