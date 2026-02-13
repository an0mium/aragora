"""Fixtures for handler utility tests.

Re-enables rate limiting for rate_limit unit tests since the parent conftest
disables it globally to prevent test pollution.
"""

import pytest


@pytest.fixture(autouse=True)
def _reenable_rate_limiting():
    """Re-enable rate limiting for utility tests that test rate limiters."""
    import sys

    rl_mod = sys.modules.get("aragora.server.handlers.utils.rate_limit")
    if rl_mod and hasattr(rl_mod, "RATE_LIMITING_DISABLED"):
        old_val = rl_mod.RATE_LIMITING_DISABLED
        rl_mod.RATE_LIMITING_DISABLED = False
        yield
        rl_mod.RATE_LIMITING_DISABLED = old_val
    else:
        yield
