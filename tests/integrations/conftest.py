"""Shared fixtures for integration tests.

Provides cleanup fixtures to prevent test ordering dependencies caused
by module-level global state in integration modules.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_email_reply_handlers():
    """Save and restore the global _reply_handlers list between tests.

    The email_reply_loop module maintains a module-level _reply_handlers list.
    Tests like test_register_reply_handler() append to this list, which persists
    across tests and causes ordering-dependent failures when
    test_handle_email_reply_default_processing() expects no handlers registered.
    """
    try:
        from aragora.integrations import email_reply_loop

        saved = list(email_reply_loop._reply_handlers)
        yield
        email_reply_loop._reply_handlers[:] = saved
    except (ImportError, AttributeError):
        yield


@pytest.fixture(autouse=True)
def _reset_platform_circuits():
    """Clear the global _platform_circuits registry between tests.

    The platform_resilience module maintains a module-level _platform_circuits
    dict that caches PlatformCircuitBreaker instances by platform name.
    Tests that create PlatformCircuitBreaker objects with the same platform
    name share the underlying circuit breaker from the global registry,
    causing ordering-dependent failures.
    """
    try:
        from aragora.integrations import platform_resilience

        saved = dict(platform_resilience._platform_circuits)
        yield
        platform_resilience._platform_circuits.clear()
        platform_resilience._platform_circuits.update(saved)
    except (ImportError, AttributeError):
        yield
