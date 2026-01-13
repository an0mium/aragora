"""Test utilities for the Aragora test suite.

This module provides fundamental primitives for test infrastructure:

- managed_fixture: Universal cleanup for fixtures with resources
- run_with_timeout: Async operations with timeout and clear errors
- gather_with_timeout: Multiple async operations with shared timeout

These utilities solve common test infrastructure issues:
1. Resource leaks from fixtures without cleanup
2. Tests hanging indefinitely on async operations
3. Unclear error messages on timeouts
"""

from tests.utils.async_helpers import (
    DEFAULT_TIMEOUT,
    AsyncTestContext,
    gather_with_timeout,
    run_with_cancellation,
    run_with_timeout,
)
from tests.utils.managed_resources import (
    managed_async_fixture,
    managed_fixture,
)

__all__ = [
    # Resource management
    "managed_fixture",
    "managed_async_fixture",
    # Async helpers
    "DEFAULT_TIMEOUT",
    "run_with_timeout",
    "run_with_cancellation",
    "gather_with_timeout",
    "AsyncTestContext",
]
