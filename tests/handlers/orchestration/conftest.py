"""
Conftest for orchestration handler tests.

Provides an autouse fixture that prevents _execute_deliberation from making
real HTTP calls via run_async.  Without this guard, certain test orderings
(e.g. seed 99999) allow _handle_deliberate(sync=True) to call the real
run_async -> _execute_deliberation path, which blocks on actual HTTP
requests and hangs the test suite indefinitely.

The fixture patches run_async *only* in the orchestration handler module so
that it returns a safe OrchestrationResult instead of blocking.  Tests that
need to verify _execute_deliberation behavior already mock its internal
components (DeliberationManager, DecisionRouter, etc.) and are unaffected.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _prevent_real_deliberation_http():
    """Prevent the orchestration handler from blocking on real HTTP calls.

    The sync deliberation path calls
    ``run_async(self._execute_deliberation(...))``.  If neither ``run_async``
    nor ``_execute_deliberation`` is explicitly mocked by a test, this blocks
    on actual HTTP requests and hangs the test process.

    This fixture replaces ``run_async`` in the orchestration handler module
    with a safe stub that closes the coroutine and returns a harmless
    ``OrchestrationResult``.  Tests that explicitly patch ``run_async`` via
    ``with patch("aragora.server.handlers.orchestration.handler.run_async", ...)``
    override this fixture's patch because the ``with`` block restores the
    attribute on exit, then this fixture restores it again in teardown.
    """
    from unittest.mock import patch
    from aragora.server.handlers.orchestration.models import OrchestrationResult

    def _safe_run_async(coro, timeout=30.0):
        """Non-blocking replacement for run_async in tests."""
        if hasattr(coro, "close"):
            coro.close()
        return OrchestrationResult(
            request_id="test-safe-fallback",
            success=True,
            final_answer="Mocked by conftest",
        )

    with patch(
        "aragora.server.handlers.orchestration.handler.run_async",
        side_effect=_safe_run_async,
    ):
        yield
