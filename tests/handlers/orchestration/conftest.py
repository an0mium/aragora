"""
Conftest for orchestration handler tests.

Provides an autouse fixture that prevents _execute_deliberation from making
real HTTP calls.  Without this guard, certain test orderings (e.g. seed 99999)
allow _handle_deliberate(sync=True) to call the real run_async →
_execute_deliberation path, which blocks on actual HTTP requests and hangs
the test suite indefinitely.

The fixture patches run_async *only* in the orchestration handler module so
that it returns a safe OrchestrationResult instead of blocking.  Tests that
need to verify _execute_deliberation behavior already mock its internal
components (DeliberationManager, DecisionRouter, etc.) and are unaffected.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture(autouse=True)
def _prevent_real_deliberation_http(monkeypatch):
    """Prevent the orchestration handler from making real HTTP calls.

    Patches two dangerous paths:

    1. ``run_async`` in the orchestration handler module -- the sync
       deliberation path calls ``run_async(self._execute_deliberation(...))``.
       If run_async is the real function and _execute_deliberation is not
       mocked, this blocks on actual HTTP.

    2. ``asyncio.create_task`` in the orchestration handler module -- the
       async deliberation path fires a background task that eventually calls
       _execute_deliberation.  Patching create_task prevents orphaned tasks
       that can hang event loop teardown.

    Tests that explicitly patch run_async or _execute_deliberation via
    ``with patch(...)`` override this fixture's patches (inner patches win).
    """
    from aragora.server.handlers.orchestration.models import OrchestrationResult

    # Safe fallback: when run_async is called without an explicit mock,
    # inspect the coroutine argument.  If it's an _execute_deliberation call,
    # close the coroutine (to avoid "coroutine never awaited" warnings) and
    # return a harmless result.  For other coroutines, also return safely.
    def _safe_run_async(coro, timeout=30.0):
        """Non-blocking replacement for run_async in tests."""
        # Close the coroutine to suppress ResourceWarning
        if hasattr(coro, "close"):
            coro.close()
        return OrchestrationResult(
            request_id="test-safe-fallback",
            success=True,
            final_answer="Mocked by conftest",
        )

    monkeypatch.setattr(
        "aragora.server.handlers.orchestration.handler.run_async",
        _safe_run_async,
    )

    # Also prevent asyncio.create_task from spawning real background work.
    # The async path does:  task = asyncio.create_task(self._execute_and_store(req))
    # We replace create_task with a no-op that returns a mock task.
    import asyncio
    from unittest.mock import MagicMock

    _original_create_task = asyncio.create_task

    def _safe_create_task(coro, *, name=None):
        """Close the coroutine instead of scheduling it."""
        if hasattr(coro, "close"):
            coro.close()
        mock_task = MagicMock()
        mock_task.cancelled.return_value = False
        mock_task.exception.return_value = None
        mock_task.add_done_callback = MagicMock()
        return mock_task

    monkeypatch.setattr(
        "aragora.server.handlers.orchestration.handler.asyncio.create_task",
        _safe_create_task,
    )
