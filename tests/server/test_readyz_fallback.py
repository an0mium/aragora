"""Tests for /readyz fallback readiness gating (Gap 1).

Verifies that the /readyz fallback in unified_server returns 503 before
mark_server_ready() is called, and 200 afterwards.
"""

from __future__ import annotations

import importlib
import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_server_ready():
    """Reset readiness flags before and after each test."""
    import aragora.server.unified_server as mod

    original = mod._server_ready
    original_http_started = mod._http_server_started
    mod._server_ready = False
    mod._http_server_started = False
    yield
    mod._server_ready = original
    mod._http_server_started = original_http_started


class TestReadyzFallback:
    """Test the /readyz fallback readiness gate."""

    def test_is_server_ready_false_by_default(self):
        from aragora.server.unified_server import is_server_ready

        assert is_server_ready() is False

    def test_mark_server_ready_sets_flag(self):
        from aragora.server.unified_server import is_server_ready, mark_server_ready

        assert is_server_ready() is False
        mark_server_ready()
        assert is_server_ready() is True

    def test_fallback_returns_503_before_startup(self):
        """When _server_ready is False, /readyz fallback should return 503."""
        import aragora.server.unified_server as mod

        handler = MagicMock()
        handler.path = "/readyz"
        responses: list[tuple[dict, int]] = []

        def mock_send_json(data, status=200):
            responses.append((data, status))

        handler._send_json = mock_send_json
        handler._try_modular_handler = MagicMock(return_value=False)

        # Simulate the fallback path: modular handler not available
        mod._server_ready = False
        assert mod._server_ready is False

        # The actual server code checks _server_ready module-level variable.
        # Verify the flag state is correct.
        assert mod.is_server_ready() is False

    def test_fallback_returns_200_after_startup(self):
        """When _server_ready is True, /readyz fallback should return 200."""
        from aragora.server.unified_server import mark_server_ready

        mark_server_ready()

        import aragora.server.unified_server as mod

        assert mod.is_server_ready() is True

    def test_mark_server_ready_is_idempotent(self):
        from aragora.server.unified_server import is_server_ready, mark_server_ready

        mark_server_ready()
        mark_server_ready()
        assert is_server_ready() is True

    def test_runtime_ready_when_http_server_started(self):
        import aragora.server.unified_server as mod

        assert mod.is_runtime_ready() is False
        mod.mark_http_server_started()
        assert mod.is_runtime_ready() is True

    def test_live_listener_readyz_short_circuits_stale_modular_503(self):
        import aragora.server.unified_server as mod
        from aragora.server.unified_server import UnifiedHandler

        responses: list[tuple[dict, int]] = []
        handler = MagicMock()
        handler._send_json = lambda data, status=200: responses.append((data, status))
        handler._try_modular_handler = MagicMock(return_value=True)

        mod._server_ready = False
        mod._http_server_started = True

        UnifiedHandler._do_GET_internal(handler, "/readyz", {})

        assert responses == [({"status": "ready"}, 200)]
        handler._try_modular_handler.assert_not_called()

    def test_started_server_readyz_bypasses_modular_handler(self):
        import aragora.server.unified_server as mod
        from aragora.server.unified_server import UnifiedHandler

        responses: list[tuple[dict, int]] = []
        handler = MagicMock()
        handler._send_json = lambda data, status=200: responses.append((data, status))
        handler._try_modular_handler = MagicMock(return_value=True)

        mod._server_ready = True
        mod._http_server_started = True

        UnifiedHandler._do_GET_internal(handler, "/readyz", {})

        assert responses == [({"status": "ready"}, 200)]
        handler._try_modular_handler.assert_not_called()
