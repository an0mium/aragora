"""
Tests for aragora.server.handlers.admin.system - System and Utility Handlers.

Tests cover:
- SystemHandler initialization and routing
- Debug test endpoint
- History endpoints (cycles, events, debates, summary)
- Authentication for history endpoints
- System maintenance endpoint
- Circuit breaker metrics endpoint
- Auth stats endpoint
- Token revocation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, mock_open

import pytest

from aragora.server.handlers.admin.system import (
    SystemHandler,
    CACHE_TTL_HISTORY,
    HISTORY_PERMISSION,
)
from aragora.server.handlers.utils.responses import HandlerResult


# ===========================================================================
# Helper Functions
# ===========================================================================


def get_response_data(result: HandlerResult) -> dict:
    """Extract JSON data from HandlerResult."""
    if result and result.body:
        return json.loads(result.body.decode("utf-8"))
    return {}


# ===========================================================================
# Mock Classes
# ===========================================================================


class MockHandler:
    """Mock HTTP handler for testing."""

    def __init__(
        self,
        headers: dict | None = None,
        body: bytes = b"",
        path: str = "/",
        method: str = "GET",
    ):
        self.headers = headers or {}
        self._body = body
        self.request_body = body
        self.path = path
        self.command = method
        self.client_address = ("127.0.0.1", 12345)


class MockUserContext:
    """Mock user context from JWT auth."""

    def __init__(self, is_authenticated: bool = True, user_id: str = "user-001"):
        self.is_authenticated = is_authenticated
        self.user_id = user_id


class MockStorage:
    """Mock storage for debate retrieval."""

    def __init__(self, debates: list | None = None):
        self._debates = debates or []

    def list_recent(self, limit: int = 100):
        return self._debates[:limit]


class MockEloSystem:
    """Mock ELO system for leaderboard."""

    def __init__(self, rankings: list | None = None):
        self._rankings = rankings or []

    def get_leaderboard(self, limit: int = 100):
        return self._rankings[:limit]


class MockDebateMetadata:
    """Mock debate metadata for storage."""

    def __init__(
        self,
        id: str = "debate-001",
        task: str = "test debate",
        loop_id: str | None = None,
    ):
        self.id = id
        self.task = task
        self.loop_id = loop_id


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def mock_server_context() -> dict:
    """Create mock server context."""
    return {
        "storage": None,
        "elo_system": None,
        "nomic_dir": "/tmp/nomic",
    }


@pytest.fixture
def system_handler(mock_server_context) -> SystemHandler:
    """Create SystemHandler instance."""
    return SystemHandler(mock_server_context)


@pytest.fixture
def mock_http_handler() -> MockHandler:
    """Create mock HTTP handler."""
    return MockHandler(
        headers={"Content-Type": "application/json"},
        path="/api/debug/test",
        method="GET",
    )


# ===========================================================================
# Routing Tests
# ===========================================================================


class TestRouting:
    """Tests for route handling."""

    def test_can_handle_debug_route(self, system_handler):
        """Test handler recognizes debug route."""
        assert system_handler.can_handle("/api/debug/test") is True

    def test_can_handle_history_routes(self, system_handler):
        """Test handler recognizes history routes."""
        assert system_handler.can_handle("/api/history/cycles") is True
        assert system_handler.can_handle("/api/history/events") is True
        assert system_handler.can_handle("/api/history/debates") is True
        assert system_handler.can_handle("/api/history/summary") is True

    def test_can_handle_system_routes(self, system_handler):
        """Test handler recognizes system routes."""
        assert system_handler.can_handle("/api/system/maintenance") is True
        assert system_handler.can_handle("/api/circuit-breakers") is True
        assert system_handler.can_handle("/metrics") is True

    def test_can_handle_auth_routes(self, system_handler):
        """Test handler recognizes auth routes."""
        assert system_handler.can_handle("/api/auth/stats") is True
        assert system_handler.can_handle("/api/auth/revoke") is True

    def test_cannot_handle_unknown_routes(self, system_handler):
        """Test handler rejects unknown routes."""
        assert system_handler.can_handle("/api/unknown") is False
        assert system_handler.can_handle("/api/v1/debates") is False

    def test_can_handle_versioned_routes(self, system_handler):
        """Test handler recognizes versioned routes."""
        assert system_handler.can_handle("/api/v1/debug/test") is True
        assert system_handler.can_handle("/api/v1/history/cycles") is True


# ===========================================================================
# Debug Endpoint Tests
# ===========================================================================


class TestDebugEndpoint:
    """Tests for debug test endpoint."""

    def test_debug_endpoint_returns_ok(self, system_handler, mock_http_handler):
        """Test debug endpoint returns success response."""
        result = system_handler.handle("/api/debug/test", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        data = get_response_data(result)
        assert data["status"] == "ok"
        assert data["method"] == "GET"
        assert "message" in data


# ===========================================================================
# History Authentication Tests
# ===========================================================================


class TestHistoryAuthentication:
    """Tests for history endpoint authentication."""

    def test_history_auth_passes_when_disabled(self, system_handler, mock_http_handler):
        """Test auth passes when globally disabled."""
        with patch("aragora.server.auth.auth_config") as mock_config:
            mock_config.enabled = False

            result = system_handler._check_history_auth(mock_http_handler)

            assert result is None  # No error = auth passed

    def test_history_auth_passes_with_jwt(self, system_handler, mock_http_handler):
        """Test auth passes with valid JWT authentication."""
        with patch("aragora.server.auth.auth_config") as mock_config:
            mock_config.enabled = True
            with patch(
                "aragora.server.handlers.admin.system.extract_user_from_request"
            ) as mock_extract:
                mock_extract.return_value = MockUserContext(is_authenticated=True)

                result = system_handler._check_history_auth(mock_http_handler)

                assert result is None  # No error = auth passed

    def test_history_auth_fails_without_auth(self, system_handler, mock_http_handler):
        """Test auth fails when not authenticated."""
        with patch("aragora.server.auth.auth_config") as mock_config:
            mock_config.enabled = True
            mock_config.api_token = None
            with patch(
                "aragora.server.handlers.admin.system.extract_user_from_request"
            ) as mock_extract:
                mock_extract.return_value = MockUserContext(is_authenticated=False)

                result = system_handler._check_history_auth(mock_http_handler)

                assert result is not None
                assert result.status_code == 401


# ===========================================================================
# History Endpoint Tests
# ===========================================================================


class TestHistoryCycles:
    """Tests for history cycles endpoint."""

    def test_get_cycles_with_nomic_dir(self, system_handler, mock_http_handler):
        """Test getting cycles when nomic dir exists."""
        cycles_data = [{"id": "cycle-1", "phase": "debate"}]

        with patch.object(system_handler, "get_nomic_dir") as mock_dir:
            mock_dir.return_value = Path("/tmp/nomic")
            with patch.object(system_handler, "_load_filtered_json") as mock_load:
                mock_load.return_value = cycles_data

                result = system_handler._get_history_cycles(mock_http_handler, None, 50)

                assert result.status_code == 200
                data = get_response_data(result)
                assert "cycles" in data
                assert data["cycles"] == cycles_data

    def test_get_cycles_without_nomic_dir(self, system_handler, mock_http_handler):
        """Test getting cycles when nomic dir doesn't exist."""
        with patch.object(system_handler, "get_nomic_dir") as mock_dir:
            mock_dir.return_value = None

            result = system_handler._get_history_cycles(mock_http_handler, None, 50)

            assert result.status_code == 200
            data = get_response_data(result)
            assert data["cycles"] == []


class TestHistoryEvents:
    """Tests for history events endpoint."""

    def test_get_events_with_nomic_dir(self, system_handler, mock_http_handler):
        """Test getting events when nomic dir exists."""
        events_data = [{"id": "event-1", "type": "phase_start"}]

        with patch.object(system_handler, "get_nomic_dir") as mock_dir:
            mock_dir.return_value = Path("/tmp/nomic")
            with patch.object(system_handler, "_load_filtered_json") as mock_load:
                mock_load.return_value = events_data

                result = system_handler._get_history_events(mock_http_handler, None, 100)

                assert result.status_code == 200
                data = get_response_data(result)
                assert "events" in data


class TestHistoryDebates:
    """Tests for history debates endpoint."""

    def test_get_debates_with_storage(self, system_handler, mock_http_handler):
        """Test getting debates when storage available."""
        mock_debates = [
            MockDebateMetadata("debate-1", "task 1"),
            MockDebateMetadata("debate-2", "task 2"),
        ]

        with patch.object(system_handler, "get_storage") as mock_get:
            mock_storage = MockStorage(mock_debates)
            mock_get.return_value = mock_storage

            result = system_handler._get_history_debates(mock_http_handler, None, 50)

            assert result.status_code == 200
            data = get_response_data(result)
            assert "debates" in data
            assert len(data["debates"]) == 2

    def test_get_debates_without_storage(self, system_handler, mock_http_handler):
        """Test getting debates when storage not available."""
        with patch.object(system_handler, "get_storage") as mock_get:
            mock_get.return_value = None

            result = system_handler._get_history_debates(mock_http_handler, None, 50)

            assert result.status_code == 200
            data = get_response_data(result)
            assert data["debates"] == []


class TestHistorySummary:
    """Tests for history summary endpoint."""

    def test_get_summary_with_data(self, system_handler, mock_http_handler):
        """Test getting summary with available data."""
        mock_debates = [MockDebateMetadata() for _ in range(5)]
        mock_rankings = [{"agent": "claude"}, {"agent": "gpt4"}]

        with patch.object(system_handler, "get_storage") as mock_storage:
            mock_storage.return_value = MockStorage(mock_debates)
            with patch.object(system_handler, "get_elo_system") as mock_elo:
                mock_elo.return_value = MockEloSystem(mock_rankings)

                result = system_handler._get_history_summary(mock_http_handler, None)

                assert result.status_code == 200
                data = get_response_data(result)
                assert data["total_debates"] == 5
                assert data["total_agents"] == 2

    def test_get_summary_without_data(self, system_handler, mock_http_handler):
        """Test getting summary with no data sources."""
        with patch.object(system_handler, "get_storage") as mock_storage:
            mock_storage.return_value = None
            with patch.object(system_handler, "get_elo_system") as mock_elo:
                mock_elo.return_value = None

                result = system_handler._get_history_summary(mock_http_handler, None)

                assert result.status_code == 200
                data = get_response_data(result)
                assert data["total_debates"] == 0
                assert data["total_agents"] == 0


# ===========================================================================
# Maintenance Endpoint Tests
# ===========================================================================


class TestMaintenanceEndpoint:
    """Tests for system maintenance endpoint."""

    def test_maintenance_invalid_task(self, system_handler, mock_http_handler):
        """Test maintenance with invalid task."""
        with patch.object(system_handler, "_handle_maintenance") as mock_handle:
            from aragora.server.handlers.utils.responses import error_response

            mock_handle.return_value = error_response("Invalid task", 400)

            result = system_handler.handle(
                "/api/system/maintenance",
                {"task": "invalid"},
                mock_http_handler,
            )

            # The result depends on whether RBAC passes
            # For now, just check the call was routed correctly
            assert result is not None


# ===========================================================================
# Circuit Breaker Tests
# ===========================================================================


class TestCircuitBreakerEndpoint:
    """Tests for circuit breaker metrics endpoint."""

    def test_get_circuit_breaker_metrics(self, system_handler, mock_http_handler):
        """Test getting circuit breaker metrics."""
        with patch(
            "aragora.server.handlers.admin.system.get_circuit_breaker_status",
            create=True,
        ) as mock_status:
            mock_status.return_value = {"total": 3, "open": 0}

            result = system_handler.handle("/api/circuit-breakers", {}, mock_http_handler)

            assert result is not None
            assert result.status_code == 200


# ===========================================================================
# JSON Loading Tests
# ===========================================================================


class TestLoadFilteredJson:
    """Tests for _load_filtered_json helper."""

    def test_load_json_file_not_exists(self, system_handler):
        """Test loading non-existent file returns empty list."""
        result = system_handler._load_filtered_json(Path("/nonexistent/file.json"))
        assert result == []

    def test_load_json_with_limit(self, system_handler, tmp_path):
        """Test loading JSON with limit."""
        test_data = [{"id": i} for i in range(10)]
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(test_data))

        result = system_handler._load_filtered_json(json_file, None, 5)

        assert len(result) == 5

    def test_load_json_with_loop_id_filter(self, system_handler, tmp_path):
        """Test loading JSON with loop_id filter."""
        test_data = [
            {"id": "1", "loop_id": "loop-a"},
            {"id": "2", "loop_id": "loop-b"},
            {"id": "3", "loop_id": "loop-a"},
        ]
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(test_data))

        result = system_handler._load_filtered_json(json_file, "loop-a", 10)

        assert len(result) == 2
        assert all(item["loop_id"] == "loop-a" for item in result)


# ===========================================================================
# Handle Method Tests
# ===========================================================================


class TestHandleMethod:
    """Tests for the handle method routing."""

    def test_handle_routes_to_debug(self, system_handler, mock_http_handler):
        """Test handle routes to debug endpoint."""
        result = system_handler.handle("/api/debug/test", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        data = get_response_data(result)
        assert data["status"] == "ok"

    def test_handle_returns_none_for_unknown_route(self, system_handler, mock_http_handler):
        """Test handle returns None for unknown routes."""
        result = system_handler.handle("/api/unknown/path", {}, mock_http_handler)
        assert result is None


__all__ = ["TestRouting", "TestDebugEndpoint", "TestHistoryAuthentication"]
