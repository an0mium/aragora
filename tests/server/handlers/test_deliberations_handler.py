"""Tests for Deliberations (Vetted Decisionmaking) handler."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.deliberations import (
    DeliberationsHandler,
    _active_deliberations,
    _stats,
    broadcast_deliberation_event,
    complete_deliberation,
    register_deliberation,
    update_deliberation,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def handler():
    """Create a DeliberationsHandler instance."""
    return DeliberationsHandler({})


@pytest.fixture
def mock_request():
    """Create a mock request object."""
    request = MagicMock()
    request.path = "/api/v1/deliberations/active"
    request.method = "GET"
    return request


@pytest.fixture(autouse=True)
def clear_state():
    """Clear module state before each test."""
    _active_deliberations.clear()
    _stats.clear()
    _stats.update(
        {
            "active_count": 0,
            "completed_today": 0,
            "average_consensus_time": 0,
            "average_rounds": 0,
            "top_agents": [],
        }
    )
    yield
    _active_deliberations.clear()


# =============================================================================
# Test Handler Initialization
# =============================================================================


class TestDeliberationsHandlerInit:
    """Tests for handler initialization."""

    def test_handler_routes(self, handler):
        """Should define correct routes."""
        assert "/api/v1/deliberations/active" in handler.ROUTES
        assert "/api/v1/deliberations/stats" in handler.ROUTES
        assert "/api/v1/deliberations/stream" in handler.ROUTES
        assert "/api/v1/deliberations/{deliberation_id}" in handler.ROUTES


# =============================================================================
# Test Active Deliberations Endpoint
# =============================================================================


class TestActiveDeliberations:
    """Tests for active deliberations endpoint."""

    @pytest.mark.asyncio
    async def test_get_active_deliberations_empty(self, handler, mock_request):
        """Should return empty list when no deliberations active."""
        mock_request.path = "/api/v1/deliberations/active"
        mock_request.method = "GET"

        with patch.object(handler, "_check_rbac_permission", return_value=None):
            result, status = await handler.handle_request(mock_request)

        assert status == 200
        assert result["count"] == 0
        assert result["deliberations"] == []

    @pytest.mark.asyncio
    async def test_get_active_deliberations_with_data(self, handler, mock_request):
        """Should return active deliberations from in-memory store."""
        # Register a deliberation
        register_deliberation(
            "test-123",
            {
                "task": "Test deliberation",
                "status": "active",
                "agents": ["claude", "gpt4"],
                "current_round": 2,
            },
        )

        mock_request.path = "/api/v1/deliberations/active"
        mock_request.method = "GET"

        with patch.object(handler, "_check_rbac_permission", return_value=None):
            result, status = await handler.handle_request(mock_request)

        assert status == 200
        assert result["count"] == 1
        assert len(result["deliberations"]) == 1
        assert result["deliberations"][0]["id"] == "test-123"


# =============================================================================
# Test Stats Endpoint
# =============================================================================


class TestDeliberationStats:
    """Tests for deliberation statistics endpoint."""

    @pytest.mark.asyncio
    async def test_get_stats_empty(self, handler, mock_request):
        """Should return default stats when no deliberations."""
        mock_request.path = "/api/v1/deliberations/stats"
        mock_request.method = "GET"

        with patch.object(handler, "_check_rbac_permission", return_value=None):
            result, status = await handler.handle_request(mock_request)

        assert status == 200
        assert "active_count" in result
        assert "completed_today" in result
        assert "average_consensus_time" in result
        assert "average_rounds" in result
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_get_stats_with_active_deliberation(self, handler, mock_request):
        """Should count active deliberations in stats."""
        # Register an active deliberation
        register_deliberation(
            "test-456",
            {
                "task": "Another deliberation",
                "status": "active",
                "agents": ["gemini"],
            },
        )

        mock_request.path = "/api/v1/deliberations/stats"
        mock_request.method = "GET"

        with patch.object(handler, "_check_rbac_permission", return_value=None):
            result, status = await handler.handle_request(mock_request)

        assert status == 200
        assert result["active_count"] >= 1


# =============================================================================
# Test Single Deliberation Endpoint
# =============================================================================


class TestGetDeliberation:
    """Tests for getting a single deliberation."""

    @pytest.mark.asyncio
    async def test_get_deliberation_found(self, handler, mock_request):
        """Should return deliberation when found."""
        register_deliberation(
            "delib-789",
            {
                "task": "Specific task",
                "status": "consensus_forming",
                "agents": ["claude", "gpt4", "gemini"],
                "current_round": 3,
            },
        )

        mock_request.path = "/api/v1/deliberations/delib-789"
        mock_request.method = "GET"

        with patch.object(handler, "_check_rbac_permission", return_value=None):
            result, status = await handler.handle_request(mock_request)

        assert status == 200
        assert result["id"] == "delib-789"
        assert result["task"] == "Specific task"

    @pytest.mark.asyncio
    async def test_get_deliberation_not_found(self, handler, mock_request):
        """Should return 404 when deliberation not found."""
        mock_request.path = "/api/v1/deliberations/nonexistent-id"
        mock_request.method = "GET"

        with patch.object(handler, "_check_rbac_permission", return_value=None):
            result, status = await handler.handle_request(mock_request)

        assert status == 404
        assert "error" in result


# =============================================================================
# Test Stream Endpoint
# =============================================================================


class TestDeliberationStream:
    """Tests for deliberation stream endpoint."""

    @pytest.mark.asyncio
    async def test_stream_endpoint(self, handler, mock_request):
        """Should return stream configuration."""
        mock_request.path = "/api/v1/deliberations/stream"
        mock_request.method = "GET"

        with patch.object(handler, "_check_rbac_permission", return_value=None):
            result, status = await handler.handle_request(mock_request)

        assert status == 200
        assert result["type"] == "websocket"
        assert "events" in result
        assert "agent_message" in result["events"]


# =============================================================================
# Test Module Functions
# =============================================================================


class TestModuleFunctions:
    """Tests for module-level functions."""

    def test_register_deliberation(self):
        """Should register a new deliberation."""
        register_deliberation(
            "new-delib",
            {
                "task": "New task",
                "status": "initializing",
            },
        )

        assert "new-delib" in _active_deliberations
        assert _active_deliberations["new-delib"]["task"] == "New task"
        assert "updated_at" in _active_deliberations["new-delib"]

    def test_update_deliberation(self):
        """Should update existing deliberation."""
        register_deliberation("update-test", {"task": "Original"})
        update_deliberation("update-test", {"task": "Updated", "current_round": 5})

        assert _active_deliberations["update-test"]["task"] == "Updated"
        assert _active_deliberations["update-test"]["current_round"] == 5

    def test_update_deliberation_not_found(self):
        """Should do nothing when updating non-existent deliberation."""
        update_deliberation("nonexistent", {"task": "Updated"})
        assert "nonexistent" not in _active_deliberations

    def test_complete_deliberation(self):
        """Should mark deliberation as complete."""
        register_deliberation("complete-test", {"task": "To complete", "status": "active"})
        complete_deliberation("complete-test")

        assert _active_deliberations["complete-test"]["status"] == "complete"
        assert _stats["completed_today"] == 1

    def test_complete_deliberation_not_found(self):
        """Should do nothing when completing non-existent deliberation."""
        initial_completed = _stats.get("completed_today", 0)
        complete_deliberation("nonexistent")
        assert _stats.get("completed_today", 0) == initial_completed

    @pytest.mark.asyncio
    async def test_broadcast_event(self):
        """Should broadcast event to stream clients."""
        from aragora.server.handlers.deliberations import _stream_clients
        import asyncio

        # Add a mock queue
        queue = asyncio.Queue()
        _stream_clients.append(queue)

        try:
            await broadcast_deliberation_event({"type": "test", "data": "hello"})
            event = await asyncio.wait_for(queue.get(), timeout=1.0)
            assert event["type"] == "test"
            assert event["data"] == "hello"
        finally:
            _stream_clients.remove(queue)


# =============================================================================
# Test Status Mapping
# =============================================================================


class TestStatusMapping:
    """Tests for debate status mapping."""

    def test_map_debate_status_pending(self, handler):
        """Should map pending to initializing."""
        assert handler._map_debate_status("pending") == "initializing"

    def test_map_debate_status_running(self, handler):
        """Should map running to active."""
        assert handler._map_debate_status("running") == "active"

    def test_map_debate_status_voting(self, handler):
        """Should map voting to consensus_forming."""
        assert handler._map_debate_status("voting") == "consensus_forming"

    def test_map_debate_status_complete(self, handler):
        """Should map complete to complete."""
        assert handler._map_debate_status("complete") == "complete"

    def test_map_debate_status_unknown(self, handler):
        """Should pass through unknown status."""
        assert handler._map_debate_status("custom") == "custom"


# =============================================================================
# Test RBAC
# =============================================================================


class TestRBAC:
    """Tests for RBAC permission checks."""

    @pytest.mark.asyncio
    async def test_rbac_denied_active(self, handler, mock_request):
        """Should return 403 when RBAC denies access to active deliberations."""
        mock_request.path = "/api/v1/deliberations/active"
        mock_request.method = "GET"

        with patch.object(
            handler,
            "_check_rbac_permission",
            return_value=({"error": "Permission denied: deliberation.read required"}, 403),
        ):
            result, status = await handler.handle_request(mock_request)

        assert status == 403
        assert "Permission denied" in result["error"]

    @pytest.mark.asyncio
    async def test_rbac_denied_stats(self, handler, mock_request):
        """Should return 403 when RBAC denies access to stats."""
        mock_request.path = "/api/v1/deliberations/stats"
        mock_request.method = "GET"

        with patch.object(
            handler,
            "_check_rbac_permission",
            return_value=({"error": "Permission denied"}, 403),
        ):
            result, status = await handler.handle_request(mock_request)

        assert status == 403


# =============================================================================
# Test Route Not Found
# =============================================================================


class TestRouteNotFound:
    """Tests for unknown routes."""

    @pytest.mark.asyncio
    async def test_unknown_route(self, handler, mock_request):
        """Should return 404 for unknown routes."""
        mock_request.path = "/api/v1/deliberations/unknown"
        mock_request.method = "POST"

        result, status = await handler.handle_request(mock_request)

        assert status == 404
        assert "error" in result
