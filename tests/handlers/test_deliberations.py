"""Tests for deliberations handler.

Tests the deliberations (vetted decisionmaking) API endpoints including:
- GET /api/v1/deliberations/active - List active deliberation sessions
- GET /api/v1/deliberations/stats - Get aggregate statistics
- GET /api/v1/deliberations/{id} - Get single deliberation
- GET /api/v1/deliberations/stream - WebSocket stream config
"""

import json
from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers import deliberations


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_deliberations_state():
    """Reset deliberations state before each test."""
    deliberations._active_deliberations.clear()
    deliberations._stream_clients.clear()
    deliberations._stats.update(
        {
            "active_count": 0,
            "completed_today": 0,
            "average_consensus_time": 0,
            "average_rounds": 0,
            "top_agents": [],
        }
    )
    yield
    deliberations._active_deliberations.clear()
    deliberations._stream_clients.clear()


@pytest.fixture
def deliberations_handler():
    """Create deliberations handler instance."""
    return deliberations.DeliberationsHandler({})


@pytest.fixture
def mock_request():
    """Create a mock request object."""
    request = MagicMock()
    request.path = "/api/v1/deliberations/active"
    request.method = "GET"
    return request


@pytest.fixture
def sample_deliberation():
    """Sample deliberation data."""
    return {
        "id": "delib-123",
        "task": "Should we adopt policy X?",
        "status": "active",
        "agents": ["claude", "gpt-4", "gemini"],
        "current_round": 2,
        "total_rounds": 5,
        "consensus_score": 0.6,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "message_count": 15,
    }


# =============================================================================
# Routing Tests
# =============================================================================


class TestDeliberationsRouting:
    """Tests for request routing."""

    @pytest.mark.asyncio
    async def test_route_active_deliberations(self, deliberations_handler):
        """Test GET /active routes to active deliberations."""
        request = MagicMock()
        request.path = "/api/v1/deliberations/active"
        request.method = "GET"

        result, status = await deliberations_handler.handle_request(request)

        assert status == 200
        assert "deliberations" in result
        assert "count" in result

    @pytest.mark.asyncio
    async def test_route_stats(self, deliberations_handler):
        """Test GET /stats routes to statistics."""
        request = MagicMock()
        request.path = "/api/v1/deliberations/stats"
        request.method = "GET"

        result, status = await deliberations_handler.handle_request(request)

        assert status == 200
        assert "active_count" in result
        assert "completed_today" in result

    @pytest.mark.asyncio
    async def test_route_single_deliberation(self, deliberations_handler, sample_deliberation):
        """Test GET /{id} routes to single deliberation."""
        # Register a deliberation first
        deliberations.register_deliberation("delib-123", sample_deliberation)

        request = MagicMock()
        request.path = "/api/v1/deliberations/delib-123"
        request.method = "GET"

        result, status = await deliberations_handler.handle_request(request)

        assert status == 200
        assert result["id"] == "delib-123"

    @pytest.mark.asyncio
    async def test_route_stream(self, deliberations_handler):
        """Test GET /stream returns WebSocket configuration."""
        request = MagicMock()
        request.path = "/api/v1/deliberations/stream"
        request.method = "GET"

        result, status = await deliberations_handler.handle_request(request)

        assert status == 200
        assert result["type"] == "websocket"
        assert "events" in result

    @pytest.mark.asyncio
    async def test_route_not_found(self, deliberations_handler):
        """Test unknown path returns 404."""
        request = MagicMock()
        request.path = "/api/v1/deliberations/unknown/path"
        request.method = "GET"

        result, status = await deliberations_handler.handle_request(request)

        assert status == 404
        assert "error" in result


# =============================================================================
# Active Deliberations Tests
# =============================================================================


class TestActiveDeliberations:
    """Tests for active deliberations retrieval."""

    @pytest.mark.asyncio
    async def test_get_active_empty(self, deliberations_handler):
        """Test getting active deliberations when none exist."""
        request = MagicMock()
        request.path = "/api/v1/deliberations/active"
        request.method = "GET"

        result, status = await deliberations_handler.handle_request(request)

        assert status == 200
        assert result["deliberations"] == []
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_get_active_with_in_memory(self, deliberations_handler, sample_deliberation):
        """Test getting active deliberations from in-memory store."""
        # Register some deliberations
        deliberations.register_deliberation("delib-1", {**sample_deliberation, "id": "delib-1"})
        deliberations.register_deliberation("delib-2", {**sample_deliberation, "id": "delib-2"})

        request = MagicMock()
        request.path = "/api/v1/deliberations/active"
        request.method = "GET"

        result, status = await deliberations_handler.handle_request(request)

        assert status == 200
        assert result["count"] >= 2
        ids = [d["id"] for d in result["deliberations"]]
        assert "delib-1" in ids
        assert "delib-2" in ids

    def test_format_deliberation(self, deliberations_handler):
        """Test deliberation formatting."""
        debate = {
            "id": "debate-123",
            "task": "Test task",
            "status": "running",
            "agents": ["claude", "gpt-4"],
            "messages": [{"round": 1}, {"round": 2}],
            "total_rounds": 5,
            "consensus_score": 0.7,
            "created_at": "2024-01-01T00:00:00Z",
        }

        formatted = deliberations_handler._format_deliberation(debate)

        assert formatted["id"] == "debate-123"
        assert formatted["task"] == "Test task"
        assert formatted["status"] == "active"  # "running" maps to "active"
        assert formatted["agents"] == ["claude", "gpt-4"]
        assert formatted["message_count"] == 2

    def test_status_mapping(self, deliberations_handler):
        """Test debate status to deliberation status mapping."""
        assert deliberations_handler._map_debate_status("pending") == "initializing"
        assert deliberations_handler._map_debate_status("running") == "active"
        assert deliberations_handler._map_debate_status("streaming") == "active"
        assert deliberations_handler._map_debate_status("voting") == "consensus_forming"
        assert deliberations_handler._map_debate_status("complete") == "complete"
        assert deliberations_handler._map_debate_status("completed") == "complete"
        assert deliberations_handler._map_debate_status("failed") == "failed"
        assert deliberations_handler._map_debate_status("unknown") == "unknown"


# =============================================================================
# Statistics Tests
# =============================================================================


class TestDeliberationStats:
    """Tests for deliberation statistics."""

    @pytest.mark.asyncio
    async def test_get_stats_default(self, deliberations_handler):
        """Test getting stats returns default values."""
        request = MagicMock()
        request.path = "/api/v1/deliberations/stats"
        request.method = "GET"

        result, status = await deliberations_handler.handle_request(request)

        assert status == 200
        assert "active_count" in result
        assert "completed_today" in result
        assert "average_consensus_time" in result
        assert "average_rounds" in result
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_stats_active_count(self, deliberations_handler, sample_deliberation):
        """Test active count reflects registered deliberations."""
        # Register active deliberations
        deliberations.register_deliberation("delib-1", {**sample_deliberation, "status": "active"})
        deliberations.register_deliberation(
            "delib-2", {**sample_deliberation, "status": "consensus_forming"}
        )
        deliberations.register_deliberation(
            "delib-3", {**sample_deliberation, "status": "complete"}
        )

        request = MagicMock()
        request.path = "/api/v1/deliberations/stats"
        request.method = "GET"

        result, status = await deliberations_handler.handle_request(request)

        assert status == 200
        # Only active and consensus_forming should be counted
        assert result["active_count"] >= 2


# =============================================================================
# Single Deliberation Tests
# =============================================================================


class TestSingleDeliberation:
    """Tests for single deliberation retrieval."""

    @pytest.mark.asyncio
    async def test_get_deliberation_found(self, deliberations_handler, sample_deliberation):
        """Test getting existing deliberation."""
        deliberations.register_deliberation("delib-123", sample_deliberation)

        request = MagicMock()
        request.path = "/api/v1/deliberations/delib-123"
        request.method = "GET"

        result, status = await deliberations_handler.handle_request(request)

        assert status == 200
        assert result["id"] == "delib-123"

    @pytest.mark.asyncio
    async def test_get_deliberation_not_found(self, deliberations_handler):
        """Test getting non-existent deliberation returns 404."""
        request = MagicMock()
        request.path = "/api/v1/deliberations/nonexistent"
        request.method = "GET"

        result, status = await deliberations_handler.handle_request(request)

        assert status == 404
        assert "error" in result


# =============================================================================
# Module Function Tests
# =============================================================================


class TestModuleFunctions:
    """Tests for module-level functions."""

    def test_register_deliberation(self):
        """Test registering a new deliberation."""
        data = {"task": "Test task", "agents": ["claude"]}
        deliberations.register_deliberation("delib-new", data)

        assert "delib-new" in deliberations._active_deliberations
        assert deliberations._active_deliberations["delib-new"]["id"] == "delib-new"
        assert deliberations._active_deliberations["delib-new"]["task"] == "Test task"
        assert "updated_at" in deliberations._active_deliberations["delib-new"]

    def test_update_deliberation(self):
        """Test updating an existing deliberation."""
        deliberations.register_deliberation("delib-123", {"task": "Original"})

        deliberations.update_deliberation("delib-123", {"current_round": 3, "consensus_score": 0.8})

        delib = deliberations._active_deliberations["delib-123"]
        assert delib["current_round"] == 3
        assert delib["consensus_score"] == 0.8
        assert "updated_at" in delib

    def test_update_deliberation_not_found(self):
        """Test updating non-existent deliberation does nothing."""
        deliberations.update_deliberation("nonexistent", {"current_round": 3})

        assert "nonexistent" not in deliberations._active_deliberations

    def test_complete_deliberation(self):
        """Test completing a deliberation."""
        deliberations.register_deliberation("delib-123", {"status": "active"})

        deliberations.complete_deliberation("delib-123")

        assert deliberations._active_deliberations["delib-123"]["status"] == "complete"
        assert deliberations._stats["completed_today"] == 1

    def test_complete_deliberation_not_found(self):
        """Test completing non-existent deliberation does nothing."""
        initial_completed = deliberations._stats["completed_today"]

        deliberations.complete_deliberation("nonexistent")

        assert deliberations._stats["completed_today"] == initial_completed


# =============================================================================
# Broadcast Tests
# =============================================================================


class TestBroadcast:
    """Tests for event broadcasting."""

    @pytest.mark.asyncio
    async def test_broadcast_deliberation_event(self):
        """Test broadcasting events to connected clients."""
        import asyncio

        # Add a mock client queue
        client_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        deliberations._stream_clients.append(client_queue)

        event = {"type": "agent_message", "agent": "claude", "content": "Test"}

        await deliberations.broadcast_deliberation_event(event)

        # Check the event was added to the queue
        received = await asyncio.wait_for(client_queue.get(), timeout=1.0)
        assert received["type"] == "agent_message"
        assert received["agent"] == "claude"

    @pytest.mark.asyncio
    async def test_broadcast_handles_failed_clients(self):
        """Test broadcast gracefully handles failed clients."""
        import asyncio

        # Add a mock client that will fail
        failing_queue = MagicMock()
        failing_queue.put = AsyncMock(side_effect=Exception("Client disconnected"))
        deliberations._stream_clients.append(failing_queue)

        # Should not raise
        await deliberations.broadcast_deliberation_event({"type": "test"})
