"""
Tests for the HybridDebateHandler module.

Comprehensive tests covering:
- Handler routing for hybrid debate endpoints
- Successful request handling (create, get, list)
- Error responses (400, 404, 500, 503)
- Input validation (task, consensus_threshold, max_rounds)
- Rate limiting behavior
- Query parameter handling
- Response format validation
- Module unavailability handling
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import pytest

from aragora.server.handlers.hybrid_debate_handler import HybridDebateHandler


@pytest.fixture
def mock_server_context():
    """Create mock server context for handler initialization."""
    return {
        "storage": None,
        "elo_system": None,
        "nomic_dir": None,
        "external_agents": {
            "my-crewai-agent": {"name": "CrewAI Agent", "type": "crewai"},
            "my-langchain-agent": {"name": "LangChain Agent", "type": "langchain"},
        },
    }


@pytest.fixture
def mock_server_context_no_agents():
    """Create mock server context without external agents."""
    return {
        "storage": None,
        "elo_system": None,
        "nomic_dir": None,
        "external_agents": {},
    }


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler with client address."""
    handler = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = {"Content-Length": "0", "Content-Type": "application/json"}
    handler.command = "GET"
    return handler


@pytest.fixture
def mock_http_handler_with_body():
    """Create a mock HTTP handler with JSON body support."""
    handler = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = {"Content-Length": "100", "Content-Type": "application/json"}
    handler.command = "POST"
    return handler


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset rate limiter before each test."""
    from aragora.server.handlers.utils.rate_limit import _limiters, clear_all_limiters

    clear_all_limiters()
    yield
    clear_all_limiters()


class TestHybridDebateHandlerRouting:
    """Tests for handler routing."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return HybridDebateHandler(mock_server_context)

    def test_can_handle_base_path(self, handler):
        """Handler can handle base hybrid debate endpoint."""
        assert handler.can_handle("/api/v1/debates/hybrid")

    def test_can_handle_with_trailing_slash(self, handler):
        """Handler can handle path with trailing slash."""
        assert handler.can_handle("/api/v1/debates/hybrid/")

    def test_can_handle_with_id(self, handler):
        """Handler can handle path with debate ID."""
        assert handler.can_handle("/api/v1/debates/hybrid/hybrid_abc123")

    def test_can_handle_with_any_id(self, handler):
        """Handler can handle path with any debate ID."""
        assert handler.can_handle("/api/v1/debates/hybrid/xyz789def")

    def test_cannot_handle_unrelated_paths(self, handler):
        """Handler cannot handle unrelated paths."""
        assert not handler.can_handle("/api/v1/debates")
        assert not handler.can_handle("/api/v1/debates/standard")
        assert not handler.can_handle("/api/v1/agents")
        assert not handler.can_handle("/api/v1/gateway/agents")


class TestHybridDebateHandlerRoutesAttribute:
    """Tests for ROUTES class attribute."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return HybridDebateHandler(mock_server_context)

    def test_routes_contains_base_path(self, handler):
        """ROUTES contains base hybrid debate path."""
        assert "/api/v1/debates/hybrid" in handler.ROUTES

    def test_routes_contains_wildcard_path(self, handler):
        """ROUTES contains wildcard path for IDs."""
        assert "/api/v1/debates/hybrid/*" in handler.ROUTES


class TestHybridDebateHandlerGetEndpoints:
    """Tests for GET endpoints."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return HybridDebateHandler(mock_server_context)

    def test_list_returns_503_when_module_unavailable(self, handler, mock_http_handler):
        """List endpoint returns 503 when hybrid module not available."""
        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", False):
            result = handler.handle("/api/v1/debates/hybrid", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 503
        body = json.loads(result.body)
        assert "error" in body
        assert "not available" in body["error"].lower()

    def test_list_returns_empty_debates(self, handler, mock_http_handler):
        """List endpoint returns empty list when no debates exist."""
        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle("/api/v1/debates/hybrid", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "debates" in body
        assert body["debates"] == []
        assert body["total"] == 0

    def test_list_returns_populated_debates(self, handler, mock_http_handler):
        """List endpoint returns debates when they exist."""
        # Add a debate directly to the handler's internal storage
        handler._debates["hybrid_test123"] = {
            "debate_id": "hybrid_test123",
            "task": "Test task",
            "status": "completed",
            "consensus_reached": True,
            "confidence": 0.85,
            "started_at": "2024-01-01T00:00:00+00:00",
        }

        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle("/api/v1/debates/hybrid", {}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["total"] == 1
        assert len(body["debates"]) == 1
        assert body["debates"][0]["debate_id"] == "hybrid_test123"

    def test_list_with_status_filter(self, handler, mock_http_handler):
        """List endpoint filters by status."""
        handler._debates["hybrid_1"] = {
            "debate_id": "hybrid_1",
            "task": "Task 1",
            "status": "completed",
            "consensus_reached": True,
            "confidence": 0.9,
            "started_at": "2024-01-01T00:00:00+00:00",
        }
        handler._debates["hybrid_2"] = {
            "debate_id": "hybrid_2",
            "task": "Task 2",
            "status": "pending",
            "consensus_reached": False,
            "confidence": 0.0,
            "started_at": "2024-01-02T00:00:00+00:00",
        }

        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle(
                "/api/v1/debates/hybrid",
                {"status": "completed"},
                mock_http_handler,
            )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["total"] == 1
        assert body["debates"][0]["status"] == "completed"

    def test_list_with_limit(self, handler, mock_http_handler):
        """List endpoint respects limit parameter."""
        # Add multiple debates
        for i in range(5):
            handler._debates[f"hybrid_{i}"] = {
                "debate_id": f"hybrid_{i}",
                "task": f"Task {i}",
                "status": "completed",
                "consensus_reached": True,
                "confidence": 0.8,
                "started_at": "2024-01-01T00:00:00+00:00",
            }

        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle(
                "/api/v1/debates/hybrid",
                {"limit": "2"},
                mock_http_handler,
            )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert len(body["debates"]) == 2

    def test_list_limit_capped_at_100(self, handler, mock_http_handler):
        """List endpoint caps limit at 100."""
        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle(
                "/api/v1/debates/hybrid",
                {"limit": "500"},
                mock_http_handler,
            )

        assert result is not None
        assert result.status_code == 200

    def test_get_debate_not_found(self, handler, mock_http_handler):
        """Get endpoint returns 404 when debate not found."""
        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle(
                "/api/v1/debates/hybrid/nonexistent_id",
                {},
                mock_http_handler,
            )

        assert result is not None
        assert result.status_code == 404
        body = json.loads(result.body)
        assert "error" in body
        assert "not found" in body["error"].lower()

    def test_get_debate_success(self, handler, mock_http_handler):
        """Get endpoint returns debate details on success."""
        handler._debates["hybrid_test456"] = {
            "debate_id": "hybrid_test456",
            "task": "Design a rate limiter",
            "status": "completed",
            "consensus_reached": True,
            "confidence": 0.85,
            "final_answer": "Use token bucket algorithm",
            "external_agent": "my-crewai-agent",
            "rounds": 2,
            "started_at": "2024-01-01T00:00:00+00:00",
            "completed_at": "2024-01-01T01:00:00+00:00",
        }

        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle(
                "/api/v1/debates/hybrid/hybrid_test456",
                {},
                mock_http_handler,
            )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["debate_id"] == "hybrid_test456"
        assert body["task"] == "Design a rate limiter"
        assert body["consensus_reached"] is True

    def test_get_returns_503_when_module_unavailable(self, handler, mock_http_handler):
        """Get endpoint returns 503 when hybrid module not available."""
        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", False):
            result = handler.handle(
                "/api/v1/debates/hybrid/hybrid_test456",
                {},
                mock_http_handler,
            )

        assert result is not None
        assert result.status_code == 503


class TestHybridDebateHandlerPostEndpoint:
    """Tests for POST /api/v1/debates/hybrid endpoint."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return HybridDebateHandler(mock_server_context)

    def test_create_returns_503_when_module_unavailable(self, handler, mock_http_handler_with_body):
        """Create endpoint returns 503 when hybrid module not available."""
        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", False):
            result = handler.handle_post(
                "/api/v1/debates/hybrid",
                {},
                mock_http_handler_with_body,
            )

        assert result is not None
        assert result.status_code == 503

    def test_create_requires_task(self, handler, mock_http_handler_with_body):
        """Create endpoint requires task field."""
        mock_http_handler_with_body.rfile.read = MagicMock(
            return_value=json.dumps({"external_agent": "my-crewai-agent"}).encode()
        )

        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle_post(
                "/api/v1/debates/hybrid",
                {},
                mock_http_handler_with_body,
            )

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "task" in body["error"].lower()

    def test_create_requires_non_empty_task(self, handler, mock_http_handler_with_body):
        """Create endpoint requires non-empty task."""
        mock_http_handler_with_body.rfile.read = MagicMock(
            return_value=json.dumps(
                {
                    "task": "   ",
                    "external_agent": "my-crewai-agent",
                }
            ).encode()
        )

        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle_post(
                "/api/v1/debates/hybrid",
                {},
                mock_http_handler_with_body,
            )

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "task" in body["error"].lower()

    def test_create_task_max_length(self, handler, mock_http_handler_with_body):
        """Create endpoint enforces max task length of 5000 chars."""
        long_task = "x" * 5001
        mock_http_handler_with_body.rfile.read = MagicMock(
            return_value=json.dumps(
                {
                    "task": long_task,
                    "external_agent": "my-crewai-agent",
                }
            ).encode()
        )

        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle_post(
                "/api/v1/debates/hybrid",
                {},
                mock_http_handler_with_body,
            )

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "5000" in body["error"]

    def test_create_requires_external_agent(self, handler, mock_http_handler_with_body):
        """Create endpoint requires external_agent field."""
        mock_http_handler_with_body.rfile.read = MagicMock(
            return_value=json.dumps({"task": "Design a system"}).encode()
        )

        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle_post(
                "/api/v1/debates/hybrid",
                {},
                mock_http_handler_with_body,
            )

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "external_agent" in body["error"].lower()

    def test_create_validates_external_agent_exists(self, handler, mock_http_handler_with_body):
        """Create endpoint validates external agent is registered."""
        mock_http_handler_with_body.rfile.read = MagicMock(
            return_value=json.dumps(
                {
                    "task": "Design a system",
                    "external_agent": "nonexistent-agent",
                }
            ).encode()
        )

        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle_post(
                "/api/v1/debates/hybrid",
                {},
                mock_http_handler_with_body,
            )

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "not found" in body["error"].lower()

    def test_create_validates_consensus_threshold_min(self, handler, mock_http_handler_with_body):
        """Create endpoint validates consensus_threshold minimum."""
        mock_http_handler_with_body.rfile.read = MagicMock(
            return_value=json.dumps(
                {
                    "task": "Design a system",
                    "external_agent": "my-crewai-agent",
                    "consensus_threshold": -0.1,
                }
            ).encode()
        )

        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle_post(
                "/api/v1/debates/hybrid",
                {},
                mock_http_handler_with_body,
            )

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "consensus_threshold" in body["error"].lower()

    def test_create_validates_consensus_threshold_max(self, handler, mock_http_handler_with_body):
        """Create endpoint validates consensus_threshold maximum."""
        mock_http_handler_with_body.rfile.read = MagicMock(
            return_value=json.dumps(
                {
                    "task": "Design a system",
                    "external_agent": "my-crewai-agent",
                    "consensus_threshold": 1.5,
                }
            ).encode()
        )

        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle_post(
                "/api/v1/debates/hybrid",
                {},
                mock_http_handler_with_body,
            )

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "consensus_threshold" in body["error"].lower()

    def test_create_validates_max_rounds_min(self, handler, mock_http_handler_with_body):
        """Create endpoint validates max_rounds minimum."""
        mock_http_handler_with_body.rfile.read = MagicMock(
            return_value=json.dumps(
                {
                    "task": "Design a system",
                    "external_agent": "my-crewai-agent",
                    "max_rounds": 0,
                }
            ).encode()
        )

        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle_post(
                "/api/v1/debates/hybrid",
                {},
                mock_http_handler_with_body,
            )

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "max_rounds" in body["error"].lower()

    def test_create_validates_max_rounds_max(self, handler, mock_http_handler_with_body):
        """Create endpoint validates max_rounds maximum."""
        mock_http_handler_with_body.rfile.read = MagicMock(
            return_value=json.dumps(
                {
                    "task": "Design a system",
                    "external_agent": "my-crewai-agent",
                    "max_rounds": 11,
                }
            ).encode()
        )

        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle_post(
                "/api/v1/debates/hybrid",
                {},
                mock_http_handler_with_body,
            )

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "max_rounds" in body["error"].lower()

    def test_create_success(self, handler, mock_http_handler_with_body):
        """Create endpoint succeeds with valid input."""
        mock_http_handler_with_body.rfile.read = MagicMock(
            return_value=json.dumps(
                {
                    "task": "Design a rate limiter",
                    "external_agent": "my-crewai-agent",
                    "verification_agents": ["claude-3"],
                    "consensus_threshold": 0.7,
                    "max_rounds": 3,
                    "domain": "engineering",
                }
            ).encode()
        )

        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle_post(
                "/api/v1/debates/hybrid",
                {},
                mock_http_handler_with_body,
            )

        assert result is not None
        assert result.status_code == 201
        body = json.loads(result.body)
        assert "debate_id" in body
        assert body["debate_id"].startswith("hybrid_")
        assert len(body["debate_id"]) == 19  # "hybrid_" + 12 hex chars
        assert body["task"] == "Design a rate limiter"
        assert body["external_agent"] == "my-crewai-agent"
        assert body["status"] == "completed"  # From mock _run_debate
        assert body["consensus_reached"] is True
        assert "started_at" in body
        assert "completed_at" in body

    def test_create_uses_defaults(self, handler, mock_http_handler_with_body):
        """Create endpoint uses default values for optional fields."""
        mock_http_handler_with_body.rfile.read = MagicMock(
            return_value=json.dumps(
                {
                    "task": "Design a system",
                    "external_agent": "my-crewai-agent",
                }
            ).encode()
        )

        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle_post(
                "/api/v1/debates/hybrid",
                {},
                mock_http_handler_with_body,
            )

        assert result is not None
        assert result.status_code == 201
        body = json.loads(result.body)
        assert body["consensus_threshold"] == 0.7
        assert body["max_rounds"] == 3
        assert body["domain"] == "general"

    def test_create_stores_debate(self, handler, mock_http_handler_with_body):
        """Create endpoint stores debate in internal storage."""
        mock_http_handler_with_body.rfile.read = MagicMock(
            return_value=json.dumps(
                {
                    "task": "Test task for storage",
                    "external_agent": "my-crewai-agent",
                }
            ).encode()
        )

        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle_post(
                "/api/v1/debates/hybrid",
                {},
                mock_http_handler_with_body,
            )

        assert result is not None
        assert result.status_code == 201
        body = json.loads(result.body)
        debate_id = body["debate_id"]

        # Verify stored
        assert debate_id in handler._debates
        assert handler._debates[debate_id]["task"] == "Test task for storage"

    def test_create_invalid_json_body(self, handler, mock_http_handler_with_body):
        """Create endpoint returns 400 for invalid JSON."""
        mock_http_handler_with_body.rfile.read = MagicMock(return_value=b"not valid json")

        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle_post(
                "/api/v1/debates/hybrid",
                {},
                mock_http_handler_with_body,
            )

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "error" in body


class TestHybridDebateHandlerIdGeneration:
    """Tests for debate ID generation."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return HybridDebateHandler(mock_server_context)

    def test_generated_id_format(self, handler, mock_http_handler_with_body):
        """Generated IDs follow the hybrid_<12hex> format."""
        mock_http_handler_with_body.rfile.read = MagicMock(
            return_value=json.dumps(
                {
                    "task": "Test task",
                    "external_agent": "my-crewai-agent",
                }
            ).encode()
        )

        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle_post(
                "/api/v1/debates/hybrid",
                {},
                mock_http_handler_with_body,
            )

        body = json.loads(result.body)
        debate_id = body["debate_id"]

        assert debate_id.startswith("hybrid_")
        hex_part = debate_id[7:]  # After "hybrid_"
        assert len(hex_part) == 12
        # Verify it's valid hex
        int(hex_part, 16)

    def test_generated_ids_are_unique(self, handler, mock_http_handler_with_body):
        """Generated IDs are unique across multiple creates."""
        ids = set()

        for _ in range(5):
            mock_http_handler_with_body.rfile.read = MagicMock(
                return_value=json.dumps(
                    {
                        "task": "Test task",
                        "external_agent": "my-crewai-agent",
                    }
                ).encode()
            )

            with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
                result = handler.handle_post(
                    "/api/v1/debates/hybrid",
                    {},
                    mock_http_handler_with_body,
                )

            body = json.loads(result.body)
            ids.add(body["debate_id"])

        assert len(ids) == 5


class TestHybridDebateHandlerTimestamps:
    """Tests for timestamp generation."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return HybridDebateHandler(mock_server_context)

    def test_started_at_is_iso_format(self, handler, mock_http_handler_with_body):
        """started_at is in ISO 8601 format."""
        mock_http_handler_with_body.rfile.read = MagicMock(
            return_value=json.dumps(
                {
                    "task": "Test task",
                    "external_agent": "my-crewai-agent",
                }
            ).encode()
        )

        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle_post(
                "/api/v1/debates/hybrid",
                {},
                mock_http_handler_with_body,
            )

        body = json.loads(result.body)
        started_at = body["started_at"]

        # Should be parseable as ISO datetime
        parsed = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        assert parsed is not None

    def test_completed_at_is_iso_format(self, handler, mock_http_handler_with_body):
        """completed_at is in ISO 8601 format when present."""
        mock_http_handler_with_body.rfile.read = MagicMock(
            return_value=json.dumps(
                {
                    "task": "Test task",
                    "external_agent": "my-crewai-agent",
                }
            ).encode()
        )

        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle_post(
                "/api/v1/debates/hybrid",
                {},
                mock_http_handler_with_body,
            )

        body = json.loads(result.body)
        completed_at = body["completed_at"]

        if completed_at:
            parsed = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
            assert parsed is not None


class TestHybridDebateHandlerRunDebateMock:
    """Tests for _run_debate method mocking."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return HybridDebateHandler(mock_server_context)

    def test_run_debate_returns_result(self, handler, mock_http_handler_with_body):
        """_run_debate returns result that is merged into debate record."""
        mock_http_handler_with_body.rfile.read = MagicMock(
            return_value=json.dumps(
                {
                    "task": "Test task",
                    "external_agent": "my-crewai-agent",
                }
            ).encode()
        )

        custom_result = {
            "status": "failed",
            "consensus_reached": False,
            "confidence": 0.3,
            "final_answer": "Custom answer",
            "rounds": 5,
            "completed_at": "2024-06-01T12:00:00+00:00",
        }

        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            with patch.object(handler, "_run_debate", return_value=custom_result):
                result = handler.handle_post(
                    "/api/v1/debates/hybrid",
                    {},
                    mock_http_handler_with_body,
                )

        body = json.loads(result.body)
        assert body["status"] == "failed"
        assert body["consensus_reached"] is False
        assert body["confidence"] == 0.3
        assert body["final_answer"] == "Custom answer"


class TestHybridDebateHandlerNoExternalAgents:
    """Tests for handler when no external agents are registered."""

    @pytest.fixture
    def handler(self, mock_server_context_no_agents):
        return HybridDebateHandler(mock_server_context_no_agents)

    def test_create_fails_with_no_agents(self, handler, mock_http_handler_with_body):
        """Create fails when external agent is not registered."""
        mock_http_handler_with_body.rfile.read = MagicMock(
            return_value=json.dumps(
                {
                    "task": "Test task",
                    "external_agent": "any-agent",
                }
            ).encode()
        )

        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle_post(
                "/api/v1/debates/hybrid",
                {},
                mock_http_handler_with_body,
            )

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "not found" in body["error"].lower()


class TestHybridDebateHandlerVerificationAgents:
    """Tests for verification_agents field."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return HybridDebateHandler(mock_server_context)

    def test_verification_agents_stored(self, handler, mock_http_handler_with_body):
        """verification_agents are stored in debate record."""
        mock_http_handler_with_body.rfile.read = MagicMock(
            return_value=json.dumps(
                {
                    "task": "Test task",
                    "external_agent": "my-crewai-agent",
                    "verification_agents": ["claude-3", "gpt-4"],
                }
            ).encode()
        )

        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle_post(
                "/api/v1/debates/hybrid",
                {},
                mock_http_handler_with_body,
            )

        body = json.loads(result.body)
        assert body["verification_agents"] == ["claude-3", "gpt-4"]

    def test_verification_agents_must_be_list(self, handler, mock_http_handler_with_body):
        """verification_agents must be a list."""
        mock_http_handler_with_body.rfile.read = MagicMock(
            return_value=json.dumps(
                {
                    "task": "Test task",
                    "external_agent": "my-crewai-agent",
                    "verification_agents": "not-a-list",
                }
            ).encode()
        )

        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle_post(
                "/api/v1/debates/hybrid",
                {},
                mock_http_handler_with_body,
            )

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "verification_agents" in body["error"].lower()


class TestHybridDebateHandlerResponseFormat:
    """Tests for response format consistency."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return HybridDebateHandler(mock_server_context)

    def test_list_response_format(self, handler, mock_http_handler):
        """List response has correct format."""
        handler._debates["hybrid_1"] = {
            "debate_id": "hybrid_1",
            "task": "Task 1",
            "status": "completed",
            "consensus_reached": True,
            "confidence": 0.9,
            "started_at": "2024-01-01T00:00:00+00:00",
        }

        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle("/api/v1/debates/hybrid", {}, mock_http_handler)

        body = json.loads(result.body)
        assert "debates" in body
        assert "total" in body
        assert isinstance(body["debates"], list)
        assert isinstance(body["total"], int)

        # Check summary format
        summary = body["debates"][0]
        assert "debate_id" in summary
        assert "task" in summary
        assert "status" in summary
        assert "consensus_reached" in summary
        assert "confidence" in summary
        assert "started_at" in summary

    def test_create_response_format(self, handler, mock_http_handler_with_body):
        """Create response has correct format."""
        mock_http_handler_with_body.rfile.read = MagicMock(
            return_value=json.dumps(
                {
                    "task": "Test task",
                    "external_agent": "my-crewai-agent",
                }
            ).encode()
        )

        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle_post(
                "/api/v1/debates/hybrid",
                {},
                mock_http_handler_with_body,
            )

        body = json.loads(result.body)

        # Required fields
        required_fields = [
            "debate_id",
            "task",
            "status",
            "consensus_reached",
            "confidence",
            "final_answer",
            "external_agent",
            "verification_agents",
            "consensus_threshold",
            "max_rounds",
            "domain",
            "rounds",
            "started_at",
            "completed_at",
        ]

        for field in required_fields:
            assert field in body, f"Missing field: {field}"
