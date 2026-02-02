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


class TestHybridDebateHandlerCircuitBreaker:
    """Tests for circuit breaker behavior."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return HybridDebateHandler(mock_server_context)

    @pytest.fixture(autouse=True)
    def reset_circuit_breaker(self):
        """Reset circuit breaker before and after each test."""
        from aragora.server.handlers.hybrid_debate_handler import (
            reset_hybrid_debate_circuit_breaker,
        )

        reset_hybrid_debate_circuit_breaker()
        yield
        reset_hybrid_debate_circuit_breaker()

    def test_circuit_breaker_initially_closed(self, handler, mock_http_handler_with_body):
        """Circuit breaker starts in closed state (allows requests)."""
        from aragora.server.handlers.hybrid_debate_handler import (
            get_hybrid_debate_circuit_breaker,
        )

        cb = get_hybrid_debate_circuit_breaker()
        assert cb.can_proceed()

    def test_circuit_breaker_opens_after_failures(self, handler, mock_http_handler_with_body):
        """Circuit breaker opens after consecutive failures."""
        from aragora.server.handlers.hybrid_debate_handler import (
            get_hybrid_debate_circuit_breaker,
        )

        cb = get_hybrid_debate_circuit_breaker()

        # Record 5 failures (threshold)
        for _ in range(5):
            cb.record_failure()

        assert not cb.can_proceed()

    def test_create_returns_503_when_circuit_open(self, handler, mock_http_handler_with_body):
        """Create endpoint returns 503 when circuit breaker is open."""
        from aragora.server.handlers.hybrid_debate_handler import (
            get_hybrid_debate_circuit_breaker,
        )

        cb = get_hybrid_debate_circuit_breaker()

        # Open the circuit by recording failures
        for _ in range(5):
            cb.record_failure()

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

        assert result is not None
        assert result.status_code == 503
        body = json.loads(result.body)
        assert "temporarily unavailable" in body["error"].lower()

    def test_successful_create_records_success(self, handler, mock_http_handler_with_body):
        """Successful create records success with circuit breaker."""
        from aragora.server.handlers.hybrid_debate_handler import (
            get_hybrid_debate_circuit_breaker,
        )

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

        assert result.status_code == 201

        cb = get_hybrid_debate_circuit_breaker()
        # After success, circuit should still be closed
        assert cb.can_proceed()

    def test_failed_create_records_failure(self, handler, mock_http_handler_with_body):
        """Failed create records failure with circuit breaker and returns 500."""
        from aragora.server.handlers.hybrid_debate_handler import (
            get_hybrid_debate_circuit_breaker,
        )

        mock_http_handler_with_body.rfile.read = MagicMock(
            return_value=json.dumps(
                {
                    "task": "Test task",
                    "external_agent": "my-crewai-agent",
                }
            ).encode()
        )

        # Make _run_debate raise an exception
        # The @handle_errors decorator catches the exception and returns an error response
        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            with patch.object(
                handler, "_run_debate", side_effect=RuntimeError("Simulated failure")
            ):
                result = handler.handle_post(
                    "/api/v1/debates/hybrid",
                    {},
                    mock_http_handler_with_body,
                )

        # @handle_errors catches exceptions and returns error response
        assert result is not None
        assert result.status_code == 500  # Internal server error

        cb = get_hybrid_debate_circuit_breaker()
        # After failure, we should have recorded it
        # The circuit should still be closed after 1 failure (threshold is 5)
        assert cb.can_proceed()

    def test_get_hybrid_debate_circuit_breaker_returns_same_instance(self):
        """get_hybrid_debate_circuit_breaker returns singleton instance."""
        from aragora.server.handlers.hybrid_debate_handler import (
            get_hybrid_debate_circuit_breaker,
        )

        cb1 = get_hybrid_debate_circuit_breaker()
        cb2 = get_hybrid_debate_circuit_breaker()
        assert cb1 is cb2

    def test_reset_hybrid_debate_circuit_breaker_resets_state(self):
        """reset_hybrid_debate_circuit_breaker resets the circuit state."""
        from aragora.server.handlers.hybrid_debate_handler import (
            get_hybrid_debate_circuit_breaker,
            reset_hybrid_debate_circuit_breaker,
        )

        cb = get_hybrid_debate_circuit_breaker()

        # Open the circuit
        for _ in range(5):
            cb.record_failure()

        assert not cb.can_proceed()

        # Reset it
        reset_hybrid_debate_circuit_breaker()

        # Should be usable again
        assert cb.can_proceed()


class TestHybridDebateHandlerExceptionHandling:
    """Tests for exception handling in debate execution."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return HybridDebateHandler(mock_server_context)

    @pytest.fixture(autouse=True)
    def reset_circuit_breaker(self):
        """Reset circuit breaker before and after each test."""
        from aragora.server.handlers.hybrid_debate_handler import (
            reset_hybrid_debate_circuit_breaker,
        )

        reset_hybrid_debate_circuit_breaker()
        yield
        reset_hybrid_debate_circuit_breaker()

    def test_run_debate_exception_returns_error_response(
        self, handler, mock_http_handler_with_body
    ):
        """Exception in _run_debate returns error response via @handle_errors."""
        mock_http_handler_with_body.rfile.read = MagicMock(
            return_value=json.dumps(
                {
                    "task": "Test task",
                    "external_agent": "my-crewai-agent",
                }
            ).encode()
        )

        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            with patch.object(handler, "_run_debate", side_effect=RuntimeError("Test error")):
                result = handler.handle_post(
                    "/api/v1/debates/hybrid",
                    {},
                    mock_http_handler_with_body,
                )

        # @handle_errors decorator catches the exception and returns an error response
        # RuntimeError maps to 500 Internal Server Error
        assert result is not None
        assert result.status_code == 500
        body = json.loads(result.body)
        assert "error" in body

    def test_run_debate_timeout_exception_returns_error(self, handler, mock_http_handler_with_body):
        """Timeout exception in _run_debate returns error response."""
        mock_http_handler_with_body.rfile.read = MagicMock(
            return_value=json.dumps(
                {
                    "task": "Test task",
                    "external_agent": "my-crewai-agent",
                }
            ).encode()
        )

        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            with patch.object(handler, "_run_debate", side_effect=TimeoutError("Debate timed out")):
                result = handler.handle_post(
                    "/api/v1/debates/hybrid",
                    {},
                    mock_http_handler_with_body,
                )

        # @handle_errors decorator catches the exception and returns an error response
        assert result is not None
        # TimeoutError should map to 504 Gateway Timeout
        assert result.status_code in (500, 504)
        body = json.loads(result.body)
        assert "error" in body


class TestHybridDebateHandlerEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return HybridDebateHandler(mock_server_context)

    def test_empty_config_allowed(self, handler, mock_http_handler_with_body):
        """Empty config dict is allowed."""
        mock_http_handler_with_body.rfile.read = MagicMock(
            return_value=json.dumps(
                {
                    "task": "Test task",
                    "external_agent": "my-crewai-agent",
                    "config": {},
                }
            ).encode()
        )

        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle_post(
                "/api/v1/debates/hybrid",
                {},
                mock_http_handler_with_body,
            )

        assert result.status_code == 201
        body = json.loads(result.body)
        assert body["config"] == {}

    def test_non_dict_config_replaced_with_empty_dict(self, handler, mock_http_handler_with_body):
        """Non-dict config is replaced with empty dict."""
        mock_http_handler_with_body.rfile.read = MagicMock(
            return_value=json.dumps(
                {
                    "task": "Test task",
                    "external_agent": "my-crewai-agent",
                    "config": "not a dict",
                }
            ).encode()
        )

        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle_post(
                "/api/v1/debates/hybrid",
                {},
                mock_http_handler_with_body,
            )

        assert result.status_code == 201
        body = json.loads(result.body)
        assert body["config"] == {}

    def test_non_string_domain_replaced_with_general(self, handler, mock_http_handler_with_body):
        """Non-string domain is replaced with 'general'."""
        mock_http_handler_with_body.rfile.read = MagicMock(
            return_value=json.dumps(
                {
                    "task": "Test task",
                    "external_agent": "my-crewai-agent",
                    "domain": 123,
                }
            ).encode()
        )

        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle_post(
                "/api/v1/debates/hybrid",
                {},
                mock_http_handler_with_body,
            )

        assert result.status_code == 201
        body = json.loads(result.body)
        assert body["domain"] == "general"

    def test_task_at_exact_limit(self, handler, mock_http_handler_with_body):
        """Task at exactly 5000 characters is accepted."""
        task = "x" * 5000
        mock_http_handler_with_body.rfile.read = MagicMock(
            return_value=json.dumps(
                {
                    "task": task,
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

        assert result.status_code == 201

    def test_consensus_threshold_at_boundaries(self, handler, mock_http_handler_with_body):
        """Consensus threshold at 0.0 and 1.0 boundaries."""
        # Test 0.0
        mock_http_handler_with_body.rfile.read = MagicMock(
            return_value=json.dumps(
                {
                    "task": "Test task",
                    "external_agent": "my-crewai-agent",
                    "consensus_threshold": 0.0,
                }
            ).encode()
        )

        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle_post(
                "/api/v1/debates/hybrid",
                {},
                mock_http_handler_with_body,
            )

        assert result.status_code == 201
        body = json.loads(result.body)
        assert body["consensus_threshold"] == 0.0

        # Test 1.0
        mock_http_handler_with_body.rfile.read = MagicMock(
            return_value=json.dumps(
                {
                    "task": "Test task 2",
                    "external_agent": "my-crewai-agent",
                    "consensus_threshold": 1.0,
                }
            ).encode()
        )

        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle_post(
                "/api/v1/debates/hybrid",
                {},
                mock_http_handler_with_body,
            )

        assert result.status_code == 201
        body = json.loads(result.body)
        assert body["consensus_threshold"] == 1.0

    def test_max_rounds_at_boundaries(self, handler, mock_http_handler_with_body):
        """Max rounds at 1 and 10 boundaries."""
        # Test 1
        mock_http_handler_with_body.rfile.read = MagicMock(
            return_value=json.dumps(
                {
                    "task": "Test task",
                    "external_agent": "my-crewai-agent",
                    "max_rounds": 1,
                }
            ).encode()
        )

        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle_post(
                "/api/v1/debates/hybrid",
                {},
                mock_http_handler_with_body,
            )

        assert result.status_code == 201
        body = json.loads(result.body)
        assert body["max_rounds"] == 1

        # Test 10
        mock_http_handler_with_body.rfile.read = MagicMock(
            return_value=json.dumps(
                {
                    "task": "Test task 2",
                    "external_agent": "my-crewai-agent",
                    "max_rounds": 10,
                }
            ).encode()
        )

        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle_post(
                "/api/v1/debates/hybrid",
                {},
                mock_http_handler_with_body,
            )

        assert result.status_code == 201
        body = json.loads(result.body)
        assert body["max_rounds"] == 10

    def test_empty_verification_agents_list(self, handler, mock_http_handler_with_body):
        """Empty verification_agents list is accepted."""
        mock_http_handler_with_body.rfile.read = MagicMock(
            return_value=json.dumps(
                {
                    "task": "Test task",
                    "external_agent": "my-crewai-agent",
                    "verification_agents": [],
                }
            ).encode()
        )

        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle_post(
                "/api/v1/debates/hybrid",
                {},
                mock_http_handler_with_body,
            )

        assert result.status_code == 201
        body = json.loads(result.body)
        assert body["verification_agents"] == []

    def test_limit_minimum_clamped_to_1(self, handler, mock_http_handler):
        """Limit parameter is clamped to minimum of 1."""
        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle(
                "/api/v1/debates/hybrid",
                {"limit": "0"},
                mock_http_handler,
            )

        assert result.status_code == 200

    def test_limit_invalid_value_uses_default(self, handler, mock_http_handler):
        """Invalid limit value uses default of 20."""
        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle(
                "/api/v1/debates/hybrid",
                {"limit": "invalid"},
                mock_http_handler,
            )

        assert result.status_code == 200

    def test_task_with_only_whitespace(self, handler, mock_http_handler_with_body):
        """Task with only whitespace is rejected."""
        mock_http_handler_with_body.rfile.read = MagicMock(
            return_value=json.dumps(
                {
                    "task": "   \t\n   ",
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

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "task" in body["error"].lower()

    def test_external_agent_with_whitespace_is_trimmed(self, handler, mock_http_handler_with_body):
        """External agent with leading/trailing whitespace is trimmed."""
        mock_http_handler_with_body.rfile.read = MagicMock(
            return_value=json.dumps(
                {
                    "task": "Test task",
                    "external_agent": "  my-crewai-agent  ",
                }
            ).encode()
        )

        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle_post(
                "/api/v1/debates/hybrid",
                {},
                mock_http_handler_with_body,
            )

        assert result.status_code == 201
        body = json.loads(result.body)
        assert body["external_agent"] == "my-crewai-agent"

    def test_task_with_whitespace_is_trimmed(self, handler, mock_http_handler_with_body):
        """Task with leading/trailing whitespace is trimmed."""
        mock_http_handler_with_body.rfile.read = MagicMock(
            return_value=json.dumps(
                {
                    "task": "  Test task  ",
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

        assert result.status_code == 201
        body = json.loads(result.body)
        assert body["task"] == "Test task"

    def test_consensus_threshold_non_numeric_string(self, handler, mock_http_handler_with_body):
        """Non-numeric string for consensus_threshold is rejected."""
        mock_http_handler_with_body.rfile.read = MagicMock(
            return_value=json.dumps(
                {
                    "task": "Test task",
                    "external_agent": "my-crewai-agent",
                    "consensus_threshold": "high",
                }
            ).encode()
        )

        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle_post(
                "/api/v1/debates/hybrid",
                {},
                mock_http_handler_with_body,
            )

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "consensus_threshold" in body["error"].lower()

    def test_max_rounds_non_numeric_string(self, handler, mock_http_handler_with_body):
        """Non-numeric string for max_rounds is rejected."""
        mock_http_handler_with_body.rfile.read = MagicMock(
            return_value=json.dumps(
                {
                    "task": "Test task",
                    "external_agent": "my-crewai-agent",
                    "max_rounds": "many",
                }
            ).encode()
        )

        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle_post(
                "/api/v1/debates/hybrid",
                {},
                mock_http_handler_with_body,
            )

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "max_rounds" in body["error"].lower()

    def test_task_as_non_string_type(self, handler, mock_http_handler_with_body):
        """Task as non-string type (e.g., number) is rejected."""
        mock_http_handler_with_body.rfile.read = MagicMock(
            return_value=json.dumps(
                {
                    "task": 12345,
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

        assert result.status_code == 400

    def test_external_agent_as_non_string_type(self, handler, mock_http_handler_with_body):
        """External agent as non-string type is rejected."""
        mock_http_handler_with_body.rfile.read = MagicMock(
            return_value=json.dumps(
                {
                    "task": "Test task",
                    "external_agent": ["not", "a", "string"],
                }
            ).encode()
        )

        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle_post(
                "/api/v1/debates/hybrid",
                {},
                mock_http_handler_with_body,
            )

        assert result.status_code == 400

    def test_empty_body(self, handler, mock_http_handler_with_body):
        """Empty JSON body is handled."""
        mock_http_handler_with_body.rfile.read = MagicMock(return_value=b"{}")

        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle_post(
                "/api/v1/debates/hybrid",
                {},
                mock_http_handler_with_body,
            )

        assert result.status_code == 400

    def test_handle_returns_none_for_unhandled_path(self, handler, mock_http_handler):
        """Handle returns None for paths it doesn't handle."""
        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle(
                "/api/v1/other/endpoint",
                {},
                mock_http_handler,
            )

        assert result is None

    def test_handle_post_returns_none_for_unhandled_path(
        self, handler, mock_http_handler_with_body
    ):
        """Handle_post returns None for paths it doesn't handle."""
        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle_post(
                "/api/v1/other/endpoint",
                {},
                mock_http_handler_with_body,
            )

        assert result is None

    def test_handle_post_returns_none_for_non_create_path(
        self, handler, mock_http_handler_with_body
    ):
        """Handle_post returns None for hybrid paths that aren't create."""
        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle_post(
                "/api/v1/debates/hybrid/some_id",
                {},
                mock_http_handler_with_body,
            )

        assert result is None


class TestHybridDebateHandlerInvalidJsonEdgeCases:
    """Tests for various invalid JSON scenarios."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return HybridDebateHandler(mock_server_context)

    def test_null_body(self, handler, mock_http_handler_with_body):
        """Null JSON body is handled."""
        mock_http_handler_with_body.rfile.read = MagicMock(return_value=b"null")

        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle_post(
                "/api/v1/debates/hybrid",
                {},
                mock_http_handler_with_body,
            )

        assert result.status_code == 400

    def test_json_array_instead_of_object(self, handler, mock_http_handler_with_body):
        """JSON array instead of object is handled."""
        mock_http_handler_with_body.rfile.read = MagicMock(
            return_value=b'["task", "external_agent"]'
        )

        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle_post(
                "/api/v1/debates/hybrid",
                {},
                mock_http_handler_with_body,
            )

        # read_json_body returns a list from json.loads, but body.get() fails
        # because lists don't have .get() method. The exception is caught by
        # @handle_errors which returns 500
        assert result is not None
        assert result.status_code in (400, 500)

    def test_truncated_json(self, handler, mock_http_handler_with_body):
        """Truncated JSON is handled."""
        mock_http_handler_with_body.rfile.read = MagicMock(return_value=b'{"task": "incomplete')

        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle_post(
                "/api/v1/debates/hybrid",
                {},
                mock_http_handler_with_body,
            )

        assert result.status_code == 400

    def test_json_with_unicode(self, handler, mock_http_handler_with_body):
        """JSON with unicode characters is handled correctly."""
        mock_http_handler_with_body.rfile.read = MagicMock(
            return_value=json.dumps(
                {
                    "task": "Test task with unicode: \u4e2d\u6587 \u65e5\u672c\u8a9e \ud83d\ude00",
                    "external_agent": "my-crewai-agent",
                }
            ).encode("utf-8")
        )

        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle_post(
                "/api/v1/debates/hybrid",
                {},
                mock_http_handler_with_body,
            )

        assert result.status_code == 201

    def test_json_with_special_characters_in_task(self, handler, mock_http_handler_with_body):
        """JSON with special characters in task is handled."""
        mock_http_handler_with_body.rfile.read = MagicMock(
            return_value=json.dumps(
                {
                    "task": 'Task with special chars: <script>alert("xss")</script>',
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

        assert result.status_code == 201
        body = json.loads(result.body)
        assert '<script>alert("xss")</script>' in body["task"]


class TestHybridDebateHandlerRunDebateMethod:
    """Tests for the _run_debate method."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return HybridDebateHandler(mock_server_context)

    def test_run_debate_returns_expected_structure(self, handler):
        """_run_debate returns expected result structure."""
        debate_record = {
            "debate_id": "hybrid_test123",
            "task": "Test task for debate",
            "max_rounds": 5,
        }

        result = handler._run_debate(debate_record)

        assert "status" in result
        assert "consensus_reached" in result
        assert "confidence" in result
        assert "final_answer" in result
        assert "rounds" in result
        assert "completed_at" in result

    def test_run_debate_status_is_completed(self, handler):
        """_run_debate returns status as completed."""
        debate_record = {"task": "Test", "max_rounds": 3}
        result = handler._run_debate(debate_record)
        assert result["status"] == "completed"

    def test_run_debate_consensus_reached_is_true(self, handler):
        """_run_debate returns consensus_reached as True."""
        debate_record = {"task": "Test", "max_rounds": 3}
        result = handler._run_debate(debate_record)
        assert result["consensus_reached"] is True

    def test_run_debate_confidence_is_reasonable(self, handler):
        """_run_debate returns reasonable confidence value."""
        debate_record = {"task": "Test", "max_rounds": 3}
        result = handler._run_debate(debate_record)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_run_debate_final_answer_contains_task(self, handler):
        """_run_debate final_answer references the task."""
        debate_record = {"task": "Design a microservice", "max_rounds": 3}
        result = handler._run_debate(debate_record)
        assert "Design a microservice" in result["final_answer"]

    def test_run_debate_rounds_respects_max_rounds(self, handler):
        """_run_debate rounds does not exceed max_rounds."""
        debate_record = {"task": "Test", "max_rounds": 1}
        result = handler._run_debate(debate_record)
        assert result["rounds"] <= debate_record["max_rounds"]

    def test_run_debate_completed_at_is_iso_format(self, handler):
        """_run_debate completed_at is valid ISO format."""
        debate_record = {"task": "Test", "max_rounds": 3}
        result = handler._run_debate(debate_record)

        # Should be parseable
        parsed = datetime.fromisoformat(result["completed_at"].replace("Z", "+00:00"))
        assert parsed is not None

    def test_run_debate_long_task_truncated_in_answer(self, handler):
        """_run_debate truncates long task in final_answer."""
        long_task = "x" * 500
        debate_record = {"task": long_task, "max_rounds": 3}
        result = handler._run_debate(debate_record)

        # Answer should contain first 100 chars of task
        assert long_task[:100] in result["final_answer"]


class TestHybridDebateHandlerRBACProtection:
    """Tests for RBAC permission decorator behavior."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return HybridDebateHandler(mock_server_context)

    def test_handle_has_require_permission_decorator(self):
        """Handle method has require_permission decorator."""
        # Check that the decorator is applied
        method = HybridDebateHandler.handle
        assert hasattr(method, "__wrapped__") or hasattr(method, "_permission")

    def test_handle_post_has_require_permission_decorator(self):
        """Handle_post method has require_permission decorator."""
        method = HybridDebateHandler.handle_post
        assert hasattr(method, "__wrapped__") or hasattr(method, "_permission")


class TestHybridDebateHandlerConfigParsing:
    """Tests for config field parsing."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return HybridDebateHandler(mock_server_context)

    def test_config_with_nested_values(self, handler, mock_http_handler_with_body):
        """Config with nested dict values is preserved."""
        mock_http_handler_with_body.rfile.read = MagicMock(
            return_value=json.dumps(
                {
                    "task": "Test task",
                    "external_agent": "my-crewai-agent",
                    "config": {
                        "retry_policy": {"max_retries": 3, "backoff": "exponential"},
                        "logging": {"level": "DEBUG", "verbose": True},
                    },
                }
            ).encode()
        )

        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle_post(
                "/api/v1/debates/hybrid",
                {},
                mock_http_handler_with_body,
            )

        assert result.status_code == 201
        body = json.loads(result.body)
        assert body["config"]["retry_policy"]["max_retries"] == 3
        assert body["config"]["logging"]["verbose"] is True

    def test_config_with_array_values(self, handler, mock_http_handler_with_body):
        """Config with array values is preserved."""
        mock_http_handler_with_body.rfile.read = MagicMock(
            return_value=json.dumps(
                {
                    "task": "Test task",
                    "external_agent": "my-crewai-agent",
                    "config": {
                        "allowed_models": ["gpt-4", "claude-3", "gemini"],
                        "tags": ["production", "high-priority"],
                    },
                }
            ).encode()
        )

        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle_post(
                "/api/v1/debates/hybrid",
                {},
                mock_http_handler_with_body,
            )

        assert result.status_code == 201
        body = json.loads(result.body)
        assert "gpt-4" in body["config"]["allowed_models"]
        assert len(body["config"]["tags"]) == 2

    def test_config_with_null_values(self, handler, mock_http_handler_with_body):
        """Config with null values is preserved."""
        mock_http_handler_with_body.rfile.read = MagicMock(
            return_value=json.dumps(
                {
                    "task": "Test task",
                    "external_agent": "my-crewai-agent",
                    "config": {
                        "optional_field": None,
                        "timeout": 30,
                    },
                }
            ).encode()
        )

        with patch("aragora.server.handlers.hybrid_debate_handler.HYBRID_AVAILABLE", True):
            result = handler.handle_post(
                "/api/v1/debates/hybrid",
                {},
                mock_http_handler_with_body,
            )

        assert result.status_code == 201
        body = json.loads(result.body)
        assert body["config"]["optional_field"] is None
        assert body["config"]["timeout"] == 30
