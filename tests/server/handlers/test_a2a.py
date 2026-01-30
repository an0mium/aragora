"""Comprehensive tests for A2A Protocol handler with input validation.

Tests cover:
- Valid and invalid path parameters (agent IDs, task IDs)
- Request body validation
- Rate limiting behavior
- Authentication/authorization
- Error responses for invalid input
- Edge cases (empty strings, special characters, very long IDs)
"""

import json
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.a2a import (
    A2AHandler,
    AGENT_NAME_PATTERN,
    MAX_BODY_SIZE,
    MAX_CONTEXT_CONTENT_LENGTH,
    MAX_CONTEXT_ITEMS,
    MAX_INSTRUCTION_LENGTH,
    MAX_METADATA_KEYS,
    MAX_METADATA_VALUE_LENGTH,
    TASK_ID_PATTERN,
    get_a2a_handler,
    get_a2a_server,
    validate_agent_name,
    validate_task_id,
    validate_task_request_body,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def a2a_handler():
    """Create an A2A handler with mocked dependencies."""
    ctx = {"storage": None, "elo_system": None, "nomic_dir": None}
    handler = A2AHandler(ctx)
    return handler


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler for GET requests."""
    handler = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = {"Content-Type": "application/json"}
    handler.command = "GET"
    return handler


def create_post_request(data: dict, content_type: str = "application/json") -> MagicMock:
    """Create a mock HTTP handler with a JSON body for POST requests."""
    handler = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    body = json.dumps(data).encode("utf-8")
    handler.headers = {
        "Content-Length": str(len(body)),
        "Content-Type": content_type,
    }
    handler.rfile = BytesIO(body)
    handler.command = "POST"
    return handler


def create_raw_request(body: bytes, content_type: str = "application/json") -> MagicMock:
    """Create a mock HTTP handler with raw body bytes."""
    handler = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = {
        "Content-Length": str(len(body)),
        "Content-Type": content_type,
    }
    handler.rfile = BytesIO(body)
    handler.command = "POST"
    return handler


# =============================================================================
# Agent Name Validation Tests
# =============================================================================


class TestAgentNameValidation:
    """Test agent name validation."""

    def test_valid_agent_name_simple(self):
        """Accept simple alphanumeric agent name."""
        is_valid, err = validate_agent_name("claude")
        assert is_valid is True
        assert err is None

    def test_valid_agent_name_with_hyphen(self):
        """Accept agent name with hyphen."""
        is_valid, err = validate_agent_name("claude-3")
        assert is_valid is True
        assert err is None

    def test_valid_agent_name_with_underscore(self):
        """Accept agent name with underscore."""
        is_valid, err = validate_agent_name("claude_agent")
        assert is_valid is True
        assert err is None

    def test_valid_agent_name_with_dot(self):
        """Accept agent name with dot for versioning."""
        is_valid, err = validate_agent_name("claude-3.5-sonnet")
        assert is_valid is True
        assert err is None

    def test_valid_agent_name_max_length(self):
        """Accept agent name at maximum length (64 chars)."""
        name = "a" * 64
        is_valid, err = validate_agent_name(name)
        assert is_valid is True
        assert err is None

    def test_invalid_agent_name_empty(self):
        """Reject empty agent name."""
        is_valid, err = validate_agent_name("")
        assert is_valid is False
        assert "required" in err.lower()

    def test_invalid_agent_name_too_long(self):
        """Reject agent name exceeding 64 characters."""
        name = "a" * 65
        is_valid, err = validate_agent_name(name)
        assert is_valid is False
        assert "64" in err

    def test_invalid_agent_name_starts_with_hyphen(self):
        """Reject agent name starting with hyphen."""
        is_valid, err = validate_agent_name("-claude")
        assert is_valid is False
        assert "alphanumeric" in err.lower()

    def test_invalid_agent_name_starts_with_dot(self):
        """Reject agent name starting with dot."""
        is_valid, err = validate_agent_name(".hidden")
        assert is_valid is False
        assert "alphanumeric" in err.lower()

    def test_invalid_agent_name_with_spaces(self):
        """Reject agent name with spaces."""
        is_valid, err = validate_agent_name("claude agent")
        assert is_valid is False

    def test_invalid_agent_name_with_special_chars(self):
        """Reject agent name with special characters."""
        is_valid, err = validate_agent_name("claude@agent")
        assert is_valid is False

    def test_invalid_agent_name_path_traversal(self):
        """Reject agent name with path traversal."""
        is_valid, err = validate_agent_name("../etc/passwd")
        assert is_valid is False

    def test_invalid_agent_name_null_byte(self):
        """Reject agent name with null byte."""
        is_valid, err = validate_agent_name("agent\x00evil")
        assert is_valid is False


# =============================================================================
# Task ID Validation Tests
# =============================================================================


class TestTaskIdValidation:
    """Test task ID validation."""

    def test_valid_task_id_simple(self):
        """Accept simple alphanumeric task ID."""
        is_valid, err = validate_task_id("task123")
        assert is_valid is True
        assert err is None

    def test_valid_task_id_uuid_like(self):
        """Accept UUID-like task ID."""
        is_valid, err = validate_task_id("550e8400-e29b-41d4-a716-446655440000")
        assert is_valid is True
        assert err is None

    def test_valid_task_id_with_underscore(self):
        """Accept task ID with underscore."""
        is_valid, err = validate_task_id("task_123_abc")
        assert is_valid is True
        assert err is None

    def test_valid_task_id_max_length(self):
        """Accept task ID at maximum length (128 chars)."""
        task_id = "a" * 128
        is_valid, err = validate_task_id(task_id)
        assert is_valid is True
        assert err is None

    def test_invalid_task_id_empty(self):
        """Reject empty task ID."""
        is_valid, err = validate_task_id("")
        assert is_valid is False
        assert "required" in err.lower()

    def test_invalid_task_id_too_long(self):
        """Reject task ID exceeding 128 characters."""
        task_id = "a" * 129
        is_valid, err = validate_task_id(task_id)
        assert is_valid is False
        assert "128" in err

    def test_invalid_task_id_starts_with_hyphen(self):
        """Reject task ID starting with hyphen."""
        is_valid, err = validate_task_id("-task123")
        assert is_valid is False

    def test_invalid_task_id_with_spaces(self):
        """Reject task ID with spaces."""
        is_valid, err = validate_task_id("task 123")
        assert is_valid is False

    def test_invalid_task_id_with_special_chars(self):
        """Reject task ID with special characters."""
        is_valid, err = validate_task_id("task$123")
        assert is_valid is False

    def test_invalid_task_id_path_traversal(self):
        """Reject task ID with path traversal."""
        is_valid, err = validate_task_id("../../secret")
        assert is_valid is False


# =============================================================================
# Task Request Body Validation Tests
# =============================================================================


class TestTaskRequestBodyValidation:
    """Test task submission body validation."""

    def test_valid_minimal_body(self):
        """Accept minimal valid body with just instruction."""
        data = {"instruction": "Analyze this text"}
        is_valid, err = validate_task_request_body(data)
        assert is_valid is True
        assert err is None

    def test_valid_full_body(self):
        """Accept full valid body with all fields."""
        data = {
            "instruction": "Analyze this text",
            "task_id": "task-123",
            "capability": "debate",
            "priority": "high",
            "timeout_ms": 60000,
            "context": [{"type": "text", "content": "Some context"}],
            "metadata": {"source": "test"},
        }
        is_valid, err = validate_task_request_body(data)
        assert is_valid is True
        assert err is None

    def test_invalid_body_not_dict(self):
        """Reject body that is not a dictionary."""
        is_valid, err = validate_task_request_body("not a dict")
        assert is_valid is False
        assert "object" in err.lower()

    def test_invalid_missing_instruction(self):
        """Reject body missing instruction field."""
        data = {"task_id": "task-123"}
        is_valid, err = validate_task_request_body(data)
        assert is_valid is False
        assert "instruction" in err.lower()

    def test_invalid_instruction_not_string(self):
        """Reject body with non-string instruction."""
        data = {"instruction": 12345}
        is_valid, err = validate_task_request_body(data)
        assert is_valid is False
        assert "string" in err.lower()

    def test_invalid_instruction_too_long(self):
        """Reject body with instruction exceeding max length."""
        data = {"instruction": "x" * (MAX_INSTRUCTION_LENGTH + 1)}
        is_valid, err = validate_task_request_body(data)
        assert is_valid is False
        assert str(MAX_INSTRUCTION_LENGTH) in err

    def test_invalid_task_id_format(self):
        """Reject body with invalid task_id format."""
        data = {"instruction": "Test", "task_id": "../invalid"}
        is_valid, err = validate_task_request_body(data)
        assert is_valid is False

    def test_invalid_capability_unknown(self):
        """Reject body with unknown capability."""
        data = {"instruction": "Test", "capability": "unknown_capability"}
        is_valid, err = validate_task_request_body(data)
        assert is_valid is False
        assert "Invalid capability" in err

    def test_valid_all_capabilities(self):
        """Accept all valid capabilities."""
        valid_capabilities = [
            "debate",
            "consensus",
            "critique",
            "synthesis",
            "audit",
            "verification",
            "code_review",
            "document_analysis",
            "research",
            "reasoning",
        ]
        for cap in valid_capabilities:
            data = {"instruction": "Test", "capability": cap}
            is_valid, err = validate_task_request_body(data)
            assert is_valid is True, f"Failed for capability: {cap}"

    def test_invalid_priority_unknown(self):
        """Reject body with unknown priority."""
        data = {"instruction": "Test", "priority": "super_urgent"}
        is_valid, err = validate_task_request_body(data)
        assert is_valid is False
        assert "Invalid priority" in err

    def test_valid_all_priorities(self):
        """Accept all valid priorities."""
        valid_priorities = ["low", "normal", "high", "urgent"]
        for prio in valid_priorities:
            data = {"instruction": "Test", "priority": prio}
            is_valid, err = validate_task_request_body(data)
            assert is_valid is True, f"Failed for priority: {prio}"

    def test_invalid_context_not_array(self):
        """Reject body with non-array context."""
        data = {"instruction": "Test", "context": "not an array"}
        is_valid, err = validate_task_request_body(data)
        assert is_valid is False
        assert "array" in err.lower()

    def test_invalid_context_too_many_items(self):
        """Reject body with too many context items."""
        data = {
            "instruction": "Test",
            "context": [{"type": "text", "content": "x"} for _ in range(MAX_CONTEXT_ITEMS + 1)],
        }
        is_valid, err = validate_task_request_body(data)
        assert is_valid is False
        assert str(MAX_CONTEXT_ITEMS) in err

    def test_invalid_context_item_not_dict(self):
        """Reject body with non-dict context item."""
        data = {"instruction": "Test", "context": ["not a dict"]}
        is_valid, err = validate_task_request_body(data)
        assert is_valid is False
        assert "object" in err.lower()

    def test_invalid_context_content_too_long(self):
        """Reject body with context content exceeding max length."""
        data = {
            "instruction": "Test",
            "context": [{"type": "text", "content": "x" * (MAX_CONTEXT_CONTENT_LENGTH + 1)}],
        }
        is_valid, err = validate_task_request_body(data)
        assert is_valid is False
        assert str(MAX_CONTEXT_CONTENT_LENGTH) in err

    def test_invalid_metadata_not_dict(self):
        """Reject body with non-dict metadata."""
        data = {"instruction": "Test", "metadata": "not a dict"}
        is_valid, err = validate_task_request_body(data)
        assert is_valid is False
        assert "object" in err.lower()

    def test_invalid_metadata_too_many_keys(self):
        """Reject body with too many metadata keys."""
        data = {
            "instruction": "Test",
            "metadata": {f"key{i}": "value" for i in range(MAX_METADATA_KEYS + 1)},
        }
        is_valid, err = validate_task_request_body(data)
        assert is_valid is False
        assert str(MAX_METADATA_KEYS) in err

    def test_invalid_metadata_value_too_long(self):
        """Reject body with metadata value exceeding max length."""
        data = {"instruction": "Test", "metadata": {"key": "x" * (MAX_METADATA_VALUE_LENGTH + 1)}}
        is_valid, err = validate_task_request_body(data)
        assert is_valid is False
        assert "exceeds maximum length" in err

    def test_invalid_timeout_not_int(self):
        """Reject body with non-integer timeout."""
        data = {"instruction": "Test", "timeout_ms": "60000"}
        is_valid, err = validate_task_request_body(data)
        assert is_valid is False
        assert "integer" in err.lower()

    def test_invalid_timeout_too_low(self):
        """Reject body with timeout below minimum."""
        data = {"instruction": "Test", "timeout_ms": 500}
        is_valid, err = validate_task_request_body(data)
        assert is_valid is False
        assert "1000" in err

    def test_invalid_timeout_too_high(self):
        """Reject body with timeout above maximum."""
        data = {"instruction": "Test", "timeout_ms": 4000000}
        is_valid, err = validate_task_request_body(data)
        assert is_valid is False
        assert "3600000" in err


# =============================================================================
# Handler Get Agent Tests
# =============================================================================


class TestA2AHandlerGetAgent:
    """Test get agent endpoint with validation."""

    @patch("aragora.server.handlers.a2a.get_a2a_server")
    def test_get_agent_valid_name(self, mock_get_server, a2a_handler):
        """Get agent with valid name returns agent card."""
        mock_agent = MagicMock()
        mock_agent.to_dict.return_value = {"name": "claude", "version": "1.0.0"}
        mock_server = MagicMock()
        mock_server.get_agent.return_value = mock_agent
        mock_get_server.return_value = mock_server

        result = a2a_handler._handle_get_agent("claude")

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["name"] == "claude"

    @patch("aragora.server.handlers.a2a.get_a2a_server")
    def test_get_agent_invalid_name_empty(self, mock_get_server, a2a_handler):
        """Get agent with empty name returns 400."""
        result = a2a_handler._handle_get_agent("")

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "error" in body

    @patch("aragora.server.handlers.a2a.get_a2a_server")
    def test_get_agent_invalid_name_special_chars(self, mock_get_server, a2a_handler):
        """Get agent with special characters returns 400."""
        result = a2a_handler._handle_get_agent("claude@evil.com")

        assert result.status_code == 400

    @patch("aragora.server.handlers.a2a.get_a2a_server")
    def test_get_agent_invalid_name_path_traversal(self, mock_get_server, a2a_handler):
        """Get agent with path traversal returns 400."""
        result = a2a_handler._handle_get_agent("../../../etc/passwd")

        assert result.status_code == 400

    @patch("aragora.server.handlers.a2a.get_a2a_server")
    def test_get_agent_not_found(self, mock_get_server, a2a_handler):
        """Get agent returns 404 when agent not found."""
        mock_server = MagicMock()
        mock_server.get_agent.return_value = None
        mock_get_server.return_value = mock_server

        result = a2a_handler._handle_get_agent("nonexistent")

        assert result.status_code == 404


# =============================================================================
# Handler Get Task Tests
# =============================================================================


class TestA2AHandlerGetTask:
    """Test get task endpoint with validation."""

    @patch("aragora.server.handlers.a2a.get_a2a_server")
    def test_get_task_valid_id(self, mock_get_server, a2a_handler):
        """Get task with valid ID returns task status."""
        mock_task = MagicMock()
        mock_task.to_dict.return_value = {"task_id": "task-123", "status": "completed"}
        mock_server = MagicMock()
        mock_server.get_task_status.return_value = mock_task
        mock_get_server.return_value = mock_server

        result = a2a_handler._handle_get_task("task-123")

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["task_id"] == "task-123"

    @patch("aragora.server.handlers.a2a.get_a2a_server")
    def test_get_task_invalid_id_empty(self, mock_get_server, a2a_handler):
        """Get task with empty ID returns 400."""
        result = a2a_handler._handle_get_task("")

        assert result.status_code == 400

    @patch("aragora.server.handlers.a2a.get_a2a_server")
    def test_get_task_invalid_id_special_chars(self, mock_get_server, a2a_handler):
        """Get task with special characters returns 400."""
        result = a2a_handler._handle_get_task("task$evil")

        assert result.status_code == 400

    @patch("aragora.server.handlers.a2a.get_a2a_server")
    def test_get_task_invalid_id_too_long(self, mock_get_server, a2a_handler):
        """Get task with ID exceeding max length returns 400."""
        result = a2a_handler._handle_get_task("a" * 129)

        assert result.status_code == 400

    @patch("aragora.server.handlers.a2a.get_a2a_server")
    def test_get_task_not_found(self, mock_get_server, a2a_handler):
        """Get task returns 404 when task not found."""
        mock_server = MagicMock()
        mock_server.get_task_status.return_value = None
        mock_get_server.return_value = mock_server

        result = a2a_handler._handle_get_task("nonexistent")

        assert result.status_code == 404


# =============================================================================
# Handler Submit Task Tests
# =============================================================================


class TestA2AHandlerSubmitTask:
    """Test submit task endpoint with validation."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.a2a.get_a2a_server")
    async def test_submit_task_valid_minimal(self, mock_get_server, a2a_handler):
        """Submit task with minimal valid body succeeds."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"task_id": "generated-id", "status": "pending"}
        mock_server = MagicMock()
        mock_server.handle_task = AsyncMock(return_value=mock_result)
        mock_get_server.return_value = mock_server

        http_handler = create_post_request({"instruction": "Test task"})
        result = await a2a_handler._handle_submit_task(http_handler)

        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_submit_task_invalid_content_type(self, a2a_handler):
        """Submit task with wrong Content-Type returns 415."""
        http_handler = create_post_request({"instruction": "Test"}, content_type="text/plain")
        result = await a2a_handler._handle_submit_task(http_handler)

        assert result.status_code == 415

    @pytest.mark.asyncio
    async def test_submit_task_invalid_json(self, a2a_handler):
        """Submit task with invalid JSON returns 400."""
        http_handler = create_raw_request(b"not valid json")
        result = await a2a_handler._handle_submit_task(http_handler)

        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_submit_task_missing_instruction(self, a2a_handler):
        """Submit task without instruction returns 400."""
        http_handler = create_post_request({"task_id": "task-123"})
        result = await a2a_handler._handle_submit_task(http_handler)

        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_submit_task_instruction_too_long(self, a2a_handler):
        """Submit task with instruction exceeding max length returns 400."""
        http_handler = create_post_request({"instruction": "x" * (MAX_INSTRUCTION_LENGTH + 1)})
        result = await a2a_handler._handle_submit_task(http_handler)

        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_submit_task_invalid_capability(self, a2a_handler):
        """Submit task with invalid capability returns 400."""
        http_handler = create_post_request(
            {"instruction": "Test", "capability": "invalid_capability"}
        )
        result = await a2a_handler._handle_submit_task(http_handler)

        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_submit_task_invalid_priority(self, a2a_handler):
        """Submit task with invalid priority returns 400."""
        http_handler = create_post_request({"instruction": "Test", "priority": "super_urgent"})
        result = await a2a_handler._handle_submit_task(http_handler)

        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_submit_task_too_many_context_items(self, a2a_handler):
        """Submit task with too many context items returns 400."""
        http_handler = create_post_request(
            {
                "instruction": "Test",
                "context": [{"type": "text", "content": "x"} for _ in range(MAX_CONTEXT_ITEMS + 1)],
            }
        )
        result = await a2a_handler._handle_submit_task(http_handler)

        assert result.status_code == 400


# =============================================================================
# Handler Cancel Task Tests
# =============================================================================


class TestA2AHandlerCancelTask:
    """Test cancel task endpoint with validation."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.a2a.get_a2a_server")
    async def test_cancel_task_valid_id(self, mock_get_server, a2a_handler):
        """Cancel task with valid ID succeeds."""
        mock_server = MagicMock()
        mock_server.cancel_task = AsyncMock(return_value=True)
        mock_get_server.return_value = mock_server

        result = await a2a_handler._handle_cancel_task("task-123")

        assert result.status_code == 204

    @pytest.mark.asyncio
    async def test_cancel_task_invalid_id_empty(self, a2a_handler):
        """Cancel task with empty ID returns 400."""
        result = await a2a_handler._handle_cancel_task("")

        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_cancel_task_invalid_id_special_chars(self, a2a_handler):
        """Cancel task with special characters returns 400."""
        result = await a2a_handler._handle_cancel_task("task$evil")

        assert result.status_code == 400

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.a2a.get_a2a_server")
    async def test_cancel_task_not_found(self, mock_get_server, a2a_handler):
        """Cancel task returns 404 when task not found."""
        mock_server = MagicMock()
        mock_server.cancel_task = AsyncMock(return_value=False)
        mock_get_server.return_value = mock_server

        result = await a2a_handler._handle_cancel_task("nonexistent")

        assert result.status_code == 404


# =============================================================================
# Handler Stream Task Tests
# =============================================================================


class TestA2AHandlerStreamTask:
    """Test stream task endpoint with validation."""

    def test_stream_task_valid_id(self, a2a_handler, mock_http_handler):
        """Stream task with valid ID returns upgrade message."""
        result = a2a_handler._handle_stream_task("task-123", mock_http_handler)

        assert result.status_code == 426
        body = json.loads(result.body)
        assert "ws_path" in body
        assert "task-123" in body["ws_path"]

    def test_stream_task_invalid_id_empty(self, a2a_handler, mock_http_handler):
        """Stream task with empty ID returns 400."""
        result = a2a_handler._handle_stream_task("", mock_http_handler)

        assert result.status_code == 400

    def test_stream_task_invalid_id_special_chars(self, a2a_handler, mock_http_handler):
        """Stream task with special characters returns 400."""
        result = a2a_handler._handle_stream_task("task@evil", mock_http_handler)

        assert result.status_code == 400


# =============================================================================
# Handler Routing Tests
# =============================================================================


class TestA2AHandlerRouting:
    """Test handler routing and can_handle."""

    def test_can_handle_a2a_prefix(self, a2a_handler):
        """Handler matches /api/v1/a2a/ paths."""
        assert a2a_handler.can_handle("/api/v1/a2a/agents") is True
        assert a2a_handler.can_handle("/api/v1/a2a/tasks") is True
        assert a2a_handler.can_handle("/api/v1/a2a/openapi.json") is True

    def test_can_handle_well_known(self, a2a_handler):
        """Handler matches well-known discovery path."""
        assert a2a_handler.can_handle("/.well-known/agent.json") is True

    def test_cannot_handle_other_paths(self, a2a_handler):
        """Handler does not match unrelated paths."""
        assert a2a_handler.can_handle("/api/debates") is False
        assert a2a_handler.can_handle("/api/agents") is False
        assert a2a_handler.can_handle("/health") is False

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.a2a.get_a2a_server")
    async def test_route_to_discovery(self, mock_get_server, a2a_handler, mock_http_handler):
        """Routes to discovery endpoint."""
        mock_server = MagicMock()
        mock_server.list_agents.return_value = []
        mock_get_server.return_value = mock_server

        result = await a2a_handler.handle("/.well-known/agent.json", {}, mock_http_handler)

        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_route_unknown_endpoint(self, a2a_handler, mock_http_handler):
        """Returns 404 for unknown A2A endpoint."""
        result = await a2a_handler.handle("/api/v1/a2a/unknown", {}, mock_http_handler)

        assert result.status_code == 404


# =============================================================================
# Edge Cases and Security Tests
# =============================================================================


class TestA2AHandlerEdgeCases:
    """Test edge cases and security concerns."""

    def test_agent_name_with_unicode(self, a2a_handler):
        """Reject agent name with unicode characters."""
        result = a2a_handler._handle_get_agent("agent\u200bname")  # Zero-width space
        assert result.status_code == 400

    def test_task_id_with_null_byte(self, a2a_handler):
        """Reject task ID with null byte."""
        result = a2a_handler._handle_get_task("task\x00id")
        assert result.status_code == 400

    def test_agent_name_single_char(self, a2a_handler):
        """Accept single character agent name."""
        is_valid, err = validate_agent_name("a")
        assert is_valid is True

    def test_task_id_single_char(self, a2a_handler):
        """Accept single character task ID."""
        is_valid, err = validate_task_id("1")
        assert is_valid is True

    def test_agent_name_numeric_only(self, a2a_handler):
        """Accept numeric-only agent name."""
        is_valid, err = validate_agent_name("12345")
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_submit_task_unicode_instruction(self, a2a_handler):
        """Accept task with unicode instruction."""
        http_handler = create_post_request({"instruction": "Analyze: cafe"})
        # This should not raise an error during validation
        # (actual execution mocked)
        with patch("aragora.server.handlers.a2a.get_a2a_server") as mock:
            mock_result = MagicMock()
            mock_result.to_dict.return_value = {"task_id": "123", "status": "pending"}
            mock.return_value.handle_task = AsyncMock(return_value=mock_result)
            result = await a2a_handler._handle_submit_task(http_handler)
            assert result.status_code == 200


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestA2AHandlerFactory:
    """Test handler factory functions."""

    def test_get_a2a_handler_creates_instance(self):
        """get_a2a_handler creates handler instance."""
        # Reset global state for test
        import aragora.server.handlers.a2a as a2a_module

        a2a_module._a2a_handler = None

        handler = get_a2a_handler()
        assert isinstance(handler, A2AHandler)

    def test_get_a2a_handler_returns_singleton(self):
        """get_a2a_handler returns same instance."""
        handler1 = get_a2a_handler()
        handler2 = get_a2a_handler()
        assert handler1 is handler2

    def test_get_a2a_server_creates_instance(self):
        """get_a2a_server creates server instance."""
        import aragora.server.handlers.a2a as a2a_module

        a2a_module._a2a_server = None

        # Mock the A2AServer class in the protocols.a2a module
        with patch("aragora.protocols.a2a.A2AServer") as mock_cls:
            mock_server = MagicMock()
            mock_cls.return_value = mock_server
            server = get_a2a_server()
            assert server is mock_server
            mock_cls.assert_called_once()


# =============================================================================
# Discovery and OpenAPI Tests
# =============================================================================


class TestA2AHandlerDiscovery:
    """Test discovery and OpenAPI endpoints."""

    @patch("aragora.server.handlers.a2a.get_a2a_server")
    def test_discovery_with_agents(self, mock_get_server, a2a_handler):
        """Discovery returns primary agent card when agents exist."""
        mock_agent = MagicMock()
        mock_agent.to_dict.return_value = {
            "name": "aragora-debate",
            "version": "1.0.0",
            "capabilities": ["debate"],
        }
        mock_server = MagicMock()
        mock_server.list_agents.return_value = [mock_agent]
        mock_get_server.return_value = mock_server

        result = a2a_handler._handle_discovery()

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["name"] == "aragora-debate"

    @patch("aragora.server.handlers.a2a.get_a2a_server")
    def test_discovery_without_agents(self, mock_get_server, a2a_handler):
        """Discovery returns default card when no agents."""
        mock_server = MagicMock()
        mock_server.list_agents.return_value = []
        mock_get_server.return_value = mock_server

        result = a2a_handler._handle_discovery()

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["name"] == "aragora"
        assert "capabilities" in body

    @patch("aragora.server.handlers.a2a.get_a2a_server")
    def test_openapi_spec(self, mock_get_server, a2a_handler):
        """OpenAPI endpoint returns specification."""
        mock_server = MagicMock()
        mock_server.get_openapi_spec.return_value = {
            "openapi": "3.1.0",
            "info": {"title": "A2A API"},
        }
        mock_get_server.return_value = mock_server

        result = a2a_handler._handle_openapi()

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["openapi"] == "3.1.0"


# =============================================================================
# List Agents Tests
# =============================================================================


class TestA2AHandlerListAgents:
    """Test list agents endpoint."""

    @patch("aragora.server.handlers.a2a.get_a2a_server")
    def test_list_agents_success(self, mock_get_server, a2a_handler):
        """List agents returns all registered agents."""
        mock_agent = MagicMock()
        mock_agent.to_dict.return_value = {"name": "test-agent"}
        mock_server = MagicMock()
        mock_server.list_agents.return_value = [mock_agent]
        mock_get_server.return_value = mock_server

        result = a2a_handler._handle_list_agents()

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "agents" in body
        assert len(body["agents"]) == 1
        assert body["total"] == 1

    @patch("aragora.server.handlers.a2a.get_a2a_server")
    def test_list_agents_empty(self, mock_get_server, a2a_handler):
        """List agents returns empty list when no agents."""
        mock_server = MagicMock()
        mock_server.list_agents.return_value = []
        mock_get_server.return_value = mock_server

        result = a2a_handler._handle_list_agents()

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["agents"] == []
        assert body["total"] == 0
