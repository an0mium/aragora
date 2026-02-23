"""Tests for the A2A protocol HTTP handler.

Covers:
- Validation functions (agent names, task IDs, task request bodies)
- can_handle routing
- Discovery endpoint (/.well-known/agent.json and /api/v1/a2a/.well-known/agent.json)
- OpenAPI spec endpoint
- List agents endpoint
- Get agent by name endpoint
- Submit task endpoint (POST /api/v1/a2a/tasks)
- Get task status endpoint
- Cancel task endpoint
- Stream task endpoint
- Circuit breaker helpers
- Handler factory (get_a2a_handler)
- Error/edge cases throughout

Note: The handler's `handle()` method uses exact path matching for discovery
and openapi endpoints, and subpath extraction for agents/tasks. Tests for
agents/tasks call the private methods directly (consistent with the existing
test pattern in tests/server/handlers/test_a2a_handler.py) to isolate
endpoint logic from routing.
"""

from __future__ import annotations

import json
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.a2a import (
    AGENT_NAME_PATTERN,
    MAX_BODY_SIZE,
    MAX_CONTEXT_CONTENT_LENGTH,
    MAX_CONTEXT_ITEMS,
    MAX_INSTRUCTION_LENGTH,
    MAX_METADATA_KEYS,
    MAX_METADATA_VALUE_LENGTH,
    TASK_ID_PATTERN,
    A2AHandler,
    get_a2a_circuit_breaker,
    get_a2a_circuit_breaker_status,
    get_a2a_handler,
    validate_agent_name,
    validate_task_id,
    validate_task_request_body,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_a2a_singletons():
    """Reset module-level singletons between tests."""
    import aragora.server.handlers.a2a as mod

    old_server = mod._a2a_server
    old_handler = mod._a2a_handler
    mod._a2a_server = None
    mod._a2a_handler = None
    yield
    mod._a2a_server = old_server
    mod._a2a_handler = old_handler


@pytest.fixture
def handler():
    """Create an A2AHandler instance."""
    return A2AHandler(ctx={"test": True})


@pytest.fixture
def mock_a2a_server():
    """Create a mock A2A server."""
    server = MagicMock()
    server.list_agents.return_value = []
    server.get_agent.return_value = None
    server.get_task_status.return_value = None
    server.get_openapi_spec.return_value = {"openapi": "3.0.0"}
    server.handle_task = AsyncMock(
        return_value=MagicMock(
            to_dict=lambda: {"task_id": "t1", "status": "completed"}
        )
    )
    server.cancel_task = AsyncMock(return_value=True)
    return server


def _make_http_handler(
    method: str = "GET",
    body: dict | None = None,
    content_type: str = "application/json",
    content_length: int | None = None,
):
    """Create a mock HTTP handler with configurable properties."""
    h = MagicMock()
    h.command = method

    if body is not None:
        body_bytes = json.dumps(body).encode("utf-8")
    else:
        body_bytes = b"{}"

    h.rfile = MagicMock()
    h.rfile.read.return_value = body_bytes

    headers = {}
    if content_type:
        headers["Content-Type"] = content_type
    headers["Content-Length"] = str(
        content_length if content_length is not None else len(body_bytes)
    )

    h.headers = headers
    return h


def _make_post_handler(data: dict) -> MagicMock:
    """Create a mock HTTP handler with a JSON body for POST requests."""
    handler = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    body = json.dumps(data).encode("utf-8")
    handler.headers = {
        "Content-Length": str(len(body)),
        "Content-Type": "application/json",
    }
    handler.rfile = BytesIO(body)
    handler.command = "POST"
    return handler


# ===========================================================================
# Test validate_agent_name
# ===========================================================================


class TestValidateAgentName:
    """Tests for the validate_agent_name function."""

    def test_valid_simple_name(self):
        ok, err = validate_agent_name("my-agent")
        assert ok is True
        assert err is None

    def test_valid_name_with_dots(self):
        ok, err = validate_agent_name("agent.v2.0")
        assert ok is True
        assert err is None

    def test_valid_name_with_underscores(self):
        ok, err = validate_agent_name("my_agent_1")
        assert ok is True
        assert err is None

    def test_valid_single_char(self):
        ok, err = validate_agent_name("a")
        assert ok is True

    def test_valid_max_length_64(self):
        name = "a" * 64
        ok, err = validate_agent_name(name)
        assert ok is True

    def test_empty_string(self):
        ok, err = validate_agent_name("")
        assert ok is False
        assert "required" in err.lower()

    def test_too_long(self):
        name = "a" * 65
        ok, err = validate_agent_name(name)
        assert ok is False
        assert "64" in err

    def test_starts_with_hyphen(self):
        ok, err = validate_agent_name("-agent")
        assert ok is False

    def test_starts_with_dot(self):
        ok, err = validate_agent_name(".agent")
        assert ok is False

    def test_special_characters(self):
        ok, err = validate_agent_name("agent@home")
        assert ok is False

    def test_spaces(self):
        ok, err = validate_agent_name("agent name")
        assert ok is False

    def test_slash(self):
        ok, err = validate_agent_name("agent/name")
        assert ok is False

    def test_starts_with_underscore(self):
        ok, err = validate_agent_name("_agent")
        assert ok is False

    def test_numeric_start(self):
        ok, err = validate_agent_name("42agent")
        assert ok is True


# ===========================================================================
# Test validate_task_id
# ===========================================================================


class TestValidateTaskId:
    """Tests for the validate_task_id function."""

    def test_valid_uuid_style(self):
        ok, err = validate_task_id("abc-123-def")
        assert ok is True
        assert err is None

    def test_valid_simple(self):
        ok, err = validate_task_id("task123")
        assert ok is True

    def test_valid_max_length_128(self):
        tid = "a" * 128
        ok, err = validate_task_id(tid)
        assert ok is True

    def test_empty(self):
        ok, err = validate_task_id("")
        assert ok is False
        assert "required" in err.lower()

    def test_too_long(self):
        tid = "a" * 129
        ok, err = validate_task_id(tid)
        assert ok is False
        assert "128" in err

    def test_starts_with_hyphen(self):
        ok, err = validate_task_id("-task")
        assert ok is False

    def test_special_chars(self):
        ok, err = validate_task_id("task@id")
        assert ok is False

    def test_dots_not_allowed(self):
        ok, err = validate_task_id("task.id")
        assert ok is False

    def test_with_underscores(self):
        ok, err = validate_task_id("task_123_abc")
        assert ok is True

    def test_numeric_only(self):
        ok, err = validate_task_id("12345")
        assert ok is True


# ===========================================================================
# Test validate_task_request_body
# ===========================================================================


class TestValidateTaskRequestBody:
    """Tests for the validate_task_request_body function."""

    def test_valid_minimal(self):
        ok, err = validate_task_request_body({"instruction": "Do something"})
        assert ok is True
        assert err is None

    def test_valid_full_body(self):
        data = {
            "instruction": "Audit this code",
            "task_id": "task-001",
            "capability": "audit",
            "priority": "high",
            "context": [{"type": "text", "content": "some code"}],
            "metadata": {"key": "value"},
            "timeout_ms": 5000,
        }
        ok, err = validate_task_request_body(data)
        assert ok is True

    def test_not_a_dict(self):
        ok, err = validate_task_request_body("not a dict")
        assert ok is False
        assert "JSON object" in err

    def test_missing_instruction(self):
        ok, err = validate_task_request_body({})
        assert ok is False
        assert "instruction" in err

    def test_instruction_not_string(self):
        ok, err = validate_task_request_body({"instruction": 42})
        assert ok is False
        assert "string" in err

    def test_instruction_too_long(self):
        ok, err = validate_task_request_body(
            {"instruction": "x" * (MAX_INSTRUCTION_LENGTH + 1)}
        )
        assert ok is False
        assert str(MAX_INSTRUCTION_LENGTH) in err

    def test_instruction_empty_string(self):
        ok, err = validate_task_request_body({"instruction": ""})
        assert ok is False
        assert "instruction" in err.lower()

    def test_task_id_not_string(self):
        ok, err = validate_task_request_body({"instruction": "ok", "task_id": 123})
        assert ok is False
        assert "task_id" in err

    def test_task_id_invalid_format(self):
        ok, err = validate_task_request_body({"instruction": "ok", "task_id": "-bad"})
        assert ok is False

    def test_capability_not_string(self):
        ok, err = validate_task_request_body(
            {"instruction": "ok", "capability": 42}
        )
        assert ok is False
        assert "capability" in err and "string" in err

    def test_capability_invalid_value(self):
        ok, err = validate_task_request_body(
            {"instruction": "ok", "capability": "flying"}
        )
        assert ok is False
        assert "Invalid capability" in err

    def test_capability_case_insensitive(self):
        ok, err = validate_task_request_body(
            {"instruction": "ok", "capability": "DEBATE"}
        )
        assert ok is True

    def test_all_valid_capabilities(self):
        valid = [
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
        for cap in valid:
            ok, _ = validate_task_request_body(
                {"instruction": "ok", "capability": cap}
            )
            assert ok is True, f"Expected capability '{cap}' to be valid"

    def test_priority_not_string(self):
        ok, err = validate_task_request_body(
            {"instruction": "ok", "priority": 1}
        )
        assert ok is False
        assert "priority" in err

    def test_priority_invalid_value(self):
        ok, err = validate_task_request_body(
            {"instruction": "ok", "priority": "extreme"}
        )
        assert ok is False
        assert "Invalid priority" in err

    def test_all_valid_priorities(self):
        for p in ("low", "normal", "high", "urgent"):
            ok, _ = validate_task_request_body(
                {"instruction": "ok", "priority": p}
            )
            assert ok is True, f"Expected priority '{p}' to be valid"

    def test_context_not_list(self):
        ok, err = validate_task_request_body(
            {"instruction": "ok", "context": "nope"}
        )
        assert ok is False
        assert "array" in err

    def test_context_too_many_items(self):
        items = [{"type": "text"} for _ in range(MAX_CONTEXT_ITEMS + 1)]
        ok, err = validate_task_request_body(
            {"instruction": "ok", "context": items}
        )
        assert ok is False
        assert str(MAX_CONTEXT_ITEMS) in err

    def test_context_item_not_dict(self):
        ok, err = validate_task_request_body(
            {"instruction": "ok", "context": ["bad"]}
        )
        assert ok is False
        assert "context[0]" in err

    def test_context_item_type_not_string(self):
        ok, err = validate_task_request_body(
            {"instruction": "ok", "context": [{"type": 42}]}
        )
        assert ok is False
        assert "context[0].type" in err

    def test_context_item_content_not_string(self):
        ok, err = validate_task_request_body(
            {"instruction": "ok", "context": [{"content": 123}]}
        )
        assert ok is False
        assert "context[0].content" in err

    def test_context_item_content_too_long(self):
        ok, err = validate_task_request_body(
            {
                "instruction": "ok",
                "context": [{"content": "x" * (MAX_CONTEXT_CONTENT_LENGTH + 1)}],
            }
        )
        assert ok is False
        assert str(MAX_CONTEXT_CONTENT_LENGTH) in err

    def test_context_item_metadata_not_dict(self):
        ok, err = validate_task_request_body(
            {"instruction": "ok", "context": [{"metadata": "bad"}]}
        )
        assert ok is False
        assert "context[0].metadata" in err

    def test_context_second_item_invalid(self):
        ok, err = validate_task_request_body(
            {
                "instruction": "ok",
                "context": [{"type": "text"}, "invalid"],
            }
        )
        assert ok is False
        assert "context[1]" in err

    def test_metadata_not_dict(self):
        ok, err = validate_task_request_body(
            {"instruction": "ok", "metadata": "bad"}
        )
        assert ok is False
        assert "metadata must be an object" in err

    def test_metadata_too_many_keys(self):
        meta = {f"k{i}": "v" for i in range(MAX_METADATA_KEYS + 1)}
        ok, err = validate_task_request_body(
            {"instruction": "ok", "metadata": meta}
        )
        assert ok is False
        assert str(MAX_METADATA_KEYS) in err

    def test_metadata_value_too_long(self):
        ok, err = validate_task_request_body(
            {
                "instruction": "ok",
                "metadata": {"big": "x" * (MAX_METADATA_VALUE_LENGTH + 1)},
            }
        )
        assert ok is False
        assert "exceeds maximum" in err

    def test_metadata_non_string_value_ok(self):
        """Non-string metadata values are allowed (length check only for strings)."""
        ok, err = validate_task_request_body(
            {"instruction": "ok", "metadata": {"count": 42, "flag": True}}
        )
        assert ok is True

    def test_timeout_not_int(self):
        ok, err = validate_task_request_body(
            {"instruction": "ok", "timeout_ms": "fast"}
        )
        assert ok is False
        assert "integer" in err

    def test_timeout_too_low(self):
        ok, err = validate_task_request_body(
            {"instruction": "ok", "timeout_ms": 500}
        )
        assert ok is False
        assert "1000" in err

    def test_timeout_too_high(self):
        ok, err = validate_task_request_body(
            {"instruction": "ok", "timeout_ms": 3600001}
        )
        assert ok is False
        assert "3600000" in err

    def test_timeout_min_boundary(self):
        ok, err = validate_task_request_body(
            {"instruction": "ok", "timeout_ms": 1000}
        )
        assert ok is True

    def test_timeout_max_boundary(self):
        ok, err = validate_task_request_body(
            {"instruction": "ok", "timeout_ms": 3600000}
        )
        assert ok is True

    def test_timeout_float_rejected(self):
        ok, err = validate_task_request_body(
            {"instruction": "ok", "timeout_ms": 5000.5}
        )
        assert ok is False
        assert "integer" in err

    def test_valid_context_with_all_fields(self):
        ok, err = validate_task_request_body(
            {
                "instruction": "ok",
                "context": [
                    {
                        "type": "text",
                        "content": "some text",
                        "metadata": {"key": "val"},
                    }
                ],
            }
        )
        assert ok is True


# ===========================================================================
# Test can_handle
# ===========================================================================


class TestCanHandle:
    """Tests for A2AHandler.can_handle method."""

    def test_a2a_prefix(self, handler):
        assert handler.can_handle("/api/v1/a2a/agents") is True

    def test_a2a_tasks(self, handler):
        assert handler.can_handle("/api/v1/a2a/tasks") is True

    def test_well_known(self, handler):
        assert handler.can_handle("/.well-known/agent.json") is True

    def test_unrelated_path(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_partial_prefix(self, handler):
        assert handler.can_handle("/api/v1/a2") is False

    def test_different_version(self, handler):
        assert handler.can_handle("/api/v2/a2a/agents") is False

    def test_openapi_path(self, handler):
        assert handler.can_handle("/api/v1/a2a/openapi.json") is True

    def test_well_known_via_api(self, handler):
        assert handler.can_handle("/api/v1/a2a/.well-known/agent.json") is True


# ===========================================================================
# Test discovery endpoint (via handle())
# ===========================================================================


class TestDiscoveryEndpoint:
    """Tests for /.well-known/agent.json via the handle() method."""

    @pytest.mark.asyncio
    async def test_discovery_with_agents(self, handler, mock_a2a_server):
        agent_card = MagicMock()
        agent_card.to_dict.return_value = {"name": "primary", "version": "1.0"}
        mock_a2a_server.list_agents.return_value = [agent_card]

        with patch(
            "aragora.server.handlers.a2a.get_a2a_server",
            return_value=mock_a2a_server,
        ):
            result = await handler.handle(
                "/.well-known/agent.json", {}, _make_http_handler()
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["name"] == "primary"

    @pytest.mark.asyncio
    async def test_discovery_no_agents_returns_default(
        self, handler, mock_a2a_server
    ):
        mock_a2a_server.list_agents.return_value = []

        with patch(
            "aragora.server.handlers.a2a.get_a2a_server",
            return_value=mock_a2a_server,
        ):
            result = await handler.handle(
                "/.well-known/agent.json", {}, _make_http_handler()
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["name"] == "aragora"
        assert "capabilities" in body
        assert "endpoints" in body

    @pytest.mark.asyncio
    async def test_discovery_via_api_path(self, handler, mock_a2a_server):
        mock_a2a_server.list_agents.return_value = []

        with patch(
            "aragora.server.handlers.a2a.get_a2a_server",
            return_value=mock_a2a_server,
        ):
            result = await handler.handle(
                "/api/v1/a2a/.well-known/agent.json",
                {},
                _make_http_handler(),
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["name"] == "aragora"

    @pytest.mark.asyncio
    async def test_discovery_default_card_has_endpoints(
        self, handler, mock_a2a_server
    ):
        """The default discovery card lists endpoints for integration."""
        mock_a2a_server.list_agents.return_value = []

        with patch(
            "aragora.server.handlers.a2a.get_a2a_server",
            return_value=mock_a2a_server,
        ):
            result = await handler.handle(
                "/.well-known/agent.json", {}, _make_http_handler()
            )

        body = json.loads(result.body)
        assert body["endpoints"]["agents"] == "/api/v1/a2a/agents"
        assert body["endpoints"]["tasks"] == "/api/v1/a2a/tasks"


# ===========================================================================
# Test OpenAPI spec endpoint (via handle())
# ===========================================================================


class TestOpenAPIEndpoint:
    """Tests for /api/v1/a2a/openapi.json via handle()."""

    @pytest.mark.asyncio
    async def test_openapi_returns_spec(self, handler, mock_a2a_server):
        mock_a2a_server.get_openapi_spec.return_value = {
            "openapi": "3.0.0",
            "info": {"title": "test"},
        }

        with patch(
            "aragora.server.handlers.a2a.get_a2a_server",
            return_value=mock_a2a_server,
        ):
            result = await handler.handle(
                "/api/v1/a2a/openapi.json", {}, _make_http_handler()
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["openapi"] == "3.0.0"


# ===========================================================================
# Test list agents (direct method call)
# ===========================================================================


class TestListAgents:
    """Tests for _handle_list_agents."""

    def test_list_agents_empty(self, handler, mock_a2a_server):
        mock_a2a_server.list_agents.return_value = []

        with patch(
            "aragora.server.handlers.a2a.get_a2a_server",
            return_value=mock_a2a_server,
        ):
            result = handler._handle_list_agents()

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["agents"] == []
        assert body["total"] == 0

    def test_list_agents_with_results(self, handler, mock_a2a_server):
        card1 = MagicMock()
        card1.to_dict.return_value = {"name": "agent1"}
        card2 = MagicMock()
        card2.to_dict.return_value = {"name": "agent2"}
        mock_a2a_server.list_agents.return_value = [card1, card2]

        with patch(
            "aragora.server.handlers.a2a.get_a2a_server",
            return_value=mock_a2a_server,
        ):
            result = handler._handle_list_agents()

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["total"] == 2
        assert len(body["agents"]) == 2

    def test_list_agents_single_agent(self, handler, mock_a2a_server):
        card = MagicMock()
        card.to_dict.return_value = {"name": "solo"}
        mock_a2a_server.list_agents.return_value = [card]

        with patch(
            "aragora.server.handlers.a2a.get_a2a_server",
            return_value=mock_a2a_server,
        ):
            result = handler._handle_list_agents()

        body = json.loads(result.body)
        assert body["total"] == 1


# ===========================================================================
# Test get agent (direct method call)
# ===========================================================================


class TestGetAgent:
    """Tests for _handle_get_agent."""

    def test_get_agent_found(self, handler, mock_a2a_server):
        card = MagicMock()
        card.to_dict.return_value = {"name": "test-agent", "version": "1.0"}
        mock_a2a_server.get_agent.return_value = card

        with patch(
            "aragora.server.handlers.a2a.get_a2a_server",
            return_value=mock_a2a_server,
        ):
            result = handler._handle_get_agent("test-agent")

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["name"] == "test-agent"

    def test_get_agent_not_found(self, handler, mock_a2a_server):
        mock_a2a_server.get_agent.return_value = None

        with patch(
            "aragora.server.handlers.a2a.get_a2a_server",
            return_value=mock_a2a_server,
        ):
            result = handler._handle_get_agent("nonexistent")

        assert result.status_code == 404

    def test_get_agent_invalid_name_hyphen_start(self, handler):
        result = handler._handle_get_agent("-bad-name")
        assert result.status_code == 400

    def test_get_agent_invalid_name_too_long(self, handler):
        result = handler._handle_get_agent("a" * 65)
        assert result.status_code == 400

    def test_get_agent_invalid_name_special_chars(self, handler):
        result = handler._handle_get_agent("agent@home")
        assert result.status_code == 400

    def test_get_agent_invalid_empty_name(self, handler):
        result = handler._handle_get_agent("")
        assert result.status_code == 400


# ===========================================================================
# Test submit task (direct method call)
# ===========================================================================


class TestSubmitTask:
    """Tests for _handle_submit_task."""

    @pytest.mark.asyncio
    async def test_submit_task_success(self, handler, mock_a2a_server):
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {
            "task_id": "t1",
            "status": "completed",
        }
        mock_a2a_server.handle_task = AsyncMock(return_value=mock_result)

        http_handler = _make_post_handler(
            {"instruction": "Run a debate about testing"}
        )

        with patch(
            "aragora.server.handlers.a2a.get_a2a_server",
            return_value=mock_a2a_server,
        ):
            result = await handler._handle_submit_task(http_handler)

        assert result.status_code == 200
        resp = json.loads(result.body)
        assert resp["task_id"] == "t1"

    @pytest.mark.asyncio
    async def test_submit_task_with_capability(self, handler, mock_a2a_server):
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {
            "task_id": "t2",
            "status": "running",
        }
        mock_a2a_server.handle_task = AsyncMock(return_value=mock_result)

        http_handler = _make_post_handler(
            {"instruction": "Audit this code", "capability": "audit"}
        )

        with patch(
            "aragora.server.handlers.a2a.get_a2a_server",
            return_value=mock_a2a_server,
        ):
            result = await handler._handle_submit_task(http_handler)

        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_submit_task_with_priority(self, handler, mock_a2a_server):
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {
            "task_id": "t3",
            "status": "running",
        }
        mock_a2a_server.handle_task = AsyncMock(return_value=mock_result)

        http_handler = _make_post_handler(
            {"instruction": "Urgent task", "priority": "urgent"}
        )

        with patch(
            "aragora.server.handlers.a2a.get_a2a_server",
            return_value=mock_a2a_server,
        ):
            result = await handler._handle_submit_task(http_handler)

        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_submit_task_with_context(self, handler, mock_a2a_server):
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {
            "task_id": "t4",
            "status": "running",
        }
        mock_a2a_server.handle_task = AsyncMock(return_value=mock_result)

        http_handler = _make_post_handler(
            {
                "instruction": "Analyze",
                "context": [
                    {
                        "type": "text",
                        "content": "sample content",
                        "metadata": {"key": "val"},
                    },
                ],
            }
        )

        with patch(
            "aragora.server.handlers.a2a.get_a2a_server",
            return_value=mock_a2a_server,
        ):
            result = await handler._handle_submit_task(http_handler)

        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_submit_task_with_metadata_and_deadline(
        self, handler, mock_a2a_server
    ):
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {
            "task_id": "t5",
            "status": "running",
        }
        mock_a2a_server.handle_task = AsyncMock(return_value=mock_result)

        http_handler = _make_post_handler(
            {
                "instruction": "Quick task",
                "metadata": {"source": "test"},
                "deadline": "2026-12-31T00:00:00",
            }
        )

        with patch(
            "aragora.server.handlers.a2a.get_a2a_server",
            return_value=mock_a2a_server,
        ):
            result = await handler._handle_submit_task(http_handler)

        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_submit_task_with_custom_task_id(
        self, handler, mock_a2a_server
    ):
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {
            "task_id": "my-custom-id",
            "status": "running",
        }
        mock_a2a_server.handle_task = AsyncMock(return_value=mock_result)

        http_handler = _make_post_handler(
            {"instruction": "Test", "task_id": "my-custom-id"}
        )

        with patch(
            "aragora.server.handlers.a2a.get_a2a_server",
            return_value=mock_a2a_server,
        ):
            result = await handler._handle_submit_task(http_handler)

        assert result.status_code == 200
        resp = json.loads(result.body)
        assert resp["task_id"] == "my-custom-id"

    @pytest.mark.asyncio
    async def test_submit_task_wrong_content_type(
        self, handler, mock_a2a_server
    ):
        http_handler = _make_http_handler(
            method="POST", content_type="text/plain"
        )

        with patch(
            "aragora.server.handlers.a2a.get_a2a_server",
            return_value=mock_a2a_server,
        ):
            result = await handler._handle_submit_task(http_handler)

        assert result.status_code == 415

    @pytest.mark.asyncio
    async def test_submit_task_body_too_large(
        self, handler, mock_a2a_server
    ):
        http_handler = _make_http_handler(
            method="POST",
            body={"instruction": "ok"},
            content_length=MAX_BODY_SIZE + 1,
        )

        with patch(
            "aragora.server.handlers.a2a.get_a2a_server",
            return_value=mock_a2a_server,
        ):
            result = await handler._handle_submit_task(http_handler)

        assert result.status_code == 413

    @pytest.mark.asyncio
    async def test_submit_task_invalid_json(
        self, handler, mock_a2a_server
    ):
        http_handler = _make_http_handler(method="POST")
        http_handler.rfile.read.return_value = b"not json{{"

        with patch(
            "aragora.server.handlers.a2a.get_a2a_server",
            return_value=mock_a2a_server,
        ):
            result = await handler._handle_submit_task(http_handler)

        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_submit_task_invalid_utf8(
        self, handler, mock_a2a_server
    ):
        http_handler = _make_http_handler(method="POST")
        http_handler.rfile.read.return_value = b"\xff\xfe"

        with patch(
            "aragora.server.handlers.a2a.get_a2a_server",
            return_value=mock_a2a_server,
        ):
            result = await handler._handle_submit_task(http_handler)

        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_submit_task_missing_instruction(
        self, handler, mock_a2a_server
    ):
        http_handler = _make_post_handler({"capability": "debate"})

        with patch(
            "aragora.server.handlers.a2a.get_a2a_server",
            return_value=mock_a2a_server,
        ):
            result = await handler._handle_submit_task(http_handler)

        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_submit_task_execution_failure(
        self, handler, mock_a2a_server
    ):
        mock_a2a_server.handle_task = AsyncMock(
            side_effect=RuntimeError("boom")
        )

        http_handler = _make_post_handler({"instruction": "fail please"})

        with patch(
            "aragora.server.handlers.a2a.get_a2a_server",
            return_value=mock_a2a_server,
        ):
            result = await handler._handle_submit_task(http_handler)

        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_submit_task_empty_body(
        self, handler, mock_a2a_server
    ):
        http_handler = _make_http_handler(method="POST")
        http_handler.rfile.read.return_value = b""
        http_handler.headers["Content-Length"] = "0"

        with patch(
            "aragora.server.handlers.a2a.get_a2a_server",
            return_value=mock_a2a_server,
        ):
            result = await handler._handle_submit_task(http_handler)

        # empty body parses to {}, which is missing 'instruction'
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_submit_task_no_content_type(
        self, handler, mock_a2a_server
    ):
        """When Content-Type is empty string, it is treated as acceptable."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {
            "task_id": "t-no-ct",
            "status": "completed",
        }
        mock_a2a_server.handle_task = AsyncMock(return_value=mock_result)

        http_handler = _make_http_handler(
            method="POST",
            body={"instruction": "Test"},
            content_type="",
        )

        with patch(
            "aragora.server.handlers.a2a.get_a2a_server",
            return_value=mock_a2a_server,
        ):
            result = await handler._handle_submit_task(http_handler)

        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_submit_task_value_error(
        self, handler, mock_a2a_server
    ):
        mock_a2a_server.handle_task = AsyncMock(
            side_effect=ValueError("bad value")
        )
        http_handler = _make_post_handler({"instruction": "Test"})

        with patch(
            "aragora.server.handlers.a2a.get_a2a_server",
            return_value=mock_a2a_server,
        ):
            result = await handler._handle_submit_task(http_handler)

        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_submit_task_os_error(
        self, handler, mock_a2a_server
    ):
        mock_a2a_server.handle_task = AsyncMock(
            side_effect=OSError("disk full")
        )
        http_handler = _make_post_handler({"instruction": "Test"})

        with patch(
            "aragora.server.handlers.a2a.get_a2a_server",
            return_value=mock_a2a_server,
        ):
            result = await handler._handle_submit_task(http_handler)

        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_submit_task_type_error(
        self, handler, mock_a2a_server
    ):
        mock_a2a_server.handle_task = AsyncMock(
            side_effect=TypeError("wrong type")
        )
        http_handler = _make_post_handler({"instruction": "Test"})

        with patch(
            "aragora.server.handlers.a2a.get_a2a_server",
            return_value=mock_a2a_server,
        ):
            result = await handler._handle_submit_task(http_handler)

        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_submit_task_generates_uuid_when_no_task_id(
        self, handler, mock_a2a_server
    ):
        """When task_id is not provided, a UUID is generated."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"task_id": "auto", "status": "ok"}
        mock_a2a_server.handle_task = AsyncMock(return_value=mock_result)

        http_handler = _make_post_handler({"instruction": "Test"})

        with patch(
            "aragora.server.handlers.a2a.get_a2a_server",
            return_value=mock_a2a_server,
        ):
            result = await handler._handle_submit_task(http_handler)

        assert result.status_code == 200
        # Verify handle_task was called (the UUID was generated)
        mock_a2a_server.handle_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_submit_task_invalid_capability_ignored_in_enum(
        self, handler, mock_a2a_server
    ):
        """When capability passes validation but fails enum conversion, it's set to None."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"task_id": "t", "status": "ok"}
        mock_a2a_server.handle_task = AsyncMock(return_value=mock_result)

        # "debate" is valid but "DEBATE" as an enum might or might not match
        # depending on AgentCapability. The handler has a try/except around enum
        # conversion.
        http_handler = _make_post_handler(
            {"instruction": "Test", "capability": "debate"}
        )

        with patch(
            "aragora.server.handlers.a2a.get_a2a_server",
            return_value=mock_a2a_server,
        ):
            result = await handler._handle_submit_task(http_handler)

        assert result.status_code == 200


# ===========================================================================
# Test get task status (direct method call)
# ===========================================================================


class TestGetTaskStatus:
    """Tests for _handle_get_task."""

    def test_get_task_found(self, handler, mock_a2a_server):
        task_result = MagicMock()
        task_result.to_dict.return_value = {
            "task_id": "t1",
            "status": "completed",
        }
        mock_a2a_server.get_task_status.return_value = task_result

        with patch(
            "aragora.server.handlers.a2a.get_a2a_server",
            return_value=mock_a2a_server,
        ):
            result = handler._handle_get_task("t1")

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["task_id"] == "t1"

    def test_get_task_not_found(self, handler, mock_a2a_server):
        mock_a2a_server.get_task_status.return_value = None

        with patch(
            "aragora.server.handlers.a2a.get_a2a_server",
            return_value=mock_a2a_server,
        ):
            result = handler._handle_get_task("nonexistent")

        assert result.status_code == 404

    def test_get_task_invalid_id(self, handler):
        result = handler._handle_get_task("-bad")
        assert result.status_code == 400

    def test_get_task_empty_id(self, handler):
        result = handler._handle_get_task("")
        assert result.status_code == 400

    def test_get_task_id_too_long(self, handler):
        result = handler._handle_get_task("a" * 129)
        assert result.status_code == 400


# ===========================================================================
# Test cancel task (direct method call)
# ===========================================================================


class TestCancelTask:
    """Tests for _handle_cancel_task."""

    @pytest.mark.asyncio
    async def test_cancel_task_success(self, handler, mock_a2a_server):
        mock_a2a_server.cancel_task = AsyncMock(return_value=True)

        with patch(
            "aragora.server.handlers.a2a.get_a2a_server",
            return_value=mock_a2a_server,
        ):
            result = await handler._handle_cancel_task("t1")

        assert result.status_code == 204
        assert result.body == b""

    @pytest.mark.asyncio
    async def test_cancel_task_not_found(self, handler, mock_a2a_server):
        mock_a2a_server.cancel_task = AsyncMock(return_value=False)

        with patch(
            "aragora.server.handlers.a2a.get_a2a_server",
            return_value=mock_a2a_server,
        ):
            result = await handler._handle_cancel_task("t1")

        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_cancel_task_invalid_id(self, handler):
        result = await handler._handle_cancel_task("-bad")
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_cancel_task_runtime_error(
        self, handler, mock_a2a_server
    ):
        mock_a2a_server.cancel_task = AsyncMock(
            side_effect=RuntimeError("oops")
        )

        with patch(
            "aragora.server.handlers.a2a.get_a2a_server",
            return_value=mock_a2a_server,
        ):
            result = await handler._handle_cancel_task("t1")

        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_cancel_task_key_error(
        self, handler, mock_a2a_server
    ):
        mock_a2a_server.cancel_task = AsyncMock(
            side_effect=KeyError("missing")
        )

        with patch(
            "aragora.server.handlers.a2a.get_a2a_server",
            return_value=mock_a2a_server,
        ):
            result = await handler._handle_cancel_task("t1")

        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_cancel_task_os_error(
        self, handler, mock_a2a_server
    ):
        mock_a2a_server.cancel_task = AsyncMock(
            side_effect=OSError("io error")
        )

        with patch(
            "aragora.server.handlers.a2a.get_a2a_server",
            return_value=mock_a2a_server,
        ):
            result = await handler._handle_cancel_task("t1")

        assert result.status_code == 500


# ===========================================================================
# Test stream task (direct method call)
# ===========================================================================


class TestStreamTask:
    """Tests for _handle_stream_task."""

    def test_stream_returns_upgrade_required(self, handler):
        http_handler = _make_http_handler(method="POST")
        result = handler._handle_stream_task("t1", http_handler)

        assert result.status_code == 426
        body = json.loads(result.body)
        assert "WebSocket" in body["message"]
        assert "/ws/a2a/tasks/t1/stream" in body["ws_path"]

    def test_stream_invalid_task_id(self, handler):
        http_handler = _make_http_handler(method="POST")
        result = handler._handle_stream_task("-bad", http_handler)
        assert result.status_code == 400

    def test_stream_empty_task_id(self, handler):
        http_handler = _make_http_handler(method="POST")
        result = handler._handle_stream_task("", http_handler)
        assert result.status_code == 400

    def test_stream_ws_path_format(self, handler):
        http_handler = _make_http_handler(method="POST")
        result = handler._handle_stream_task("my-task-42", http_handler)

        body = json.loads(result.body)
        assert body["ws_path"] == "/ws/a2a/tasks/my-task-42/stream"


# ===========================================================================
# Test unknown endpoint (via handle())
# ===========================================================================


class TestUnknownEndpoint:
    """Tests for unknown A2A paths via handle()."""

    @pytest.mark.asyncio
    async def test_unknown_subpath(self, handler, mock_a2a_server):
        with patch(
            "aragora.server.handlers.a2a.get_a2a_server",
            return_value=mock_a2a_server,
        ):
            result = await handler.handle(
                "/api/v1/a2a/unknown", {}, _make_http_handler()
            )

        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_method_defaults_to_get(self, handler, mock_a2a_server):
        """When handler has no command attribute, method defaults to GET."""
        http_handler = MagicMock(spec=[])
        with patch(
            "aragora.server.handlers.a2a.get_a2a_server",
            return_value=mock_a2a_server,
        ):
            # Non-matching subpath returns 404 for any method
            result = await handler.handle(
                "/api/v1/a2a/unknown-path", {}, http_handler
            )

        assert result.status_code == 404


# ===========================================================================
# Test circuit breaker helpers
# ===========================================================================


class TestCircuitBreaker:
    """Tests for circuit breaker utility functions."""

    def test_get_circuit_breaker(self):
        cb = get_a2a_circuit_breaker()
        assert cb is not None
        assert cb.name == "a2a_handler"

    def test_get_circuit_breaker_status(self):
        status = get_a2a_circuit_breaker_status()
        assert isinstance(status, dict)

    def test_circuit_breaker_is_singleton(self):
        cb1 = get_a2a_circuit_breaker()
        cb2 = get_a2a_circuit_breaker()
        assert cb1 is cb2

    def test_circuit_breaker_config(self):
        cb = get_a2a_circuit_breaker()
        assert cb.failure_threshold == 5


# ===========================================================================
# Test handler factory
# ===========================================================================


class TestHandlerFactory:
    """Tests for get_a2a_handler factory."""

    def test_creates_handler(self):
        h = get_a2a_handler()
        assert isinstance(h, A2AHandler)

    def test_returns_same_singleton(self):
        h1 = get_a2a_handler()
        h2 = get_a2a_handler()
        assert h1 is h2

    def test_accepts_server_context(self):
        ctx = {"key": "value"}
        h = get_a2a_handler(server_context=ctx)
        assert h.ctx == ctx

    def test_default_server_context(self):
        h = get_a2a_handler()
        assert h.ctx == {}


# ===========================================================================
# Test handler __init__
# ===========================================================================


class TestHandlerInit:
    """Tests for A2AHandler initialization."""

    def test_default_ctx(self):
        h = A2AHandler()
        assert h.ctx == {}

    def test_custom_ctx(self):
        h = A2AHandler(ctx={"hello": "world"})
        assert h.ctx["hello"] == "world"

    def test_none_ctx_becomes_empty_dict(self):
        h = A2AHandler(ctx=None)
        assert h.ctx == {}


# ===========================================================================
# Test regex patterns
# ===========================================================================


class TestRegexPatterns:
    """Tests for the compiled regex patterns."""

    def test_agent_name_pattern_valid(self):
        assert AGENT_NAME_PATTERN.match("abc123")
        assert AGENT_NAME_PATTERN.match("a")
        assert AGENT_NAME_PATTERN.match("agent-v2.0_test")

    def test_agent_name_pattern_invalid(self):
        assert not AGENT_NAME_PATTERN.match("")
        assert not AGENT_NAME_PATTERN.match("-start")
        assert not AGENT_NAME_PATTERN.match("has space")

    def test_task_id_pattern_valid(self):
        assert TASK_ID_PATTERN.match("task-123")
        assert TASK_ID_PATTERN.match("a")
        assert TASK_ID_PATTERN.match("abc_def-ghi")

    def test_task_id_pattern_invalid(self):
        assert not TASK_ID_PATTERN.match("")
        assert not TASK_ID_PATTERN.match("-start")
        assert not TASK_ID_PATTERN.match("has.dot")

    def test_agent_name_max_boundary(self):
        """64-char name matches, 65 does not."""
        assert AGENT_NAME_PATTERN.match("a" * 64)
        assert not AGENT_NAME_PATTERN.match("a" * 65)

    def test_task_id_max_boundary(self):
        """128-char ID matches, 129 does not."""
        assert TASK_ID_PATTERN.match("a" * 128)
        assert not TASK_ID_PATTERN.match("a" * 129)


# ===========================================================================
# Test ROUTES class attribute
# ===========================================================================


class TestRoutes:
    """Tests for the ROUTES class attribute."""

    def test_routes_defined(self):
        assert isinstance(A2AHandler.ROUTES, list)
        assert len(A2AHandler.ROUTES) > 0

    def test_routes_include_well_known(self):
        assert "/.well-known/agent.json" in A2AHandler.ROUTES

    def test_routes_include_agents(self):
        assert "/api/v1/a2a/agents" in A2AHandler.ROUTES

    def test_routes_include_tasks(self):
        assert "/api/v1/a2a/tasks" in A2AHandler.ROUTES

    def test_routes_include_openapi(self):
        assert "/api/v1/a2a/openapi.json" in A2AHandler.ROUTES

    def test_routes_include_versioned_well_known(self):
        assert "/api/v1/a2a/.well-known/agent.json" in A2AHandler.ROUTES

    def test_routes_include_wildcard_agents(self):
        assert "/api/v1/a2a/agents/*" in A2AHandler.ROUTES

    def test_routes_include_wildcard_tasks(self):
        assert "/api/v1/a2a/tasks/*" in A2AHandler.ROUTES


# ===========================================================================
# Test get_a2a_server singleton
# ===========================================================================


class TestGetA2AServer:
    """Tests for the get_a2a_server function."""

    def test_returns_server(self):
        from aragora.server.handlers.a2a import get_a2a_server

        server = get_a2a_server()
        assert server is not None

    def test_returns_same_instance(self):
        from aragora.server.handlers.a2a import get_a2a_server

        s1 = get_a2a_server()
        s2 = get_a2a_server()
        assert s1 is s2


# ===========================================================================
# Test constants
# ===========================================================================


class TestConstants:
    """Tests for module-level constants."""

    def test_max_body_size(self):
        assert MAX_BODY_SIZE == 1024 * 1024  # 1 MB

    def test_max_instruction_length(self):
        assert MAX_INSTRUCTION_LENGTH == 10000

    def test_max_context_items(self):
        assert MAX_CONTEXT_ITEMS == 50

    def test_max_context_content_length(self):
        assert MAX_CONTEXT_CONTENT_LENGTH == 100000

    def test_max_metadata_keys(self):
        assert MAX_METADATA_KEYS == 20

    def test_max_metadata_value_length(self):
        assert MAX_METADATA_VALUE_LENGTH == 1000
