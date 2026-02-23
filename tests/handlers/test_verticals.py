"""Tests for verticals handler (aragora/server/handlers/verticals.py).

Covers all routes and behavior of the VerticalsHandler class:
- can_handle() routing for all ROUTES
- GET    /api/verticals            - List all verticals
- GET    /api/verticals/:id        - Get vertical config
- PUT    /api/verticals/:id/config - Update vertical configuration
- GET    /api/verticals/:id/tools  - Get vertical tools
- GET    /api/verticals/:id/compliance - Get compliance frameworks
- POST   /api/verticals/:id/debate - Create vertical-specific debate
- POST   /api/verticals/:id/agent  - Create specialist agent instance
- GET    /api/verticals/suggest    - Suggest vertical for a task
- Error handling (invalid IDs, missing registry, circuit breaker)
- Validation (topic, agent name, task, additional agents, rounds)
- Version prefix stripping (v1 paths)
- Circuit breaker integration
"""

from __future__ import annotations

import json
import sys
from typing import Any
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from aragora.server.handlers.verticals import (
    VerticalsHandler,
    get_verticals_circuit_breaker,
    reset_verticals_circuit_breaker,
)
from aragora.server.handlers.base import HandlerResult, json_response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract the body dict from a HandlerResult."""
    if result is None:
        return {}
    if hasattr(result, "body"):
        try:
            return json.loads(result.body.decode("utf-8")) if result.body else {}
        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
            return {}
    if isinstance(result, dict):
        return result.get("body", result)
    return {}


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    if hasattr(result, "status_code"):
        return result.status_code
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return 0


class _MockHTTPHandler:
    """Mock HTTP handler for simulating requests with JSON body."""

    def __init__(self, body: dict | None = None, method: str = "GET"):
        self.command = method
        self.rfile = MagicMock()
        self.headers = {}
        if body is not None:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers["Content-Length"] = str(len(body_bytes))
        else:
            self.rfile.read.return_value = b"{}"
            self.headers["Content-Length"] = "2"


# ---------------------------------------------------------------------------
# Mock registry helpers
# ---------------------------------------------------------------------------


class _MockModelConfig:
    def __init__(self):
        self.primary_model = "claude-3"
        self.primary_provider = "anthropic"
        self.temperature = 0.7
        self.max_tokens = 4096

    def to_dict(self):
        return {
            "primary_model": self.primary_model,
            "primary_provider": self.primary_provider,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


class _MockToolConfig:
    def __init__(self, name="tool1", enabled=True):
        self.name = name
        self.enabled = enabled
        self.description = f"{name} description"
        self.connector_type = None
        self.parameters = {}

    def to_dict(self):
        return {"name": self.name, "enabled": self.enabled, "description": self.description}


class _MockComplianceConfig:
    def __init__(self, framework="SOC2", level="required"):
        self.framework = framework
        self.level = level
        self.version = "latest"
        self.rules = []
        self.exemptions = []

    def to_dict(self):
        return {"framework": self.framework, "level": self.level, "version": self.version}


class _MockVerticalConfig:
    def __init__(self, vertical_id="healthcare"):
        self.display_name = f"{vertical_id.title()} Vertical"
        self.domain_keywords = [vertical_id, "domain"]
        self.expertise_areas = ["analysis", "compliance"]
        self.tools = [_MockToolConfig("tool1"), _MockToolConfig("tool2", enabled=False)]
        self.compliance_frameworks = [_MockComplianceConfig()]
        self.model_config = _MockModelConfig()
        self.version = "1.0.0"
        self.author = "test"
        self.tags = ["test"]

    def get_enabled_tools(self):
        return [t for t in self.tools if t.enabled]

    def get_compliance_frameworks(self, level=None):
        if level is not None:
            return [f for f in self.compliance_frameworks if f.level == level.value]
        return self.compliance_frameworks


class _MockVerticalSpec:
    def __init__(self, vertical_id="healthcare"):
        self.config = _MockVerticalConfig(vertical_id)
        self.description = f"A {vertical_id} vertical for testing"


class _MockSpecialist:
    def __init__(self, vertical_id="healthcare", name=None, model=None, role="specialist"):
        self.name = name or f"{vertical_id}-specialist"
        self.model = model or "claude-3"
        self.role = role
        self.expertise_areas = ["analysis"]

    def to_dict(self):
        return {"name": self.name, "model": self.model, "role": self.role}

    def get_enabled_tools(self):
        return [_MockToolConfig("specialist-tool")]


def _make_mock_registry():
    """Create a mock VerticalRegistry with standard methods."""
    registry = MagicMock()
    registry.list_all.return_value = {
        "healthcare": {"description": "Healthcare vertical", "keywords": ["health"]},
        "financial": {"description": "Financial vertical", "keywords": ["finance"]},
    }
    registry.get_registered_ids.return_value = ["healthcare", "financial"]
    registry.get.return_value = _MockVerticalSpec("healthcare")
    registry.get_config.return_value = _MockVerticalConfig("healthcare")
    registry.is_registered.return_value = True
    registry.get_by_keyword.return_value = ["healthcare"]
    registry.get_for_task.return_value = "healthcare"
    registry.create_specialist.return_value = _MockSpecialist()
    return registry


def _debate_success_response(vertical_id="healthcare", **overrides):
    """Build a json_response matching what _create_debate returns on success."""
    body = {
        "debate_id": overrides.get("debate_id", "debate-123"),
        "vertical_id": vertical_id,
        "topic": overrides.get("topic", "Test topic"),
        "consensus_reached": overrides.get("consensus_reached", True),
        "final_answer": overrides.get("final_answer", "Answer"),
        "confidence": overrides.get("confidence", 0.95),
        "participants": overrides.get("participants", ["healthcare-specialist"]),
    }
    return json_response(body)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a VerticalsHandler with empty context."""
    return VerticalsHandler(ctx={})


@pytest.fixture(autouse=True)
def _reset_cb():
    """Reset the circuit breaker between tests."""
    reset_verticals_circuit_breaker()
    yield
    reset_verticals_circuit_breaker()


# ============================================================================
# can_handle routing
# ============================================================================


class TestCanHandle:
    """Verify that can_handle correctly accepts or rejects paths."""

    def test_list_verticals(self, handler):
        assert handler.can_handle("/api/verticals")

    def test_list_verticals_v1(self, handler):
        assert handler.can_handle("/api/v1/verticals")

    def test_suggest_path(self, handler):
        assert handler.can_handle("/api/verticals/suggest")

    def test_suggest_path_v1(self, handler):
        assert handler.can_handle("/api/v1/verticals/suggest")

    def test_vertical_by_id(self, handler):
        assert handler.can_handle("/api/verticals/healthcare")

    def test_vertical_config_path(self, handler):
        assert handler.can_handle("/api/verticals/healthcare/config")

    def test_vertical_tools_path(self, handler):
        assert handler.can_handle("/api/verticals/healthcare/tools")

    def test_vertical_compliance_path(self, handler):
        assert handler.can_handle("/api/verticals/healthcare/compliance")

    def test_vertical_debate_path(self, handler):
        assert handler.can_handle("/api/verticals/healthcare/debate")

    def test_vertical_agent_path(self, handler):
        assert handler.can_handle("/api/verticals/healthcare/agent")

    def test_rejects_unrelated_path(self, handler):
        assert not handler.can_handle("/api/agents")

    def test_rejects_empty_path(self, handler):
        assert not handler.can_handle("")

    def test_rejects_root(self, handler):
        assert not handler.can_handle("/")

    def test_rejects_partial_prefix(self, handler):
        assert not handler.can_handle("/api/vertical")

    def test_v1_vertical_by_id(self, handler):
        assert handler.can_handle("/api/v1/verticals/healthcare")

    def test_v1_sub_path(self, handler):
        assert handler.can_handle("/api/v1/verticals/healthcare/tools")


# ============================================================================
# Initialization
# ============================================================================


class TestHandlerInit:
    def test_init_with_empty_context(self):
        h = VerticalsHandler(ctx={})
        assert h.ctx == {}

    def test_init_with_none_context(self):
        h = VerticalsHandler(ctx=None)
        assert h.ctx == {}

    def test_init_with_context(self):
        ctx = {"document_store": "store"}
        h = VerticalsHandler(ctx=ctx)
        assert h.ctx == ctx

    def test_resource_type(self, handler):
        assert handler.RESOURCE_TYPE == "verticals"

    def test_routes_defined(self, handler):
        assert len(handler.ROUTES) > 0
        assert "/api/verticals" in handler.ROUTES

    def test_circuit_breaker_attached(self, handler):
        assert handler._circuit_breaker is not None


# ============================================================================
# Validation helpers
# ============================================================================


class TestValidateKeyword:
    def test_none_is_valid(self, handler):
        is_valid, err = handler._validate_keyword(None)
        assert is_valid
        assert err == ""

    def test_valid_keyword(self, handler):
        is_valid, err = handler._validate_keyword("health")
        assert is_valid

    def test_too_long_keyword(self, handler):
        is_valid, err = handler._validate_keyword("x" * 201)
        assert not is_valid
        assert "maximum length" in err

    def test_max_length_keyword(self, handler):
        is_valid, err = handler._validate_keyword("x" * 200)
        assert is_valid


class TestValidateTask:
    def test_none_is_invalid(self, handler):
        is_valid, err = handler._validate_task(None)
        assert not is_valid
        assert "required" in err.lower()

    def test_empty_string(self, handler):
        is_valid, err = handler._validate_task("")
        assert not is_valid
        assert "empty" in err.lower()

    def test_whitespace_only(self, handler):
        is_valid, err = handler._validate_task("   ")
        assert not is_valid
        assert "empty" in err.lower()

    def test_valid_task(self, handler):
        is_valid, err = handler._validate_task("Analyze patient data")
        assert is_valid
        assert err == ""

    def test_too_long_task(self, handler):
        is_valid, err = handler._validate_task("x" * 100_001)
        assert not is_valid
        assert "maximum length" in err


class TestValidateTopic:
    def test_none_is_invalid(self, handler):
        is_valid, err = handler._validate_topic(None)
        assert not is_valid
        assert "required" in err.lower()

    def test_empty_string(self, handler):
        is_valid, err = handler._validate_topic("")
        assert not is_valid
        assert "empty" in err.lower()

    def test_whitespace_only(self, handler):
        is_valid, err = handler._validate_topic("   ")
        assert not is_valid
        assert "empty" in err.lower()

    def test_valid_topic(self, handler):
        is_valid, err = handler._validate_topic("Patient diagnosis approach")
        assert is_valid
        assert err == ""

    def test_too_long_topic(self, handler):
        is_valid, err = handler._validate_topic("x" * 100_001)
        assert not is_valid
        assert "maximum length" in err


class TestValidateAgentName:
    def test_none_is_valid(self, handler):
        is_valid, err = handler._validate_agent_name(None)
        assert is_valid

    def test_valid_name(self, handler):
        is_valid, err = handler._validate_agent_name("my-agent-1")
        assert is_valid

    def test_too_long_name(self, handler):
        is_valid, err = handler._validate_agent_name("x" * 101)
        assert not is_valid
        assert "maximum length" in err

    def test_invalid_characters(self, handler):
        is_valid, err = handler._validate_agent_name("agent name with spaces")
        assert not is_valid


class TestValidateAdditionalAgents:
    def test_none_is_valid(self, handler):
        is_valid, err = handler._validate_additional_agents(None)
        assert is_valid

    def test_valid_list(self, handler):
        is_valid, err = handler._validate_additional_agents(["agent1", "agent2"])
        assert is_valid

    def test_not_a_list(self, handler):
        is_valid, err = handler._validate_additional_agents("not-a-list")
        assert not is_valid
        assert "must be a list" in err

    def test_too_many_agents(self, handler):
        is_valid, err = handler._validate_additional_agents(["a"] * 11)
        assert not is_valid
        assert "maximum count" in err

    def test_empty_list(self, handler):
        is_valid, err = handler._validate_additional_agents([])
        assert is_valid

    def test_max_agents(self, handler):
        is_valid, err = handler._validate_additional_agents(["a"] * 10)
        assert is_valid


# ============================================================================
# GET /api/verticals - List all verticals
# ============================================================================


class TestListVerticals:
    @pytest.mark.asyncio
    async def test_list_all(self, handler):
        mock_reg = _make_mock_registry()
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle("/api/verticals", {}, None)
        assert _status(result) == 200
        body = _body(result)
        assert "verticals" in body
        assert body["total"] == 2

    @pytest.mark.asyncio
    async def test_list_with_keyword(self, handler):
        mock_reg = _make_mock_registry()
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle("/api/verticals", {"keyword": "health"}, None)
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1

    @pytest.mark.asyncio
    async def test_list_no_registry(self, handler):
        with patch.object(handler, "_get_registry", return_value=None):
            result = await handler.handle("/api/verticals", {}, None)
        assert _status(result) == 503
        body = _body(result)
        assert body["verticals"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_list_keyword_too_long(self, handler):
        mock_reg = _make_mock_registry()
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle("/api/verticals", {"keyword": "x" * 201}, None)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_list_v1_path(self, handler):
        mock_reg = _make_mock_registry()
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle("/api/v1/verticals", {}, None)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_list_key_error(self, handler):
        mock_reg = _make_mock_registry()
        mock_reg.list_all.side_effect = KeyError("bad key")
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle("/api/verticals", {}, None)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_list_runtime_error(self, handler):
        mock_reg = _make_mock_registry()
        mock_reg.list_all.side_effect = RuntimeError("crash")
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle("/api/verticals", {}, None)
        assert _status(result) == 500


# ============================================================================
# GET /api/verticals/:id - Get vertical config
# ============================================================================


class TestGetVertical:
    @pytest.mark.asyncio
    async def test_get_existing(self, handler):
        mock_reg = _make_mock_registry()
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle("/api/verticals/healthcare", {}, None)
        assert _status(result) == 200
        body = _body(result)
        assert body["vertical_id"] == "healthcare"
        assert "display_name" in body
        assert "tools" in body

    @pytest.mark.asyncio
    async def test_get_not_found(self, handler):
        mock_reg = _make_mock_registry()
        mock_reg.get.return_value = None
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle("/api/verticals/nonexistent", {}, None)
        assert _status(result) == 404
        assert "not found" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_get_no_registry(self, handler):
        with patch.object(handler, "_get_registry", return_value=None):
            result = await handler.handle("/api/verticals/healthcare", {}, None)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_get_invalid_id(self, handler):
        result = await handler.handle("/api/verticals/invalid id!!", {}, None)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_get_key_error(self, handler):
        mock_reg = _make_mock_registry()
        mock_reg.get.side_effect = KeyError("bad")
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle("/api/verticals/healthcare", {}, None)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_get_runtime_error(self, handler):
        mock_reg = _make_mock_registry()
        mock_reg.get.side_effect = RuntimeError("crash")
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle("/api/verticals/healthcare", {}, None)
        assert _status(result) == 500


# ============================================================================
# GET /api/verticals/:id/tools
# ============================================================================


class TestGetTools:
    @pytest.mark.asyncio
    async def test_get_tools(self, handler):
        mock_reg = _make_mock_registry()
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle("/api/verticals/healthcare/tools", {}, None)
        assert _status(result) == 200
        body = _body(result)
        assert body["vertical_id"] == "healthcare"
        assert "tools" in body
        assert body["total_count"] == 2
        assert body["enabled_count"] == 1

    @pytest.mark.asyncio
    async def test_get_tools_not_found(self, handler):
        mock_reg = _make_mock_registry()
        mock_reg.get_config.return_value = None
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle("/api/verticals/nonexistent/tools", {}, None)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_get_tools_no_registry(self, handler):
        with patch.object(handler, "_get_registry", return_value=None):
            result = await handler.handle("/api/verticals/healthcare/tools", {}, None)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_get_tools_invalid_id(self, handler):
        result = await handler.handle("/api/verticals/bad id!!/tools", {}, None)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_get_tools_key_error(self, handler):
        mock_reg = _make_mock_registry()
        mock_reg.get_config.side_effect = KeyError("bad")
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle("/api/verticals/healthcare/tools", {}, None)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_get_tools_runtime_error(self, handler):
        mock_reg = _make_mock_registry()
        mock_reg.get_config.side_effect = RuntimeError("crash")
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle("/api/verticals/healthcare/tools", {}, None)
        assert _status(result) == 500


# ============================================================================
# GET /api/verticals/:id/compliance
# ============================================================================


class TestGetCompliance:
    @pytest.mark.asyncio
    async def test_get_compliance_success_or_unavailable(self, handler):
        """Compliance endpoint returns 200 or 503 depending on verticals.config import."""
        mock_reg = _make_mock_registry()
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle("/api/verticals/healthcare/compliance", {}, None)
        assert _status(result) in (200, 503)

    @pytest.mark.asyncio
    async def test_get_compliance_not_found(self, handler):
        mock_reg = _make_mock_registry()
        mock_reg.get_config.return_value = None
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = handler._get_compliance("nonexistent", {})
        assert _status(result) in (404, 503)

    @pytest.mark.asyncio
    async def test_get_compliance_no_registry(self, handler):
        with patch.object(handler, "_get_registry", return_value=None):
            result = await handler.handle("/api/verticals/healthcare/compliance", {}, None)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_get_compliance_invalid_id(self, handler):
        result = await handler.handle("/api/verticals/bad id!!/compliance", {}, None)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_get_compliance_key_error(self, handler):
        mock_reg = _make_mock_registry()
        mock_reg.get_config.side_effect = KeyError("bad")
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = handler._get_compliance("healthcare", {})
        assert _status(result) in (400, 503)

    @pytest.mark.asyncio
    async def test_get_compliance_runtime_error(self, handler):
        mock_reg = _make_mock_registry()
        mock_reg.get_config.side_effect = RuntimeError("crash")
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = handler._get_compliance("healthcare", {})
        assert _status(result) in (500, 503)


# ============================================================================
# GET /api/verticals/suggest
# ============================================================================


class TestSuggestVertical:
    @pytest.mark.asyncio
    async def test_suggest_success(self, handler):
        mock_reg = _make_mock_registry()
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle(
                "/api/verticals/suggest", {"task": "Analyze patient records"}, None
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["suggestion"]["vertical_id"] == "healthcare"
        assert body["task"] == "Analyze patient records"

    @pytest.mark.asyncio
    async def test_suggest_no_match(self, handler):
        mock_reg = _make_mock_registry()
        mock_reg.get_for_task.return_value = None
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle(
                "/api/verticals/suggest", {"task": "Random task"}, None
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["suggestion"] is None
        assert "No vertical matches" in body["message"]

    @pytest.mark.asyncio
    async def test_suggest_missing_task(self, handler):
        mock_reg = _make_mock_registry()
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle("/api/verticals/suggest", {}, None)
        assert _status(result) == 400
        assert "required" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_suggest_empty_task(self, handler):
        mock_reg = _make_mock_registry()
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle("/api/verticals/suggest", {"task": ""}, None)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_suggest_whitespace_task(self, handler):
        mock_reg = _make_mock_registry()
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle("/api/verticals/suggest", {"task": "   "}, None)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_suggest_task_too_long(self, handler):
        mock_reg = _make_mock_registry()
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle(
                "/api/verticals/suggest", {"task": "x" * 100_001}, None
            )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_suggest_no_registry(self, handler):
        with patch.object(handler, "_get_registry", return_value=None):
            result = await handler.handle("/api/verticals/suggest", {"task": "test"}, None)
        assert _status(result) in (400, 503)

    @pytest.mark.asyncio
    async def test_suggest_v1_path(self, handler):
        mock_reg = _make_mock_registry()
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle(
                "/api/v1/verticals/suggest", {"task": "Analyze data"}, None
            )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_suggest_key_error(self, handler):
        mock_reg = _make_mock_registry()
        mock_reg.get_for_task.side_effect = KeyError("bad")
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle("/api/verticals/suggest", {"task": "test"}, None)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_suggest_runtime_error(self, handler):
        mock_reg = _make_mock_registry()
        mock_reg.get_for_task.side_effect = RuntimeError("crash")
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle("/api/verticals/suggest", {"task": "test"}, None)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_suggest_spec_is_none(self, handler):
        """When get_for_task returns a vertical but get() returns None."""
        mock_reg = _make_mock_registry()
        mock_reg.get_for_task.return_value = "healthcare"
        mock_reg.get.return_value = None
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle("/api/verticals/suggest", {"task": "test"}, None)
        assert _status(result) == 200
        body = _body(result)
        assert body["suggestion"]["vertical_id"] == "healthcare"


# ============================================================================
# POST /api/verticals/:id/debate
# ============================================================================


class TestCreateDebate:
    """Test creating a debate via vertical endpoint.

    Since _create_debate uses local imports (from aragora.core, from aragora.debate),
    we patch at the source modules rather than on the handler module.
    """

    @pytest.mark.asyncio
    async def test_create_debate_success(self, handler):
        mock_reg = _make_mock_registry()
        mock_handler = _MockHTTPHandler(
            body={"topic": "Patient diagnosis approach", "rounds": 3},
            method="POST",
        )
        mock_result = MagicMock()
        mock_result.debate_id = "debate-123"
        mock_result.consensus_reached = True
        mock_result.final_answer = "Answer"
        mock_result.confidence = 0.95

        with patch.object(handler, "_get_registry", return_value=mock_reg):
            with patch("aragora.debate.orchestrator.Arena") as MockArena:
                MockArena.return_value.run = AsyncMock(return_value=mock_result)
                result = await handler.handle(
                    "/api/verticals/healthcare/debate", {}, mock_handler
                )
        assert _status(result) == 200
        body = _body(result)
        assert body["debate_id"] == "debate-123"
        assert body["vertical_id"] == "healthcare"
        assert body["consensus_reached"] is True

    @pytest.mark.asyncio
    async def test_create_debate_no_registry(self, handler):
        mock_handler = _MockHTTPHandler(body={"topic": "Test"}, method="POST")
        with patch.object(handler, "_get_registry", return_value=None):
            result = await handler.handle(
                "/api/verticals/healthcare/debate", {}, mock_handler
            )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_create_debate_vertical_not_found(self, handler):
        mock_reg = _make_mock_registry()
        mock_reg.is_registered.return_value = False
        mock_handler = _MockHTTPHandler(body={"topic": "Test"}, method="POST")
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle(
                "/api/verticals/nonexistent/debate", {}, mock_handler
            )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_create_debate_missing_topic(self, handler):
        mock_reg = _make_mock_registry()
        mock_handler = _MockHTTPHandler(body={}, method="POST")
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle(
                "/api/verticals/healthcare/debate", {}, mock_handler
            )
        assert _status(result) == 400
        assert "required" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_create_debate_empty_topic(self, handler):
        mock_reg = _make_mock_registry()
        mock_handler = _MockHTTPHandler(body={"topic": ""}, method="POST")
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle(
                "/api/verticals/healthcare/debate", {}, mock_handler
            )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_debate_invalid_rounds_zero(self, handler):
        mock_reg = _make_mock_registry()
        mock_handler = _MockHTTPHandler(
            body={"topic": "Test", "rounds": 0}, method="POST"
        )
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle(
                "/api/verticals/healthcare/debate", {}, mock_handler
            )
        assert _status(result) == 400
        assert "rounds" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_create_debate_invalid_rounds_over_20(self, handler):
        mock_reg = _make_mock_registry()
        mock_handler = _MockHTTPHandler(
            body={"topic": "Test", "rounds": 21}, method="POST"
        )
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle(
                "/api/verticals/healthcare/debate", {}, mock_handler
            )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_debate_invalid_rounds_string(self, handler):
        mock_reg = _make_mock_registry()
        mock_handler = _MockHTTPHandler(
            body={"topic": "Test", "rounds": "not_a_number"}, method="POST"
        )
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle(
                "/api/verticals/healthcare/debate", {}, mock_handler
            )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_debate_invalid_agent_name(self, handler):
        mock_reg = _make_mock_registry()
        mock_handler = _MockHTTPHandler(
            body={"topic": "Test", "agent_name": "agent with spaces"}, method="POST"
        )
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle(
                "/api/verticals/healthcare/debate", {}, mock_handler
            )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_debate_too_many_additional_agents(self, handler):
        mock_reg = _make_mock_registry()
        mock_handler = _MockHTTPHandler(
            body={"topic": "Test", "additional_agents": ["a"] * 11}, method="POST"
        )
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle(
                "/api/verticals/healthcare/debate", {}, mock_handler
            )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_debate_invalid_body(self, handler):
        mock_reg = _make_mock_registry()
        mock_handler = _MockHTTPHandler(method="POST")
        mock_handler.headers["Content-Length"] = "99999999999"
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle(
                "/api/verticals/healthcare/debate", {}, mock_handler
            )
        assert _status(result) == 400
        assert "body" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_create_debate_import_error(self, handler):
        """ImportError for debate infrastructure returns 503."""
        mock_reg = _make_mock_registry()
        mock_handler = _MockHTTPHandler(body={"topic": "Test topic"}, method="POST")
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            with patch.dict(sys.modules, {"aragora.core": None}):
                result = await handler.handle(
                    "/api/verticals/healthcare/debate", {}, mock_handler
                )
        assert _status(result) in (400, 500, 503)

    @pytest.mark.asyncio
    async def test_create_debate_invalid_id(self, handler):
        mock_handler = _MockHTTPHandler(body={"topic": "Test"}, method="POST")
        result = await handler.handle(
            "/api/verticals/bad id!!/debate", {}, mock_handler
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_debate_with_additional_agents_strings(self, handler):
        mock_reg = _make_mock_registry()
        mock_handler = _MockHTTPHandler(
            body={"topic": "Test topic", "additional_agents": ["anthropic-api"]},
            method="POST",
        )
        mock_result = MagicMock()
        mock_result.debate_id = "d-1"
        mock_result.consensus_reached = False
        mock_result.final_answer = None
        mock_result.confidence = 0.5

        mock_agent = MagicMock()
        mock_agent.name = "test-agent"

        with patch.object(handler, "_get_registry", return_value=mock_reg):
            with patch("aragora.debate.orchestrator.Arena") as MockArena:
                with patch("aragora.agents.base.create_agent", return_value=mock_agent):
                    MockArena.return_value.run = AsyncMock(return_value=mock_result)
                    result = await handler.handle(
                        "/api/verticals/healthcare/debate", {}, mock_handler
                    )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_create_debate_with_additional_agents_dicts(self, handler):
        mock_reg = _make_mock_registry()
        mock_handler = _MockHTTPHandler(
            body={
                "topic": "Test topic",
                "additional_agents": [
                    {"type": "openai-api", "name": "gpt-agent", "role": "critic"}
                ],
            },
            method="POST",
        )
        mock_result = MagicMock()
        mock_result.debate_id = "d-2"
        mock_result.consensus_reached = True
        mock_result.final_answer = "Yes"
        mock_result.confidence = 0.8

        mock_agent = MagicMock()
        mock_agent.name = "gpt-agent"

        with patch.object(handler, "_get_registry", return_value=mock_reg):
            with patch("aragora.debate.orchestrator.Arena") as MockArena:
                with patch("aragora.agents.base.create_agent", return_value=mock_agent):
                    MockArena.return_value.run = AsyncMock(return_value=mock_result)
                    result = await handler.handle(
                        "/api/verticals/healthcare/debate", {}, mock_handler
                    )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_create_debate_value_error(self, handler):
        mock_reg = _make_mock_registry()
        mock_reg.create_specialist.side_effect = ValueError("bad config")
        mock_handler = _MockHTTPHandler(body={"topic": "Test topic"}, method="POST")
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle(
                "/api/verticals/healthcare/debate", {}, mock_handler
            )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_debate_runtime_error(self, handler):
        mock_reg = _make_mock_registry()
        mock_reg.create_specialist.side_effect = RuntimeError("crash")
        mock_handler = _MockHTTPHandler(body={"topic": "Test topic"}, method="POST")
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle(
                "/api/verticals/healthcare/debate", {}, mock_handler
            )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_create_debate_negative_rounds(self, handler):
        mock_reg = _make_mock_registry()
        mock_handler = _MockHTTPHandler(
            body={"topic": "Test", "rounds": -1}, method="POST"
        )
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle(
                "/api/verticals/healthcare/debate", {}, mock_handler
            )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_debate_additional_agents_not_list(self, handler):
        mock_reg = _make_mock_registry()
        mock_handler = _MockHTTPHandler(
            body={"topic": "Test", "additional_agents": "not-a-list"}, method="POST"
        )
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle(
                "/api/verticals/healthcare/debate", {}, mock_handler
            )
        assert _status(result) == 400


# ============================================================================
# POST /api/verticals/:id/agent
# ============================================================================


class TestCreateAgent:
    @pytest.mark.asyncio
    async def test_create_agent_success(self, handler):
        mock_reg = _make_mock_registry()
        mock_handler = _MockHTTPHandler(
            body={"name": "my-agent", "model": "claude-3", "role": "specialist"},
            method="POST",
        )
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle(
                "/api/verticals/healthcare/agent", {}, mock_handler
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["vertical_id"] == "healthcare"
        assert "agent" in body
        assert "message" in body

    @pytest.mark.asyncio
    async def test_create_agent_default_name(self, handler):
        mock_reg = _make_mock_registry()
        mock_handler = _MockHTTPHandler(body={}, method="POST")
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle(
                "/api/verticals/healthcare/agent", {}, mock_handler
            )
        assert _status(result) == 200
        call_kwargs = mock_reg.create_specialist.call_args
        assert call_kwargs.kwargs.get("name") == "healthcare-agent"

    @pytest.mark.asyncio
    async def test_create_agent_no_registry(self, handler):
        mock_handler = _MockHTTPHandler(body={}, method="POST")
        with patch.object(handler, "_get_registry", return_value=None):
            result = await handler.handle(
                "/api/verticals/healthcare/agent", {}, mock_handler
            )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_create_agent_vertical_not_found(self, handler):
        mock_reg = _make_mock_registry()
        mock_reg.is_registered.return_value = False
        mock_handler = _MockHTTPHandler(body={}, method="POST")
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle(
                "/api/verticals/nonexistent/agent", {}, mock_handler
            )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_create_agent_invalid_name(self, handler):
        mock_reg = _make_mock_registry()
        mock_handler = _MockHTTPHandler(
            body={"name": "bad name with spaces"}, method="POST"
        )
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle(
                "/api/verticals/healthcare/agent", {}, mock_handler
            )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_agent_name_too_long(self, handler):
        mock_reg = _make_mock_registry()
        mock_handler = _MockHTTPHandler(body={"name": "x" * 101}, method="POST")
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle(
                "/api/verticals/healthcare/agent", {}, mock_handler
            )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_agent_invalid_body(self, handler):
        mock_reg = _make_mock_registry()
        mock_handler = _MockHTTPHandler(method="POST")
        mock_handler.headers["Content-Length"] = "99999999999"
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle(
                "/api/verticals/healthcare/agent", {}, mock_handler
            )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_agent_value_error(self, handler):
        mock_reg = _make_mock_registry()
        mock_reg.create_specialist.side_effect = ValueError("bad")
        mock_handler = _MockHTTPHandler(body={}, method="POST")
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle(
                "/api/verticals/healthcare/agent", {}, mock_handler
            )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_agent_runtime_error(self, handler):
        mock_reg = _make_mock_registry()
        mock_reg.create_specialist.side_effect = RuntimeError("crash")
        mock_handler = _MockHTTPHandler(body={}, method="POST")
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle(
                "/api/verticals/healthcare/agent", {}, mock_handler
            )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_create_agent_invalid_vertical_id(self, handler):
        mock_handler = _MockHTTPHandler(body={}, method="POST")
        result = await handler.handle(
            "/api/verticals/bad id!!/agent", {}, mock_handler
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_agent_type_error(self, handler):
        mock_reg = _make_mock_registry()
        mock_reg.create_specialist.side_effect = TypeError("bad type")
        mock_handler = _MockHTTPHandler(body={}, method="POST")
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle(
                "/api/verticals/healthcare/agent", {}, mock_handler
            )
        assert _status(result) == 400


# ============================================================================
# PUT /api/verticals/:id/config
# ============================================================================


class TestUpdateConfig:
    """Test updating vertical configuration.

    The _update_config method imports ToolConfig/ModelConfig/ComplianceConfig/ComplianceLevel
    locally from aragora.verticals.config. We patch at the source module.
    """

    @pytest.mark.asyncio
    async def test_update_tools(self, handler):
        mock_reg = _make_mock_registry()
        mock_handler = _MockHTTPHandler(
            body={"tools": [{"name": "new-tool", "description": "A new tool", "enabled": True}]},
            method="PUT",
        )
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            with patch(
                "aragora.verticals.config.ToolConfig",
                side_effect=lambda **kw: _MockToolConfig(kw.get("name", "tool")),
            ):
                result = await handler.handle(
                    "/api/verticals/healthcare/config", {}, mock_handler
                )
        assert _status(result) == 200
        body = _body(result)
        assert "tools" in body["updated_fields"]

    @pytest.mark.asyncio
    async def test_update_model_config(self, handler):
        mock_reg = _make_mock_registry()
        mock_handler = _MockHTTPHandler(
            body={"model_config": {"temperature": 0.5, "max_tokens": 2048}},
            method="PUT",
        )
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            with patch(
                "aragora.verticals.config.ModelConfig",
                return_value=_MockModelConfig(),
            ):
                result = await handler.handle(
                    "/api/verticals/healthcare/config", {}, mock_handler
                )
        assert _status(result) == 200
        body = _body(result)
        assert "model_config" in body["updated_fields"]

    @pytest.mark.asyncio
    async def test_update_compliance_frameworks(self, handler):
        mock_reg = _make_mock_registry()
        mock_handler = _MockHTTPHandler(
            body={"compliance_frameworks": [{"framework": "HIPAA", "level": "required"}]},
            method="PUT",
        )
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            with patch(
                "aragora.verticals.config.ComplianceConfig",
                return_value=_MockComplianceConfig(),
            ):
                result = await handler.handle(
                    "/api/verticals/healthcare/config", {}, mock_handler
                )
        assert _status(result) == 200
        body = _body(result)
        assert "compliance_frameworks" in body["updated_fields"]

    @pytest.mark.asyncio
    async def test_update_no_registry(self, handler):
        mock_handler = _MockHTTPHandler(body={"tools": []}, method="PUT")
        with patch.object(handler, "_get_registry", return_value=None):
            result = await handler.handle(
                "/api/verticals/healthcare/config", {}, mock_handler
            )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_update_vertical_not_registered(self, handler):
        mock_reg = _make_mock_registry()
        mock_reg.is_registered.return_value = False
        mock_handler = _MockHTTPHandler(body={"tools": []}, method="PUT")
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle(
                "/api/verticals/nonexistent/config", {}, mock_handler
            )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_update_invalid_body(self, handler):
        mock_reg = _make_mock_registry()
        mock_handler = _MockHTTPHandler(method="PUT")
        mock_handler.headers["Content-Length"] = "99999999999"
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle(
                "/api/verticals/healthcare/config", {}, mock_handler
            )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_update_no_valid_fields(self, handler):
        mock_reg = _make_mock_registry()
        mock_handler = _MockHTTPHandler(body={"invalid_field": "value"}, method="PUT")
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle(
                "/api/verticals/healthcare/config", {}, mock_handler
            )
        assert _status(result) in (400, 503)

    @pytest.mark.asyncio
    async def test_update_tools_not_a_list(self, handler):
        mock_reg = _make_mock_registry()
        mock_handler = _MockHTTPHandler(body={"tools": "not-a-list"}, method="PUT")
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle(
                "/api/verticals/healthcare/config", {}, mock_handler
            )
        assert _status(result) in (400, 503)

    @pytest.mark.asyncio
    async def test_update_tools_too_many(self, handler):
        mock_reg = _make_mock_registry()
        many_tools = [{"name": f"tool-{i}"} for i in range(51)]
        mock_handler = _MockHTTPHandler(body={"tools": many_tools}, method="PUT")
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle(
                "/api/verticals/healthcare/config", {}, mock_handler
            )
        assert _status(result) in (400, 503)

    @pytest.mark.asyncio
    async def test_update_tool_name_too_long(self, handler):
        mock_reg = _make_mock_registry()
        mock_handler = _MockHTTPHandler(
            body={"tools": [{"name": "x" * 101}]}, method="PUT"
        )
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle(
                "/api/verticals/healthcare/config", {}, mock_handler
            )
        assert _status(result) in (400, 503)

    @pytest.mark.asyncio
    async def test_update_compliance_not_a_list(self, handler):
        mock_reg = _make_mock_registry()
        mock_handler = _MockHTTPHandler(
            body={"compliance_frameworks": "not-a-list"}, method="PUT"
        )
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle(
                "/api/verticals/healthcare/config", {}, mock_handler
            )
        assert _status(result) in (400, 503)

    @pytest.mark.asyncio
    async def test_update_compliance_too_many(self, handler):
        mock_reg = _make_mock_registry()
        many_fw = [{"framework": f"fw-{i}"} for i in range(21)]
        mock_handler = _MockHTTPHandler(
            body={"compliance_frameworks": many_fw}, method="PUT"
        )
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle(
                "/api/verticals/healthcare/config", {}, mock_handler
            )
        assert _status(result) in (400, 503)

    @pytest.mark.asyncio
    async def test_update_model_config_not_dict(self, handler):
        mock_reg = _make_mock_registry()
        mock_handler = _MockHTTPHandler(
            body={"model_config": "not-a-dict"}, method="PUT"
        )
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle(
                "/api/verticals/healthcare/config", {}, mock_handler
            )
        assert _status(result) in (400, 503)

    @pytest.mark.asyncio
    async def test_update_invalid_temperature(self, handler):
        mock_reg = _make_mock_registry()
        mock_handler = _MockHTTPHandler(
            body={"model_config": {"temperature": 3.0}}, method="PUT"
        )
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle(
                "/api/verticals/healthcare/config", {}, mock_handler
            )
        assert _status(result) in (400, 503)

    @pytest.mark.asyncio
    async def test_update_invalid_max_tokens(self, handler):
        mock_reg = _make_mock_registry()
        mock_handler = _MockHTTPHandler(
            body={"model_config": {"max_tokens": 0}}, method="PUT"
        )
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle(
                "/api/verticals/healthcare/config", {}, mock_handler
            )
        assert _status(result) in (400, 503)

    @pytest.mark.asyncio
    async def test_update_max_tokens_too_large(self, handler):
        mock_reg = _make_mock_registry()
        mock_handler = _MockHTTPHandler(
            body={"model_config": {"max_tokens": 200001}}, method="PUT"
        )
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle(
                "/api/verticals/healthcare/config", {}, mock_handler
            )
        assert _status(result) in (400, 503)

    @pytest.mark.asyncio
    async def test_update_invalid_vertical_id(self, handler):
        mock_handler = _MockHTTPHandler(body={"tools": []}, method="PUT")
        result = await handler.handle(
            "/api/verticals/bad id!!/config", {}, mock_handler
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_update_spec_not_found(self, handler):
        mock_reg = _make_mock_registry()
        mock_reg.get.return_value = None
        mock_handler = _MockHTTPHandler(
            body={"tools": [{"name": "t1"}]}, method="PUT"
        )
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle(
                "/api/verticals/healthcare/config", {}, mock_handler
            )
        assert _status(result) in (404, 503)

    @pytest.mark.asyncio
    async def test_update_negative_temperature(self, handler):
        mock_reg = _make_mock_registry()
        mock_handler = _MockHTTPHandler(
            body={"model_config": {"temperature": -0.5}}, method="PUT"
        )
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle(
                "/api/verticals/healthcare/config", {}, mock_handler
            )
        assert _status(result) in (400, 503)


# ============================================================================
# RBAC
# ============================================================================


class TestRBACEnforcement:
    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_post_requires_auth(self, handler):
        mock_handler = _MockHTTPHandler(body={"topic": "Test"}, method="POST")
        result = await handler.handle(
            "/api/verticals/healthcare/debate", {}, mock_handler
        )
        assert _status(result) == 401

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_put_requires_auth(self, handler):
        mock_handler = _MockHTTPHandler(body={"tools": []}, method="PUT")
        result = await handler.handle(
            "/api/verticals/healthcare/config", {}, mock_handler
        )
        assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_get_does_not_require_auth(self, handler):
        mock_reg = _make_mock_registry()
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle("/api/verticals", {}, None)
        assert _status(result) == 200


# ============================================================================
# Circuit breaker
# ============================================================================


class TestCircuitBreaker:
    def test_get_circuit_breaker(self):
        cb = get_verticals_circuit_breaker()
        assert cb is not None
        assert cb.name == "verticals"

    def test_reset_circuit_breaker(self):
        cb = get_verticals_circuit_breaker()
        for _ in range(5):
            cb.record_failure()
        assert cb.state == "open"
        reset_verticals_circuit_breaker()
        assert cb.state == "closed"

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_rejects(self, handler):
        cb = get_verticals_circuit_breaker()
        for _ in range(3):
            cb.record_failure()
        assert cb.state == "open"
        result = await handler.handle("/api/verticals", {}, None)
        assert _status(result) == 503
        assert "temporarily unavailable" in _body(result).get("error", "").lower()

    def test_circuit_breaker_status(self, handler):
        status = handler.get_circuit_breaker_status()
        assert "state" in status
        assert status["state"] == "closed"
        assert "failure_count" in status

    def test_registry_import_failure_records_failure(self, handler):
        with patch.dict(sys.modules, {"aragora.verticals": None}):
            reg = handler._get_registry_with_circuit_breaker()
        assert reg is None


# ============================================================================
# Route handling
# ============================================================================


class TestRouteHandling:
    @pytest.mark.asyncio
    async def test_unknown_path_returns_none(self, handler):
        result = await handler.handle("/api/unknown", {}, None)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_on_debate_path_returns_none(self, handler):
        mock_handler = _MockHTTPHandler(method="GET")
        result = await handler.handle(
            "/api/verticals/healthcare/debate", {}, mock_handler
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_method_not_matched(self, handler):
        mock_handler = _MockHTTPHandler(method="DELETE")
        result = await handler.handle(
            "/api/verticals/healthcare/tools", {}, mock_handler
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_post_on_tools_returns_none(self, handler):
        mock_handler = _MockHTTPHandler(method="POST")
        result = await handler.handle(
            "/api/verticals/healthcare/tools", {}, mock_handler
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_get_request_without_handler(self, handler):
        mock_reg = _make_mock_registry()
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle("/api/verticals", {}, None)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_handler_with_command_attribute(self, handler):
        mock_reg = _make_mock_registry()
        mock_handler = _MockHTTPHandler(method="GET")
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle("/api/verticals", {}, mock_handler)
        assert _status(result) == 200


# ============================================================================
# Registry helper
# ============================================================================


class TestRegistryHelper:
    def test_success_returns_registry(self, handler):
        with patch("aragora.verticals.VerticalRegistry") as mock_vr:
            result = handler._get_registry_with_circuit_breaker()
        assert result is mock_vr

    def test_import_error_returns_none(self, handler):
        with patch.dict(sys.modules, {"aragora.verticals": None}):
            result = handler._get_registry_with_circuit_breaker()
        assert result is None

    def test_get_registry_delegates(self, handler):
        with patch.object(
            handler, "_get_registry_with_circuit_breaker", return_value="mock_registry"
        ):
            result = handler._get_registry()
        assert result == "mock_registry"


# ============================================================================
# Edge cases
# ============================================================================


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_path_with_extra_segments(self, handler):
        result = await handler.handle(
            "/api/verticals/healthcare/tools/extra", {}, None
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_suggest_with_config_none(self, handler):
        mock_reg = _make_mock_registry()
        spec_no_config = MagicMock()
        spec_no_config.config = None
        spec_no_config.description = "test"
        mock_reg.get.return_value = spec_no_config
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            result = await handler.handle(
                "/api/verticals/suggest", {"task": "test task"}, None
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["suggestion"]["vertical_id"] == "healthcare"

    @pytest.mark.asyncio
    async def test_debate_default_agent_name(self, handler):
        mock_reg = _make_mock_registry()
        mock_handler = _MockHTTPHandler(body={"topic": "Test topic"}, method="POST")
        mock_result = MagicMock()
        mock_result.debate_id = "d-1"
        mock_result.consensus_reached = False
        mock_result.final_answer = None
        mock_result.confidence = 0.0

        with patch.object(handler, "_get_registry", return_value=mock_reg):
            with patch("aragora.debate.orchestrator.Arena") as MockArena:
                MockArena.return_value.run = AsyncMock(return_value=mock_result)
                result = await handler.handle(
                    "/api/verticals/healthcare/debate", {}, mock_handler
                )
        assert _status(result) == 200
        call_kwargs = mock_reg.create_specialist.call_args
        assert call_kwargs.kwargs.get("name") == "healthcare-specialist"

    @pytest.mark.asyncio
    async def test_debate_with_consensus_param(self, handler):
        mock_reg = _make_mock_registry()
        mock_handler = _MockHTTPHandler(
            body={"topic": "Test topic", "rounds": 5, "consensus": "unanimous"},
            method="POST",
        )
        mock_result = MagicMock()
        mock_result.debate_id = "d-3"
        mock_result.consensus_reached = True
        mock_result.final_answer = "Result"
        mock_result.confidence = 0.99

        with patch.object(handler, "_get_registry", return_value=mock_reg):
            with patch("aragora.debate.orchestrator.Arena") as MockArena:
                with patch("aragora.core.DebateProtocol") as MockDP:
                    MockArena.return_value.run = AsyncMock(return_value=mock_result)
                    result = await handler.handle(
                        "/api/verticals/healthcare/debate", {}, mock_handler
                    )
        assert _status(result) == 200
        MockDP.assert_called_once_with(rounds=5, consensus="unanimous")

    @pytest.mark.asyncio
    async def test_debate_uses_ctx_stores(self, handler):
        handler_with_ctx = VerticalsHandler(
            ctx={"document_store": "doc_store", "evidence_store": "ev_store"}
        )
        mock_reg = _make_mock_registry()
        mock_handler = _MockHTTPHandler(body={"topic": "Test topic"}, method="POST")
        mock_result = MagicMock()
        mock_result.debate_id = "d-4"
        mock_result.consensus_reached = False
        mock_result.final_answer = None
        mock_result.confidence = 0.0

        with patch.object(handler_with_ctx, "_get_registry", return_value=mock_reg):
            with patch("aragora.debate.orchestrator.Arena") as MockArena:
                MockArena.return_value.run = AsyncMock(return_value=mock_result)
                result = await handler_with_ctx.handle(
                    "/api/verticals/healthcare/debate", {}, mock_handler
                )
        assert _status(result) == 200
        arena_call = MockArena.call_args
        assert arena_call.kwargs.get("document_store") == "doc_store"
        assert arena_call.kwargs.get("evidence_store") == "ev_store"

    @pytest.mark.asyncio
    async def test_rounds_boundary_1(self, handler):
        mock_reg = _make_mock_registry()
        mock_handler = _MockHTTPHandler(
            body={"topic": "Test", "rounds": 1}, method="POST"
        )
        mock_result = MagicMock()
        mock_result.debate_id = "d-5"
        mock_result.consensus_reached = False
        mock_result.final_answer = None
        mock_result.confidence = 0.0

        with patch.object(handler, "_get_registry", return_value=mock_reg):
            with patch("aragora.debate.orchestrator.Arena") as MockArena:
                MockArena.return_value.run = AsyncMock(return_value=mock_result)
                result = await handler.handle(
                    "/api/verticals/healthcare/debate", {}, mock_handler
                )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_rounds_boundary_20(self, handler):
        mock_reg = _make_mock_registry()
        mock_handler = _MockHTTPHandler(
            body={"topic": "Test", "rounds": 20}, method="POST"
        )
        mock_result = MagicMock()
        mock_result.debate_id = "d-6"
        mock_result.consensus_reached = False
        mock_result.final_answer = None
        mock_result.confidence = 0.0

        with patch.object(handler, "_get_registry", return_value=mock_reg):
            with patch("aragora.debate.orchestrator.Arena") as MockArena:
                MockArena.return_value.run = AsyncMock(return_value=mock_result)
                result = await handler.handle(
                    "/api/verticals/healthcare/debate", {}, mock_handler
                )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_temperature_boundary_zero(self, handler):
        mock_reg = _make_mock_registry()
        mock_handler = _MockHTTPHandler(
            body={"model_config": {"temperature": 0, "max_tokens": 1024}},
            method="PUT",
        )
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            with patch(
                "aragora.verticals.config.ModelConfig", return_value=_MockModelConfig()
            ):
                result = await handler.handle(
                    "/api/verticals/healthcare/config", {}, mock_handler
                )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_temperature_boundary_two(self, handler):
        mock_reg = _make_mock_registry()
        mock_handler = _MockHTTPHandler(
            body={"model_config": {"temperature": 2, "max_tokens": 1024}},
            method="PUT",
        )
        with patch.object(handler, "_get_registry", return_value=mock_reg):
            with patch(
                "aragora.verticals.config.ModelConfig", return_value=_MockModelConfig()
            ):
                result = await handler.handle(
                    "/api/verticals/healthcare/config", {}, mock_handler
                )
        assert _status(result) == 200
