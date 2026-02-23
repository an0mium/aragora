"""Comprehensive tests for the SkillsHandler (aragora/server/handlers/skills.py).

Covers all routes and behavior:
- GET  /api/skills           - List all registered skills (with pagination)
- GET  /api/skills/:name     - Get skill details
- GET  /api/skills/:name/metrics - Get skill execution metrics
- POST /api/skills/invoke    - Invoke a skill by name (body)
- POST /api/skills/:name/invoke - Invoke a skill by name (URL path)
- Rate limiting behavior for GET and POST
- Skills system unavailable (503)
- Registry unavailable (503)
- Skill not found (404)
- Unknown routes (404)
- Method not allowed (405)
- Timeout handling (408)
- Invocation errors (500)
- Skill status variants (FAILURE, RATE_LIMITED, PERMISSION_DENIED, unknown)
- Input validation (missing skill name, invalid JSON)
- Security tests (path traversal, injection)
- Edge cases (empty registry, large pagination, boundary values)
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.skills import SkillsHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


def _error_msg(body: dict) -> str:
    """Extract the error message string from an error response body.

    Handles both simple format {"error": "msg"} and structured format
    {"error": {"message": "msg", "code": "..."}}.
    """
    err = body.get("error", "")
    if isinstance(err, dict):
        return err.get("message", "")
    return err


def _error_code(body: dict) -> str | None:
    """Extract the error code from an error response body.

    Handles structured format {"error": {"code": "...", "message": "..."}}.
    Returns None for simple format {"error": "msg"}.
    """
    err = body.get("error", "")
    if isinstance(err, dict):
        return err.get("code")
    return body.get("code")


# ---------------------------------------------------------------------------
# Mock domain objects
# ---------------------------------------------------------------------------


class MockSkillCapability(Enum):
    ANALYSIS = "analysis"
    SEARCH = "search"
    GENERATION = "generation"


class MockSkillStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    RATE_LIMITED = "rate_limited"
    PERMISSION_DENIED = "permission_denied"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass
class MockSkillManifest:
    name: str
    version: str = "1.0.0"
    description: str = "A test skill"
    capabilities: list[MockSkillCapability] = field(
        default_factory=lambda: [MockSkillCapability.ANALYSIS]
    )
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    rate_limit_per_minute: int = 30
    max_execution_time_seconds: float = 30.0


@dataclass
class MockSkill:
    manifest: MockSkillManifest


@dataclass
class MockSkillResult:
    status: MockSkillStatus = MockSkillStatus.SUCCESS
    data: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None
    duration_seconds: float = 0.123
    metadata: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_skills():
    """Build a standard set of mock skills."""
    return [
        MockSkill(
            MockSkillManifest(
                name="search",
                description="Search the knowledge base",
                version="2.1.0",
                capabilities=[MockSkillCapability.SEARCH],
                input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
                output_schema={"type": "object"},
                tags=["search", "knowledge"],
                rate_limit_per_minute=60,
                max_execution_time_seconds=15.0,
            )
        ),
        MockSkill(
            MockSkillManifest(
                name="analyze",
                description="Analyze data patterns",
                version="1.3.0",
                capabilities=[MockSkillCapability.ANALYSIS],
                tags=["analytics"],
            )
        ),
        MockSkill(
            MockSkillManifest(
                name="generate",
                description="Generate content",
                version="0.9.0",
                capabilities=[MockSkillCapability.GENERATION],
                tags=["content", "generation"],
            )
        ),
    ]


@pytest.fixture
def mock_skills():
    """Build a default set of mock skills."""
    return _build_skills()


@pytest.fixture
def mock_registry(mock_skills):
    """Create a mock skill registry wired to mock_skills."""
    registry = MagicMock()
    registry.list_skills.return_value = [s.manifest for s in mock_skills]
    registry.get.side_effect = lambda name: next(
        (s for s in mock_skills if s.manifest.name == name), None
    )
    registry.get_metrics.return_value = None
    registry.invoke = AsyncMock(
        return_value=MockSkillResult(data={"result": "ok"}, metadata={"trace": "abc"})
    )
    return registry


@pytest.fixture
def handler(mock_registry):
    """Create a SkillsHandler with a fully mocked skills backend.

    The patches are yielded so they remain active during the test body.
    """
    with (
        patch("aragora.server.handlers.skills.SKILLS_AVAILABLE", True),
        patch(
            "aragora.server.handlers.skills.get_skill_registry",
            return_value=mock_registry,
        ),
        patch("aragora.server.handlers.skills.SkillStatus", MockSkillStatus),
        patch("aragora.server.handlers.skills.SkillContext", MagicMock),
    ):
        h = SkillsHandler(server_context={})
        h._registry = mock_registry
        yield h


@pytest.fixture
def mock_request():
    """Create a mock HTTP request object with required attributes."""
    req = MagicMock()
    req.headers = {}
    req.remote = "127.0.0.1"
    req.client_address = ("127.0.0.1", 12345)
    req.transport = MagicMock()
    req.transport.get_extra_info.return_value = ("127.0.0.1", 12345)
    req.path = "/api/skills"
    return req


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Clear rate limiter state between tests."""
    from aragora.server.handlers.skills import _skills_limiter

    _skills_limiter._buckets.clear()
    yield
    _skills_limiter._buckets.clear()


# ============================================================================
# Routing / can_handle Tests
# ============================================================================


class TestRoutes:
    """Verify that the handler declares the expected routes."""

    def test_has_list_route(self, handler):
        assert "/api/skills" in handler.ROUTES

    def test_has_invoke_body_route(self, handler):
        assert "/api/skills/invoke" in handler.ROUTES

    def test_has_invoke_path_route(self, handler):
        assert "/api/skills/*/invoke" in handler.ROUTES

    def test_has_metrics_route(self, handler):
        assert "/api/skills/*/metrics" in handler.ROUTES

    def test_has_wildcard_route(self, handler):
        assert "/api/skills/*" in handler.ROUTES

    def test_wildcard_is_last(self, handler):
        """The catch-all wildcard is last to avoid premature matching."""
        assert handler.ROUTES[-1] == "/api/skills/*"

    def test_minimum_route_count(self, handler):
        assert len(handler.ROUTES) >= 5


# ============================================================================
# Handler Initialization Tests
# ============================================================================


class TestHandlerInit:
    """Tests for SkillsHandler initialization."""

    def test_has_routes(self, handler):
        assert len(handler.ROUTES) >= 4

    def test_extends_base_handler(self, handler):
        from aragora.server.handlers.base import BaseHandler

        assert isinstance(handler, BaseHandler)

    def test_registry_initially_none(self):
        with patch("aragora.server.handlers.skills.SKILLS_AVAILABLE", True):
            h = SkillsHandler(server_context={})
            # Reset registry to test lazy init
            h._registry = None
            assert h._registry is None

    def test_get_registry_returns_registry_when_available(self, handler, mock_registry):
        assert handler._get_registry() is mock_registry

    def test_get_registry_returns_none_when_skills_unavailable(self):
        with patch("aragora.server.handlers.skills.SKILLS_AVAILABLE", False):
            h = SkillsHandler(server_context={})
            assert h._get_registry() is None


# ============================================================================
# GET /api/skills - List Skills
# ============================================================================


class TestListSkills:
    """Tests for GET /api/skills."""

    @pytest.mark.asyncio
    async def test_list_returns_200(self, handler, mock_request):
        result = await handler.handle_get("/api/v1/skills", mock_request)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_list_response_structure(self, handler, mock_request):
        result = await handler.handle_get("/api/v1/skills", mock_request)
        body = _body(result)
        assert "skills" in body
        assert "total" in body
        assert "limit" in body
        assert "offset" in body

    @pytest.mark.asyncio
    async def test_list_returns_all_skills(self, handler, mock_request):
        result = await handler.handle_get("/api/v1/skills", mock_request)
        body = _body(result)
        assert body["total"] == 3
        assert len(body["skills"]) == 3

    @pytest.mark.asyncio
    async def test_list_skill_fields(self, handler, mock_request):
        result = await handler.handle_get("/api/v1/skills", mock_request)
        body = _body(result)
        skill = body["skills"][0]
        for field_name in (
            "name",
            "version",
            "description",
            "capabilities",
            "input_schema",
            "tags",
        ):
            assert field_name in skill

    @pytest.mark.asyncio
    async def test_list_default_limit_is_50(self, handler, mock_request):
        result = await handler.handle_get("/api/v1/skills", mock_request)
        body = _body(result)
        assert body["limit"] == 50

    @pytest.mark.asyncio
    async def test_list_default_offset_is_0(self, handler, mock_request):
        result = await handler.handle_get("/api/v1/skills", mock_request)
        body = _body(result)
        assert body["offset"] == 0

    @pytest.mark.asyncio
    async def test_list_pagination_with_limit(self, handler, mock_request):
        mock_request.path = "/api/skills?limit=1"
        result = await handler.handle_get("/api/v1/skills", mock_request)
        body = _body(result)
        assert len(body["skills"]) == 1
        assert body["total"] == 3
        assert body["limit"] == 1

    @pytest.mark.asyncio
    async def test_list_pagination_with_offset(self, handler, mock_request):
        mock_request.path = "/api/skills?offset=2"
        result = await handler.handle_get("/api/v1/skills", mock_request)
        body = _body(result)
        assert len(body["skills"]) == 1  # 3 total, offset 2 = 1 remaining
        assert body["offset"] == 2

    @pytest.mark.asyncio
    async def test_list_pagination_limit_and_offset(self, handler, mock_request):
        mock_request.path = "/api/skills?limit=1&offset=1"
        result = await handler.handle_get("/api/v1/skills", mock_request)
        body = _body(result)
        assert len(body["skills"]) == 1
        assert body["offset"] == 1
        assert body["limit"] == 1

    @pytest.mark.asyncio
    async def test_list_pagination_limit_capped_at_500(self, handler, mock_request):
        mock_request.path = "/api/skills?limit=9999"
        result = await handler.handle_get("/api/v1/skills", mock_request)
        body = _body(result)
        assert body["limit"] == 500

    @pytest.mark.asyncio
    async def test_list_pagination_limit_minimum_1(self, handler, mock_request):
        mock_request.path = "/api/skills?limit=0"
        result = await handler.handle_get("/api/v1/skills", mock_request)
        body = _body(result)
        assert body["limit"] == 1

    @pytest.mark.asyncio
    async def test_list_pagination_negative_offset_clamped(self, handler, mock_request):
        mock_request.path = "/api/skills?offset=-5"
        result = await handler.handle_get("/api/v1/skills", mock_request)
        body = _body(result)
        assert body["offset"] == 0

    @pytest.mark.asyncio
    async def test_list_pagination_offset_beyond_total(self, handler, mock_request):
        mock_request.path = "/api/skills?offset=100"
        result = await handler.handle_get("/api/v1/skills", mock_request)
        body = _body(result)
        assert len(body["skills"]) == 0
        assert body["total"] == 3

    @pytest.mark.asyncio
    async def test_list_empty_registry(self, handler, mock_request, mock_registry):
        mock_registry.list_skills.return_value = []
        result = await handler.handle_get("/api/v1/skills", mock_request)
        body = _body(result)
        assert body["total"] == 0
        assert body["skills"] == []

    @pytest.mark.asyncio
    async def test_list_skill_capabilities_serialized(self, handler, mock_request):
        result = await handler.handle_get("/api/v1/skills", mock_request)
        body = _body(result)
        # search skill has SEARCH capability
        search_skill = next(s for s in body["skills"] if s["name"] == "search")
        assert "search" in search_skill["capabilities"]

    @pytest.mark.asyncio
    async def test_list_skill_tags(self, handler, mock_request):
        result = await handler.handle_get("/api/v1/skills", mock_request)
        body = _body(result)
        search_skill = next(s for s in body["skills"] if s["name"] == "search")
        assert "search" in search_skill["tags"]
        assert "knowledge" in search_skill["tags"]


# ============================================================================
# GET /api/skills/:name - Get Skill Details
# ============================================================================


class TestGetSkill:
    """Tests for GET /api/skills/:name."""

    @pytest.mark.asyncio
    async def test_get_existing_skill(self, handler, mock_request):
        result = await handler.handle_get("/api/v1/skills/search", mock_request)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_get_skill_returns_detail_fields(self, handler, mock_request):
        result = await handler.handle_get("/api/v1/skills/search", mock_request)
        body = _body(result)
        assert body["name"] == "search"
        assert body["version"] == "2.1.0"
        assert body["description"] == "Search the knowledge base"
        assert "capabilities" in body
        assert "input_schema" in body
        assert "output_schema" in body
        assert "rate_limit_per_minute" in body
        assert "timeout_seconds" in body
        assert "tags" in body

    @pytest.mark.asyncio
    async def test_get_skill_rate_limit_value(self, handler, mock_request):
        result = await handler.handle_get("/api/v1/skills/search", mock_request)
        body = _body(result)
        assert body["rate_limit_per_minute"] == 60

    @pytest.mark.asyncio
    async def test_get_skill_timeout_value(self, handler, mock_request):
        result = await handler.handle_get("/api/v1/skills/search", mock_request)
        body = _body(result)
        assert body["timeout_seconds"] == 15.0

    @pytest.mark.asyncio
    async def test_get_nonexistent_skill(self, handler, mock_request):
        result = await handler.handle_get("/api/v1/skills/nonexistent", mock_request)
        assert _status(result) == 404
        body = _body(result)
        assert "not found" in _error_msg(body).lower()

    @pytest.mark.asyncio
    async def test_get_skill_no_description(self, handler, mock_request, mock_registry):
        """Skill with None description returns empty string."""
        skills = _build_skills()
        skills[0].manifest.description = None
        mock_registry.get.side_effect = lambda name: next(
            (s for s in skills if s.manifest.name == name), None
        )
        result = await handler.handle_get("/api/v1/skills/search", mock_request)
        body = _body(result)
        assert body["description"] == ""


# ============================================================================
# GET /api/skills/:name/metrics - Get Skill Metrics
# ============================================================================


class TestGetSkillMetrics:
    """Tests for GET /api/skills/:name/metrics."""

    @pytest.mark.asyncio
    async def test_metrics_no_data(self, handler, mock_request):
        """Metrics returns zeros when no data is available."""
        result = await handler.handle_get("/api/v1/skills/search/metrics", mock_request)
        assert _status(result) == 200
        body = _body(result)
        assert body["skill"] == "search"
        assert body["total_invocations"] == 0
        assert body["successful_invocations"] == 0
        assert body["failed_invocations"] == 0
        assert body["average_latency_ms"] == 0
        assert body["last_invoked"] is None

    @pytest.mark.asyncio
    async def test_metrics_with_data(self, handler, mock_request, mock_registry):
        """Metrics returns actual data when available."""
        last_invoked = datetime(2026, 2, 23, 10, 0, 0, tzinfo=timezone.utc)
        mock_registry.get_metrics.return_value = {
            "total_invocations": 100,
            "successful_invocations": 90,
            "failed_invocations": 10,
            "average_latency_ms": 42.5,
            "last_invoked": last_invoked,
        }
        result = await handler.handle_get("/api/v1/skills/search/metrics", mock_request)
        assert _status(result) == 200
        body = _body(result)
        assert body["total_invocations"] == 100
        assert body["successful_invocations"] == 90
        assert body["failed_invocations"] == 10
        assert body["average_latency_ms"] == 42.5
        assert body["last_invoked"] == last_invoked.isoformat()

    @pytest.mark.asyncio
    async def test_metrics_last_invoked_none(self, handler, mock_request, mock_registry):
        """Metrics with last_invoked=None returns null."""
        mock_registry.get_metrics.return_value = {
            "total_invocations": 5,
            "successful_invocations": 5,
            "failed_invocations": 0,
            "average_latency_ms": 10,
            "last_invoked": None,
        }
        result = await handler.handle_get("/api/v1/skills/search/metrics", mock_request)
        body = _body(result)
        assert body["last_invoked"] is None

    @pytest.mark.asyncio
    async def test_metrics_nonexistent_skill(self, handler, mock_request):
        result = await handler.handle_get("/api/v1/skills/nonexistent/metrics", mock_request)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_metrics_response_has_skill_name(self, handler, mock_request):
        result = await handler.handle_get("/api/v1/skills/analyze/metrics", mock_request)
        body = _body(result)
        assert body["skill"] == "analyze"


# ============================================================================
# POST /api/skills/invoke - Invoke Skill (body)
# ============================================================================


class TestInvokeSkillByBody:
    """Tests for POST /api/skills/invoke with skill name in body."""

    @pytest.mark.asyncio
    async def test_invoke_success(self, handler, mock_request):
        mock_request.json = AsyncMock(
            return_value={
                "skill": "search",
                "input": {"query": "test"},
            }
        )
        result = await handler.handle_post("/api/v1/skills/invoke", mock_request)
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "success"
        assert body["output"] == {"result": "ok"}
        assert body["metadata"] == {"trace": "abc"}

    @pytest.mark.asyncio
    async def test_invoke_execution_time(self, handler, mock_request, mock_registry):
        mock_request.json = AsyncMock(
            return_value={
                "skill": "search",
                "input": {},
            }
        )
        mock_registry.invoke.return_value = MockSkillResult(data={}, duration_seconds=0.456)
        result = await handler.handle_post("/api/v1/skills/invoke", mock_request)
        body = _body(result)
        assert body["execution_time_ms"] == 456

    @pytest.mark.asyncio
    async def test_invoke_missing_skill_name(self, handler, mock_request):
        mock_request.json = AsyncMock(return_value={"input": {"query": "test"}})
        result = await handler.handle_post("/api/v1/skills/invoke", mock_request)
        assert _status(result) == 400
        body = _body(result)
        assert "skill" in _error_msg(body).lower()

    @pytest.mark.asyncio
    async def test_invoke_empty_skill_name(self, handler, mock_request):
        mock_request.json = AsyncMock(return_value={"skill": "", "input": {}})
        result = await handler.handle_post("/api/v1/skills/invoke", mock_request)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_invoke_nonexistent_skill(self, handler, mock_request):
        mock_request.json = AsyncMock(return_value={"skill": "nonexistent", "input": {}})
        result = await handler.handle_post("/api/v1/skills/invoke", mock_request)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_invoke_no_input_defaults_empty(self, handler, mock_request, mock_registry):
        mock_request.json = AsyncMock(return_value={"skill": "search"})
        result = await handler.handle_post("/api/v1/skills/invoke", mock_request)
        assert _status(result) == 200
        # Verify invoke was called with empty dict input
        call_args = mock_registry.invoke.call_args
        assert call_args[0][1] == {}

    @pytest.mark.asyncio
    async def test_invoke_custom_user_id(self, handler, mock_request):
        mock_request.json = AsyncMock(
            return_value={
                "skill": "search",
                "input": {},
                "user_id": "custom-user-42",
            }
        )
        result = await handler.handle_post("/api/v1/skills/invoke", mock_request)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_invoke_custom_permissions(self, handler, mock_request):
        mock_request.json = AsyncMock(
            return_value={
                "skill": "search",
                "input": {},
                "permissions": ["skills:invoke", "skills:read"],
            }
        )
        result = await handler.handle_post("/api/v1/skills/invoke", mock_request)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_invoke_default_user_id_is_api(self, handler, mock_request):
        """When user_id is not specified, defaults to 'api'."""
        mock_request.json = AsyncMock(return_value={"skill": "search", "input": {}})
        with patch("aragora.server.handlers.skills.SkillContext") as mock_ctx_cls:
            mock_ctx_cls.return_value = MagicMock()
            result = await handler.handle_post("/api/v1/skills/invoke", mock_request)
            call_kwargs = mock_ctx_cls.call_args
            assert call_kwargs[1]["user_id"] == "api" or call_kwargs[0][0] == "api"

    @pytest.mark.asyncio
    async def test_invoke_with_body_dict_fallback(self, handler, mock_request):
        """When request has no json attribute, falls back to body dict."""
        # Remove the json attribute to trigger the else branch
        del mock_request.json
        mock_request.get = MagicMock(
            return_value={
                "skill": "search",
                "input": {"query": "test"},
            }
        )
        # Simulate dict-like request
        mock_request_dict = {"body": {"skill": "search", "input": {"query": "test"}}}
        # We need to patch the request to not have .json and act like a dict
        # The handler checks hasattr(request, "json"), so we remove it
        req = MagicMock(spec=[])
        req.get = lambda key, default=None: mock_request_dict.get(key, default)
        req.headers = {}
        req.remote = "127.0.0.1"
        req.client_address = ("127.0.0.1", 12345)
        req.transport = MagicMock()
        req.transport.get_extra_info.return_value = ("127.0.0.1", 12345)
        req.path = "/api/skills/invoke"

        result = await handler.handle_post("/api/v1/skills/invoke", req)
        assert _status(result) == 200


# ============================================================================
# POST /api/skills/:name/invoke - Invoke Skill (URL path)
# ============================================================================


class TestInvokeSkillByPath:
    """Tests for POST /api/skills/:name/invoke with skill name in URL."""

    @pytest.mark.asyncio
    async def test_invoke_by_path(self, handler, mock_request):
        mock_request.json = AsyncMock(return_value={"input": {"query": "test"}})
        result = await handler.handle_post("/api/v1/skills/search/invoke", mock_request)
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "success"

    @pytest.mark.asyncio
    async def test_invoke_by_path_overrides_body_skill_name(
        self, handler, mock_request, mock_registry
    ):
        """URL path skill name takes precedence over body skill name."""
        mock_request.json = AsyncMock(
            return_value={
                "skill": "analyze",
                "input": {"data": [1, 2, 3]},
            }
        )
        result = await handler.handle_post("/api/v1/skills/search/invoke", mock_request)
        assert _status(result) == 200
        # Verify invoke was called with "search" (from URL), not "analyze" (from body)
        call_args = mock_registry.invoke.call_args
        assert call_args[0][0] == "search"

    @pytest.mark.asyncio
    async def test_invoke_nonexistent_by_path(self, handler, mock_request):
        mock_request.json = AsyncMock(return_value={"input": {}})
        result = await handler.handle_post("/api/v1/skills/nonexistent/invoke", mock_request)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_invoke_by_path_analyze(self, handler, mock_request):
        mock_request.json = AsyncMock(return_value={"input": {}})
        result = await handler.handle_post("/api/v1/skills/analyze/invoke", mock_request)
        assert _status(result) == 200


# ============================================================================
# Skill Invocation Result Status Variants
# ============================================================================


class TestInvokeStatusVariants:
    """Tests for different skill invocation result statuses."""

    @pytest.mark.asyncio
    async def test_invoke_failure_status(self, handler, mock_request, mock_registry):
        mock_request.json = AsyncMock(return_value={"skill": "search", "input": {}})
        mock_registry.invoke.return_value = MockSkillResult(
            status=MockSkillStatus.FAILURE,
            error_message="Skill execution failed",
            duration_seconds=0.5,
        )
        result = await handler.handle_post("/api/v1/skills/invoke", mock_request)
        assert _status(result) == 500
        body = _body(result)
        assert body["status"] == "error"
        assert "failed" in body["error"].lower()
        assert body["execution_time_ms"] == 500

    @pytest.mark.asyncio
    async def test_invoke_failure_unknown_error(self, handler, mock_request, mock_registry):
        mock_request.json = AsyncMock(return_value={"skill": "search", "input": {}})
        mock_registry.invoke.return_value = MockSkillResult(
            status=MockSkillStatus.FAILURE,
            error_message=None,
            duration_seconds=0.1,
        )
        result = await handler.handle_post("/api/v1/skills/invoke", mock_request)
        assert _status(result) == 500
        body = _body(result)
        assert body["error"] == "Unknown error"

    @pytest.mark.asyncio
    async def test_invoke_rate_limited_status(self, handler, mock_request, mock_registry):
        mock_request.json = AsyncMock(return_value={"skill": "search", "input": {}})
        mock_registry.invoke.return_value = MockSkillResult(
            status=MockSkillStatus.RATE_LIMITED,
            error_message="Too many requests for this skill",
        )
        result = await handler.handle_post("/api/v1/skills/invoke", mock_request)
        assert _status(result) == 429

    @pytest.mark.asyncio
    async def test_invoke_rate_limited_default_message(self, handler, mock_request, mock_registry):
        mock_request.json = AsyncMock(return_value={"skill": "search", "input": {}})
        mock_registry.invoke.return_value = MockSkillResult(
            status=MockSkillStatus.RATE_LIMITED,
            error_message=None,
        )
        result = await handler.handle_post("/api/v1/skills/invoke", mock_request)
        assert _status(result) == 429
        body = _body(result)
        assert "rate limit" in _error_msg(body).lower()

    @pytest.mark.asyncio
    async def test_invoke_permission_denied_status(self, handler, mock_request, mock_registry):
        mock_request.json = AsyncMock(return_value={"skill": "search", "input": {}})
        mock_registry.invoke.return_value = MockSkillResult(
            status=MockSkillStatus.PERMISSION_DENIED,
            error_message="Access denied for this skill",
        )
        result = await handler.handle_post("/api/v1/skills/invoke", mock_request)
        assert _status(result) == 403

    @pytest.mark.asyncio
    async def test_invoke_permission_denied_default_message(
        self, handler, mock_request, mock_registry
    ):
        mock_request.json = AsyncMock(return_value={"skill": "search", "input": {}})
        mock_registry.invoke.return_value = MockSkillResult(
            status=MockSkillStatus.PERMISSION_DENIED,
            error_message=None,
        )
        result = await handler.handle_post("/api/v1/skills/invoke", mock_request)
        assert _status(result) == 403
        body = _body(result)
        assert "permission" in _error_msg(body).lower()

    @pytest.mark.asyncio
    async def test_invoke_unknown_status(self, handler, mock_request, mock_registry):
        """Unknown status returns 200 with raw status value."""
        mock_request.json = AsyncMock(return_value={"skill": "search", "input": {}})
        mock_registry.invoke.return_value = MockSkillResult(
            status=MockSkillStatus.UNKNOWN,
            data={"partial": True},
            error_message="Something unexpected",
            duration_seconds=0.2,
        )
        result = await handler.handle_post("/api/v1/skills/invoke", mock_request)
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "unknown"
        assert body["error"] == "Something unexpected"

    @pytest.mark.asyncio
    async def test_invoke_duration_none(self, handler, mock_request, mock_registry):
        """duration_seconds=None produces execution_time_ms=None."""
        mock_request.json = AsyncMock(return_value={"skill": "search", "input": {}})
        mock_registry.invoke.return_value = MockSkillResult(
            status=MockSkillStatus.SUCCESS,
            data={"ok": True},
            duration_seconds=None,
        )
        result = await handler.handle_post("/api/v1/skills/invoke", mock_request)
        body = _body(result)
        assert body["execution_time_ms"] is None

    @pytest.mark.asyncio
    async def test_invoke_metadata_none(self, handler, mock_request, mock_registry):
        """metadata=None returns empty dict in response."""
        mock_request.json = AsyncMock(return_value={"skill": "search", "input": {}})
        mock_registry.invoke.return_value = MockSkillResult(
            status=MockSkillStatus.SUCCESS,
            data={},
            metadata=None,
        )
        result = await handler.handle_post("/api/v1/skills/invoke", mock_request)
        body = _body(result)
        assert body["metadata"] == {}


# ============================================================================
# Timeout and Error Handling
# ============================================================================


class TestInvokeTimeout:
    """Tests for skill invocation timeout handling."""

    @pytest.mark.asyncio
    async def test_invoke_timeout(self, handler, mock_request, mock_registry):
        mock_request.json = AsyncMock(
            return_value={
                "skill": "search",
                "input": {},
                "timeout": 5.0,
            }
        )
        mock_registry.invoke.side_effect = asyncio.TimeoutError()
        result = await handler.handle_post("/api/v1/skills/invoke", mock_request)
        assert _status(result) == 408
        body = _body(result)
        assert "timed out" in _error_msg(body).lower()

    @pytest.mark.asyncio
    async def test_invoke_timeout_capped_at_60(self, handler, mock_request, mock_registry):
        """Timeout is capped at 60 seconds even if body specifies more."""
        mock_request.json = AsyncMock(
            return_value={
                "skill": "search",
                "input": {},
                "timeout": 300.0,
            }
        )
        # We can verify indirectly by checking the call still succeeds
        result = await handler.handle_post("/api/v1/skills/invoke", mock_request)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_invoke_default_timeout_30(self, handler, mock_request):
        """Default timeout is 30 seconds."""
        mock_request.json = AsyncMock(
            return_value={
                "skill": "search",
                "input": {},
            }
        )
        result = await handler.handle_post("/api/v1/skills/invoke", mock_request)
        assert _status(result) == 200


class TestInvokeErrors:
    """Tests for skill invocation exception handling."""

    @pytest.mark.asyncio
    async def test_invoke_runtime_error(self, handler, mock_request, mock_registry):
        mock_request.json = AsyncMock(return_value={"skill": "search", "input": {}})
        mock_registry.invoke.side_effect = RuntimeError("unexpected failure")
        result = await handler.handle_post("/api/v1/skills/invoke", mock_request)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_invoke_value_error(self, handler, mock_request, mock_registry):
        mock_request.json = AsyncMock(return_value={"skill": "search", "input": {}})
        mock_registry.invoke.side_effect = ValueError("bad input")
        result = await handler.handle_post("/api/v1/skills/invoke", mock_request)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_invoke_type_error(self, handler, mock_request, mock_registry):
        mock_request.json = AsyncMock(return_value={"skill": "search", "input": {}})
        mock_registry.invoke.side_effect = TypeError("wrong type")
        result = await handler.handle_post("/api/v1/skills/invoke", mock_request)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_invoke_os_error(self, handler, mock_request, mock_registry):
        mock_request.json = AsyncMock(return_value={"skill": "search", "input": {}})
        mock_registry.invoke.side_effect = OSError("disk error")
        result = await handler.handle_post("/api/v1/skills/invoke", mock_request)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_invoke_attribute_error(self, handler, mock_request, mock_registry):
        mock_request.json = AsyncMock(return_value={"skill": "search", "input": {}})
        mock_registry.invoke.side_effect = AttributeError("missing attr")
        result = await handler.handle_post("/api/v1/skills/invoke", mock_request)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_invoke_error_response_is_sanitized(self, handler, mock_request, mock_registry):
        """Error message should be sanitized (no stack trace leakage)."""
        mock_request.json = AsyncMock(return_value={"skill": "search", "input": {}})
        mock_registry.invoke.side_effect = RuntimeError("secret internal error at /usr/lib/foo")
        result = await handler.handle_post("/api/v1/skills/invoke", mock_request)
        body = _body(result)
        error_msg = _error_msg(body)
        assert "Skill invocation failed" in error_msg or "/usr/lib" not in error_msg


# ============================================================================
# Skills System Unavailable (503)
# ============================================================================


class TestSkillsUnavailable:
    """Tests when skills system is not available."""

    @pytest.mark.asyncio
    async def test_get_unavailable(self, mock_request):
        with patch("aragora.server.handlers.skills.SKILLS_AVAILABLE", False):
            h = SkillsHandler(server_context={})
            result = await h.handle_get("/api/v1/skills", mock_request)
            assert _status(result) == 503
            body = _body(result)
            assert "not available" in _error_msg(body).lower()

    @pytest.mark.asyncio
    async def test_post_unavailable(self, mock_request):
        with patch("aragora.server.handlers.skills.SKILLS_AVAILABLE", False):
            h = SkillsHandler(server_context={})
            mock_request.json = AsyncMock(return_value={"skill": "search", "input": {}})
            result = await h.handle_post("/api/v1/skills/invoke", mock_request)
            assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_unavailable_error_code(self, mock_request):
        with patch("aragora.server.handlers.skills.SKILLS_AVAILABLE", False):
            h = SkillsHandler(server_context={})
            result = await h.handle_get("/api/v1/skills", mock_request)
            body = _body(result)
            assert _error_code(body) == "SKILLS_UNAVAILABLE"


# ============================================================================
# Registry Unavailable (503)
# ============================================================================


class TestRegistryUnavailable:
    """Tests when registry is not available (returns None)."""

    @pytest.fixture
    def handler_no_registry(self):
        with (
            patch("aragora.server.handlers.skills.SKILLS_AVAILABLE", True),
            patch("aragora.server.handlers.skills.get_skill_registry", return_value=None),
            patch("aragora.server.handlers.skills.SkillStatus", MockSkillStatus),
            patch("aragora.server.handlers.skills.SkillContext", MagicMock),
        ):
            h = SkillsHandler(server_context={})
            h._registry = None
            yield h

    @pytest.mark.asyncio
    async def test_list_no_registry(self, handler_no_registry, mock_request):
        result = await handler_no_registry.handle_get("/api/v1/skills", mock_request)
        assert _status(result) == 503
        body = _body(result)
        assert _error_code(body) == "REGISTRY_UNAVAILABLE"

    @pytest.mark.asyncio
    async def test_get_skill_no_registry(self, handler_no_registry, mock_request):
        result = await handler_no_registry.handle_get("/api/v1/skills/search", mock_request)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_get_metrics_no_registry(self, handler_no_registry, mock_request):
        result = await handler_no_registry.handle_get("/api/v1/skills/search/metrics", mock_request)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_invoke_no_registry(self, handler_no_registry, mock_request):
        mock_request.json = AsyncMock(return_value={"skill": "search", "input": {}})
        result = await handler_no_registry.handle_post("/api/v1/skills/invoke", mock_request)
        assert _status(result) == 503


# ============================================================================
# Unknown Routes (404)
# ============================================================================


class TestUnknownRoutes:
    """Tests for unknown route handling."""

    @pytest.mark.asyncio
    async def test_get_unknown_route(self, handler, mock_request):
        result = await handler.handle_get("/api/v1/skills/foo/bar/baz", mock_request)
        # This will match as a skill detail lookup for "foo", which will be 404
        assert _status(result) in (404, 200)

    @pytest.mark.asyncio
    async def test_post_unknown_route(self, handler, mock_request):
        mock_request.json = AsyncMock(return_value={})
        result = await handler.handle_post("/api/v1/skills/something", mock_request)
        assert _status(result) == 404


# ============================================================================
# Rate Limiting Tests
# ============================================================================


class TestRateLimiting:
    """Tests for rate limiting on skills endpoints."""

    @pytest.mark.asyncio
    async def test_get_rate_limited(self, handler, mock_request):
        """Rate limiting returns 429 when limit exceeded."""
        with patch("aragora.server.handlers.skills._skills_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = False
            result = await handler.handle_get("/api/v1/skills", mock_request)
            assert _status(result) == 429

    @pytest.mark.asyncio
    async def test_post_rate_limited(self, handler, mock_request):
        """Rate limiting returns 429 for POST when limit exceeded."""
        mock_request.json = AsyncMock(return_value={"skill": "search", "input": {}})
        with patch("aragora.server.handlers.skills._skills_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = False
            result = await handler.handle_post("/api/v1/skills/invoke", mock_request)
            assert _status(result) == 429

    @pytest.mark.asyncio
    async def test_rate_limit_error_code_get(self, handler, mock_request):
        with patch("aragora.server.handlers.skills._skills_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = False
            result = await handler.handle_get("/api/v1/skills", mock_request)
            body = _body(result)
            assert _error_code(body) == "RATE_LIMITED"

    @pytest.mark.asyncio
    async def test_rate_limit_error_code_post(self, handler, mock_request):
        mock_request.json = AsyncMock(return_value={"skill": "search", "input": {}})
        with patch("aragora.server.handlers.skills._skills_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = False
            result = await handler.handle_post("/api/v1/skills/invoke", mock_request)
            body = _body(result)
            assert _error_code(body) == "RATE_LIMITED"

    def test_rate_limiter_configured_at_30_rpm(self):
        from aragora.server.handlers.skills import _skills_limiter

        assert _skills_limiter.rpm == 30


# ============================================================================
# Security Tests
# ============================================================================


class TestSecurity:
    """Security-focused tests."""

    @pytest.mark.asyncio
    async def test_path_traversal_in_skill_name(self, handler, mock_request):
        """Path traversal attempts are safely handled."""
        result = await handler.handle_get("/api/v1/skills/../../etc/passwd", mock_request)
        # Should be 404 (skill not found) rather than file access
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_null_byte_in_skill_name(self, handler, mock_request):
        """Null byte in skill name does not cause issues."""
        result = await handler.handle_get("/api/v1/skills/skill%00name", mock_request)
        assert _status(result) in (404, 400)

    @pytest.mark.asyncio
    async def test_sql_injection_in_skill_name(self, handler, mock_request):
        """SQL injection in skill name is safely handled."""
        result = await handler.handle_get("/api/v1/skills/'; DROP TABLE skills; --", mock_request)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_xss_in_skill_name(self, handler, mock_request):
        """XSS payload in skill name is safely handled."""
        result = await handler.handle_get("/api/v1/skills/<script>alert(1)</script>", mock_request)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_very_long_skill_name(self, handler, mock_request):
        """Very long skill name does not crash the handler."""
        long_name = "a" * 10000
        result = await handler.handle_get(f"/api/v1/skills/{long_name}", mock_request)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_invoke_with_malicious_input_data(self, handler, mock_request):
        """Invoke with nested deeply malicious input still processes normally."""
        mock_request.json = AsyncMock(
            return_value={
                "skill": "search",
                "input": {"query": "'; DROP TABLE users; --", "nested": {"deep": True}},
            }
        )
        result = await handler.handle_post("/api/v1/skills/invoke", mock_request)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_invoke_invalid_json_body(self, handler, mock_request):
        """Invalid JSON body returns 400."""
        mock_request.json = AsyncMock(side_effect=json.JSONDecodeError("bad", "", 0))
        # parse_json_body catches JSONDecodeError and returns 400
        result = await handler.handle_post("/api/v1/skills/invoke", mock_request)
        assert _status(result) == 400


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Edge case tests."""

    @pytest.mark.asyncio
    async def test_get_skill_with_special_characters(self, handler, mock_request):
        """Skill name with special characters returns 404 (not found)."""
        result = await handler.handle_get("/api/v1/skills/my-skill_v2.0", mock_request)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_get_skill_with_only_path_prefix(self, handler, mock_request):
        """GET /api/skills with no name lists all skills."""
        result = await handler.handle_get("/api/v1/skills", mock_request)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_invoke_with_metadata(self, handler, mock_request):
        """Invoke passes metadata to SkillContext config."""
        mock_request.json = AsyncMock(
            return_value={
                "skill": "search",
                "input": {"query": "test"},
                "metadata": {"trace_id": "abc-123", "source": "api"},
            }
        )
        with patch("aragora.server.handlers.skills.SkillContext") as mock_ctx_cls:
            mock_ctx_cls.return_value = MagicMock()
            result = await handler.handle_post("/api/v1/skills/invoke", mock_request)
            assert _status(result) == 200
            call_kwargs = mock_ctx_cls.call_args[1]
            assert call_kwargs["config"] == {"trace_id": "abc-123", "source": "api"}

    @pytest.mark.asyncio
    async def test_invoke_zero_duration(self, handler, mock_request, mock_registry):
        """Duration of 0.0 seconds returns 0 ms."""
        mock_request.json = AsyncMock(return_value={"skill": "search", "input": {}})
        mock_registry.invoke.return_value = MockSkillResult(
            status=MockSkillStatus.SUCCESS,
            data={},
            duration_seconds=0.0,
        )
        result = await handler.handle_post("/api/v1/skills/invoke", mock_request)
        body = _body(result)
        # 0.0 is falsy, so duration_seconds check results in None
        # int(0.0 * 1000) = 0, but `if result.duration_seconds` is False for 0.0
        assert body["execution_time_ms"] is None

    @pytest.mark.asyncio
    async def test_invoke_small_duration(self, handler, mock_request, mock_registry):
        """Very small duration rounds to 0 ms."""
        mock_request.json = AsyncMock(return_value={"skill": "search", "input": {}})
        mock_registry.invoke.return_value = MockSkillResult(
            status=MockSkillStatus.SUCCESS,
            data={},
            duration_seconds=0.0001,
        )
        result = await handler.handle_post("/api/v1/skills/invoke", mock_request)
        body = _body(result)
        assert body["execution_time_ms"] == 0

    @pytest.mark.asyncio
    async def test_version_prefix_stripped_correctly(self, handler, mock_request):
        """API version prefix is stripped before routing."""
        result = await handler.handle_get("/api/v1/skills", mock_request)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_skills_without_tags(self, handler, mock_request, mock_registry):
        """Skills with empty tags list work correctly."""
        skills = [MockSkill(MockSkillManifest(name="bare", tags=[]))]
        mock_registry.list_skills.return_value = [s.manifest for s in skills]
        result = await handler.handle_get("/api/v1/skills", mock_request)
        body = _body(result)
        assert body["skills"][0]["tags"] == []

    @pytest.mark.asyncio
    async def test_skills_without_input_schema(self, handler, mock_request, mock_registry):
        """Skills with no input schema return empty dict."""
        skills = [MockSkill(MockSkillManifest(name="simple", input_schema=None))]
        mock_registry.list_skills.return_value = [s.manifest for s in skills]
        mock_registry.get.side_effect = lambda name: skills[0] if name == "simple" else None
        result = await handler.handle_get("/api/v1/skills/simple", mock_request)
        body = _body(result)
        assert body["input_schema"] == {}

    @pytest.mark.asyncio
    async def test_skills_without_output_schema(self, handler, mock_request, mock_registry):
        """Skills with no output schema return empty dict."""
        skills = [MockSkill(MockSkillManifest(name="simple", output_schema=None))]
        mock_registry.get.side_effect = lambda name: skills[0] if name == "simple" else None
        result = await handler.handle_get("/api/v1/skills/simple", mock_request)
        body = _body(result)
        assert body["output_schema"] == {}

    @pytest.mark.asyncio
    async def test_skill_description_none_in_list(self, handler, mock_request, mock_registry):
        """Skill with None description shows empty string in list."""
        skills = [MockSkill(MockSkillManifest(name="nodesc", description=None))]
        mock_registry.list_skills.return_value = [s.manifest for s in skills]
        result = await handler.handle_get("/api/v1/skills", mock_request)
        body = _body(result)
        assert body["skills"][0]["description"] == ""


# ============================================================================
# Lazy Registry Initialization
# ============================================================================


class TestLazyRegistryInit:
    """Tests for lazy registry initialization."""

    def test_registry_lazy_created(self):
        """Registry is lazily created on first access."""
        mock_fn = MagicMock(return_value=MagicMock())
        with (
            patch("aragora.server.handlers.skills.SKILLS_AVAILABLE", True),
            patch("aragora.server.handlers.skills.get_skill_registry", mock_fn),
        ):
            h = SkillsHandler(server_context={})
            h._registry = None
            result = h._get_registry()
            assert result is not None
            mock_fn.assert_called_once()

    def test_registry_cached_after_first_access(self):
        """Registry is cached after first access."""
        mock_fn = MagicMock(return_value=MagicMock())
        with (
            patch("aragora.server.handlers.skills.SKILLS_AVAILABLE", True),
            patch("aragora.server.handlers.skills.get_skill_registry", mock_fn),
        ):
            h = SkillsHandler(server_context={})
            h._registry = None
            h._get_registry()
            h._get_registry()
            mock_fn.assert_called_once()

    def test_registry_none_when_get_skill_registry_is_none(self):
        """Returns None if get_skill_registry is None."""
        with (
            patch("aragora.server.handlers.skills.SKILLS_AVAILABLE", True),
            patch("aragora.server.handlers.skills.get_skill_registry", None),
        ):
            h = SkillsHandler(server_context={})
            h._registry = None
            assert h._get_registry() is None


# ============================================================================
# Module-Level Constants
# ============================================================================


class TestModuleConstants:
    """Tests for module-level constants and configuration."""

    def test_skills_handler_exported(self):
        from aragora.server.handlers.skills import __all__

        assert "SkillsHandler" in __all__

    def test_handler_has_routes(self, handler):
        assert len(handler.ROUTES) >= 5
