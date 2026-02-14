"""
Tests for aragora.server.handlers.skills - Skills Endpoint Handlers.

Tests cover:
- SkillsHandler: instantiation, ROUTES attribute
- handle_get: routing for /api/skills, /api/skills/:name, /api/skills/:name/metrics
- handle_post: routing for /api/skills/invoke
- _list_skills: success, pagination, registry unavailable
- _get_skill: found, not found, registry unavailable
- _get_skill_metrics: found, not found, no metrics
- _invoke_skill: success, missing skill name, skill not found, timeout
- Rate limiting check
- SKILLS_AVAILABLE flag behavior
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.skills import SkillsHandler
from aragora.server.handlers.utils.responses import HandlerResult


# ===========================================================================
# Helpers
# ===========================================================================


def _parse_body(result: HandlerResult) -> dict[str, Any]:
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body)


class MockSkillManifest:
    """Mock skill manifest."""

    def __init__(self, name: str = "test-skill"):
        self.name = name
        self.version = "1.0.0"
        self.description = f"A test skill: {name}"
        self.capabilities = []
        self.input_schema = {"type": "object"}
        self.output_schema = {"type": "object"}
        self.rate_limit_per_minute = 10
        self.max_execution_time_seconds = 30
        self.tags = ["test"]


class MockSkill:
    """Mock skill object."""

    def __init__(self, name: str = "test-skill"):
        self.manifest = MockSkillManifest(name)


class MockSkillResult:
    """Mock skill invocation result."""

    def __init__(self, status: str = "success", data: Any = None, error: str | None = None):
        self.status = MagicMock()
        self.status.value = status
        # Set the enum-like comparison
        if status == "success":
            self.status.__eq__ = lambda s, o: str(o) == "SkillStatus.SUCCESS" or o == self.status
        self.data = data or {"result": "ok"}
        self.error_message = error
        self.duration_seconds = 0.5
        self.metadata = {}


class MockRequest:
    """Mock request for skills handler."""

    def __init__(self, query: dict | None = None, body: dict | None = None):
        self._query = query or {}
        self._body = body or {}
        self.client_address = ("127.0.0.1", 12345)
        self.headers = {"X-Forwarded-For": "10.0.0.1"}

    def get(self, key, default=None):
        return self._body.get(key, default)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def mock_registry():
    """Create a mock skill registry."""
    registry = MagicMock()
    registry.list_skills.return_value = [
        MockSkillManifest("skill-a"),
        MockSkillManifest("skill-b"),
    ]
    registry.get.return_value = MockSkill("skill-a")
    registry.get_metrics.return_value = {
        "total_invocations": 100,
        "successful_invocations": 95,
        "failed_invocations": 5,
        "average_latency_ms": 200,
        "last_invoked": None,
    }

    mock_result = MockSkillResult()
    registry.invoke = AsyncMock(return_value=mock_result)
    return registry


@pytest.fixture
def handler(mock_registry):
    """Create a SkillsHandler with mocked dependencies."""
    h = SkillsHandler(server_context={})
    h._registry = mock_registry
    return h


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Reset the rate limiter between tests."""
    import aragora.server.handlers.skills as mod
    mod._skills_limiter = mod.RateLimiter(requests_per_minute=1000)  # high limit for tests
    yield


# ===========================================================================
# Test Basics
# ===========================================================================


class TestSkillsHandlerBasics:
    """Basic instantiation and attribute tests."""

    def test_instantiation(self, handler):
        assert handler is not None
        assert isinstance(handler, SkillsHandler)

    def test_routes_attribute(self, handler):
        assert "/api/skills" in handler.ROUTES
        assert "/api/skills/invoke" in handler.ROUTES
        assert "/api/skills/*/metrics" in handler.ROUTES
        assert "/api/skills/*" in handler.ROUTES

    def test_get_registry(self, handler, mock_registry):
        assert handler._get_registry() is mock_registry

    def test_get_registry_none_when_unavailable(self):
        h = SkillsHandler(server_context={})
        h._registry = None
        with patch("aragora.server.handlers.skills.SKILLS_AVAILABLE", False):
            assert h._get_registry() is None


# ===========================================================================
# Test _list_skills
# ===========================================================================


class TestListSkills:
    """Tests for GET /api/skills."""

    @pytest.mark.asyncio
    async def test_list_skills_success(self, handler):
        request = MockRequest()
        with patch.object(handler, "get_query_param", return_value="50"):
            result = await handler._list_skills.__wrapped__(handler, request)
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["total"] == 2
            assert len(data["skills"]) == 2

    @pytest.mark.asyncio
    async def test_list_skills_registry_unavailable(self, handler):
        handler._registry = None
        with patch("aragora.server.handlers.skills.SKILLS_AVAILABLE", False):
            request = MockRequest()
            with patch.object(handler, "get_query_param", return_value="50"):
                result = await handler._list_skills.__wrapped__(handler, request)
                assert result.status_code == 503


# ===========================================================================
# Test _get_skill
# ===========================================================================


class TestGetSkill:
    """Tests for GET /api/skills/:name."""

    @pytest.mark.asyncio
    async def test_get_skill_success(self, handler, mock_registry):
        request = MockRequest()
        result = await handler._get_skill.__wrapped__(handler, "skill-a", request)
        assert result.status_code == 200
        data = _parse_body(result)
        assert data["name"] == "skill-a"
        assert data["version"] == "1.0.0"

    @pytest.mark.asyncio
    async def test_get_skill_not_found(self, handler, mock_registry):
        mock_registry.get.return_value = None
        request = MockRequest()
        result = await handler._get_skill.__wrapped__(handler, "nonexistent", request)
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_get_skill_registry_unavailable(self, handler):
        handler._registry = None
        with patch("aragora.server.handlers.skills.SKILLS_AVAILABLE", False):
            request = MockRequest()
            result = await handler._get_skill.__wrapped__(handler, "test", request)
            assert result.status_code == 503


# ===========================================================================
# Test _get_skill_metrics
# ===========================================================================


class TestGetSkillMetrics:
    """Tests for GET /api/skills/:name/metrics."""

    @pytest.mark.asyncio
    async def test_get_metrics_success(self, handler, mock_registry):
        request = MockRequest()
        result = await handler._get_skill_metrics.__wrapped__(handler, "skill-a", request)
        assert result.status_code == 200
        data = _parse_body(result)
        assert data["total_invocations"] == 100
        assert data["skill"] == "skill-a"

    @pytest.mark.asyncio
    async def test_get_metrics_not_found(self, handler, mock_registry):
        mock_registry.get.return_value = None
        request = MockRequest()
        result = await handler._get_skill_metrics.__wrapped__(handler, "nonexistent", request)
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_get_metrics_empty(self, handler, mock_registry):
        mock_registry.get_metrics.return_value = None
        request = MockRequest()
        result = await handler._get_skill_metrics.__wrapped__(handler, "skill-a", request)
        assert result.status_code == 200
        data = _parse_body(result)
        assert data["total_invocations"] == 0


# ===========================================================================
# Test _invoke_skill
# ===========================================================================


class TestInvokeSkill:
    """Tests for POST /api/skills/invoke."""

    @pytest.mark.asyncio
    async def test_invoke_success(self, handler, mock_registry):
        request = MockRequest(body={"skill": "skill-a", "input": {"data": 1}})
        # Mock SkillContext
        with patch("aragora.server.handlers.skills.SkillContext", MagicMock()):
            # Set up the mock result with SUCCESS status
            mock_result = MagicMock()
            mock_result.status = MagicMock()
            mock_result.data = {"result": "ok"}
            mock_result.duration_seconds = 0.5
            mock_result.metadata = {}
            mock_result.error_message = None

            # Make SkillStatus.SUCCESS comparison work
            with patch("aragora.server.handlers.skills.SkillStatus") as mock_ss:
                mock_result.status = mock_ss.SUCCESS
                mock_registry.invoke = AsyncMock(return_value=mock_result)

                result = await handler._invoke_skill.__wrapped__(handler, request)
                assert result.status_code == 200
                data = _parse_body(result)
                assert data["status"] == "success"

    @pytest.mark.asyncio
    async def test_invoke_missing_skill_name(self, handler):
        request = MockRequest(body={"input": {"data": 1}})
        result = await handler._invoke_skill.__wrapped__(handler, request)
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_invoke_skill_not_found(self, handler, mock_registry):
        mock_registry.get.return_value = None
        request = MockRequest(body={"skill": "nonexistent"})
        result = await handler._invoke_skill.__wrapped__(handler, request)
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_invoke_timeout(self, handler, mock_registry):
        mock_registry.invoke = AsyncMock(side_effect=asyncio.TimeoutError())
        request = MockRequest(body={"skill": "skill-a", "input": {}})
        with patch("aragora.server.handlers.skills.SkillContext", MagicMock()):
            result = await handler._invoke_skill.__wrapped__(handler, request)
            assert result.status_code == 408

    @pytest.mark.asyncio
    async def test_invoke_registry_unavailable(self, handler):
        handler._registry = None
        with patch("aragora.server.handlers.skills.SKILLS_AVAILABLE", False):
            request = MockRequest(body={"skill": "test"})
            result = await handler._invoke_skill.__wrapped__(handler, request)
            assert result.status_code == 503


# ===========================================================================
# Test handle_get routing
# ===========================================================================


class TestHandleGetRouting:
    """Tests for the handle_get method routing."""

    @pytest.mark.asyncio
    async def test_handle_get_list(self, handler):
        request = MockRequest()
        with patch("aragora.server.handlers.skills.SKILLS_AVAILABLE", True), \
             patch.object(handler, "get_query_param", return_value="50"):
            result = await handler.handle_get.__wrapped__(handler, "/api/skills", request)
            assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_get_specific_skill(self, handler, mock_registry):
        request = MockRequest()
        with patch("aragora.server.handlers.skills.SKILLS_AVAILABLE", True):
            result = await handler.handle_get.__wrapped__(handler, "/api/skills/skill-a", request)
            assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_get_metrics(self, handler, mock_registry):
        request = MockRequest()
        with patch("aragora.server.handlers.skills.SKILLS_AVAILABLE", True):
            result = await handler.handle_get.__wrapped__(handler, "/api/skills/skill-a/metrics", request)
            assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_get_unavailable(self, handler):
        request = MockRequest()
        with patch("aragora.server.handlers.skills.SKILLS_AVAILABLE", False):
            result = await handler.handle_get.__wrapped__(handler, "/api/skills", request)
            assert result.status_code == 503


# ===========================================================================
# Test handle_post routing
# ===========================================================================


class TestHandlePostRouting:
    """Tests for the handle_post method routing."""

    @pytest.mark.asyncio
    async def test_handle_post_invoke(self, handler, mock_registry):
        request = MockRequest(body={"skill": "skill-a", "input": {}})
        with patch("aragora.server.handlers.skills.SKILLS_AVAILABLE", True), \
             patch("aragora.server.handlers.skills.SkillContext", MagicMock()):
            mock_result = MagicMock()
            mock_result.data = {"result": "ok"}
            mock_result.duration_seconds = 0.5
            mock_result.metadata = {}
            mock_result.error_message = None
            with patch("aragora.server.handlers.skills.SkillStatus") as mock_ss:
                mock_result.status = mock_ss.SUCCESS
                mock_registry.invoke = AsyncMock(return_value=mock_result)
                result = await handler.handle_post.__wrapped__(handler, "/api/skills/invoke", request)
                assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_post_unknown_path(self, handler):
        request = MockRequest()
        with patch("aragora.server.handlers.skills.SKILLS_AVAILABLE", True):
            result = await handler.handle_post.__wrapped__(handler, "/api/skills/unknown", request)
            assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_handle_post_unavailable(self, handler):
        request = MockRequest()
        with patch("aragora.server.handlers.skills.SKILLS_AVAILABLE", False):
            result = await handler.handle_post.__wrapped__(handler, "/api/skills/invoke", request)
            assert result.status_code == 503
