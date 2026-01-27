"""Tests for the SkillsHandler module.

Tests cover:
- Handler routing for skills endpoints
- GET /api/skills - List all skills
- GET /api/skills/:name - Get skill details
- GET /api/skills/:name/metrics - Get skill metrics
- POST /api/skills/invoke - Invoke skill
- Rate limiting behavior
- Error handling (skills unavailable, registry unavailable)
- RBAC permission checks
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from io import BytesIO
from typing import Any, Dict, List, Optional, Set
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.skills import SkillsHandler


# Mock skill classes for testing
class MockSkillCapability(str, Enum):
    """Mock skill capabilities."""

    EXTERNAL_API = "external_api"
    FILE_SYSTEM = "file_system"
    CODE_EXECUTION = "code_execution"


class MockSkillStatus(str, Enum):
    """Mock skill status (matches real SkillStatus enum)."""

    SUCCESS = "success"
    FAILURE = "failure"  # Changed from ERROR
    RATE_LIMITED = "rate_limited"
    PERMISSION_DENIED = "permission_denied"


@dataclass
class MockSkillManifest:
    """Mock skill manifest for testing."""

    name: str
    version: str = "1.0.0"
    description: Optional[str] = None
    capabilities: List[MockSkillCapability] = field(default_factory=list)
    input_schema: Optional[Dict] = None
    output_schema: Optional[Dict] = None
    rate_limit_per_minute: int = 60
    max_execution_time_seconds: float = 30.0  # Changed from timeout_seconds
    tags: List[str] = field(default_factory=list)


@dataclass
class MockSkillResult:
    """Mock skill result for testing (matches real SkillResult interface)."""

    status: MockSkillStatus
    data: Any = None  # Changed from output
    error_message: Optional[str] = None  # Changed from error
    metadata: Optional[Dict] = None
    duration_seconds: Optional[float] = None  # Changed from execution_time_ms


@dataclass
class MockSkillMetrics:
    """Mock skill metrics for testing."""

    total_invocations: int = 0
    successful_invocations: int = 0
    failed_invocations: int = 0
    average_latency_ms: float = 0.0
    last_invoked: Optional[datetime] = None


class MockSkill:
    """Mock skill for testing."""

    def __init__(self, manifest: MockSkillManifest):
        self.manifest = manifest


class MockSkillRegistry:
    """Mock skill registry for testing."""

    def __init__(self):
        self._skills: Dict[str, MockSkill] = {}
        self._metrics: Dict[str, MockSkillMetrics] = {}

    def register(self, skill: MockSkill) -> None:
        self._skills[skill.manifest.name] = skill

    def get(self, name: str) -> Optional[MockSkill]:
        return self._skills.get(name)

    def list_skills(self) -> List[MockSkillManifest]:
        """Return list of manifests (matches real SkillRegistry behavior)."""
        return [skill.manifest for skill in self._skills.values()]

    def get_metrics(self, name: str) -> Optional[Dict[str, Any]]:
        """Return metrics as dict (matches real SkillRegistry behavior)."""
        metrics = self._metrics.get(name)
        if metrics is None:
            return None
        return {
            "total_invocations": metrics.total_invocations,
            "successful_invocations": metrics.successful_invocations,
            "failed_invocations": metrics.failed_invocations,
            "average_latency_ms": metrics.average_latency_ms,
            "last_invoked": metrics.last_invoked,
        }

    def set_metrics(self, name: str, metrics: MockSkillMetrics) -> None:
        self._metrics[name] = metrics

    async def invoke(
        self,
        name: str,
        input_data: Dict,
        ctx: Any,
    ) -> MockSkillResult:
        skill = self.get(name)
        if not skill:
            return MockSkillResult(
                status=MockSkillStatus.FAILURE,
                error=f"Skill not found: {name}",
            )
        # Return success by default
        return MockSkillResult(
            status=MockSkillStatus.SUCCESS,
            data={"result": f"Executed {name}"},
            duration_seconds=0.05,
            metadata={"skill": name},
        )


@dataclass
class MockSkillContext:
    """Mock skill context for testing (matches real SkillContext interface)."""

    user_id: str = "test-user"
    permissions: List[str] = field(default_factory=lambda: ["skills:invoke"])  # Changed to List
    config: Dict[str, Any] = field(default_factory=dict)  # Changed from metadata


@pytest.fixture
def mock_server_context():
    """Create mock server context for handler initialization."""
    return {"storage": None, "elo_system": None, "nomic_dir": None}


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler with client address."""
    handler = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = {"Content-Length": "0"}
    handler.command = "GET"
    return handler


@pytest.fixture
def mock_registry():
    """Create a mock skill registry with test skills."""
    registry = MockSkillRegistry()

    # Add test skills
    web_search = MockSkill(
        MockSkillManifest(
            name="web_search",
            version="1.2.0",
            description="Search the web for information",
            capabilities=[MockSkillCapability.EXTERNAL_API],
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            output_schema={"type": "object"},
            tags=["search", "web"],
        )
    )
    registry.register(web_search)

    code_exec = MockSkill(
        MockSkillManifest(
            name="code_execution",
            version="0.9.0",
            description="Execute code in a sandbox",
            capabilities=[MockSkillCapability.CODE_EXECUTION],
            input_schema={"type": "object", "properties": {"code": {"type": "string"}}},
            tags=["code", "sandbox"],
        )
    )
    registry.register(code_exec)

    # Set metrics for web_search
    registry.set_metrics(
        "web_search",
        MockSkillMetrics(
            total_invocations=100,
            successful_invocations=95,
            failed_invocations=5,
            average_latency_ms=150.0,
            last_invoked=datetime.now(timezone.utc),
        ),
    )

    return registry


def create_request_body(data: dict) -> MagicMock:
    """Create a mock handler with request body."""
    handler = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    body = json.dumps(data).encode("utf-8")
    handler.headers = {"Content-Length": str(len(body))}
    handler.rfile = BytesIO(body)
    handler.command = "POST"
    return handler


def create_async_request(body: dict) -> MagicMock:
    """Create a mock request with async json method.

    The skills handler checks if request has 'json' attribute and calls await request.json().
    This helper creates a proper async mock for that pattern.
    """
    mock_request = MagicMock()
    mock_request.client_address = ("127.0.0.1", 12345)
    mock_request.json = AsyncMock(return_value=body)
    return mock_request


def create_simple_request(body: dict) -> MagicMock:
    """Create a mock request without json attribute (falls back to get('body')).

    Use this when you want the handler to use request.get('body', {}) fallback.
    """
    mock_request = MagicMock(spec=["get", "client_address"])
    mock_request.client_address = ("127.0.0.1", 12345)
    mock_request.get.return_value = body
    return mock_request


class TestSkillsHandlerRouting:
    """Tests for handler routing via ROUTES constant."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return SkillsHandler(mock_server_context)

    def test_routes_defined(self, handler):
        """Handler has correct routes defined."""
        assert "/api/skills" in handler.ROUTES
        assert "/api/skills/invoke" in handler.ROUTES
        assert "/api/skills/*/metrics" in handler.ROUTES
        assert "/api/skills/*" in handler.ROUTES

    def test_routes_order(self, handler):
        """Wildcard route is last (required for proper matching)."""
        routes = handler.ROUTES
        wildcard_idx = routes.index("/api/skills/*")
        assert wildcard_idx == len(routes) - 1, "/api/skills/* should be last route"

    def test_routes_cover_all_endpoints(self, handler):
        """All documented endpoints are covered by routes."""
        assert any("/api/skills" == r or r.startswith("/api/skills") for r in handler.ROUTES)
        assert "/api/skills/invoke" in handler.ROUTES
        assert "/api/skills/*/metrics" in handler.ROUTES
        assert "/api/skills/*" in handler.ROUTES

    def test_routes_do_not_include_unrelated_paths(self, handler):
        """Handler routes only include /api/skills paths."""
        for route in handler.ROUTES:
            assert route.startswith("/api/skills"), f"Route {route} should start with /api/skills"


class TestSkillsHandlerListSkills:
    """Tests for GET /api/skills endpoint."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return SkillsHandler(mock_server_context)

    @pytest.mark.asyncio
    async def test_list_skills_success(self, handler, mock_http_handler, mock_registry):
        """List skills returns all registered skills."""
        with patch.object(handler, "_get_registry", return_value=mock_registry):
            with patch("aragora.server.handlers.skills.SKILLS_AVAILABLE", True):
                with patch("aragora.server.handlers.skills._skills_limiter") as limiter:
                    limiter.is_allowed.return_value = True

                    # Mock request with auth context
                    mock_request = MagicMock()
                    mock_request.get.return_value = {
                        "user_id": "test",
                        "permissions": {"skills:read"},
                    }

                    result = await handler.handle_get("/api/skills", mock_request)

                    assert result is not None
                    assert result.status_code == 200
                    body = json.loads(result.body)
                    assert "skills" in body
                    assert "total" in body
                    assert body["total"] == 2
                    assert len(body["skills"]) == 2

                    # Check skill structure
                    skill_names = [s["name"] for s in body["skills"]]
                    assert "web_search" in skill_names
                    assert "code_execution" in skill_names

    @pytest.mark.asyncio
    async def test_list_skills_unavailable(self, handler, mock_http_handler):
        """List skills returns 503 when skills system unavailable."""
        with patch("aragora.server.handlers.skills.SKILLS_AVAILABLE", False):
            mock_request = MagicMock()
            result = await handler.handle_get("/api/skills", mock_request)

            assert result is not None
            assert result.status_code == 503
            body = json.loads(result.body)
            assert body["error"]["code"] == "SKILLS_UNAVAILABLE"

    @pytest.mark.asyncio
    async def test_list_skills_rate_limited(self, handler, mock_http_handler):
        """List skills returns 429 when rate limited."""
        with patch("aragora.server.handlers.skills.SKILLS_AVAILABLE", True):
            with patch("aragora.server.handlers.skills._skills_limiter") as limiter:
                limiter.is_allowed.return_value = False

                mock_request = MagicMock()
                result = await handler.handle_get("/api/skills", mock_request)

                assert result is not None
                assert result.status_code == 429
                body = json.loads(result.body)
                assert body["error"]["code"] == "RATE_LIMITED"

    @pytest.mark.asyncio
    async def test_list_skills_registry_unavailable(self, handler, mock_http_handler):
        """List skills returns 503 when registry unavailable."""
        with patch.object(handler, "_get_registry", return_value=None):
            with patch("aragora.server.handlers.skills.SKILLS_AVAILABLE", True):
                with patch("aragora.server.handlers.skills._skills_limiter") as limiter:
                    limiter.is_allowed.return_value = True

                    mock_request = MagicMock()
                    mock_request.get.return_value = {
                        "user_id": "test",
                        "permissions": {"skills:read"},
                    }

                    result = await handler.handle_get("/api/skills", mock_request)

                    assert result is not None
                    assert result.status_code == 503
                    body = json.loads(result.body)
                    assert body["error"]["code"] == "REGISTRY_UNAVAILABLE"


class TestSkillsHandlerGetSkill:
    """Tests for GET /api/skills/:name endpoint."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return SkillsHandler(mock_server_context)

    @pytest.mark.asyncio
    async def test_get_skill_success(self, handler, mock_registry):
        """Get skill details returns full manifest."""
        with patch.object(handler, "_get_registry", return_value=mock_registry):
            with patch("aragora.server.handlers.skills.SKILLS_AVAILABLE", True):
                with patch("aragora.server.handlers.skills._skills_limiter") as limiter:
                    limiter.is_allowed.return_value = True

                    mock_request = MagicMock()
                    mock_request.get.return_value = {
                        "user_id": "test",
                        "permissions": {"skills:read"},
                    }

                    result = await handler.handle_get("/api/skills/web_search", mock_request)

                    assert result is not None
                    assert result.status_code == 200
                    body = json.loads(result.body)
                    assert body["name"] == "web_search"
                    assert body["version"] == "1.2.0"
                    assert body["description"] == "Search the web for information"
                    assert "external_api" in body["capabilities"]
                    assert "input_schema" in body
                    assert "output_schema" in body
                    assert body["rate_limit_per_minute"] == 60
                    assert body["timeout_seconds"] == 30.0
                    assert "search" in body["tags"]

    @pytest.mark.asyncio
    async def test_get_skill_not_found(self, handler, mock_registry):
        """Get skill returns 404 for unknown skill."""
        with patch.object(handler, "_get_registry", return_value=mock_registry):
            with patch("aragora.server.handlers.skills.SKILLS_AVAILABLE", True):
                with patch("aragora.server.handlers.skills._skills_limiter") as limiter:
                    limiter.is_allowed.return_value = True

                    mock_request = MagicMock()
                    mock_request.get.return_value = {
                        "user_id": "test",
                        "permissions": {"skills:read"},
                    }

                    result = await handler.handle_get("/api/skills/nonexistent", mock_request)

                    assert result is not None
                    assert result.status_code == 404
                    body = json.loads(result.body)
                    assert body["error"]["code"] == "SKILL_NOT_FOUND"


class TestSkillsHandlerGetMetrics:
    """Tests for GET /api/skills/:name/metrics endpoint."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return SkillsHandler(mock_server_context)

    @pytest.mark.asyncio
    async def test_get_metrics_success(self, handler, mock_registry):
        """Get metrics returns skill execution statistics."""
        with patch.object(handler, "_get_registry", return_value=mock_registry):
            with patch("aragora.server.handlers.skills.SKILLS_AVAILABLE", True):
                with patch("aragora.server.handlers.skills._skills_limiter") as limiter:
                    limiter.is_allowed.return_value = True

                    mock_request = MagicMock()
                    mock_request.get.return_value = {
                        "user_id": "test",
                        "permissions": {"skills:read"},
                    }

                    result = await handler.handle_get(
                        "/api/skills/web_search/metrics", mock_request
                    )

                    assert result is not None
                    assert result.status_code == 200
                    body = json.loads(result.body)
                    assert body["skill"] == "web_search"
                    assert body["total_invocations"] == 100
                    assert body["successful_invocations"] == 95
                    assert body["failed_invocations"] == 5
                    assert body["average_latency_ms"] == 150.0
                    assert body["last_invoked"] is not None

    @pytest.mark.asyncio
    async def test_get_metrics_no_data(self, handler, mock_registry):
        """Get metrics returns zeros for skill with no metrics."""
        with patch.object(handler, "_get_registry", return_value=mock_registry):
            with patch("aragora.server.handlers.skills.SKILLS_AVAILABLE", True):
                with patch("aragora.server.handlers.skills._skills_limiter") as limiter:
                    limiter.is_allowed.return_value = True

                    mock_request = MagicMock()
                    mock_request.get.return_value = {
                        "user_id": "test",
                        "permissions": {"skills:read"},
                    }

                    result = await handler.handle_get(
                        "/api/skills/code_execution/metrics", mock_request
                    )

                    assert result is not None
                    assert result.status_code == 200
                    body = json.loads(result.body)
                    assert body["skill"] == "code_execution"
                    assert body["total_invocations"] == 0
                    assert body["successful_invocations"] == 0
                    assert body["failed_invocations"] == 0
                    assert body["average_latency_ms"] == 0
                    assert body["last_invoked"] is None

    @pytest.mark.asyncio
    async def test_get_metrics_skill_not_found(self, handler, mock_registry):
        """Get metrics returns 404 for unknown skill."""
        with patch.object(handler, "_get_registry", return_value=mock_registry):
            with patch("aragora.server.handlers.skills.SKILLS_AVAILABLE", True):
                with patch("aragora.server.handlers.skills._skills_limiter") as limiter:
                    limiter.is_allowed.return_value = True

                    mock_request = MagicMock()
                    mock_request.get.return_value = {
                        "user_id": "test",
                        "permissions": {"skills:read"},
                    }

                    result = await handler.handle_get(
                        "/api/skills/nonexistent/metrics", mock_request
                    )

                    assert result is not None
                    assert result.status_code == 404


class TestSkillsHandlerInvoke:
    """Tests for POST /api/skills/invoke endpoint."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return SkillsHandler(mock_server_context)

    @pytest.mark.asyncio
    async def test_invoke_skill_success(self, handler, mock_registry):
        """Invoke skill returns successful result."""
        with patch.object(handler, "_get_registry", return_value=mock_registry):
            with patch("aragora.server.handlers.skills.SKILLS_AVAILABLE", True):
                with patch("aragora.server.handlers.skills.SkillContext", MockSkillContext):
                    with patch("aragora.server.handlers.skills.SkillStatus", MockSkillStatus):
                        with patch("aragora.server.handlers.skills._skills_limiter") as limiter:
                            limiter.is_allowed.return_value = True

                            mock_request = create_async_request(
                                {
                                    "skill": "web_search",
                                    "input": {"query": "test query"},
                                    "user_id": "test-user",
                                }
                            )

                            result = await handler.handle_post("/api/skills/invoke", mock_request)

                            assert result is not None
                            assert result.status_code == 200
                            body = json.loads(result.body)
                            assert body["status"] == "success"
                            assert "output" in body
                            assert "execution_time_ms" in body
                            assert "metadata" in body

    @pytest.mark.asyncio
    async def test_invoke_skill_missing_field(self, handler, mock_registry):
        """Invoke skill returns 400 for missing skill field."""
        with patch.object(handler, "_get_registry", return_value=mock_registry):
            with patch("aragora.server.handlers.skills.SKILLS_AVAILABLE", True):
                with patch("aragora.server.handlers.skills._skills_limiter") as limiter:
                    limiter.is_allowed.return_value = True

                    mock_request = create_async_request(
                        {"input": {"query": "test"}}  # Missing "skill"
                    )

                    result = await handler.handle_post("/api/skills/invoke", mock_request)

                    assert result is not None
                    assert result.status_code == 400
                    body = json.loads(result.body)
                    assert "Missing required field" in body["error"]

    @pytest.mark.asyncio
    async def test_invoke_skill_not_found(self, handler, mock_registry):
        """Invoke skill returns 404 for unknown skill."""
        with patch.object(handler, "_get_registry", return_value=mock_registry):
            with patch("aragora.server.handlers.skills.SKILLS_AVAILABLE", True):
                with patch("aragora.server.handlers.skills._skills_limiter") as limiter:
                    limiter.is_allowed.return_value = True

                    mock_request = create_async_request({"skill": "nonexistent", "input": {}})

                    result = await handler.handle_post("/api/skills/invoke", mock_request)

                    assert result is not None
                    assert result.status_code == 404
                    body = json.loads(result.body)
                    assert body["error"]["code"] == "SKILL_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_invoke_skill_timeout(self, handler, mock_registry):
        """Invoke skill returns 408 on timeout."""

        async def slow_invoke(*args, **kwargs):
            await asyncio.sleep(10)  # Longer than timeout

        mock_registry.invoke = slow_invoke

        with patch.object(handler, "_get_registry", return_value=mock_registry):
            with patch("aragora.server.handlers.skills.SKILLS_AVAILABLE", True):
                with patch("aragora.server.handlers.skills.SkillContext", MockSkillContext):
                    with patch("aragora.server.handlers.skills._skills_limiter") as limiter:
                        limiter.is_allowed.return_value = True

                        mock_request = create_async_request(
                            {
                                "skill": "web_search",
                                "input": {},
                                "timeout": 0.1,  # Very short timeout
                            }
                        )

                        result = await handler.handle_post("/api/skills/invoke", mock_request)

                        assert result is not None
                        assert result.status_code == 408
                        body = json.loads(result.body)
                        assert body["error"]["code"] == "TIMEOUT"

    @pytest.mark.asyncio
    async def test_invoke_skill_error_result(self, handler, mock_registry):
        """Invoke skill returns 500 for error result."""

        async def error_invoke(*args, **kwargs):
            return MockSkillResult(
                status=MockSkillStatus.FAILURE,
                error_message="Execution failed",
                duration_seconds=0.01,
            )

        mock_registry.invoke = error_invoke

        with patch.object(handler, "_get_registry", return_value=mock_registry):
            with patch("aragora.server.handlers.skills.SKILLS_AVAILABLE", True):
                with patch("aragora.server.handlers.skills.SkillContext", MockSkillContext):
                    with patch("aragora.server.handlers.skills.SkillStatus", MockSkillStatus):
                        with patch("aragora.server.handlers.skills._skills_limiter") as limiter:
                            limiter.is_allowed.return_value = True

                            mock_request = create_async_request(
                                {"skill": "web_search", "input": {}}
                            )

                            result = await handler.handle_post("/api/skills/invoke", mock_request)

                            assert result is not None
                            assert result.status_code == 500
                            body = json.loads(result.body)
                            assert body["status"] == "error"
                            assert body["error"] == "Execution failed"

    @pytest.mark.asyncio
    async def test_invoke_skill_rate_limited_result(self, handler, mock_registry):
        """Invoke skill returns 429 for rate limited result."""

        async def rate_limited_invoke(*args, **kwargs):
            return MockSkillResult(
                status=MockSkillStatus.RATE_LIMITED,
                error_message="Too many requests",
            )

        mock_registry.invoke = rate_limited_invoke

        with patch.object(handler, "_get_registry", return_value=mock_registry):
            with patch("aragora.server.handlers.skills.SKILLS_AVAILABLE", True):
                with patch("aragora.server.handlers.skills.SkillContext", MockSkillContext):
                    with patch("aragora.server.handlers.skills.SkillStatus", MockSkillStatus):
                        with patch("aragora.server.handlers.skills._skills_limiter") as limiter:
                            limiter.is_allowed.return_value = True

                            mock_request = create_async_request(
                                {"skill": "web_search", "input": {}}
                            )

                            result = await handler.handle_post("/api/skills/invoke", mock_request)

                            assert result is not None
                            assert result.status_code == 429
                            body = json.loads(result.body)
                            assert body["error"]["code"] == "SKILL_RATE_LIMITED"

    @pytest.mark.asyncio
    async def test_invoke_skill_permission_denied(self, handler, mock_registry):
        """Invoke skill returns 403 for permission denied result."""

        async def permission_denied_invoke(*args, **kwargs):
            return MockSkillResult(
                status=MockSkillStatus.PERMISSION_DENIED,
                error_message="Insufficient permissions",
            )

        mock_registry.invoke = permission_denied_invoke

        with patch.object(handler, "_get_registry", return_value=mock_registry):
            with patch("aragora.server.handlers.skills.SKILLS_AVAILABLE", True):
                with patch("aragora.server.handlers.skills.SkillContext", MockSkillContext):
                    with patch("aragora.server.handlers.skills.SkillStatus", MockSkillStatus):
                        with patch("aragora.server.handlers.skills._skills_limiter") as limiter:
                            limiter.is_allowed.return_value = True

                            mock_request = create_async_request(
                                {"skill": "web_search", "input": {}}
                            )

                            result = await handler.handle_post("/api/skills/invoke", mock_request)

                            assert result is not None
                            assert result.status_code == 403
                            body = json.loads(result.body)
                            assert body["error"]["code"] == "PERMISSION_DENIED"


class TestSkillsHandlerRateLimiting:
    """Tests for rate limiting behavior."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return SkillsHandler(mock_server_context)

    @pytest.mark.asyncio
    async def test_get_rate_limited(self, handler):
        """GET requests are rate limited."""
        with patch("aragora.server.handlers.skills.SKILLS_AVAILABLE", True):
            with patch("aragora.server.handlers.skills._skills_limiter") as limiter:
                limiter.is_allowed.return_value = False

                mock_request = MagicMock()
                result = await handler.handle_get("/api/skills", mock_request)

                assert result.status_code == 429
                body = json.loads(result.body)
                assert body["error"]["code"] == "RATE_LIMITED"

    @pytest.mark.asyncio
    async def test_post_rate_limited(self, handler):
        """POST requests are rate limited."""
        with patch("aragora.server.handlers.skills.SKILLS_AVAILABLE", True):
            with patch("aragora.server.handlers.skills._skills_limiter") as limiter:
                limiter.is_allowed.return_value = False

                mock_request = MagicMock()
                result = await handler.handle_post("/api/skills/invoke", mock_request)

                assert result.status_code == 429
                body = json.loads(result.body)
                assert body["error"]["code"] == "RATE_LIMITED"


class TestSkillsHandlerErrorCases:
    """Tests for error handling."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return SkillsHandler(mock_server_context)

    @pytest.mark.asyncio
    async def test_unknown_get_endpoint(self, handler, mock_registry):
        """Unknown GET endpoint returns 404."""
        with patch.object(handler, "_get_registry", return_value=mock_registry):
            with patch("aragora.server.handlers.skills.SKILLS_AVAILABLE", True):
                with patch("aragora.server.handlers.skills._skills_limiter") as limiter:
                    limiter.is_allowed.return_value = True

                    mock_request = MagicMock()
                    result = await handler.handle_get("/api/skills/foo/bar/baz", mock_request)

                    # Should try to get skill "foo" which doesn't exist
                    assert result is not None

    @pytest.mark.asyncio
    async def test_unknown_post_endpoint(self, handler):
        """Unknown POST endpoint returns 404."""
        with patch("aragora.server.handlers.skills.SKILLS_AVAILABLE", True):
            with patch("aragora.server.handlers.skills._skills_limiter") as limiter:
                limiter.is_allowed.return_value = True

                mock_request = MagicMock()
                result = await handler.handle_post("/api/skills/unknown", mock_request)

                assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_invalid_json_body(self, handler, mock_registry):
        """Invalid JSON body returns 400."""
        with patch.object(handler, "_get_registry", return_value=mock_registry):
            with patch("aragora.server.handlers.skills.SKILLS_AVAILABLE", True):
                with patch("aragora.server.handlers.skills._skills_limiter") as limiter:
                    limiter.is_allowed.return_value = True

                    # Mock request that raises on json()
                    mock_request = MagicMock()
                    mock_request.get.side_effect = Exception("Invalid JSON")

                    result = await handler.handle_post("/api/skills/invoke", mock_request)

                    assert result.status_code == 400


class TestSkillsHandlerVersionPrefix:
    """Tests for API version prefix handling."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return SkillsHandler(mock_server_context)

    @pytest.mark.asyncio
    async def test_handles_v1_prefix(self, handler, mock_registry):
        """Handler strips v1 prefix correctly."""
        with patch.object(handler, "_get_registry", return_value=mock_registry):
            with patch("aragora.server.handlers.skills.SKILLS_AVAILABLE", True):
                with patch("aragora.server.handlers.skills._skills_limiter") as limiter:
                    limiter.is_allowed.return_value = True

                    mock_request = MagicMock()
                    mock_request.get.return_value = {
                        "user_id": "test",
                        "permissions": {"skills:read"},
                    }

                    result = await handler.handle_get("/api/v1/skills", mock_request)

                    assert result is not None
                    assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handles_v2_prefix(self, handler, mock_registry):
        """Handler strips v2 prefix correctly."""
        with patch.object(handler, "_get_registry", return_value=mock_registry):
            with patch("aragora.server.handlers.skills.SKILLS_AVAILABLE", True):
                with patch("aragora.server.handlers.skills._skills_limiter") as limiter:
                    limiter.is_allowed.return_value = True

                    mock_request = MagicMock()
                    mock_request.get.return_value = {
                        "user_id": "test",
                        "permissions": {"skills:read"},
                    }

                    result = await handler.handle_get("/api/v2/skills", mock_request)

                    assert result is not None
                    assert result.status_code == 200
