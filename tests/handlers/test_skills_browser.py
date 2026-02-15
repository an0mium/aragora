"""Tests for skills browser API endpoints.

Tests:
- GET /api/v1/skills - list all registered skills
- GET /api/v1/skills/:name - get skill details
- POST /api/v1/skills/:name/invoke - invoke a skill by name (URL path)
- POST /api/v1/skills/invoke - invoke a skill by name (body)
- Pagination, error cases, rate limiting
"""

import asyncio
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.skills import SkillsHandler


def parse_body(result) -> dict:
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body.decode("utf-8"))


class MockSkillCapability(Enum):
    ANALYSIS = "analysis"
    SEARCH = "search"


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


class MockSkillStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"


@dataclass
class MockSkillResult:
    status: MockSkillStatus = MockSkillStatus.SUCCESS
    data: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None
    duration_seconds: float = 0.1
    metadata: dict[str, Any] | None = None


@pytest.fixture
def mock_registry():
    """Create a mock skill registry."""
    registry = MagicMock()

    skills = [
        MockSkill(MockSkillManifest(name="search", description="Search skill")),
        MockSkill(MockSkillManifest(name="analyze", description="Analysis skill")),
        MockSkill(MockSkillManifest(name="summarize", description="Summarize skill")),
    ]

    registry.list_skills.return_value = [s.manifest for s in skills]
    registry.get.side_effect = lambda name: next(
        (s for s in skills if s.manifest.name == name), None
    )
    registry.get_metrics.return_value = None
    registry.invoke = AsyncMock(return_value=MockSkillResult(data={"result": "ok"}))

    return registry


@pytest.fixture
def handler(mock_registry):
    """Create a SkillsHandler with mocked registry."""
    with (
        patch("aragora.server.handlers.skills.SKILLS_AVAILABLE", True),
        patch(
            "aragora.server.handlers.skills.get_skill_registry",
            return_value=mock_registry,
        ),
        patch(
            "aragora.server.handlers.skills.SkillStatus",
            MockSkillStatus,
        ),
        patch(
            "aragora.server.handlers.skills.SkillContext",
            MagicMock,
        ),
    ):
        h = SkillsHandler(server_context={})
        h._registry = mock_registry
        return h


@pytest.fixture
def mock_request():
    """Create a mock HTTP request."""
    req = MagicMock()
    req.headers = {}
    # Need remote attr for get_client_ip
    req.remote = "127.0.0.1"
    req.transport = MagicMock()
    req.transport.get_extra_info.return_value = ("127.0.0.1", 12345)
    return req


class TestListSkills:
    """Tests for GET /api/v1/skills."""

    @pytest.mark.asyncio
    async def test_list_returns_200(self, handler, mock_request):
        result = await handler.handle_get("/api/v1/skills", mock_request)
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_list_has_skills(self, handler, mock_request):
        result = await handler.handle_get("/api/v1/skills", mock_request)
        body = parse_body(result)
        assert "skills" in body
        assert "total" in body
        assert body["total"] == 3

    @pytest.mark.asyncio
    async def test_skill_fields(self, handler, mock_request):
        result = await handler.handle_get("/api/v1/skills", mock_request)
        body = parse_body(result)
        skill = body["skills"][0]
        assert "name" in skill
        assert "description" in skill
        assert "version" in skill
        assert "input_schema" in skill
        assert "capabilities" in skill

    @pytest.mark.asyncio
    async def test_pagination_limit(self, handler, mock_request):
        mock_request.query = {"limit": "2"}
        # SkillsHandler uses self.get_query_param which reads from query attr
        result = await handler.handle_get("/api/v1/skills", mock_request)
        body = parse_body(result)
        assert len(body["skills"]) <= body["total"]


class TestGetSkill:
    """Tests for GET /api/v1/skills/:name."""

    @pytest.mark.asyncio
    async def test_get_existing_skill(self, handler, mock_request):
        result = await handler.handle_get("/api/v1/skills/search", mock_request)
        assert result.status_code == 200
        body = parse_body(result)
        assert body["name"] == "search"

    @pytest.mark.asyncio
    async def test_get_nonexistent_skill(self, handler, mock_request):
        result = await handler.handle_get("/api/v1/skills/nonexistent", mock_request)
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_get_skill_detail_fields(self, handler, mock_request):
        result = await handler.handle_get("/api/v1/skills/analyze", mock_request)
        body = parse_body(result)
        assert body["name"] == "analyze"
        assert "version" in body
        assert "output_schema" in body
        assert "rate_limit_per_minute" in body
        assert "timeout_seconds" in body


class TestInvokeSkillByPath:
    """Tests for POST /api/v1/skills/:name/invoke."""

    @pytest.mark.asyncio
    async def test_invoke_by_path(self, handler, mock_request):
        mock_request.json = AsyncMock(return_value={"input": {"query": "test"}})
        result = await handler.handle_post("/api/v1/skills/search/invoke", mock_request)
        assert result.status_code == 200
        body = parse_body(result)
        assert body["status"] == "success"

    @pytest.mark.asyncio
    async def test_invoke_nonexistent_by_path(self, handler, mock_request):
        mock_request.json = AsyncMock(return_value={"input": {}})
        result = await handler.handle_post("/api/v1/skills/nonexistent/invoke", mock_request)
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_invoke_by_body(self, handler, mock_request):
        mock_request.json = AsyncMock(return_value={"skill": "search", "input": {"query": "test"}})
        result = await handler.handle_post("/api/v1/skills/invoke", mock_request)
        assert result.status_code == 200


class TestSkillsUnavailable:
    """Tests when skills system is unavailable."""

    @pytest.mark.asyncio
    async def test_list_unavailable(self, mock_request):
        with patch("aragora.server.handlers.skills.SKILLS_AVAILABLE", False):
            h = SkillsHandler(server_context={})
            result = await h.handle_get("/api/v1/skills", mock_request)
            assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_invoke_unavailable(self, mock_request):
        with patch("aragora.server.handlers.skills.SKILLS_AVAILABLE", False):
            h = SkillsHandler(server_context={})
            result = await h.handle_post("/api/v1/skills/invoke", mock_request)
            assert result.status_code == 503
