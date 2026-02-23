"""Tests for skill marketplace handler (aragora/server/handlers/skill_marketplace.py).

Covers all routes, validation, error handling, auth/permission checks, and edge cases:
- GET  /api/v1/skills/marketplace/search       - Search skills (public)
- GET  /api/v1/skills/marketplace/stats        - Marketplace stats (public)
- GET  /api/v1/skills/marketplace/installed     - List installed skills (auth)
- POST /api/v1/skills/marketplace/publish       - Publish a skill (auth)
- GET  /api/v1/skills/marketplace/{id}         - Get skill details (public)
- GET  /api/v1/skills/marketplace/{id}/versions - Get skill versions (public)
- GET  /api/v1/skills/marketplace/{id}/ratings  - Get skill ratings (public)
- POST /api/v1/skills/marketplace/{id}/install  - Install a skill (auth)
- DELETE /api/v1/skills/marketplace/{id}/install - Uninstall a skill (auth)
- POST /api/v1/skills/marketplace/{id}/rate     - Rate a skill (auth)
- PUT  /api/v1/skills/marketplace/{id}/verify   - Verify a skill (admin)
- DELETE /api/v1/skills/marketplace/{id}/verify  - Revoke verification (admin)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.base import HandlerResult
from aragora.server.handlers.skill_marketplace import SkillMarketplaceHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: HandlerResult | None) -> dict:
    """Extract JSON body from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, HandlerResult):
        if isinstance(result.body, bytes):
            return json.loads(result.body.decode("utf-8"))
        return result.body
    if isinstance(result, dict):
        return result.get("body", result)
    return {}


def _status(result: HandlerResult | None) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    if isinstance(result, HandlerResult):
        return result.status_code
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return 200


# ---------------------------------------------------------------------------
# Mock data classes
# ---------------------------------------------------------------------------


@dataclass
class MockSkillListing:
    """Mock skill listing from marketplace."""

    id: str = "skill-1"
    name: str = "Test Skill"
    description: str = "A test skill"
    version: str = "1.0.0"
    author_id: str = "user-1"
    author_name: str = "Test Author"
    category: str = "custom"
    tier: str = "free"
    is_verified: bool = False
    downloads: int = 100
    average_rating: float = 4.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "author_id": self.author_id,
            "author_name": self.author_name,
            "category": self.category,
            "tier": self.tier,
            "is_verified": self.is_verified,
            "downloads": self.downloads,
            "average_rating": self.average_rating,
        }


@dataclass
class MockSkillVersion:
    """Mock skill version."""

    version: str = "1.0.0"
    changelog: str = "Initial release"
    published_at: str = "2025-01-01T00:00:00Z"

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "changelog": self.changelog,
            "published_at": self.published_at,
        }


@dataclass
class MockSkillRating:
    """Mock skill rating."""

    user_id: str = "user-1"
    rating: int = 5
    review: str | None = "Great skill!"
    created_at: str = "2025-01-01T00:00:00Z"

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "rating": self.rating,
            "review": self.review,
            "created_at": self.created_at,
        }


@dataclass
class MockInstallResult:
    """Mock install result."""

    success: bool = True
    error: str | None = None
    skill_id: str = "skill-1"
    version: str = "1.0.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "skill_id": self.skill_id,
            "version": self.version,
        }


@dataclass
class MockInstalledSkill:
    """Mock installed skill."""

    skill_id: str = "skill-1"
    name: str = "Test Skill"
    version: str = "1.0.0"
    installed_at: str = "2025-01-01T00:00:00Z"

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "version": self.version,
            "installed_at": self.installed_at,
        }


@dataclass
class MockPublishIssue:
    """Mock publish issue."""

    severity: str = "warning"
    message: str = "Minor issue"

    def to_dict(self) -> dict[str, Any]:
        return {"severity": self.severity, "message": self.message}


class MockSkillCategory:
    """Mock SkillCategory enum."""

    def __init__(self, value):
        self.value = value


class MockSkillTier:
    """Mock SkillTier enum."""

    def __init__(self, value):
        self.value = value


class MockHTTPHandler:
    """Mock HTTP handler simulating BaseHTTPRequestHandler."""

    def __init__(self, body: dict[str, Any] | None = None):
        self.rfile = MagicMock()
        self.command = "GET"
        self._body = body
        if body:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers = {"Content-Length": str(len(body_bytes))}
        else:
            self.rfile.read.return_value = b"{}"
            self.headers = {"Content-Length": "2"}
        self.client_address = ("127.0.0.1", 54321)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a SkillMarketplaceHandler instance."""
    return SkillMarketplaceHandler(ctx={})


@pytest.fixture
def http_handler():
    """Create a mock HTTP handler (no body)."""
    return MockHTTPHandler()


@pytest.fixture
def mock_marketplace():
    """Create a mock marketplace instance."""
    marketplace = AsyncMock()

    listings = [
        MockSkillListing(id="skill-1", name="Skill One"),
        MockSkillListing(id="skill-2", name="Skill Two"),
        MockSkillListing(id="skill-3", name="Skill Three"),
    ]

    marketplace.search = AsyncMock(return_value=listings)
    marketplace.get_skill = AsyncMock(
        side_effect=lambda sid: next((s for s in listings if s.id == sid), None)
    )
    marketplace.get_versions = AsyncMock(
        return_value=[
            MockSkillVersion(version="1.0.0"),
            MockSkillVersion(version="1.1.0", changelog="Bug fixes"),
        ]
    )
    marketplace.get_ratings = AsyncMock(
        return_value=[
            MockSkillRating(user_id="user-1", rating=5),
            MockSkillRating(user_id="user-2", rating=4),
        ]
    )
    marketplace.rate = AsyncMock(return_value=MockSkillRating(user_id="user-1", rating=5))
    marketplace.set_verified = AsyncMock(return_value=True)
    marketplace.get_stats = AsyncMock(
        return_value={
            "total_skills": 100,
            "total_installs": 5000,
            "total_ratings": 200,
        }
    )

    return marketplace


@pytest.fixture
def mock_installer():
    """Create a mock SkillInstaller instance."""
    installer = AsyncMock()
    installer.install = AsyncMock(return_value=MockInstallResult())
    installer.uninstall = AsyncMock(return_value=True)
    installer.get_installed = AsyncMock(
        return_value=[
            MockInstalledSkill(skill_id="skill-1"),
            MockInstalledSkill(skill_id="skill-2"),
        ]
    )
    return installer


@pytest.fixture
def mock_publisher():
    """Create a mock SkillPublisher instance."""
    publisher = AsyncMock()
    publisher.publish = AsyncMock(
        return_value=(True, MockSkillListing(), [])
    )
    return publisher


@pytest.fixture
def mock_registry():
    """Create a mock SkillRegistry instance."""
    registry = MagicMock()
    skill = MagicMock()
    registry.get.return_value = skill
    return registry


# ===========================================================================
# can_handle Tests
# ===========================================================================


class TestCanHandle:
    """Tests for can_handle routing."""

    def test_search_path(self, handler):
        assert handler.can_handle("/api/v1/skills/marketplace/search") is True

    def test_publish_path(self, handler):
        assert handler.can_handle("/api/v1/skills/marketplace/publish", "POST") is True

    def test_installed_path(self, handler):
        assert handler.can_handle("/api/v1/skills/marketplace/installed") is True

    def test_stats_path(self, handler):
        assert handler.can_handle("/api/v1/skills/marketplace/stats") is True

    def test_skill_detail_path(self, handler):
        assert handler.can_handle("/api/v1/skills/marketplace/skill-1") is True

    def test_skill_versions_path(self, handler):
        assert handler.can_handle("/api/v1/skills/marketplace/skill-1/versions") is True

    def test_skill_ratings_path(self, handler):
        assert handler.can_handle("/api/v1/skills/marketplace/skill-1/ratings") is True

    def test_skill_install_path(self, handler):
        assert handler.can_handle("/api/v1/skills/marketplace/skill-1/install", "POST") is True

    def test_skill_rate_path(self, handler):
        assert handler.can_handle("/api/v1/skills/marketplace/skill-1/rate", "POST") is True

    def test_skill_verify_path(self, handler):
        assert handler.can_handle("/api/v1/skills/marketplace/skill-1/verify", "PUT") is True

    def test_unrelated_path(self, handler):
        assert handler.can_handle("/api/v1/debates/list") is False

    def test_partial_match_path(self, handler):
        assert handler.can_handle("/api/v1/skills/registry") is False

    def test_path_without_version_prefix(self, handler):
        assert handler.can_handle("/api/skills/marketplace/search") is True


# ===========================================================================
# Search Skills Tests
# ===========================================================================


class TestSearchSkills:
    """Tests for GET /api/v1/skills/marketplace/search."""

    @pytest.mark.asyncio
    async def test_search_returns_200(self, handler, http_handler, mock_marketplace):
        with patch(
            "aragora.server.handlers.skill_marketplace.SkillMarketplaceHandler._search_skills",
            new_callable=AsyncMock,
        ) as mock_search:
            from aragora.server.handlers.base import json_response

            mock_search.return_value = json_response(
                {"query": "", "count": 3, "limit": 20, "offset": 0, "results": []}
            )
            result = await handler.handle(
                "/api/v1/skills/marketplace/search", {}, http_handler, "GET"
            )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_search_with_query(self, handler, http_handler, mock_marketplace):
        with patch(
            "aragora.skills.marketplace.get_marketplace", return_value=mock_marketplace
        ), patch(
            "aragora.skills.marketplace.SkillCategory", MockSkillCategory
        ), patch(
            "aragora.skills.marketplace.SkillTier", MockSkillTier
        ):
            result = await handler._search_skills({"q": "test"})
        assert _status(result) == 200
        body = _body(result)
        assert body["query"] == "test"
        assert body["count"] == 3

    @pytest.mark.asyncio
    async def test_search_with_category(self, handler, mock_marketplace):
        with patch(
            "aragora.skills.marketplace.get_marketplace", return_value=mock_marketplace
        ), patch(
            "aragora.skills.marketplace.SkillCategory", MockSkillCategory
        ), patch(
            "aragora.skills.marketplace.SkillTier", MockSkillTier
        ):
            result = await handler._search_skills({"category": "analysis"})
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_search_with_tier(self, handler, mock_marketplace):
        with patch(
            "aragora.skills.marketplace.get_marketplace", return_value=mock_marketplace
        ), patch(
            "aragora.skills.marketplace.SkillCategory", MockSkillCategory
        ), patch(
            "aragora.skills.marketplace.SkillTier", MockSkillTier
        ):
            result = await handler._search_skills({"tier": "free"})
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_search_with_tags(self, handler, mock_marketplace):
        with patch(
            "aragora.skills.marketplace.get_marketplace", return_value=mock_marketplace
        ), patch(
            "aragora.skills.marketplace.SkillCategory", MockSkillCategory
        ), patch(
            "aragora.skills.marketplace.SkillTier", MockSkillTier
        ):
            result = await handler._search_skills({"tags": "ai,ml,nlp"})
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_search_with_author(self, handler, mock_marketplace):
        with patch(
            "aragora.skills.marketplace.get_marketplace", return_value=mock_marketplace
        ), patch(
            "aragora.skills.marketplace.SkillCategory", MockSkillCategory
        ), patch(
            "aragora.skills.marketplace.SkillTier", MockSkillTier
        ):
            result = await handler._search_skills({"author": "user-1"})
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_search_with_sort(self, handler, mock_marketplace):
        with patch(
            "aragora.skills.marketplace.get_marketplace", return_value=mock_marketplace
        ), patch(
            "aragora.skills.marketplace.SkillCategory", MockSkillCategory
        ), patch(
            "aragora.skills.marketplace.SkillTier", MockSkillTier
        ):
            result = await handler._search_skills({"sort": "downloads"})
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_search_with_pagination(self, handler, mock_marketplace):
        with patch(
            "aragora.skills.marketplace.get_marketplace", return_value=mock_marketplace
        ), patch(
            "aragora.skills.marketplace.SkillCategory", MockSkillCategory
        ), patch(
            "aragora.skills.marketplace.SkillTier", MockSkillTier
        ):
            result = await handler._search_skills({"limit": "10", "offset": "5"})
        body = _body(result)
        assert body["limit"] == 10
        assert body["offset"] == 5

    @pytest.mark.asyncio
    async def test_search_empty_query(self, handler, mock_marketplace):
        with patch(
            "aragora.skills.marketplace.get_marketplace", return_value=mock_marketplace
        ), patch(
            "aragora.skills.marketplace.SkillCategory", MockSkillCategory
        ), patch(
            "aragora.skills.marketplace.SkillTier", MockSkillTier
        ):
            result = await handler._search_skills({})
        assert _status(result) == 200
        body = _body(result)
        assert body["query"] == ""

    @pytest.mark.asyncio
    async def test_search_import_error(self, handler):
        with patch(
            "builtins.__import__", side_effect=ImportError("not available")
        ):
            result = await handler._search_skills({})
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_search_value_error(self, handler, mock_marketplace):
        mock_marketplace.search.side_effect = ValueError("bad value")
        with patch(
            "aragora.skills.marketplace.get_marketplace", return_value=mock_marketplace
        ), patch(
            "aragora.skills.marketplace.SkillCategory", MockSkillCategory
        ), patch(
            "aragora.skills.marketplace.SkillTier", MockSkillTier
        ):
            result = await handler._search_skills({})
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_search_no_tags_param(self, handler, mock_marketplace):
        with patch(
            "aragora.skills.marketplace.get_marketplace", return_value=mock_marketplace
        ), patch(
            "aragora.skills.marketplace.SkillCategory", MockSkillCategory
        ), patch(
            "aragora.skills.marketplace.SkillTier", MockSkillTier
        ):
            result = await handler._search_skills({"tags": ""})
        assert _status(result) == 200


# ===========================================================================
# Get Skill Details Tests
# ===========================================================================


class TestGetSkill:
    """Tests for GET /api/v1/skills/marketplace/{id}."""

    @pytest.mark.asyncio
    async def test_get_skill_returns_200(self, handler, mock_marketplace):
        with patch(
            "aragora.skills.marketplace.get_marketplace", return_value=mock_marketplace
        ):
            result = await handler._get_skill("skill-1")
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_get_skill_returns_data(self, handler, mock_marketplace):
        with patch(
            "aragora.skills.marketplace.get_marketplace", return_value=mock_marketplace
        ):
            result = await handler._get_skill("skill-1")
        body = _body(result)
        assert body["id"] == "skill-1"
        assert body["name"] == "Skill One"

    @pytest.mark.asyncio
    async def test_get_skill_not_found(self, handler, mock_marketplace):
        with patch(
            "aragora.skills.marketplace.get_marketplace", return_value=mock_marketplace
        ):
            result = await handler._get_skill("nonexistent")
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_get_skill_import_error(self, handler):
        with patch("builtins.__import__", side_effect=ImportError("no module")):
            result = await handler._get_skill("skill-1")
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_get_skill_internal_error(self, handler, mock_marketplace):
        mock_marketplace.get_skill.side_effect = TypeError("bad type")
        with patch(
            "aragora.skills.marketplace.get_marketplace", return_value=mock_marketplace
        ):
            result = await handler._get_skill("skill-1")
        assert _status(result) == 500


# ===========================================================================
# Get Versions Tests
# ===========================================================================


class TestGetVersions:
    """Tests for GET /api/v1/skills/marketplace/{id}/versions."""

    @pytest.mark.asyncio
    async def test_get_versions_returns_200(self, handler, mock_marketplace):
        with patch(
            "aragora.skills.marketplace.get_marketplace", return_value=mock_marketplace
        ):
            result = await handler._get_versions("skill-1")
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_get_versions_returns_list(self, handler, mock_marketplace):
        with patch(
            "aragora.skills.marketplace.get_marketplace", return_value=mock_marketplace
        ):
            result = await handler._get_versions("skill-1")
        body = _body(result)
        assert body["skill_id"] == "skill-1"
        assert len(body["versions"]) == 2

    @pytest.mark.asyncio
    async def test_get_versions_not_found(self, handler, mock_marketplace):
        mock_marketplace.get_versions.return_value = None
        with patch(
            "aragora.skills.marketplace.get_marketplace", return_value=mock_marketplace
        ):
            result = await handler._get_versions("nonexistent")
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_get_versions_empty_list(self, handler, mock_marketplace):
        mock_marketplace.get_versions.return_value = []
        with patch(
            "aragora.skills.marketplace.get_marketplace", return_value=mock_marketplace
        ):
            result = await handler._get_versions("skill-1")
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_get_versions_import_error(self, handler):
        with patch("builtins.__import__", side_effect=ImportError("no module")):
            result = await handler._get_versions("skill-1")
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_get_versions_internal_error(self, handler, mock_marketplace):
        mock_marketplace.get_versions.side_effect = OSError("disk error")
        with patch(
            "aragora.skills.marketplace.get_marketplace", return_value=mock_marketplace
        ):
            result = await handler._get_versions("skill-1")
        assert _status(result) == 500


# ===========================================================================
# Get Ratings Tests
# ===========================================================================


class TestGetRatings:
    """Tests for GET /api/v1/skills/marketplace/{id}/ratings."""

    @pytest.mark.asyncio
    async def test_get_ratings_returns_200(self, handler, mock_marketplace):
        with patch(
            "aragora.skills.marketplace.get_marketplace", return_value=mock_marketplace
        ):
            result = await handler._get_ratings("skill-1", {})
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_get_ratings_returns_list(self, handler, mock_marketplace):
        with patch(
            "aragora.skills.marketplace.get_marketplace", return_value=mock_marketplace
        ):
            result = await handler._get_ratings("skill-1", {})
        body = _body(result)
        assert body["skill_id"] == "skill-1"
        assert body["count"] == 2

    @pytest.mark.asyncio
    async def test_get_ratings_with_pagination(self, handler, mock_marketplace):
        with patch(
            "aragora.skills.marketplace.get_marketplace", return_value=mock_marketplace
        ):
            result = await handler._get_ratings("skill-1", {"limit": "5", "offset": "10"})
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_get_ratings_import_error(self, handler):
        with patch("builtins.__import__", side_effect=ImportError("no module")):
            result = await handler._get_ratings("skill-1", {})
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_get_ratings_internal_error(self, handler, mock_marketplace):
        mock_marketplace.get_ratings.side_effect = KeyError("missing key")
        with patch(
            "aragora.skills.marketplace.get_marketplace", return_value=mock_marketplace
        ):
            result = await handler._get_ratings("skill-1", {})
        assert _status(result) == 500


# ===========================================================================
# Publish Skill Tests
# ===========================================================================


class TestPublishSkill:
    """Tests for POST /api/v1/skills/marketplace/publish."""

    @pytest.mark.asyncio
    async def test_publish_success(self, handler, mock_publisher, mock_registry):
        auth_ctx = {"user_id": "user-1", "display_name": "Tester"}
        body = {"skill_name": "my-skill", "category": "custom", "tier": "free"}
        with patch(
            "aragora.skills.publisher.SkillPublisher", return_value=mock_publisher
        ), patch(
            "aragora.skills.registry.get_skill_registry", return_value=mock_registry
        ), patch(
            "aragora.skills.marketplace.SkillCategory", MockSkillCategory
        ), patch(
            "aragora.skills.marketplace.SkillTier", MockSkillTier
        ):
            result = await handler._publish_skill(body, auth_ctx)
        assert _status(result) == 200
        body_data = _body(result)
        assert body_data["success"] is True

    @pytest.mark.asyncio
    async def test_publish_missing_skill_name(self, handler):
        auth_ctx = {"user_id": "user-1"}
        result = await handler._publish_skill({}, auth_ctx)
        assert _status(result) == 400
        assert "skill_name" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_publish_empty_skill_name(self, handler):
        auth_ctx = {"user_id": "user-1"}
        result = await handler._publish_skill({"skill_name": ""}, auth_ctx)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_publish_skill_not_in_registry(self, handler, mock_registry):
        mock_registry.get.return_value = None
        auth_ctx = {"user_id": "user-1"}
        body = {"skill_name": "nonexistent-skill"}
        with patch(
            "aragora.skills.registry.get_skill_registry", return_value=mock_registry
        ), patch(
            "aragora.skills.marketplace.SkillCategory", MockSkillCategory
        ), patch(
            "aragora.skills.marketplace.SkillTier", MockSkillTier
        ):
            result = await handler._publish_skill(body, auth_ctx)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_publish_invalid_category(self, handler, mock_registry):
        auth_ctx = {"user_id": "user-1"}
        body = {"skill_name": "my-skill", "category": "invalid_cat"}

        def bad_category(val):
            raise ValueError(f"Invalid category: {val}")

        with patch(
            "aragora.skills.registry.get_skill_registry", return_value=mock_registry
        ), patch(
            "aragora.skills.marketplace.SkillCategory", bad_category
        ), patch(
            "aragora.skills.marketplace.SkillTier", MockSkillTier
        ):
            result = await handler._publish_skill(body, auth_ctx)
        assert _status(result) == 400
        assert "category" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_publish_invalid_tier(self, handler, mock_registry):
        auth_ctx = {"user_id": "user-1"}
        body = {"skill_name": "my-skill", "tier": "invalid_tier"}

        def bad_tier(val):
            raise ValueError(f"Invalid tier: {val}")

        with patch(
            "aragora.skills.registry.get_skill_registry", return_value=mock_registry
        ), patch(
            "aragora.skills.marketplace.SkillCategory", MockSkillCategory
        ), patch(
            "aragora.skills.marketplace.SkillTier", bad_tier
        ):
            result = await handler._publish_skill(body, auth_ctx)
        assert _status(result) == 400
        assert "tier" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_publish_with_scan_issues(self, handler, mock_publisher, mock_registry):
        mock_publisher.publish.return_value = (
            False,
            None,
            [MockPublishIssue(severity="error", message="Dangerous code")],
        )
        auth_ctx = {"user_id": "user-1"}
        body = {"skill_name": "my-skill"}
        with patch(
            "aragora.skills.publisher.SkillPublisher", return_value=mock_publisher
        ), patch(
            "aragora.skills.registry.get_skill_registry", return_value=mock_registry
        ), patch(
            "aragora.skills.marketplace.SkillCategory", MockSkillCategory
        ), patch(
            "aragora.skills.marketplace.SkillTier", MockSkillTier
        ):
            result = await handler._publish_skill(body, auth_ctx)
        assert _status(result) == 400
        body_data = _body(result)
        assert body_data["success"] is False
        assert len(body_data["issues"]) == 1

    @pytest.mark.asyncio
    async def test_publish_with_auth_context_object(self, handler, mock_publisher, mock_registry):
        from aragora.rbac.models import AuthorizationContext

        auth_ctx = AuthorizationContext(
            user_id="user-1",
            user_email="test@example.com",
            org_id="org-1",
            roles={"admin"},
            permissions={"skills:publish"},
        )
        body = {"skill_name": "my-skill"}
        with patch(
            "aragora.skills.publisher.SkillPublisher", return_value=mock_publisher
        ), patch(
            "aragora.skills.registry.get_skill_registry", return_value=mock_registry
        ), patch(
            "aragora.skills.marketplace.SkillCategory", MockSkillCategory
        ), patch(
            "aragora.skills.marketplace.SkillTier", MockSkillTier
        ):
            result = await handler._publish_skill(body, auth_ctx)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_publish_import_error(self, handler):
        auth_ctx = {"user_id": "user-1"}
        body = {"skill_name": "my-skill"}
        with patch("builtins.__import__", side_effect=ImportError("no module")):
            result = await handler._publish_skill(body, auth_ctx)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_publish_internal_error(self, handler, mock_publisher, mock_registry):
        mock_publisher.publish.side_effect = OSError("disk full")
        auth_ctx = {"user_id": "user-1"}
        body = {"skill_name": "my-skill"}
        with patch(
            "aragora.skills.publisher.SkillPublisher", return_value=mock_publisher
        ), patch(
            "aragora.skills.registry.get_skill_registry", return_value=mock_registry
        ), patch(
            "aragora.skills.marketplace.SkillCategory", MockSkillCategory
        ), patch(
            "aragora.skills.marketplace.SkillTier", MockSkillTier
        ):
            result = await handler._publish_skill(body, auth_ctx)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_publish_with_optional_urls(self, handler, mock_publisher, mock_registry):
        auth_ctx = {"user_id": "user-1", "display_name": "Tester"}
        body = {
            "skill_name": "my-skill",
            "homepage_url": "https://example.com",
            "repository_url": "https://github.com/example/skill",
            "documentation_url": "https://docs.example.com",
            "changelog": "Added features",
        }
        with patch(
            "aragora.skills.publisher.SkillPublisher", return_value=mock_publisher
        ), patch(
            "aragora.skills.registry.get_skill_registry", return_value=mock_registry
        ), patch(
            "aragora.skills.marketplace.SkillCategory", MockSkillCategory
        ), patch(
            "aragora.skills.marketplace.SkillTier", MockSkillTier
        ):
            result = await handler._publish_skill(body, auth_ctx)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_publish_display_name_fallback_to_user_id(
        self, handler, mock_publisher, mock_registry
    ):
        auth_ctx = {"user_id": "user-1"}
        body = {"skill_name": "my-skill"}
        with patch(
            "aragora.skills.publisher.SkillPublisher", return_value=mock_publisher
        ), patch(
            "aragora.skills.registry.get_skill_registry", return_value=mock_registry
        ), patch(
            "aragora.skills.marketplace.SkillCategory", MockSkillCategory
        ), patch(
            "aragora.skills.marketplace.SkillTier", MockSkillTier
        ):
            result = await handler._publish_skill(body, auth_ctx)
        assert _status(result) == 200
        # Verify display_name falls back to user_id
        call_kwargs = mock_publisher.publish.call_args
        assert call_kwargs.kwargs.get("author_name") == "user-1"


# ===========================================================================
# Install Skill Tests
# ===========================================================================


class TestInstallSkill:
    """Tests for POST /api/v1/skills/marketplace/{id}/install."""

    @pytest.mark.asyncio
    async def test_install_success(self, handler, mock_installer):
        auth_ctx = {"user_id": "user-1", "tenant_id": "tenant-1", "permissions": {"skills:install"}}
        with patch(
            "aragora.skills.installer.SkillInstaller", return_value=mock_installer
        ):
            result = await handler._install_skill("skill-1", {}, auth_ctx)
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True

    @pytest.mark.asyncio
    async def test_install_with_version(self, handler, mock_installer):
        auth_ctx = {"user_id": "user-1", "tenant_id": "tenant-1", "permissions": set()}
        with patch(
            "aragora.skills.installer.SkillInstaller", return_value=mock_installer
        ):
            result = await handler._install_skill("skill-1", {"version": "2.0.0"}, auth_ctx)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_install_failure(self, handler, mock_installer):
        mock_installer.install.return_value = MockInstallResult(
            success=False, error="Incompatible version"
        )
        auth_ctx = {"user_id": "user-1", "tenant_id": "tenant-1", "permissions": set()}
        with patch(
            "aragora.skills.installer.SkillInstaller", return_value=mock_installer
        ):
            result = await handler._install_skill("skill-1", {}, auth_ctx)
        assert _status(result) == 400
        body = _body(result)
        assert body["success"] is False

    @pytest.mark.asyncio
    async def test_install_default_tenant(self, handler, mock_installer):
        auth_ctx = {"user_id": "user-1", "permissions": set()}
        with patch(
            "aragora.skills.installer.SkillInstaller", return_value=mock_installer
        ):
            result = await handler._install_skill("skill-1", {}, auth_ctx)
        assert _status(result) == 200
        call_kwargs = mock_installer.install.call_args
        assert call_kwargs.kwargs.get("tenant_id") == "default"

    @pytest.mark.asyncio
    async def test_install_import_error(self, handler):
        auth_ctx = {"user_id": "user-1", "permissions": set()}
        with patch("builtins.__import__", side_effect=ImportError("no module")):
            result = await handler._install_skill("skill-1", {}, auth_ctx)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_install_internal_error(self, handler, mock_installer):
        mock_installer.install.side_effect = ValueError("bad input")
        auth_ctx = {"user_id": "user-1", "tenant_id": "t-1", "permissions": set()}
        with patch(
            "aragora.skills.installer.SkillInstaller", return_value=mock_installer
        ):
            result = await handler._install_skill("skill-1", {}, auth_ctx)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_install_with_auth_context_object(self, handler, mock_installer):
        from aragora.rbac.models import AuthorizationContext

        auth_ctx = AuthorizationContext(
            user_id="user-1",
            user_email="test@example.com",
            org_id="org-1",
            roles={"admin"},
            permissions={"skills:install"},
        )
        # tenant_id is extracted via getattr, set it as an attribute
        object.__setattr__(auth_ctx, "tenant_id", "tenant-1")
        with patch(
            "aragora.skills.installer.SkillInstaller", return_value=mock_installer
        ):
            result = await handler._install_skill("skill-1", {}, auth_ctx)
        assert _status(result) == 200


# ===========================================================================
# Uninstall Skill Tests
# ===========================================================================


class TestUninstallSkill:
    """Tests for DELETE /api/v1/skills/marketplace/{id}/install."""

    @pytest.mark.asyncio
    async def test_uninstall_success(self, handler, mock_installer):
        auth_ctx = {"user_id": "user-1", "tenant_id": "tenant-1", "permissions": set()}
        with patch(
            "aragora.skills.installer.SkillInstaller", return_value=mock_installer
        ):
            result = await handler._uninstall_skill("skill-1", auth_ctx)
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["skill_id"] == "skill-1"
        assert "uninstalled_at" in body

    @pytest.mark.asyncio
    async def test_uninstall_failure(self, handler, mock_installer):
        mock_installer.uninstall.return_value = False
        auth_ctx = {"user_id": "user-1", "tenant_id": "tenant-1", "permissions": set()}
        with patch(
            "aragora.skills.installer.SkillInstaller", return_value=mock_installer
        ):
            result = await handler._uninstall_skill("skill-1", auth_ctx)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_uninstall_default_tenant(self, handler, mock_installer):
        auth_ctx = {"user_id": "user-1", "permissions": set()}
        with patch(
            "aragora.skills.installer.SkillInstaller", return_value=mock_installer
        ):
            result = await handler._uninstall_skill("skill-1", auth_ctx)
        assert _status(result) == 200
        call_kwargs = mock_installer.uninstall.call_args
        assert call_kwargs.kwargs.get("tenant_id") == "default"

    @pytest.mark.asyncio
    async def test_uninstall_import_error(self, handler):
        auth_ctx = {"user_id": "user-1", "permissions": set()}
        with patch("builtins.__import__", side_effect=ImportError("no module")):
            result = await handler._uninstall_skill("skill-1", auth_ctx)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_uninstall_internal_error(self, handler, mock_installer):
        mock_installer.uninstall.side_effect = AttributeError("bad attr")
        auth_ctx = {"user_id": "user-1", "tenant_id": "t-1", "permissions": set()}
        with patch(
            "aragora.skills.installer.SkillInstaller", return_value=mock_installer
        ):
            result = await handler._uninstall_skill("skill-1", auth_ctx)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_uninstall_with_auth_context_object(self, handler, mock_installer):
        from aragora.rbac.models import AuthorizationContext

        auth_ctx = AuthorizationContext(
            user_id="user-1",
            user_email="test@example.com",
            org_id="org-1",
            roles={"admin"},
            permissions={"skills:install"},
        )
        object.__setattr__(auth_ctx, "tenant_id", "tenant-1")
        with patch(
            "aragora.skills.installer.SkillInstaller", return_value=mock_installer
        ):
            result = await handler._uninstall_skill("skill-1", auth_ctx)
        assert _status(result) == 200


# ===========================================================================
# Rate Skill Tests
# ===========================================================================


class TestRateSkill:
    """Tests for POST /api/v1/skills/marketplace/{id}/rate."""

    @pytest.mark.asyncio
    async def test_rate_success(self, handler, mock_marketplace):
        auth_ctx = {"user_id": "user-1"}
        body = {"rating": 5, "review": "Excellent!"}
        with patch(
            "aragora.skills.marketplace.get_marketplace", return_value=mock_marketplace
        ):
            result = await handler._rate_skill("skill-1", body, auth_ctx)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_rate_without_review(self, handler, mock_marketplace):
        auth_ctx = {"user_id": "user-1"}
        body = {"rating": 4}
        with patch(
            "aragora.skills.marketplace.get_marketplace", return_value=mock_marketplace
        ):
            result = await handler._rate_skill("skill-1", body, auth_ctx)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_rate_missing_rating(self, handler):
        auth_ctx = {"user_id": "user-1"}
        result = await handler._rate_skill("skill-1", {"review": "no rating"}, auth_ctx)
        assert _status(result) == 400
        assert "rating" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_rate_zero_value(self, handler):
        auth_ctx = {"user_id": "user-1"}
        result = await handler._rate_skill("skill-1", {"rating": 0}, auth_ctx)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_rate_too_low(self, handler):
        auth_ctx = {"user_id": "user-1"}
        result = await handler._rate_skill("skill-1", {"rating": -1}, auth_ctx)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_rate_too_high(self, handler):
        auth_ctx = {"user_id": "user-1"}
        result = await handler._rate_skill("skill-1", {"rating": 6}, auth_ctx)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_rate_non_integer(self, handler):
        auth_ctx = {"user_id": "user-1"}
        result = await handler._rate_skill("skill-1", {"rating": "five"}, auth_ctx)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_rate_float_value(self, handler):
        auth_ctx = {"user_id": "user-1"}
        result = await handler._rate_skill("skill-1", {"rating": 3.5}, auth_ctx)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_rate_import_error(self, handler):
        auth_ctx = {"user_id": "user-1"}
        body = {"rating": 5}
        with patch("builtins.__import__", side_effect=ImportError("no module")):
            result = await handler._rate_skill("skill-1", body, auth_ctx)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_rate_value_error(self, handler, mock_marketplace):
        mock_marketplace.rate.side_effect = ValueError("already rated")
        auth_ctx = {"user_id": "user-1"}
        body = {"rating": 5}
        with patch(
            "aragora.skills.marketplace.get_marketplace", return_value=mock_marketplace
        ):
            result = await handler._rate_skill("skill-1", body, auth_ctx)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_rate_internal_error(self, handler, mock_marketplace):
        mock_marketplace.rate.side_effect = TypeError("bad type")
        auth_ctx = {"user_id": "user-1"}
        body = {"rating": 5}
        with patch(
            "aragora.skills.marketplace.get_marketplace", return_value=mock_marketplace
        ):
            result = await handler._rate_skill("skill-1", body, auth_ctx)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_rate_with_auth_context_object(self, handler, mock_marketplace):
        from aragora.rbac.models import AuthorizationContext

        auth_ctx = AuthorizationContext(
            user_id="user-1",
            user_email="test@example.com",
            org_id="org-1",
            roles={"admin"},
            permissions={"skills:rate"},
        )
        body = {"rating": 5}
        with patch(
            "aragora.skills.marketplace.get_marketplace", return_value=mock_marketplace
        ):
            result = await handler._rate_skill("skill-1", body, auth_ctx)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_rate_boundary_min(self, handler, mock_marketplace):
        auth_ctx = {"user_id": "user-1"}
        body = {"rating": 1}
        with patch(
            "aragora.skills.marketplace.get_marketplace", return_value=mock_marketplace
        ):
            result = await handler._rate_skill("skill-1", body, auth_ctx)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_rate_boundary_max(self, handler, mock_marketplace):
        auth_ctx = {"user_id": "user-1"}
        body = {"rating": 5}
        with patch(
            "aragora.skills.marketplace.get_marketplace", return_value=mock_marketplace
        ):
            result = await handler._rate_skill("skill-1", body, auth_ctx)
        assert _status(result) == 200


# ===========================================================================
# Verify Skill Tests
# ===========================================================================


class TestVerifySkill:
    """Tests for PUT/DELETE /api/v1/skills/marketplace/{id}/verify."""

    @pytest.mark.asyncio
    async def test_verify_success(self, handler, mock_marketplace):
        auth_ctx = {"user_id": "admin-1"}
        with patch(
            "aragora.skills.marketplace.get_marketplace", return_value=mock_marketplace
        ):
            result = await handler._set_verification("skill-1", True, auth_ctx)
        assert _status(result) == 200
        body = _body(result)
        assert body["skill_id"] == "skill-1"
        assert body["is_verified"] is True
        assert body["changed_by"] == "admin-1"
        assert "changed_at" in body

    @pytest.mark.asyncio
    async def test_revoke_verification(self, handler, mock_marketplace):
        auth_ctx = {"user_id": "admin-1"}
        with patch(
            "aragora.skills.marketplace.get_marketplace", return_value=mock_marketplace
        ):
            result = await handler._set_verification("skill-1", False, auth_ctx)
        assert _status(result) == 200
        body = _body(result)
        assert body["is_verified"] is False

    @pytest.mark.asyncio
    async def test_verify_skill_not_found(self, handler, mock_marketplace):
        mock_marketplace.get_skill.return_value = None
        auth_ctx = {"user_id": "admin-1"}
        with patch(
            "aragora.skills.marketplace.get_marketplace", return_value=mock_marketplace
        ):
            result = await handler._set_verification("nonexistent", True, auth_ctx)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_verify_set_verified_fails(self, handler, mock_marketplace):
        mock_marketplace.set_verified.return_value = False
        auth_ctx = {"user_id": "admin-1"}
        with patch(
            "aragora.skills.marketplace.get_marketplace", return_value=mock_marketplace
        ):
            result = await handler._set_verification("skill-1", True, auth_ctx)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_verify_import_error(self, handler):
        auth_ctx = {"user_id": "admin-1"}
        with patch("builtins.__import__", side_effect=ImportError("no module")):
            result = await handler._set_verification("skill-1", True, auth_ctx)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_verify_internal_error(self, handler, mock_marketplace):
        mock_marketplace.get_skill.side_effect = KeyError("missing")
        auth_ctx = {"user_id": "admin-1"}
        with patch(
            "aragora.skills.marketplace.get_marketplace", return_value=mock_marketplace
        ):
            result = await handler._set_verification("skill-1", True, auth_ctx)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_verify_with_auth_context_object(self, handler, mock_marketplace):
        from aragora.rbac.models import AuthorizationContext

        auth_ctx = AuthorizationContext(
            user_id="admin-1",
            user_email="admin@example.com",
            org_id="org-1",
            roles={"admin"},
            permissions={"skills:admin"},
        )
        with patch(
            "aragora.skills.marketplace.get_marketplace", return_value=mock_marketplace
        ):
            result = await handler._set_verification("skill-1", True, auth_ctx)
        assert _status(result) == 200


# ===========================================================================
# List Installed Tests
# ===========================================================================


class TestListInstalled:
    """Tests for GET /api/v1/skills/marketplace/installed."""

    @pytest.mark.asyncio
    async def test_list_installed_success(self, handler, mock_installer):
        from aragora.rbac.models import AuthorizationContext

        auth_ctx = AuthorizationContext(
            user_id="user-1",
            user_email="test@example.com",
            org_id="org-1",
            roles={"admin"},
            permissions={"skills:read"},
        )
        object.__setattr__(auth_ctx, "tenant_id", "tenant-1")
        with patch(
            "aragora.skills.installer.SkillInstaller", return_value=mock_installer
        ):
            result = await handler._list_installed(auth_ctx)
        assert _status(result) == 200
        body = _body(result)
        assert body["tenant_id"] == "tenant-1"
        assert body["count"] == 2

    @pytest.mark.asyncio
    async def test_list_installed_default_tenant(self, handler, mock_installer):
        from aragora.rbac.models import AuthorizationContext

        auth_ctx = AuthorizationContext(
            user_id="user-1",
            user_email="test@example.com",
            org_id="org-1",
            roles={"admin"},
            permissions={"skills:read"},
        )
        with patch(
            "aragora.skills.installer.SkillInstaller", return_value=mock_installer
        ):
            result = await handler._list_installed(auth_ctx)
        assert _status(result) == 200
        body = _body(result)
        assert body["tenant_id"] == "default"

    @pytest.mark.asyncio
    async def test_list_installed_import_error(self, handler):
        from aragora.rbac.models import AuthorizationContext

        auth_ctx = AuthorizationContext(
            user_id="user-1",
            user_email="test@example.com",
            org_id="org-1",
            roles={"admin"},
            permissions={"skills:read"},
        )
        with patch("builtins.__import__", side_effect=ImportError("no module")):
            result = await handler._list_installed(auth_ctx)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_list_installed_internal_error(self, handler, mock_installer):
        from aragora.rbac.models import AuthorizationContext

        mock_installer.get_installed.side_effect = OSError("disk error")
        auth_ctx = AuthorizationContext(
            user_id="user-1",
            user_email="test@example.com",
            org_id="org-1",
            roles={"admin"},
            permissions={"skills:read"},
        )
        with patch(
            "aragora.skills.installer.SkillInstaller", return_value=mock_installer
        ):
            result = await handler._list_installed(auth_ctx)
        assert _status(result) == 500


# ===========================================================================
# Get Stats Tests
# ===========================================================================


class TestGetStats:
    """Tests for GET /api/v1/skills/marketplace/stats."""

    @pytest.mark.asyncio
    async def test_get_stats_success(self, handler, mock_marketplace):
        with patch(
            "aragora.skills.marketplace.get_marketplace", return_value=mock_marketplace
        ):
            result = await handler._get_stats()
        assert _status(result) == 200
        body = _body(result)
        assert body["total_skills"] == 100
        assert body["total_installs"] == 5000

    @pytest.mark.asyncio
    async def test_get_stats_import_error(self, handler):
        with patch("builtins.__import__", side_effect=ImportError("no module")):
            result = await handler._get_stats()
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_get_stats_internal_error(self, handler, mock_marketplace):
        mock_marketplace.get_stats.side_effect = KeyError("missing")
        with patch(
            "aragora.skills.marketplace.get_marketplace", return_value=mock_marketplace
        ):
            result = await handler._get_stats()
        assert _status(result) == 500


# ===========================================================================
# Route Dispatch Tests (handle method)
# ===========================================================================


class TestHandleRouting:
    """Tests for the main handle() method routing."""

    @pytest.mark.asyncio
    async def test_route_search(self, handler, http_handler):
        with patch.object(handler, "_search_skills", new_callable=AsyncMock) as mock_fn:
            from aragora.server.handlers.base import json_response

            mock_fn.return_value = json_response({"results": []})
            result = await handler.handle(
                "/api/v1/skills/marketplace/search", {"q": "test"}, http_handler, "GET"
            )
        assert result is not None
        mock_fn.assert_called_once_with({"q": "test"})

    @pytest.mark.asyncio
    async def test_route_stats(self, handler, http_handler):
        with patch.object(handler, "_get_stats", new_callable=AsyncMock) as mock_fn:
            from aragora.server.handlers.base import json_response

            mock_fn.return_value = json_response({"stats": {}})
            result = await handler.handle(
                "/api/v1/skills/marketplace/stats", {}, http_handler, "GET"
            )
        assert result is not None
        mock_fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_skill_detail(self, handler, http_handler):
        with patch.object(handler, "_get_skill", new_callable=AsyncMock) as mock_fn:
            from aragora.server.handlers.base import json_response

            mock_fn.return_value = json_response({"id": "skill-1"})
            result = await handler.handle(
                "/api/v1/skills/marketplace/skill-1", {}, http_handler, "GET"
            )
        assert result is not None
        mock_fn.assert_called_once_with("skill-1")

    @pytest.mark.asyncio
    async def test_route_versions(self, handler, http_handler):
        with patch.object(handler, "_get_versions", new_callable=AsyncMock) as mock_fn:
            from aragora.server.handlers.base import json_response

            mock_fn.return_value = json_response({"versions": []})
            result = await handler.handle(
                "/api/v1/skills/marketplace/skill-1/versions", {}, http_handler, "GET"
            )
        assert result is not None
        mock_fn.assert_called_once_with("skill-1")

    @pytest.mark.asyncio
    async def test_route_ratings(self, handler, http_handler):
        with patch.object(handler, "_get_ratings", new_callable=AsyncMock) as mock_fn:
            from aragora.server.handlers.base import json_response

            mock_fn.return_value = json_response({"ratings": []})
            result = await handler.handle(
                "/api/v1/skills/marketplace/skill-1/ratings", {}, http_handler, "GET"
            )
        assert result is not None
        mock_fn.assert_called_once_with("skill-1", {})

    @pytest.mark.asyncio
    async def test_route_install(self, handler, http_handler):
        with patch.object(handler, "_install_skill", new_callable=AsyncMock) as mock_fn:
            from aragora.server.handlers.base import json_response

            mock_fn.return_value = json_response({"success": True})
            result = await handler.handle(
                "/api/v1/skills/marketplace/skill-1/install",
                {},
                http_handler,
                "POST",
                body={"version": "1.0"},
            )
        assert result is not None

    @pytest.mark.asyncio
    async def test_route_uninstall(self, handler, http_handler):
        with patch.object(handler, "_uninstall_skill", new_callable=AsyncMock) as mock_fn:
            from aragora.server.handlers.base import json_response

            mock_fn.return_value = json_response({"success": True})
            result = await handler.handle(
                "/api/v1/skills/marketplace/skill-1/install",
                {},
                http_handler,
                "DELETE",
            )
        assert result is not None

    @pytest.mark.asyncio
    async def test_route_rate(self, handler, http_handler):
        with patch.object(handler, "_rate_skill", new_callable=AsyncMock) as mock_fn:
            from aragora.server.handlers.base import json_response

            mock_fn.return_value = json_response({"rating": 5})
            result = await handler.handle(
                "/api/v1/skills/marketplace/skill-1/rate",
                {},
                http_handler,
                "POST",
                body={"rating": 5},
            )
        assert result is not None

    @pytest.mark.asyncio
    async def test_route_verify(self, handler, http_handler):
        with patch.object(handler, "_set_verification", new_callable=AsyncMock) as mock_fn:
            from aragora.server.handlers.base import json_response

            mock_fn.return_value = json_response({"is_verified": True})
            result = await handler.handle(
                "/api/v1/skills/marketplace/skill-1/verify",
                {},
                http_handler,
                "PUT",
            )
        assert result is not None

    @pytest.mark.asyncio
    async def test_route_revoke_verify(self, handler, http_handler):
        with patch.object(handler, "_set_verification", new_callable=AsyncMock) as mock_fn:
            from aragora.server.handlers.base import json_response

            mock_fn.return_value = json_response({"is_verified": False})
            result = await handler.handle(
                "/api/v1/skills/marketplace/skill-1/verify",
                {},
                http_handler,
                "DELETE",
            )
        assert result is not None

    @pytest.mark.asyncio
    async def test_route_publish(self, handler, http_handler):
        with patch.object(handler, "_publish_skill", new_callable=AsyncMock) as mock_fn:
            from aragora.server.handlers.base import json_response

            mock_fn.return_value = json_response({"success": True})
            result = await handler.handle(
                "/api/v1/skills/marketplace/publish",
                {},
                http_handler,
                "POST",
                body={"skill_name": "test"},
            )
        assert result is not None

    @pytest.mark.asyncio
    async def test_route_installed(self, handler, http_handler):
        with patch.object(handler, "_list_installed", new_callable=AsyncMock) as mock_fn:
            from aragora.server.handlers.base import json_response

            mock_fn.return_value = json_response({"skills": []})
            result = await handler.handle(
                "/api/v1/skills/marketplace/installed",
                {},
                http_handler,
                "GET",
            )
        assert result is not None

    @pytest.mark.asyncio
    async def test_route_unknown_returns_none(self, handler, http_handler):
        result = await handler.handle(
            "/api/v1/skills/other/endpoint", {}, http_handler, "GET"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_route_short_path_returns_none(self, handler, http_handler):
        result = await handler.handle(
            "/api/v1/skills", {}, http_handler, "GET"
        )
        assert result is None


# ===========================================================================
# Authentication Tests (no_auto_auth)
# ===========================================================================


class TestAuthentication:
    """Tests for authentication enforcement."""

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_installed_unauthenticated(self, handler, http_handler):
        """Installed endpoint returns 401 when auth context has no user_id."""
        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = None
            result = await handler.handle(
                "/api/v1/skills/marketplace/installed", {}, http_handler, "GET"
            )
        assert _status(result) == 401

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_publish_unauthenticated(self, handler, http_handler):
        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = None
            result = await handler.handle(
                "/api/v1/skills/marketplace/publish",
                {},
                http_handler,
                "POST",
                body={"skill_name": "test"},
            )
        assert _status(result) == 401

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_install_unauthenticated(self, handler, http_handler):
        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = None
            result = await handler.handle(
                "/api/v1/skills/marketplace/skill-1/install",
                {},
                http_handler,
                "POST",
            )
        assert _status(result) == 401

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_uninstall_unauthenticated(self, handler, http_handler):
        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = None
            result = await handler.handle(
                "/api/v1/skills/marketplace/skill-1/install",
                {},
                http_handler,
                "DELETE",
            )
        assert _status(result) == 401

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_rate_unauthenticated(self, handler, http_handler):
        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = None
            result = await handler.handle(
                "/api/v1/skills/marketplace/skill-1/rate",
                {},
                http_handler,
                "POST",
                body={"rating": 5},
            )
        assert _status(result) == 401

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_verify_unauthenticated(self, handler, http_handler):
        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = None
            result = await handler.handle(
                "/api/v1/skills/marketplace/skill-1/verify",
                {},
                http_handler,
                "PUT",
            )
        assert _status(result) == 401

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_revoke_verify_unauthenticated(self, handler, http_handler):
        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = None
            result = await handler.handle(
                "/api/v1/skills/marketplace/skill-1/verify",
                {},
                http_handler,
                "DELETE",
            )
        assert _status(result) == 401

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_search_allows_anonymous(self, handler, http_handler):
        """Search endpoint works without authentication."""
        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = None
            with patch.object(handler, "_search_skills", new_callable=AsyncMock) as mock_search:
                from aragora.server.handlers.base import json_response

                mock_search.return_value = json_response({"results": []})
                result = await handler.handle(
                    "/api/v1/skills/marketplace/search", {}, http_handler, "GET"
                )
        assert _status(result) == 200

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_stats_allows_anonymous(self, handler, http_handler):
        """Stats endpoint works without authentication."""
        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = None
            with patch.object(handler, "_get_stats", new_callable=AsyncMock) as mock_stats:
                from aragora.server.handlers.base import json_response

                mock_stats.return_value = json_response({"stats": {}})
                result = await handler.handle(
                    "/api/v1/skills/marketplace/stats", {}, http_handler, "GET"
                )
        assert _status(result) == 200

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_get_skill_allows_anonymous(self, handler, http_handler):
        """Get skill detail endpoint works without authentication."""
        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = None
            with patch.object(handler, "_get_skill", new_callable=AsyncMock) as mock_get:
                from aragora.server.handlers.base import json_response

                mock_get.return_value = json_response({"id": "skill-1"})
                result = await handler.handle(
                    "/api/v1/skills/marketplace/skill-1", {}, http_handler, "GET"
                )
        assert _status(result) == 200

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_anonymous_user_id_treated_as_unauthenticated(self, handler, http_handler):
        """A user_id of 'anonymous' should be treated as unauthenticated."""
        from aragora.rbac.models import AuthorizationContext

        anon_ctx = AuthorizationContext(
            user_id="anonymous",
            user_email="",
            org_id="",
            roles=set(),
            permissions=set(),
        )
        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = anon_ctx
            result = await handler.handle(
                "/api/v1/skills/marketplace/installed", {}, http_handler, "GET"
            )
        assert _status(result) == 401


# ===========================================================================
# Permission Denial Tests
# ===========================================================================


class TestPermissionDenied:
    """Tests for permission check failures (403)."""

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_installed_permission_denied(self, handler, http_handler):
        from aragora.rbac.models import AuthorizationContext
        from aragora.server.handlers.secure import ForbiddenError

        auth_ctx = AuthorizationContext(
            user_id="user-1",
            user_email="test@example.com",
            org_id="org-1",
            roles=set(),
            permissions=set(),
        )
        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = auth_ctx
            with patch.object(
                handler, "check_permission", side_effect=ForbiddenError("denied")
            ):
                result = await handler.handle(
                    "/api/v1/skills/marketplace/installed", {}, http_handler, "GET"
                )
        assert _status(result) == 403

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_publish_permission_denied(self, handler, http_handler):
        from aragora.rbac.models import AuthorizationContext
        from aragora.server.handlers.secure import ForbiddenError

        auth_ctx = AuthorizationContext(
            user_id="user-1",
            user_email="test@example.com",
            org_id="org-1",
            roles=set(),
            permissions=set(),
        )
        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = auth_ctx
            with patch.object(
                handler, "check_permission", side_effect=ForbiddenError("denied")
            ):
                result = await handler.handle(
                    "/api/v1/skills/marketplace/publish",
                    {},
                    http_handler,
                    "POST",
                    body={"skill_name": "test"},
                )
        assert _status(result) == 403

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_install_permission_denied(self, handler, http_handler):
        from aragora.rbac.models import AuthorizationContext
        from aragora.server.handlers.secure import ForbiddenError

        auth_ctx = AuthorizationContext(
            user_id="user-1",
            user_email="test@example.com",
            org_id="org-1",
            roles=set(),
            permissions=set(),
        )
        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = auth_ctx
            with patch.object(
                handler, "check_permission", side_effect=ForbiddenError("denied")
            ):
                result = await handler.handle(
                    "/api/v1/skills/marketplace/skill-1/install",
                    {},
                    http_handler,
                    "POST",
                )
        assert _status(result) == 403

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_verify_permission_denied(self, handler, http_handler):
        from aragora.rbac.models import AuthorizationContext
        from aragora.server.handlers.secure import ForbiddenError

        auth_ctx = AuthorizationContext(
            user_id="user-1",
            user_email="test@example.com",
            org_id="org-1",
            roles=set(),
            permissions=set(),
        )
        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = auth_ctx
            with patch.object(
                handler, "check_permission", side_effect=ForbiddenError("denied")
            ):
                result = await handler.handle(
                    "/api/v1/skills/marketplace/skill-1/verify",
                    {},
                    http_handler,
                    "PUT",
                )
        assert _status(result) == 403


# ===========================================================================
# Handler Initialization Tests
# ===========================================================================


class TestHandlerInit:
    """Tests for SkillMarketplaceHandler initialization."""

    def test_init_with_ctx(self):
        h = SkillMarketplaceHandler(ctx={"key": "value"})
        assert h.ctx == {"key": "value"}

    def test_init_with_none(self):
        h = SkillMarketplaceHandler(ctx=None)
        assert h.ctx == {}

    def test_init_with_empty(self):
        h = SkillMarketplaceHandler(ctx={})
        assert h.ctx == {}

    def test_resource_type(self):
        h = SkillMarketplaceHandler(ctx={})
        assert h.RESOURCE_TYPE == "skills"

    def test_routes_defined(self):
        h = SkillMarketplaceHandler(ctx={})
        assert "/api/skills/marketplace/search" in h.ROUTES
        assert "/api/skills/marketplace/publish" in h.ROUTES
        assert "/api/skills/marketplace/installed" in h.ROUTES
        assert "/api/skills/marketplace/stats" in h.ROUTES

    def test_pattern_prefixes(self):
        h = SkillMarketplaceHandler(ctx={})
        assert "/api/skills/marketplace/" in h.PATTERN_PREFIXES
