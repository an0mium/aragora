"""
Tests for aragora.server.handlers.skill_marketplace - Skill Marketplace API Handler.

Tests cover:
- Route registration and can_handle
- GET /api/v1/skills/marketplace/search - Search skills
- GET /api/v1/skills/marketplace/{skill_id} - Get skill details
- GET /api/v1/skills/marketplace/{skill_id}/versions - Get skill versions
- GET /api/v1/skills/marketplace/{skill_id}/ratings - Get skill ratings
- POST /api/v1/skills/marketplace/publish - Publish a skill
- POST /api/v1/skills/marketplace/{skill_id}/install - Install a skill
- DELETE /api/v1/skills/marketplace/{skill_id}/install - Uninstall a skill
- POST /api/v1/skills/marketplace/{skill_id}/rate - Rate a skill
- GET /api/v1/skills/marketplace/installed - List installed skills
- GET /api/v1/skills/marketplace/stats - Get marketplace statistics
- Authentication and RBAC
- Error handling
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, AsyncMock, patch

import pytest


# ---------------------------------------------------------------------------
# Import the module under test with Slack stub workaround
# ---------------------------------------------------------------------------


def _import_skill_marketplace_module():
    """Import skill_marketplace module, working around broken sibling imports."""
    try:
        import aragora.server.handlers.skill_marketplace as mod

        return mod
    except (ImportError, ModuleNotFoundError):
        pass

    # Clear partially loaded modules and stub broken imports
    to_remove = [k for k in sys.modules if k.startswith("aragora.server.handlers")]
    for k in to_remove:
        del sys.modules[k]

    _slack_stubs = [
        "aragora.server.handlers.social._slack_impl",
        "aragora.server.handlers.social._slack_impl.config",
        "aragora.server.handlers.social._slack_impl.handler",
        "aragora.server.handlers.social._slack_impl.commands",
        "aragora.server.handlers.social._slack_impl.events",
        "aragora.server.handlers.social._slack_impl.blocks",
        "aragora.server.handlers.social._slack_impl.interactions",
        "aragora.server.handlers.social.slack",
        "aragora.server.handlers.social.slack.handler",
    ]
    for name in _slack_stubs:
        if name not in sys.modules:
            stub = MagicMock()
            stub.__path__ = []
            stub.__file__ = f"<stub:{name}>"
            sys.modules[name] = stub

    import aragora.server.handlers.skill_marketplace as mod

    return mod


skill_module = _import_skill_marketplace_module()
SkillMarketplaceHandler = skill_module.SkillMarketplaceHandler


# ===========================================================================
# Test Fixtures and Mocks
# ===========================================================================


@dataclass
class MockSkillListing:
    """Mock skill listing."""

    skill_id: str = "skill-test-123"
    name: str = "Test Skill"
    description: str = "A test skill for unit tests"
    version: str = "1.0.0"
    author_id: str = "author-123"
    author_name: str = "Test Author"
    category: str = "custom"
    tier: str = "free"
    rating: float = 4.5
    install_count: int = 100
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "author_id": self.author_id,
            "author_name": self.author_name,
            "category": self.category,
            "tier": self.tier,
            "rating": self.rating,
            "install_count": self.install_count,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class MockSkillVersion:
    """Mock skill version."""

    version: str = "1.0.0"
    changelog: str = "Initial release"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "changelog": self.changelog,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class MockSkillRating:
    """Mock skill rating."""

    user_id: str = "user-123"
    rating: int = 5
    review: str = "Great skill!"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "rating": self.rating,
            "review": self.review,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class MockInstalledSkill:
    """Mock installed skill."""

    skill_id: str = "skill-test-123"
    name: str = "Test Skill"
    version: str = "1.0.0"
    installed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "version": self.version,
            "installed_at": self.installed_at.isoformat(),
        }


@dataclass
class MockInstallResult:
    """Mock skill installation result."""

    success: bool = True
    skill_id: str = "skill-test-123"
    version: str = "1.0.0"
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "skill_id": self.skill_id,
            "version": self.version,
            "error": self.error,
        }


@dataclass
class MockPublishIssue:
    """Mock skill publish issue."""

    severity: str = "warning"
    message: str = "Consider adding more documentation"

    def to_dict(self) -> dict:
        return {"severity": self.severity, "message": self.message}


class MockMarketplace:
    """Mock skill marketplace."""

    async def search(self, **kwargs) -> list[MockSkillListing]:
        return [MockSkillListing()]

    async def get_skill(self, skill_id: str) -> MockSkillListing | None:
        if skill_id == "nonexistent":
            return None
        return MockSkillListing(skill_id=skill_id)

    async def get_versions(self, skill_id: str) -> list[MockSkillVersion] | None:
        if skill_id == "nonexistent":
            return None
        return [MockSkillVersion(), MockSkillVersion(version="0.9.0")]

    async def get_ratings(
        self, skill_id: str, limit: int = 20, offset: int = 0
    ) -> list[MockSkillRating]:
        return [MockSkillRating()]

    async def rate(
        self, skill_id: str, user_id: str, rating: int, review: str | None = None
    ) -> MockSkillRating:
        return MockSkillRating(user_id=user_id, rating=rating, review=review or "")

    async def get_stats(self) -> dict:
        return {
            "total_skills": 150,
            "total_installs": 5000,
            "categories": {"custom": 50, "data": 30, "integration": 70},
        }


class MockSkillInstaller:
    """Mock skill installer."""

    async def install(self, **kwargs) -> MockInstallResult:
        return MockInstallResult(skill_id=kwargs.get("skill_id", "skill-123"))

    async def uninstall(self, **kwargs) -> bool:
        return kwargs.get("skill_id") != "nonexistent"

    async def get_installed(self, tenant_id: str) -> list[MockInstalledSkill]:
        return [MockInstalledSkill()]


class MockSkillRegistry:
    """Mock skill registry."""

    def __init__(self):
        self._skills = {"test-skill": MagicMock()}

    def get(self, name: str):
        return self._skills.get(name)


class MockSkillPublisher:
    """Mock skill publisher."""

    async def publish(self, **kwargs) -> tuple[bool, MockSkillListing | None, list]:
        return True, MockSkillListing(), []


def make_mock_handler(
    body: dict | None = None,
    method: str = "GET",
    path: str = "/api/v1/skills/marketplace/search",
):
    """Create a mock HTTP handler."""
    handler = MagicMock()
    handler.command = method
    handler.path = path
    handler.headers = {}
    handler.client_address = ("127.0.0.1", 12345)

    if body is not None:
        body_bytes = json.dumps(body).encode("utf-8")
        handler.headers["Content-Length"] = str(len(body_bytes))
        handler.rfile = BytesIO(body_bytes)
    else:
        handler.rfile = BytesIO(b"")
        handler.headers["Content-Length"] = "0"

    return handler


@pytest.fixture
def skill_handler():
    """Create SkillMarketplaceHandler with mock context."""
    ctx = {}
    handler = SkillMarketplaceHandler(ctx)
    return handler


# ===========================================================================
# Test Routing (can_handle)
# ===========================================================================


class TestSkillMarketplaceRouting:
    """Tests for SkillMarketplaceHandler.can_handle."""

    def test_can_handle_search(self, skill_handler):
        assert skill_handler.can_handle("/api/v1/skills/marketplace/search") is True

    def test_can_handle_stats(self, skill_handler):
        assert skill_handler.can_handle("/api/v1/skills/marketplace/stats") is True

    def test_can_handle_installed(self, skill_handler):
        assert skill_handler.can_handle("/api/v1/skills/marketplace/installed") is True

    def test_can_handle_publish(self, skill_handler):
        assert skill_handler.can_handle("/api/v1/skills/marketplace/publish", "POST") is True

    def test_can_handle_skill_details(self, skill_handler):
        assert skill_handler.can_handle("/api/v1/skills/marketplace/skill-123") is True

    def test_can_handle_skill_versions(self, skill_handler):
        assert skill_handler.can_handle("/api/v1/skills/marketplace/skill-123/versions") is True

    def test_can_handle_skill_ratings(self, skill_handler):
        assert skill_handler.can_handle("/api/v1/skills/marketplace/skill-123/ratings") is True

    def test_can_handle_install(self, skill_handler):
        assert (
            skill_handler.can_handle("/api/v1/skills/marketplace/skill-123/install", "POST") is True
        )

    def test_can_handle_uninstall(self, skill_handler):
        assert (
            skill_handler.can_handle("/api/v1/skills/marketplace/skill-123/install", "DELETE")
            is True
        )

    def test_can_handle_rate(self, skill_handler):
        assert skill_handler.can_handle("/api/v1/skills/marketplace/skill-123/rate", "POST") is True

    def test_cannot_handle_other_paths(self, skill_handler):
        assert skill_handler.can_handle("/api/v1/debates") is False


# ===========================================================================
# Test Search Skills (GET /api/v1/skills/marketplace/search)
# ===========================================================================


class TestSkillMarketplaceSearch:
    """Tests for skill search endpoint."""

    @pytest.mark.asyncio
    async def test_search_skills_success(self, skill_handler):
        """Happy path: search skills."""
        handler = make_mock_handler()

        with patch("aragora.skills.marketplace.get_marketplace", return_value=MockMarketplace()):
            result = await skill_handler._search_skills({"q": "test"})

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "results" in data
        assert data["count"] == 1

    @pytest.mark.asyncio
    async def test_search_skills_with_filters(self, skill_handler):
        """Search with category and tier filters."""
        with patch("aragora.skills.marketplace.get_marketplace", return_value=MockMarketplace()):
            result = await skill_handler._search_skills(
                {"q": "test", "category": "custom", "tier": "free"}
            )

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_search_skills_module_unavailable(self, skill_handler):
        """Search when marketplace module unavailable returns 503."""
        with patch("aragora.skills.marketplace.get_marketplace", side_effect=ImportError()):
            result = await skill_handler._search_skills({})

        assert result is not None
        assert result.status_code == 503


# ===========================================================================
# Test Get Skill (GET /api/v1/skills/marketplace/{skill_id})
# ===========================================================================


class TestSkillMarketplaceGetSkill:
    """Tests for get skill details endpoint."""

    @pytest.mark.asyncio
    async def test_get_skill_success(self, skill_handler):
        """Happy path: get skill details."""
        with patch("aragora.skills.marketplace.get_marketplace", return_value=MockMarketplace()):
            result = await skill_handler._get_skill("skill-123")

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["skill_id"] == "skill-123"

    @pytest.mark.asyncio
    async def test_get_skill_not_found(self, skill_handler):
        """Skill not found returns 404."""
        with patch("aragora.skills.marketplace.get_marketplace", return_value=MockMarketplace()):
            result = await skill_handler._get_skill("nonexistent")

        assert result is not None
        assert result.status_code == 404


# ===========================================================================
# Test Get Versions (GET /api/v1/skills/marketplace/{skill_id}/versions)
# ===========================================================================


class TestSkillMarketplaceGetVersions:
    """Tests for get skill versions endpoint."""

    @pytest.mark.asyncio
    async def test_get_versions_success(self, skill_handler):
        """Happy path: get skill versions."""
        with patch("aragora.skills.marketplace.get_marketplace", return_value=MockMarketplace()):
            result = await skill_handler._get_versions("skill-123")

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["skill_id"] == "skill-123"
        assert len(data["versions"]) == 2

    @pytest.mark.asyncio
    async def test_get_versions_not_found(self, skill_handler):
        """Skill not found returns 404."""
        with patch("aragora.skills.marketplace.get_marketplace", return_value=MockMarketplace()):
            result = await skill_handler._get_versions("nonexistent")

        assert result is not None
        assert result.status_code == 404


# ===========================================================================
# Test Get Ratings (GET /api/v1/skills/marketplace/{skill_id}/ratings)
# ===========================================================================


class TestSkillMarketplaceGetRatings:
    """Tests for get skill ratings endpoint."""

    @pytest.mark.asyncio
    async def test_get_ratings_success(self, skill_handler):
        """Happy path: get skill ratings."""
        with patch("aragora.skills.marketplace.get_marketplace", return_value=MockMarketplace()):
            result = await skill_handler._get_ratings("skill-123", {})

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["skill_id"] == "skill-123"
        assert len(data["ratings"]) == 1

    @pytest.mark.asyncio
    async def test_get_ratings_with_pagination(self, skill_handler):
        """Get ratings with pagination params."""
        with patch("aragora.skills.marketplace.get_marketplace", return_value=MockMarketplace()):
            result = await skill_handler._get_ratings("skill-123", {"limit": "10", "offset": "5"})

        assert result is not None
        assert result.status_code == 200


# ===========================================================================
# Test Publish Skill (POST /api/v1/skills/marketplace/publish)
# ===========================================================================


class TestSkillMarketplacePublish:
    """Tests for publish skill endpoint."""

    @pytest.mark.asyncio
    async def test_publish_skill_success(self, skill_handler):
        """Happy path: publish skill."""
        auth_context = {"user_id": "user-123", "display_name": "Test User"}

        with (
            patch(
                "aragora.skills.registry.get_skill_registry",
                return_value=MockSkillRegistry(),
            ),
            patch("aragora.skills.publisher.SkillPublisher", return_value=MockSkillPublisher()),
        ):
            result = await skill_handler._publish_skill({"skill_name": "test-skill"}, auth_context)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_publish_skill_missing_name(self, skill_handler):
        """Missing skill_name returns 400."""
        result = await skill_handler._publish_skill({}, {"user_id": "user-123"})

        assert result is not None
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_publish_skill_not_found(self, skill_handler):
        """Skill not in registry returns 404."""
        auth_context = {"user_id": "user-123"}

        with patch(
            "aragora.skills.registry.get_skill_registry",
            return_value=MockSkillRegistry(),
        ):
            result = await skill_handler._publish_skill(
                {"skill_name": "nonexistent-skill"}, auth_context
            )

        assert result is not None
        assert result.status_code == 404


# ===========================================================================
# Test Install Skill (POST /api/v1/skills/marketplace/{skill_id}/install)
# ===========================================================================


class TestSkillMarketplaceInstall:
    """Tests for install skill endpoint."""

    @pytest.mark.asyncio
    async def test_install_skill_success(self, skill_handler):
        """Happy path: install skill."""
        auth_context = {"user_id": "user-123", "tenant_id": "tenant-1"}

        with patch("aragora.skills.installer.SkillInstaller", return_value=MockSkillInstaller()):
            result = await skill_handler._install_skill("skill-123", {}, auth_context)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_install_skill_with_version(self, skill_handler):
        """Install skill with specific version."""
        auth_context = {"user_id": "user-123", "tenant_id": "tenant-1"}

        with patch("aragora.skills.installer.SkillInstaller", return_value=MockSkillInstaller()):
            result = await skill_handler._install_skill(
                "skill-123", {"version": "1.0.0"}, auth_context
            )

        assert result is not None
        assert result.status_code == 200


# ===========================================================================
# Test Uninstall Skill (DELETE /api/v1/skills/marketplace/{skill_id}/install)
# ===========================================================================


class TestSkillMarketplaceUninstall:
    """Tests for uninstall skill endpoint."""

    @pytest.mark.asyncio
    async def test_uninstall_skill_success(self, skill_handler):
        """Happy path: uninstall skill."""
        auth_context = {"user_id": "user-123", "tenant_id": "tenant-1"}

        with patch("aragora.skills.installer.SkillInstaller", return_value=MockSkillInstaller()):
            result = await skill_handler._uninstall_skill("skill-123", auth_context)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_uninstall_skill_failed(self, skill_handler):
        """Uninstall failed returns 400."""
        auth_context = {"user_id": "user-123", "tenant_id": "tenant-1"}

        with patch("aragora.skills.installer.SkillInstaller", return_value=MockSkillInstaller()):
            result = await skill_handler._uninstall_skill("nonexistent", auth_context)

        assert result is not None
        assert result.status_code == 400


# ===========================================================================
# Test Rate Skill (POST /api/v1/skills/marketplace/{skill_id}/rate)
# ===========================================================================


class TestSkillMarketplaceRate:
    """Tests for rate skill endpoint."""

    @pytest.mark.asyncio
    async def test_rate_skill_success(self, skill_handler):
        """Happy path: rate skill."""
        auth_context = {"user_id": "user-123"}

        with patch("aragora.skills.marketplace.get_marketplace", return_value=MockMarketplace()):
            result = await skill_handler._rate_skill(
                "skill-123", {"rating": 5, "review": "Great!"}, auth_context
            )

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["rating"] == 5

    @pytest.mark.asyncio
    async def test_rate_skill_invalid_rating(self, skill_handler):
        """Invalid rating returns 400."""
        auth_context = {"user_id": "user-123"}

        result = await skill_handler._rate_skill("skill-123", {"rating": 6}, auth_context)

        assert result is not None
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_rate_skill_missing_rating(self, skill_handler):
        """Missing rating returns 400."""
        auth_context = {"user_id": "user-123"}

        result = await skill_handler._rate_skill("skill-123", {}, auth_context)

        assert result is not None
        assert result.status_code == 400


# ===========================================================================
# Test List Installed (GET /api/v1/skills/marketplace/installed)
# ===========================================================================


class TestSkillMarketplaceListInstalled:
    """Tests for list installed skills endpoint."""

    @pytest.mark.asyncio
    async def test_list_installed_success(self, skill_handler):
        """Happy path: list installed skills."""
        auth_context = {"user_id": "user-123", "tenant_id": "tenant-1"}

        with patch("aragora.skills.installer.SkillInstaller", return_value=MockSkillInstaller()):
            result = await skill_handler._list_installed(auth_context)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["count"] == 1
        assert len(data["skills"]) == 1


# ===========================================================================
# Test Get Stats (GET /api/v1/skills/marketplace/stats)
# ===========================================================================


class TestSkillMarketplaceGetStats:
    """Tests for get marketplace stats endpoint."""

    @pytest.mark.asyncio
    async def test_get_stats_success(self, skill_handler):
        """Happy path: get marketplace stats."""
        with patch("aragora.skills.marketplace.get_marketplace", return_value=MockMarketplace()):
            result = await skill_handler._get_stats()

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["total_skills"] == 150
        assert data["total_installs"] == 5000


# ===========================================================================
# Test Authentication
# ===========================================================================


class TestSkillMarketplaceAuth:
    """Tests for authentication on protected endpoints."""

    @pytest.mark.asyncio
    async def test_installed_requires_auth(self, skill_handler):
        """List installed requires authentication."""
        handler = make_mock_handler()

        result = await skill_handler.handle(
            "/api/v1/skills/marketplace/installed", {}, handler, "GET"
        )

        assert result is not None
        assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_publish_requires_auth(self, skill_handler):
        """Publish requires authentication."""
        handler = make_mock_handler(body={"skill_name": "test"}, method="POST")

        result = await skill_handler.handle(
            "/api/v1/skills/marketplace/publish", {}, handler, "POST", {}
        )

        assert result is not None
        assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_install_requires_auth(self, skill_handler):
        """Install requires authentication."""
        handler = make_mock_handler(method="POST")

        result = await skill_handler.handle(
            "/api/v1/skills/marketplace/skill-123/install", {}, handler, "POST", {}
        )

        assert result is not None
        assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_rate_requires_auth(self, skill_handler):
        """Rate requires authentication."""
        handler = make_mock_handler(body={"rating": 5}, method="POST")

        result = await skill_handler.handle(
            "/api/v1/skills/marketplace/skill-123/rate", {}, handler, "POST", {}
        )

        assert result is not None
        assert result.status_code == 401


# ===========================================================================
# Test Error Handling
# ===========================================================================


class TestSkillMarketplaceErrorHandling:
    """Tests for error handling in skill marketplace handler."""

    @pytest.mark.asyncio
    async def test_search_error_handling(self, skill_handler):
        """Search error returns 500."""
        with patch("aragora.skills.marketplace.get_marketplace") as mock_marketplace:
            mock_mp = MagicMock()
            mock_mp.search = AsyncMock(side_effect=Exception("Database error"))
            mock_marketplace.return_value = mock_mp

            result = await skill_handler._search_skills({})

        assert result is not None
        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_invalid_category_enum(self, skill_handler):
        """Invalid category enum is handled gracefully."""
        with (
            patch("aragora.skills.marketplace.get_marketplace", return_value=MockMarketplace()),
            patch("aragora.skills.marketplace.SkillCategory") as mock_category,
        ):
            mock_category.side_effect = ValueError("Invalid category")

            result = await skill_handler._search_skills({"category": "invalid"})

        # Should catch the ValueError and return 500
        assert result is not None
        assert result.status_code == 500
