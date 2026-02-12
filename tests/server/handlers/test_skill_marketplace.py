"""Tests for SkillMarketplaceHandler."""

from __future__ import annotations

import sys
import types as _types_mod

# Pre-stub Slack modules to prevent import chain failures
_SLACK_ATTRS = [
    "SlackHandler",
    "get_slack_handler",
    "get_slack_integration",
    "get_workspace_store",
    "resolve_workspace",
    "create_tracked_task",
    "_validate_slack_url",
    "SLACK_SIGNING_SECRET",
    "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL",
    "SLACK_ALLOWED_DOMAINS",
    "SignatureVerifierMixin",
    "CommandsMixin",
    "EventsMixin",
    "init_slack_handler",
]
for _mod_name in (
    "aragora.server.handlers.social.slack.handler",
    "aragora.server.handlers.social.slack",
    "aragora.server.handlers.social._slack_impl",
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m


import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.skill_marketplace import SkillMarketplaceHandler


def parse_response(result):
    """Parse HandlerResult body to dict."""
    return json.loads(result.body.decode("utf-8"))


# =============================================================================
# Mock Objects
# =============================================================================


@dataclass
class MockSkillListing:
    """Mock skill listing."""

    skill_id: str
    name: str
    description: str
    version: str = "1.0.0"
    category: str = "custom"
    tier: str = "free"
    author_id: str = "author-123"
    author_name: str = "Test Author"
    rating: float = 4.5
    downloads: int = 100

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "category": self.category,
            "tier": self.tier,
            "author_id": self.author_id,
            "author_name": self.author_name,
            "rating": self.rating,
            "downloads": self.downloads,
        }


@dataclass
class MockSkillVersion:
    """Mock skill version."""

    version: str
    changelog: str = ""
    published_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "changelog": self.changelog,
            "published_at": self.published_at,
        }


@dataclass
class MockSkillRating:
    """Mock skill rating."""

    user_id: str
    rating: int
    review: str | None = None
    created_at: str | None = None

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

    success: bool
    skill_id: str
    version: str = "1.0.0"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "skill_id": self.skill_id,
            "version": self.version,
            "error": self.error,
        }


@dataclass
class MockInstalledSkill:
    """Mock installed skill."""

    skill_id: str
    version: str
    installed_at: str = "2026-01-27T12:00:00Z"

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "version": self.version,
            "installed_at": self.installed_at,
        }


class MockMarketplace:
    """Mock skill marketplace."""

    async def search(
        self,
        query: str = "",
        category=None,
        tier=None,
        tags=None,
        author_id=None,
        sort_by="rating",
        limit=20,
        offset=0,
    ) -> list[MockSkillListing]:
        return [
            MockSkillListing(
                skill_id="skill-1",
                name="Test Skill",
                description="A test skill",
            )
        ]

    async def get_skill(self, skill_id: str) -> MockSkillListing | None:
        if skill_id == "nonexistent":
            return None
        return MockSkillListing(
            skill_id=skill_id,
            name="Test Skill",
            description="A test skill",
        )

    async def get_versions(self, skill_id: str) -> list[MockSkillVersion]:
        return [
            MockSkillVersion(version="1.0.0", changelog="Initial release"),
            MockSkillVersion(version="1.1.0", changelog="Bug fixes"),
        ]

    async def get_ratings(
        self, skill_id: str, limit: int = 20, offset: int = 0
    ) -> list[MockSkillRating]:
        return [MockSkillRating(user_id="user-1", rating=5, review="Great skill!")]

    async def rate(
        self, skill_id: str, user_id: str, rating: int, review: str | None = None
    ) -> MockSkillRating:
        return MockSkillRating(user_id=user_id, rating=rating, review=review)

    async def get_stats(self) -> dict[str, Any]:
        return {
            "total_skills": 100,
            "total_downloads": 5000,
            "categories": {"custom": 50, "automation": 30, "analytics": 20},
        }


class MockInstaller:
    """Mock skill installer."""

    async def install(
        self,
        skill_id: str,
        tenant_id: str,
        user_id: str,
        version: str | None = None,
        permissions: list[str] = None,
    ) -> MockInstallResult:
        return MockInstallResult(success=True, skill_id=skill_id)

    async def uninstall(
        self,
        skill_id: str,
        tenant_id: str,
        user_id: str,
        permissions: list[str] = None,
    ) -> bool:
        return True

    async def get_installed(self, tenant_id: str) -> list[MockInstalledSkill]:
        return [MockInstalledSkill(skill_id="skill-1", version="1.0.0")]


class MockHandler:
    """Mock HTTP handler for testing."""

    def __init__(self, client_ip: str = "127.0.0.1"):
        self.headers = {"X-Forwarded-For": client_ip}


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def handler():
    """Create a test handler."""
    return SkillMarketplaceHandler(server_context={})


@pytest.fixture(autouse=True)
def clear_module_state():
    """Clear any module-level state between tests."""
    yield


# =============================================================================
# Test Handler Routing
# =============================================================================


class TestHandlerRouting:
    """Tests for handler routing."""

    def test_can_handle_non_versioned_search(self, handler):
        """Test can_handle for non-versioned search route."""
        assert handler.can_handle("/api/skills/marketplace/search") is True

    def test_can_handle_versioned_search(self, handler):
        """Test can_handle for versioned search route."""
        assert handler.can_handle("/api/v1/skills/marketplace/search") is True

    def test_cannot_handle_invalid_route(self, handler):
        """Test can_handle for invalid route."""
        assert handler.can_handle("/api/v1/other/route") is False
        assert handler.can_handle("/api/other/route") is False


# =============================================================================
# Test Search Skills
# =============================================================================


class TestSearchSkills:
    """Tests for search skills endpoint."""

    @pytest.mark.asyncio
    async def test_search_success(self, handler):
        """Test successful skill search."""
        with patch("aragora.skills.marketplace.get_marketplace") as mock_get:
            mock_get.return_value = MockMarketplace()

            result = await handler._search_skills({"q": "test"})

            assert result.status_code == 200
            data = parse_response(result)
            assert "results" in data
            assert data["count"] == 1

    @pytest.mark.asyncio
    async def test_search_with_filters(self, handler):
        """Test search with category and tier filters."""
        with patch("aragora.skills.marketplace.get_marketplace") as mock_get:
            mock_get.return_value = MockMarketplace()

            result = await handler._search_skills(
                {
                    "q": "test",
                    "category": "automation",
                    "tier": "premium",
                    "tags": "ai,ml",
                }
            )

            assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_search_marketplace_unavailable(self, handler):
        """Test search when marketplace not available."""
        with patch("aragora.skills.marketplace.get_marketplace") as mock_get:
            mock_get.side_effect = ImportError("Not available")

            result = await handler._search_skills({})

            assert result.status_code == 503


# =============================================================================
# Test Get Skill
# =============================================================================


class TestGetSkill:
    """Tests for get skill endpoint."""

    @pytest.mark.asyncio
    async def test_get_skill_success(self, handler):
        """Test successful skill retrieval."""
        with patch("aragora.skills.marketplace.get_marketplace") as mock_get:
            mock_get.return_value = MockMarketplace()

            result = await handler._get_skill("skill-123")

            assert result.status_code == 200
            data = parse_response(result)
            assert data["skill_id"] == "skill-123"

    @pytest.mark.asyncio
    async def test_get_skill_not_found(self, handler):
        """Test skill not found."""
        with patch("aragora.skills.marketplace.get_marketplace") as mock_get:
            mock_get.return_value = MockMarketplace()

            result = await handler._get_skill("nonexistent")

            assert result.status_code == 404


# =============================================================================
# Test Get Versions
# =============================================================================


class TestGetVersions:
    """Tests for get versions endpoint."""

    @pytest.mark.asyncio
    async def test_get_versions_success(self, handler):
        """Test successful version listing."""
        with patch("aragora.skills.marketplace.get_marketplace") as mock_get:
            mock_get.return_value = MockMarketplace()

            result = await handler._get_versions("skill-123")

            assert result.status_code == 200
            data = parse_response(result)
            assert data["skill_id"] == "skill-123"
            assert len(data["versions"]) == 2


# =============================================================================
# Test Get Ratings
# =============================================================================


class TestGetRatings:
    """Tests for get ratings endpoint."""

    @pytest.mark.asyncio
    async def test_get_ratings_success(self, handler):
        """Test successful ratings retrieval."""
        with patch("aragora.skills.marketplace.get_marketplace") as mock_get:
            mock_get.return_value = MockMarketplace()

            result = await handler._get_ratings("skill-123", {})

            assert result.status_code == 200
            data = parse_response(result)
            assert data["skill_id"] == "skill-123"
            assert len(data["ratings"]) == 1


# =============================================================================
# Test Install Skill
# =============================================================================


class TestInstallSkill:
    """Tests for install skill endpoint."""

    @pytest.mark.asyncio
    async def test_install_success(self, handler):
        """Test successful skill installation."""
        with patch("aragora.skills.installer.SkillInstaller") as mock_installer_cls:
            mock_installer_cls.return_value = MockInstaller()

            auth_context = {"user_id": "user-123", "tenant_id": "tenant-1"}
            result = await handler._install_skill("skill-123", {}, auth_context)

            assert result.status_code == 200
            data = parse_response(result)
            assert data["success"] is True

    @pytest.mark.asyncio
    async def test_install_with_version(self, handler):
        """Test installation with specific version."""
        with patch("aragora.skills.installer.SkillInstaller") as mock_installer_cls:
            mock_installer_cls.return_value = MockInstaller()

            auth_context = {"user_id": "user-123", "tenant_id": "tenant-1"}
            result = await handler._install_skill("skill-123", {"version": "1.1.0"}, auth_context)

            assert result.status_code == 200


# =============================================================================
# Test Uninstall Skill
# =============================================================================


class TestUninstallSkill:
    """Tests for uninstall skill endpoint."""

    @pytest.mark.asyncio
    async def test_uninstall_success(self, handler):
        """Test successful skill uninstallation."""
        with patch("aragora.skills.installer.SkillInstaller") as mock_installer_cls:
            mock_installer_cls.return_value = MockInstaller()

            auth_context = {"user_id": "user-123", "tenant_id": "tenant-1"}
            result = await handler._uninstall_skill("skill-123", auth_context)

            assert result.status_code == 200
            data = parse_response(result)
            assert data["success"] is True


# =============================================================================
# Test Rate Skill
# =============================================================================


class TestRateSkill:
    """Tests for rate skill endpoint."""

    @pytest.mark.asyncio
    async def test_rate_success(self, handler):
        """Test successful skill rating."""
        with patch("aragora.skills.marketplace.get_marketplace") as mock_get:
            mock_get.return_value = MockMarketplace()

            auth_context = {"user_id": "user-123"}
            result = await handler._rate_skill(
                "skill-123", {"rating": 5, "review": "Excellent!"}, auth_context
            )

            assert result.status_code == 200
            data = parse_response(result)
            assert data["rating"] == 5

    @pytest.mark.asyncio
    async def test_rate_invalid_rating(self, handler):
        """Test rating with invalid value."""
        auth_context = {"user_id": "user-123"}
        result = await handler._rate_skill("skill-123", {"rating": 6}, auth_context)

        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_rate_missing_rating(self, handler):
        """Test rating without rating value."""
        auth_context = {"user_id": "user-123"}
        result = await handler._rate_skill("skill-123", {}, auth_context)

        assert result.status_code == 400


# =============================================================================
# Test List Installed
# =============================================================================


class TestListInstalled:
    """Tests for list installed skills endpoint."""

    @pytest.mark.asyncio
    async def test_list_installed_success(self, handler):
        """Test successful installed skills listing."""
        with patch("aragora.skills.installer.SkillInstaller") as mock_installer_cls:
            mock_installer_cls.return_value = MockInstaller()

            auth_context = {"user_id": "user-123", "tenant_id": "tenant-1"}
            result = await handler._list_installed(auth_context)

            assert result.status_code == 200
            data = parse_response(result)
            assert data["count"] == 1


# =============================================================================
# Test Get Stats
# =============================================================================


class TestGetStats:
    """Tests for get marketplace stats endpoint."""

    @pytest.mark.asyncio
    async def test_get_stats_success(self, handler):
        """Test successful stats retrieval."""
        with patch("aragora.skills.marketplace.get_marketplace") as mock_get:
            mock_get.return_value = MockMarketplace()

            result = await handler._get_stats()

            assert result.status_code == 200
            data = parse_response(result)
            assert data["total_skills"] == 100
            assert data["total_downloads"] == 5000


# =============================================================================
# Test Authentication
# =============================================================================


class TestAuthentication:
    """Tests for authentication requirements."""

    @pytest.mark.asyncio
    async def test_versioned_paths_matched(self, handler):
        """Test that versioned paths route to the handler."""
        with patch.object(handler, "get_auth_context", new=AsyncMock(return_value={})):
            result = await handler.handle(
                "/api/v1/skills/marketplace/installed", {}, MockHandler(), method="GET"
            )
            assert result is not None
            assert result.status_code == 401

            result = await handler.handle(
                "/api/v1/skills/marketplace/publish",
                {},
                MockHandler(),
                method="POST",
                body={},
            )
            assert result is not None
            assert result.status_code == 401

        with patch("aragora.skills.marketplace.get_marketplace") as mock_get:
            mock_get.return_value = MockMarketplace()
            result = await handler.handle(
                "/api/v1/skills/marketplace/search",
                {"q": "test"},
                MockHandler(),
                method="GET",
            )
            assert result is not None
            assert result.status_code == 200

            result = await handler.handle(
                "/api/v1/skills/marketplace/stats",
                {},
                MockHandler(),
                method="GET",
            )
            assert result is not None
            assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_search_method_directly(self, handler):
        """Test _search_skills method directly to verify it works."""
        with patch("aragora.skills.marketplace.get_marketplace") as mock_get:
            mock_get.return_value = MockMarketplace()

            result = await handler._search_skills({})

            assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_get_stats_method_directly(self, handler):
        """Test _get_stats method directly to verify it works."""
        with patch("aragora.skills.marketplace.get_marketplace") as mock_get:
            mock_get.return_value = MockMarketplace()

            result = await handler._get_stats()

            assert result.status_code == 200
