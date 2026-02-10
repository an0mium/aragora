"""
Tests for skill verification enforcement and admin verification API.

Tests cover:
- InstallationPolicy.require_verified_only enforcement in SkillInstaller
- Admin verification endpoint (PUT/DELETE /api/v1/skills/marketplace/{skill_id}/verify)
- Non-admin cannot set verification status
- Audit logging for verification changes
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Import the modules under test with Slack stub workaround
# ---------------------------------------------------------------------------


def _ensure_slack_stubs():
    """Stub broken slack imports if needed."""
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


_ensure_slack_stubs()

from aragora.skills.installer import InstallationPolicy, SkillInstaller
from aragora.skills.marketplace import (
    InstallResult,
    SkillCategory,
    SkillListing,
    SkillMarketplace,
    SkillTier,
)


def _import_skill_marketplace_handler():
    """Import handler module with slack stubs."""
    try:
        import aragora.server.handlers.skill_marketplace as mod

        return mod
    except (ImportError, ModuleNotFoundError):
        pass

    to_remove = [k for k in sys.modules if k.startswith("aragora.server.handlers")]
    for k in to_remove:
        del sys.modules[k]

    _ensure_slack_stubs()

    import aragora.server.handlers.skill_marketplace as mod

    return mod


handler_module = _import_skill_marketplace_handler()
SkillMarketplaceHandler = handler_module.SkillMarketplaceHandler


# ===========================================================================
# Helpers
# ===========================================================================


def _make_listing(
    skill_id: str = "test-skill",
    is_published: bool = True,
    is_verified: bool = False,
    is_deprecated: bool = False,
    tier: SkillTier = SkillTier.FREE,
) -> SkillListing:
    """Create a SkillListing for tests."""
    return SkillListing(
        skill_id=skill_id,
        name="Test Skill",
        description="A skill for testing",
        author_id="author-1",
        author_name="Author One",
        category=SkillCategory.CUSTOM,
        tier=tier,
        is_published=is_published,
        is_verified=is_verified,
        is_deprecated=is_deprecated,
    )


class MockMarketplaceForVerification:
    """Mock marketplace that tracks get_skill and set_verified calls."""

    def __init__(self, listing: SkillListing | None = None):
        self._listing = listing or _make_listing()
        self.set_verified_calls: list[tuple[str, bool]] = []

    async def get_skill(self, skill_id: str) -> SkillListing | None:
        if skill_id == "nonexistent":
            return None
        return self._listing

    async def get_installed_skills(self, tenant_id: str) -> list[SkillListing]:
        return []

    async def is_installed(self, skill_id: str, tenant_id: str) -> bool:
        return False

    async def install(self, **kwargs) -> InstallResult:
        return InstallResult(
            success=True,
            skill_id=kwargs.get("skill_id", "test-skill"),
            version=kwargs.get("version") or "1.0.0",
        )

    async def set_verified(self, skill_id: str, verified: bool) -> bool:
        self.set_verified_calls.append((skill_id, verified))
        return True

    async def get_stats(self) -> dict:
        return {"published_skills": 0, "total_installs": 0}


# ===========================================================================
# Test: require_verified_only policy enforcement
# ===========================================================================


class TestVerifiedOnlyPolicyEnforcement:
    """Tests that require_verified_only policy blocks unverified skill installation."""

    @pytest.mark.asyncio
    async def test_verified_only_blocks_unverified_skill(self):
        """When require_verified_only=True, installing an unverified skill is rejected."""
        mock_mp = MockMarketplaceForVerification(
            listing=_make_listing(is_verified=False)
        )
        installer = SkillInstaller(marketplace=mock_mp)
        installer.set_policy(
            "tenant-1",
            InstallationPolicy(require_verified_only=True),
        )

        result = await installer.install(
            skill_id="test-skill",
            tenant_id="tenant-1",
            user_id="user-1",
            permissions=["skills:install"],
        )

        assert result.success is False
        assert "verified" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_verified_only_allows_verified_skill(self):
        """When require_verified_only=True, installing a verified skill succeeds."""
        mock_mp = MockMarketplaceForVerification(
            listing=_make_listing(is_verified=True)
        )
        installer = SkillInstaller(marketplace=mock_mp)
        installer.set_policy(
            "tenant-1",
            InstallationPolicy(require_verified_only=True),
        )

        result = await installer.install(
            skill_id="test-skill",
            tenant_id="tenant-1",
            user_id="user-1",
            permissions=["skills:install"],
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_no_verified_requirement_allows_unverified(self):
        """When require_verified_only=False (default), unverified skills install fine."""
        mock_mp = MockMarketplaceForVerification(
            listing=_make_listing(is_verified=False)
        )
        installer = SkillInstaller(marketplace=mock_mp)
        # Default policy has require_verified_only=False
        installer.set_policy("tenant-1", InstallationPolicy())

        result = await installer.install(
            skill_id="test-skill",
            tenant_id="tenant-1",
            user_id="user-1",
            permissions=["skills:install"],
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_can_install_check_blocks_unverified(self):
        """can_install returns not allowed for unverified skills under strict policy."""
        mock_mp = MockMarketplaceForVerification(
            listing=_make_listing(is_verified=False)
        )
        installer = SkillInstaller(marketplace=mock_mp)
        installer.set_policy(
            "tenant-1",
            InstallationPolicy(require_verified_only=True),
        )

        check = await installer.can_install(
            skill_id="test-skill",
            tenant_id="tenant-1",
            user_id="user-1",
            permissions=["skills:install"],
        )

        assert check.allowed is False
        assert "verified" in (check.reason or "").lower()

    @pytest.mark.asyncio
    async def test_can_install_check_allows_verified(self):
        """can_install returns allowed for verified skills under strict policy."""
        mock_mp = MockMarketplaceForVerification(
            listing=_make_listing(is_verified=True)
        )
        installer = SkillInstaller(marketplace=mock_mp)
        installer.set_policy(
            "tenant-1",
            InstallationPolicy(require_verified_only=True),
        )

        check = await installer.can_install(
            skill_id="test-skill",
            tenant_id="tenant-1",
            user_id="user-1",
            permissions=["skills:install"],
        )

        assert check.allowed is True


# ===========================================================================
# Test: SkillMarketplace.set_verified
# ===========================================================================


class TestMarketplaceSetVerified:
    """Tests for SkillMarketplace.set_verified method."""

    @pytest.mark.asyncio
    async def test_set_verified_true(self):
        """set_verified(True) marks a skill as verified in the DB."""
        mp = SkillMarketplace(db_path=":memory:")

        # Publish a skill first
        mock_skill = MagicMock()
        mock_skill.manifest.name = "test-skill"
        mock_skill.manifest.version = "1.0.0"
        mock_skill.manifest.description = "Test"
        mock_skill.manifest.capabilities = []
        mock_skill.manifest.required_permissions = []
        mock_skill.manifest.tags = []

        listing = await mp.publish(
            skill=mock_skill,
            author_id="author-1",
            author_name="Author",
        )

        assert listing.is_verified is False

        success = await mp.set_verified(listing.skill_id, True)
        assert success is True

        updated = await mp.get_skill(listing.skill_id)
        assert updated is not None
        assert updated.is_verified is True

    @pytest.mark.asyncio
    async def test_set_verified_false(self):
        """set_verified(False) revokes verification."""
        mp = SkillMarketplace(db_path=":memory:")

        mock_skill = MagicMock()
        mock_skill.manifest.name = "test-skill"
        mock_skill.manifest.version = "1.0.0"
        mock_skill.manifest.description = "Test"
        mock_skill.manifest.capabilities = []
        mock_skill.manifest.required_permissions = []
        mock_skill.manifest.tags = []

        listing = await mp.publish(
            skill=mock_skill,
            author_id="author-1",
            author_name="Author",
        )

        await mp.set_verified(listing.skill_id, True)
        success = await mp.set_verified(listing.skill_id, False)
        assert success is True

        updated = await mp.get_skill(listing.skill_id)
        assert updated is not None
        assert updated.is_verified is False

    @pytest.mark.asyncio
    async def test_set_verified_nonexistent_skill(self):
        """set_verified returns False for a nonexistent skill."""
        mp = SkillMarketplace(db_path=":memory:")

        success = await mp.set_verified("nonexistent-skill", True)
        assert success is False


# ===========================================================================
# Test: Verification admin endpoint in handler
# ===========================================================================


class TestVerificationEndpoint:
    """Tests for PUT/DELETE /api/v1/skills/marketplace/{skill_id}/verify."""

    @pytest.fixture
    def skill_handler(self):
        """Create SkillMarketplaceHandler for testing."""
        return SkillMarketplaceHandler({})

    @pytest.mark.asyncio
    async def test_set_verification_success(self, skill_handler):
        """Admin can set verification to True."""
        auth_context = {"user_id": "admin-1", "tenant_id": "tenant-1"}
        mock_mp = MockMarketplaceForVerification()

        with patch(
            "aragora.skills.marketplace.get_marketplace",
            return_value=mock_mp,
        ):
            result = await skill_handler._set_verification("test-skill", True, auth_context)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["skill_id"] == "test-skill"
        assert data["is_verified"] is True
        assert data["changed_by"] == "admin-1"
        assert mock_mp.set_verified_calls == [("test-skill", True)]

    @pytest.mark.asyncio
    async def test_revoke_verification_success(self, skill_handler):
        """Admin can revoke verification."""
        auth_context = {"user_id": "admin-1", "tenant_id": "tenant-1"}
        mock_mp = MockMarketplaceForVerification()

        with patch(
            "aragora.skills.marketplace.get_marketplace",
            return_value=mock_mp,
        ):
            result = await skill_handler._set_verification("test-skill", False, auth_context)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["is_verified"] is False
        assert mock_mp.set_verified_calls == [("test-skill", False)]

    @pytest.mark.asyncio
    async def test_verify_nonexistent_skill(self, skill_handler):
        """Verifying a nonexistent skill returns 404."""
        auth_context = {"user_id": "admin-1"}
        mock_mp = MockMarketplaceForVerification()

        with patch(
            "aragora.skills.marketplace.get_marketplace",
            return_value=mock_mp,
        ):
            result = await skill_handler._set_verification("nonexistent", True, auth_context)

        assert result is not None
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_verify_marketplace_unavailable(self, skill_handler):
        """Returns 503 when marketplace is not available."""
        auth_context = {"user_id": "admin-1"}

        with patch(
            "aragora.skills.marketplace.get_marketplace",
            side_effect=ImportError("not available"),
        ):
            result = await skill_handler._set_verification("test-skill", True, auth_context)

        assert result is not None
        assert result.status_code == 503


# ===========================================================================
# Test: Routing for verify endpoints
# ===========================================================================


class TestVerificationRouting:
    """Tests for routing of verify endpoints."""

    @pytest.fixture
    def skill_handler(self):
        return SkillMarketplaceHandler({})

    def test_can_handle_verify_put(self, skill_handler):
        """Handler recognizes PUT verify path."""
        assert skill_handler.can_handle(
            "/api/v1/skills/marketplace/skill-123/verify", "PUT"
        ) is True

    def test_can_handle_verify_delete(self, skill_handler):
        """Handler recognizes DELETE verify path."""
        assert skill_handler.can_handle(
            "/api/v1/skills/marketplace/skill-123/verify", "DELETE"
        ) is True

    @pytest.mark.asyncio
    async def test_verify_requires_auth(self, skill_handler):
        """PUT verify requires authentication."""
        handler = MagicMock()
        handler.command = "PUT"
        handler.path = "/api/v1/skills/marketplace/skill-123/verify"
        handler.headers = {}
        handler.client_address = ("127.0.0.1", 12345)

        result = await skill_handler.handle(
            "/api/v1/skills/marketplace/skill-123/verify",
            {},
            handler,
            "PUT",
        )

        assert result is not None
        assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_revoke_verify_requires_auth(self, skill_handler):
        """DELETE verify requires authentication."""
        handler = MagicMock()
        handler.command = "DELETE"
        handler.path = "/api/v1/skills/marketplace/skill-123/verify"
        handler.headers = {}
        handler.client_address = ("127.0.0.1", 12345)

        result = await skill_handler.handle(
            "/api/v1/skills/marketplace/skill-123/verify",
            {},
            handler,
            "DELETE",
        )

        assert result is not None
        assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_verify_requires_admin_permission(self, skill_handler):
        """PUT verify returns 403 when user lacks skills:admin."""
        from aragora.server.handlers.secure import ForbiddenError

        handler = MagicMock()
        handler.command = "PUT"
        handler.path = "/api/v1/skills/marketplace/skill-123/verify"
        handler.headers = {}
        handler.client_address = ("127.0.0.1", 12345)

        # Mock get_auth_context to return a non-admin user
        mock_auth = MagicMock()
        mock_auth.user_id = "regular-user"
        mock_auth.tenant_id = "tenant-1"

        with (
            patch.object(
                skill_handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth
            ),
            patch.object(
                skill_handler,
                "check_permission",
                side_effect=ForbiddenError("Permission denied: skills:admin"),
            ),
        ):
            result = await skill_handler.handle(
                "/api/v1/skills/marketplace/skill-123/verify",
                {},
                handler,
                "PUT",
            )

        assert result is not None
        assert result.status_code == 403

    @pytest.mark.asyncio
    async def test_revoke_verify_requires_admin_permission(self, skill_handler):
        """DELETE verify returns 403 when user lacks skills:admin."""
        from aragora.server.handlers.secure import ForbiddenError

        handler = MagicMock()

        mock_auth = MagicMock()
        mock_auth.user_id = "regular-user"
        mock_auth.tenant_id = "tenant-1"

        with (
            patch.object(
                skill_handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth
            ),
            patch.object(
                skill_handler,
                "check_permission",
                side_effect=ForbiddenError("Permission denied: skills:admin"),
            ),
        ):
            result = await skill_handler.handle(
                "/api/v1/skills/marketplace/skill-123/verify",
                {},
                handler,
                "DELETE",
            )

        assert result is not None
        assert result.status_code == 403
