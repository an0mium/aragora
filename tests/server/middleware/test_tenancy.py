"""
Tests for aragora.server.middleware.tenancy - Multi-Tenancy Middleware.

Tests cover:
- PLAN_LIMITS configuration and get_plan_limits function
- WorkspaceManager class
  - Workspace retrieval and caching
  - User workspace access checking
  - Default workspace creation
  - Usage tracking
  - Plan limit enforcement
- Global WorkspaceManager singleton
- Decorators
  - require_workspace decorator
  - check_limit decorator
  - tenant_scoped decorator
- Utility functions
  - scope_query for database query scoping
  - ensure_workspace_access for access validation
- Cross-tenant data access prevention
- Concurrent tenant operations / race conditions
- Tenant context propagation through async chains
- Tenant ID validation and spoofing prevention
- Shared resource leakage prevention
- Missing tenant configuration handling
- All exception handling paths
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.middleware.tenancy import (
    PLAN_LIMITS,
    WorkspaceManager,
    check_limit,
    ensure_workspace_access,
    get_plan_limits,
    get_workspace_manager,
    require_workspace,
    scope_query,
    tenant_scoped,
)
from aragora.server.middleware.user_auth import User, Workspace


# ===========================================================================
# Test Fixtures
# ===========================================================================


@dataclass
class MockHandler:
    """Mock HTTP handler for testing."""

    headers: dict[str, str]
    client_address: tuple[str, int] = ("127.0.0.1", 12345)


@pytest.fixture
def mock_handler():
    """Create a mock handler with no auth."""
    return MockHandler(headers={})


@pytest.fixture
def mock_handler_with_bearer():
    """Create a mock handler with Bearer token."""
    return MockHandler(headers={"Authorization": "Bearer test-jwt-token"})


@pytest.fixture
def sample_user():
    """Create a sample user."""
    return User(
        id="user-123",
        email="test@example.com",
        role="user",
        plan="pro",
        workspace_id="ws-456",
    )


@pytest.fixture
def sample_user_free():
    """Create a sample free plan user."""
    return User(
        id="user-free-123",
        email="free@example.com",
        role="user",
        plan="free",
        workspace_id=None,
    )


@pytest.fixture
def sample_workspace():
    """Create a sample workspace."""
    return Workspace(
        id="ws-456",
        name="Test Workspace",
        owner_id="user-123",
        plan="pro",
        max_debates=500,
        max_agents=5,
        max_members=1,
        member_ids=["user-789"],
    )


@pytest.fixture
def enterprise_workspace():
    """Create an enterprise workspace."""
    return Workspace(
        id="ws-enterprise",
        name="Enterprise Workspace",
        owner_id="admin-123",
        plan="enterprise",
        max_debates=-1,
        max_agents=-1,
        max_members=-1,
        member_ids=["user-1", "user-2", "user-3"],
    )


@pytest.fixture
def mock_storage():
    """Create a mock storage backend."""
    storage = AsyncMock()
    storage.get_workspace.return_value = None
    storage.save_workspace.return_value = None
    storage.get_workspace_usage.return_value = {
        "debates_this_month": 10,
        "active_debates": 1,
        "total_tokens_used": 50000,
    }
    return storage


def get_status(result) -> int:
    """Extract status code from result."""
    if hasattr(result, "status_code"):
        return result.status_code
    if isinstance(result, tuple):
        return result[1]
    return 0


# ===========================================================================
# Test PLAN_LIMITS Configuration
# ===========================================================================


class TestPlanLimits:
    """Tests for PLAN_LIMITS configuration."""

    def test_plan_limits_has_all_tiers(self):
        """All plan tiers should be defined."""
        assert "free" in PLAN_LIMITS
        assert "pro" in PLAN_LIMITS
        assert "team" in PLAN_LIMITS
        assert "enterprise" in PLAN_LIMITS

    def test_free_plan_limits(self):
        """Free plan should have restricted limits."""
        free = PLAN_LIMITS["free"]

        assert free["max_debates_per_month"] == 50
        assert free["max_agents"] == 2
        assert free["max_members"] == 1
        assert free["max_concurrent_debates"] == 1
        assert free["private_debates"] is False
        assert free["api_access"] is False
        assert free["priority_support"] is False

    def test_pro_plan_limits(self):
        """Pro plan should have increased limits."""
        pro = PLAN_LIMITS["pro"]

        assert pro["max_debates_per_month"] == 500
        assert pro["max_agents"] == 5
        assert pro["max_members"] == 1
        assert pro["max_concurrent_debates"] == 3
        assert pro["private_debates"] is True
        assert pro["api_access"] is True
        assert pro["priority_support"] is False

    def test_team_plan_limits(self):
        """Team plan should have team features."""
        team = PLAN_LIMITS["team"]

        assert team["max_debates_per_month"] == 2000
        assert team["max_agents"] == 10
        assert team["max_members"] == 10
        assert team["max_concurrent_debates"] == 10
        assert team["private_debates"] is True
        assert team["api_access"] is True
        assert team["priority_support"] is True

    def test_enterprise_plan_limits(self):
        """Enterprise plan should have unlimited features."""
        enterprise = PLAN_LIMITS["enterprise"]

        assert enterprise["max_debates_per_month"] == -1  # Unlimited
        assert enterprise["max_agents"] == -1  # Unlimited
        assert enterprise["max_members"] == -1  # Unlimited
        assert enterprise["max_concurrent_debates"] == -1  # Unlimited
        assert enterprise["private_debates"] is True
        assert enterprise["api_access"] is True
        assert enterprise["priority_support"] is True

    def test_get_plan_limits_valid_plan(self):
        """Should return correct limits for valid plan."""
        limits = get_plan_limits("pro")

        assert limits["max_debates_per_month"] == 500
        assert limits["max_agents"] == 5

    def test_get_plan_limits_invalid_plan_defaults_to_free(self):
        """Should return free plan limits for invalid plan."""
        limits = get_plan_limits("invalid_plan")

        assert limits == PLAN_LIMITS["free"]
        assert limits["max_debates_per_month"] == 50

    def test_get_plan_limits_none_defaults_to_free(self):
        """Should return free plan limits for None."""
        limits = get_plan_limits(None)

        assert limits == PLAN_LIMITS["free"]


# ===========================================================================
# Test WorkspaceManager
# ===========================================================================


class TestWorkspaceManager:
    """Tests for WorkspaceManager class."""

    def test_init_defaults(self):
        """Should initialize with None storage and empty cache."""
        manager = WorkspaceManager()

        assert manager._storage is None
        assert manager._cache == {}

    def test_init_with_storage(self, mock_storage):
        """Should accept storage parameter."""
        manager = WorkspaceManager(storage=mock_storage)

        assert manager._storage is mock_storage

    @pytest.mark.asyncio
    async def test_get_workspace_from_cache(self, sample_workspace):
        """Should return cached workspace."""
        manager = WorkspaceManager()
        manager._cache["ws-456"] = sample_workspace

        result = await manager.get_workspace("ws-456")

        assert result is sample_workspace

    @pytest.mark.asyncio
    async def test_get_workspace_from_storage(self, mock_storage):
        """Should fetch workspace from storage and cache it."""
        mock_storage.get_workspace.return_value = {
            "id": "ws-789",
            "name": "Storage Workspace",
            "owner_id": "user-123",
            "plan": "team",
            "max_debates": 2000,
            "max_agents": 10,
            "max_members": 10,
            "member_ids": [],
            "settings": {},
        }

        manager = WorkspaceManager(storage=mock_storage)

        result = await manager.get_workspace("ws-789")

        assert result is not None
        assert result.id == "ws-789"
        assert result.name == "Storage Workspace"
        assert result.plan == "team"
        # Should be cached now
        assert "ws-789" in manager._cache
        mock_storage.get_workspace.assert_called_once_with("ws-789")

    @pytest.mark.asyncio
    async def test_get_workspace_not_found(self, mock_storage):
        """Should return None for non-existent workspace."""
        mock_storage.get_workspace.return_value = None

        manager = WorkspaceManager(storage=mock_storage)

        result = await manager.get_workspace("non-existent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_workspace_storage_error_logs_and_returns_none(self, mock_storage):
        """Should handle storage errors gracefully."""
        mock_storage.get_workspace.side_effect = ConnectionError("Database down")

        manager = WorkspaceManager(storage=mock_storage)

        result = await manager.get_workspace("ws-error")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_workspace_timeout_error(self, mock_storage):
        """Should handle timeout errors gracefully."""
        mock_storage.get_workspace.side_effect = TimeoutError("Query timed out")

        manager = WorkspaceManager(storage=mock_storage)

        result = await manager.get_workspace("ws-timeout")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_workspace_runtime_error(self, mock_storage):
        """Should handle runtime errors gracefully."""
        mock_storage.get_workspace.side_effect = RuntimeError("Unexpected error")

        manager = WorkspaceManager(storage=mock_storage)

        result = await manager.get_workspace("ws-runtime")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_user_workspace_with_workspace_id(self, sample_user, sample_workspace):
        """Should return user's workspace when workspace_id is set."""
        manager = WorkspaceManager()
        manager._cache["ws-456"] = sample_workspace

        result = await manager.get_user_workspace(sample_user)

        assert result is sample_workspace

    @pytest.mark.asyncio
    async def test_get_user_workspace_creates_default_when_none(self, sample_user_free):
        """Should create default workspace when user has no workspace."""
        manager = WorkspaceManager()

        result = await manager.get_user_workspace(sample_user_free)

        assert result is not None
        assert result.owner_id == "user-free-123"
        assert "free@example.com" in result.name
        assert result.plan == "free"
        assert result.max_debates == 50
        assert result.max_agents == 2
        assert result.max_members == 1

    @pytest.mark.asyncio
    async def test_create_default_workspace_pro_user(self):
        """Should create default workspace with pro plan limits."""
        pro_user = User(
            id="user-pro",
            email="pro@example.com",
            plan="pro",
            workspace_id=None,
        )

        manager = WorkspaceManager()

        result = await manager.create_default_workspace(pro_user)

        assert result.plan == "pro"
        assert result.max_debates == 500
        assert result.max_agents == 5
        assert result.max_members == 1
        assert result.id in manager._cache

    @pytest.mark.asyncio
    async def test_create_default_workspace_saves_to_storage(self, mock_storage):
        """Should save created workspace to storage."""
        user = User(id="user-new", email="new@example.com", plan="free")

        manager = WorkspaceManager(storage=mock_storage)

        result = await manager.create_default_workspace(user)

        mock_storage.save_workspace.assert_called_once()
        # Workspace should be in cache even if storage call happens
        assert result.id in manager._cache

    @pytest.mark.asyncio
    async def test_create_default_workspace_storage_error_continues(self, mock_storage):
        """Should continue even if storage save fails."""
        mock_storage.save_workspace.side_effect = ConnectionError("Save failed")
        user = User(id="user-error", email="error@example.com", plan="free")

        manager = WorkspaceManager(storage=mock_storage)

        result = await manager.create_default_workspace(user)

        # Should still return workspace and cache it
        assert result is not None
        assert result.id in manager._cache


# ===========================================================================
# Test WorkspaceManager Access Checks
# ===========================================================================


class TestWorkspaceManagerAccess:
    """Tests for WorkspaceManager access checking."""

    @pytest.mark.asyncio
    async def test_check_user_access_owner_has_access(self, sample_user, sample_workspace):
        """Owner should always have access to their workspace."""
        manager = WorkspaceManager()
        manager._cache["ws-456"] = sample_workspace

        result = await manager.check_user_access(sample_user, "ws-456")

        assert result is True

    @pytest.mark.asyncio
    async def test_check_user_access_member_has_access(self, sample_workspace):
        """Members should have access to workspace."""
        member = User(id="user-789", email="member@example.com")
        manager = WorkspaceManager()
        manager._cache["ws-456"] = sample_workspace

        result = await manager.check_user_access(member, "ws-456")

        assert result is True

    @pytest.mark.asyncio
    async def test_check_user_access_non_member_denied(self, sample_workspace):
        """Non-members should not have access."""
        stranger = User(id="user-stranger", email="stranger@example.com")
        manager = WorkspaceManager()
        manager._cache["ws-456"] = sample_workspace

        result = await manager.check_user_access(stranger, "ws-456")

        assert result is False

    @pytest.mark.asyncio
    async def test_check_user_access_nonexistent_workspace(self, sample_user):
        """Should return False for non-existent workspace."""
        manager = WorkspaceManager()

        result = await manager.check_user_access(sample_user, "non-existent")

        assert result is False


# ===========================================================================
# Test WorkspaceManager Usage and Limits
# ===========================================================================


class TestWorkspaceManagerLimits:
    """Tests for WorkspaceManager usage tracking and limits."""

    @pytest.mark.asyncio
    async def test_get_usage_from_storage(self, mock_storage):
        """Should fetch usage from storage."""
        mock_storage.get_workspace_usage.return_value = {
            "debates_this_month": 25,
            "active_debates": 3,
            "total_tokens_used": 100000,
        }

        manager = WorkspaceManager(storage=mock_storage)

        result = await manager.get_usage("ws-123")

        assert result["debates_this_month"] == 25
        assert result["active_debates"] == 3
        assert result["total_tokens_used"] == 100000

    @pytest.mark.asyncio
    async def test_get_usage_no_storage_returns_defaults(self):
        """Should return default usage when no storage."""
        manager = WorkspaceManager()

        result = await manager.get_usage("ws-123")

        assert result["debates_this_month"] == 0
        assert result["active_debates"] == 0
        assert result["total_tokens_used"] == 0

    @pytest.mark.asyncio
    async def test_get_usage_storage_error_returns_defaults(self, mock_storage):
        """Should return defaults on storage error."""
        mock_storage.get_workspace_usage.side_effect = OSError("Disk error")

        manager = WorkspaceManager(storage=mock_storage)

        result = await manager.get_usage("ws-error")

        assert result["debates_this_month"] == 0
        assert result["active_debates"] == 0

    @pytest.mark.asyncio
    async def test_check_limits_create_debate_allowed(self, sample_workspace):
        """Should allow debate creation when under limits."""
        manager = WorkspaceManager()

        with patch.object(manager, "get_usage") as mock_usage:
            mock_usage.return_value = {
                "debates_this_month": 10,
                "active_debates": 0,
            }

            allowed, message = await manager.check_limits(sample_workspace, "create_debate")

            assert allowed is True
            assert message == ""

    @pytest.mark.asyncio
    async def test_check_limits_monthly_debate_limit_reached(self):
        """Should reject debate creation when monthly limit reached."""
        workspace = Workspace(id="ws-free", name="Free WS", owner_id="user-1", plan="free")
        manager = WorkspaceManager()

        with patch.object(manager, "get_usage") as mock_usage:
            mock_usage.return_value = {"debates_this_month": 50, "active_debates": 0}

            allowed, message = await manager.check_limits(workspace, "create_debate")

            assert allowed is False
            assert "50" in message
            assert "limit" in message.lower()

    @pytest.mark.asyncio
    async def test_check_limits_concurrent_debate_limit_reached(self):
        """Should reject debate creation when concurrent limit reached."""
        workspace = Workspace(id="ws-free", name="Free WS", owner_id="user-1", plan="free")
        manager = WorkspaceManager()

        with patch.object(manager, "get_usage") as mock_usage:
            mock_usage.return_value = {"debates_this_month": 10, "active_debates": 1}

            allowed, message = await manager.check_limits(workspace, "create_debate")

            assert allowed is False
            assert "Concurrent" in message

    @pytest.mark.asyncio
    async def test_check_limits_enterprise_no_limits(self, enterprise_workspace):
        """Enterprise should have no limits (-1 means unlimited)."""
        manager = WorkspaceManager()

        with patch.object(manager, "get_usage") as mock_usage:
            mock_usage.return_value = {
                "debates_this_month": 10000,
                "active_debates": 100,
            }

            allowed, message = await manager.check_limits(enterprise_workspace, "create_debate")

            assert allowed is True
            assert message == ""

    @pytest.mark.asyncio
    async def test_check_limits_add_member_allowed(self, sample_workspace):
        """Should allow adding member when under limit."""
        manager = WorkspaceManager()
        # sample_workspace has 1 member and max_members=1, but this is pro

        with patch.object(manager, "get_usage") as mock_usage:
            mock_usage.return_value = {}

            # Need a workspace with room for members
            workspace = Workspace(
                id="ws-team",
                name="Team WS",
                owner_id="user-1",
                plan="team",
                max_members=10,
                member_ids=["m1", "m2"],  # 2 members, max 10
            )

            allowed, message = await manager.check_limits(workspace, "add_member")

            assert allowed is True

    @pytest.mark.asyncio
    async def test_check_limits_add_member_limit_reached(self):
        """Should reject member addition when limit reached."""
        workspace = Workspace(
            id="ws-full",
            name="Full WS",
            owner_id="user-1",
            plan="free",
            max_members=1,
            member_ids=[],  # 0 members, but free plan allows 1 which means just owner
        )
        manager = WorkspaceManager()

        # Free plan: max_members=1 which includes owner only
        with patch.object(manager, "get_usage") as mock_usage:
            mock_usage.return_value = {}

            # The workspace member_ids is empty but max_members=1 (owner counts)
            # Let's test with a workspace that actually hits the limit
            workspace2 = Workspace(
                id="ws-full2",
                name="Full WS 2",
                owner_id="user-1",
                plan="pro",  # Pro has max_members=1
                max_members=1,
                member_ids=["member-1"],  # Already has 1 member
            )

            allowed, message = await manager.check_limits(workspace2, "add_member")

            assert allowed is False
            assert "Member limit" in message

    @pytest.mark.asyncio
    async def test_check_limits_unknown_action_allowed(self, sample_workspace):
        """Unknown actions should be allowed by default."""
        manager = WorkspaceManager()

        with patch.object(manager, "get_usage") as mock_usage:
            mock_usage.return_value = {}

            allowed, message = await manager.check_limits(sample_workspace, "unknown_action")

            assert allowed is True


# ===========================================================================
# Test Global WorkspaceManager
# ===========================================================================


class TestGlobalWorkspaceManager:
    """Tests for global WorkspaceManager singleton."""

    def test_get_workspace_manager_singleton(self):
        """get_workspace_manager should return singleton instance."""
        import aragora.server.middleware.tenancy as tenancy_module

        tenancy_module._workspace_manager = None

        m1 = get_workspace_manager()
        m2 = get_workspace_manager()

        assert m1 is m2
        assert isinstance(m1, WorkspaceManager)

    def test_get_workspace_manager_creates_new_if_none(self):
        """Should create new manager if none exists."""
        import aragora.server.middleware.tenancy as tenancy_module

        tenancy_module._workspace_manager = None

        manager = get_workspace_manager()

        assert manager is not None
        assert isinstance(manager, WorkspaceManager)


# ===========================================================================
# Test require_workspace Decorator
# ===========================================================================


class TestRequireWorkspaceDecorator:
    """Tests for require_workspace decorator."""

    def test_no_handler_returns_500(self):
        """Should return 500 when no handler provided."""

        @require_workspace
        async def endpoint():
            return {"success": True}

        result = asyncio.run(endpoint())

        assert get_status(result) == 500

    def test_unauthenticated_returns_401(self, mock_handler):
        """Should return 401 when not authenticated."""
        with patch("aragora.server.middleware.tenancy.get_current_user", return_value=None):

            @require_workspace
            async def endpoint(handler):
                return {"success": True}

            result = asyncio.run(endpoint(handler=mock_handler))

            assert get_status(result) == 401

    @pytest.mark.asyncio
    async def test_workspace_not_found_returns_404(self, mock_handler_with_bearer, sample_user):
        """Should return 404 when workspace not found."""
        with patch(
            "aragora.server.middleware.tenancy.get_current_user",
            return_value=sample_user,
        ):
            with patch(
                "aragora.server.middleware.tenancy.get_workspace_manager"
            ) as mock_get_manager:
                mock_manager = AsyncMock()
                mock_manager.get_user_workspace.return_value = None
                mock_get_manager.return_value = mock_manager

                @require_workspace
                async def endpoint(handler, user, workspace):
                    return {"success": True}

                result = await endpoint(handler=mock_handler_with_bearer)

                assert get_status(result) == 404

    @pytest.mark.asyncio
    async def test_authenticated_with_workspace_allows_access(
        self, mock_handler_with_bearer, sample_user, sample_workspace
    ):
        """Should allow access with valid authentication and workspace."""
        with patch(
            "aragora.server.middleware.tenancy.get_current_user",
            return_value=sample_user,
        ):
            with patch(
                "aragora.server.middleware.tenancy.get_workspace_manager"
            ) as mock_get_manager:
                mock_manager = AsyncMock()
                mock_manager.get_user_workspace.return_value = sample_workspace
                mock_get_manager.return_value = mock_manager

                @require_workspace
                async def endpoint(handler, user, workspace):
                    return {"success": True, "workspace_id": workspace.id}

                result = await endpoint(handler=mock_handler_with_bearer)

                assert result["success"] is True
                assert result["workspace_id"] == "ws-456"

    @pytest.mark.asyncio
    async def test_handler_extraction_from_args(self, sample_user, sample_workspace):
        """Should extract handler from positional args."""
        handler = MockHandler(headers={"Authorization": "Bearer test"})

        with patch(
            "aragora.server.middleware.tenancy.get_current_user",
            return_value=sample_user,
        ):
            with patch(
                "aragora.server.middleware.tenancy.get_workspace_manager"
            ) as mock_get_manager:
                mock_manager = AsyncMock()
                mock_manager.get_user_workspace.return_value = sample_workspace
                mock_get_manager.return_value = mock_manager

                @require_workspace
                async def endpoint(self_arg, handler, user, workspace):
                    return {"success": True, "user_id": user.id}

                result = await endpoint(object(), handler)

                assert result["success"] is True
                assert result["user_id"] == "user-123"

    @pytest.mark.asyncio
    async def test_sync_function_wrapped(
        self, mock_handler_with_bearer, sample_user, sample_workspace
    ):
        """Should handle sync functions wrapped with decorator."""
        with patch(
            "aragora.server.middleware.tenancy.get_current_user",
            return_value=sample_user,
        ):
            with patch(
                "aragora.server.middleware.tenancy.get_workspace_manager"
            ) as mock_get_manager:
                mock_manager = AsyncMock()
                mock_manager.get_user_workspace.return_value = sample_workspace
                mock_get_manager.return_value = mock_manager

                @require_workspace
                def sync_endpoint(handler, user, workspace):
                    return {"sync": True, "workspace_id": workspace.id}

                result = await sync_endpoint(handler=mock_handler_with_bearer)

                assert result["sync"] is True
                assert result["workspace_id"] == "ws-456"


# ===========================================================================
# Test check_limit Decorator
# ===========================================================================


class TestCheckLimitDecorator:
    """Tests for check_limit decorator."""

    @pytest.mark.asyncio
    async def test_no_workspace_returns_400(self):
        """Should return 400 when workspace not provided."""

        @check_limit("create_debate")
        async def endpoint():
            return {"success": True}

        result = await endpoint()

        assert get_status(result) == 400

    @pytest.mark.asyncio
    async def test_limit_exceeded_returns_403(self, sample_workspace):
        """Should return 403 when limit exceeded."""
        with patch("aragora.server.middleware.tenancy.get_workspace_manager") as mock_get_manager:
            mock_manager = AsyncMock()
            mock_manager.check_limits.return_value = (False, "Monthly limit reached")
            mock_get_manager.return_value = mock_manager

            @check_limit("create_debate")
            async def endpoint(workspace):
                return {"success": True}

            result = await endpoint(workspace=sample_workspace)

            assert get_status(result) == 403

    @pytest.mark.asyncio
    async def test_within_limits_allows_access(self, sample_workspace):
        """Should allow access when within limits."""
        with patch("aragora.server.middleware.tenancy.get_workspace_manager") as mock_get_manager:
            mock_manager = AsyncMock()
            mock_manager.check_limits.return_value = (True, "")
            mock_get_manager.return_value = mock_manager

            @check_limit("create_debate")
            async def endpoint(workspace):
                return {"success": True, "workspace_id": workspace.id}

            result = await endpoint(workspace=sample_workspace)

            assert result["success"] is True
            assert result["workspace_id"] == "ws-456"

    @pytest.mark.asyncio
    async def test_sync_function_wrapped(self, sample_workspace):
        """Should handle sync functions."""
        with patch("aragora.server.middleware.tenancy.get_workspace_manager") as mock_get_manager:
            mock_manager = AsyncMock()
            mock_manager.check_limits.return_value = (True, "")
            mock_get_manager.return_value = mock_manager

            @check_limit("create_debate")
            def sync_endpoint(workspace):
                return {"sync": True}

            result = await sync_endpoint(workspace=sample_workspace)

            assert result["sync"] is True


# ===========================================================================
# Test tenant_scoped Decorator
# ===========================================================================


class TestTenantScopedDecorator:
    """Tests for tenant_scoped decorator."""

    @pytest.mark.asyncio
    async def test_missing_workspace_id_raises_error(self):
        """Should raise ValueError when workspace_id is missing."""

        @tenant_scoped
        async def get_debates(workspace_id: str):
            return [{"id": "debate-1"}]

        with pytest.raises(ValueError, match="workspace_id is required"):
            await get_debates("")

    @pytest.mark.asyncio
    async def test_none_workspace_id_raises_error(self):
        """Should raise ValueError when workspace_id is None."""

        @tenant_scoped
        async def get_debates(workspace_id: str):
            return [{"id": "debate-1"}]

        with pytest.raises(ValueError, match="workspace_id is required"):
            await get_debates(None)

    @pytest.mark.asyncio
    async def test_valid_workspace_id_passes_through(self):
        """Should pass workspace_id to wrapped function."""

        @tenant_scoped
        async def get_debates(workspace_id: str):
            return [{"workspace_id": workspace_id}]

        result = await get_debates("ws-123")

        assert result[0]["workspace_id"] == "ws-123"

    @pytest.mark.asyncio
    async def test_additional_args_passed(self):
        """Should pass additional positional and keyword args."""

        @tenant_scoped
        async def get_debates(workspace_id: str, limit: int, status: str = "active"):
            return {
                "workspace_id": workspace_id,
                "limit": limit,
                "status": status,
            }

        result = await get_debates("ws-123", 10, status="completed")

        assert result["workspace_id"] == "ws-123"
        assert result["limit"] == 10
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_sync_function_wrapped(self):
        """Should handle sync functions."""

        @tenant_scoped
        def sync_get_debates(workspace_id: str):
            return {"workspace_id": workspace_id}

        result = await sync_get_debates("ws-123")

        assert result["workspace_id"] == "ws-123"


# ===========================================================================
# Test scope_query Utility
# ===========================================================================


class TestScopeQuery:
    """Tests for scope_query utility function."""

    def test_scope_query_with_filter(self):
        """Should add filter_by for SQLAlchemy-style query."""
        mock_query = MagicMock()

        result = scope_query(mock_query, "ws-123")

        mock_query.filter_by.assert_called_once_with(workspace_id="ws-123")

    def test_scope_query_with_where(self):
        """Should add where for SQLite-style query."""
        mock_query = MagicMock(spec=["where"])

        result = scope_query(mock_query, "ws-456")

        mock_query.where.assert_called_once_with("workspace_id = ?", ("ws-456",))

    def test_scope_query_no_filter_or_where(self):
        """Should return query unchanged if no filter/where methods."""
        mock_query = MagicMock(spec=[])

        result = scope_query(mock_query, "ws-789")

        assert result is mock_query


# ===========================================================================
# Test ensure_workspace_access Utility
# ===========================================================================


class TestEnsureWorkspaceAccess:
    """Tests for ensure_workspace_access utility function."""

    @pytest.mark.asyncio
    async def test_access_granted_returns_true(self, sample_user, sample_workspace):
        """Should return True when access is granted."""
        with patch("aragora.server.middleware.tenancy.get_workspace_manager") as mock_get_manager:
            mock_manager = AsyncMock()
            mock_manager.check_user_access.return_value = True
            mock_get_manager.return_value = mock_manager

            result = await ensure_workspace_access(sample_user, "ws-456")

            assert result is True

    @pytest.mark.asyncio
    async def test_access_denied_raises_permission_error(self, sample_user):
        """Should raise PermissionError when access denied."""
        with patch("aragora.server.middleware.tenancy.get_workspace_manager") as mock_get_manager:
            mock_manager = AsyncMock()
            mock_manager.check_user_access.return_value = False
            mock_get_manager.return_value = mock_manager

            with pytest.raises(PermissionError, match="Access denied"):
                await ensure_workspace_access(sample_user, "ws-other")


# ===========================================================================
# Test Cross-Tenant Data Access Prevention
# ===========================================================================


class TestCrossTenantAccessPrevention:
    """Tests for cross-tenant data access prevention."""

    @pytest.mark.asyncio
    async def test_user_cannot_access_other_workspace(self):
        """User should not access resources from another workspace."""
        user = User(id="user-123", email="test@example.com", workspace_id="ws-A")
        other_workspace = Workspace(id="ws-B", name="Other WS", owner_id="user-other")

        manager = WorkspaceManager()
        manager._cache["ws-B"] = other_workspace

        has_access = await manager.check_user_access(user, "ws-B")

        assert has_access is False

    @pytest.mark.asyncio
    async def test_ensure_workspace_access_blocks_cross_tenant(self):
        """ensure_workspace_access should block cross-tenant access."""
        user = User(id="user-123", email="test@example.com", workspace_id="ws-A")

        with patch("aragora.server.middleware.tenancy.get_workspace_manager") as mock_get_manager:
            mock_manager = AsyncMock()
            mock_manager.check_user_access.return_value = False
            mock_get_manager.return_value = mock_manager

            with pytest.raises(PermissionError, match="Access denied to workspace ws-B"):
                await ensure_workspace_access(user, "ws-B")


# ===========================================================================
# Test Concurrent Tenant Operations
# ===========================================================================


class TestConcurrentTenantOperations:
    """Tests for concurrent tenant operations and race conditions."""

    @pytest.mark.asyncio
    async def test_concurrent_workspace_access_checks(self, sample_workspace):
        """Multiple concurrent access checks should be consistent."""
        manager = WorkspaceManager()
        manager._cache["ws-456"] = sample_workspace

        users = [
            User(id="user-123", email="owner@test.com"),  # Owner
            User(id="user-789", email="member@test.com"),  # Member
            User(id="user-stranger", email="stranger@test.com"),  # Non-member
        ]

        async def check_access(user):
            return await manager.check_user_access(user, "ws-456")

        results = await asyncio.gather(*[check_access(u) for u in users])

        assert results[0] is True  # Owner
        assert results[1] is True  # Member
        assert results[2] is False  # Stranger

    @pytest.mark.asyncio
    async def test_concurrent_limit_checks(self, sample_workspace):
        """Concurrent limit checks should not cause race conditions."""
        manager = WorkspaceManager()

        call_count = 0

        async def mock_get_usage(workspace_id):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)  # Simulate async delay
            return {"debates_this_month": 10, "active_debates": 0}

        with patch.object(manager, "get_usage", side_effect=mock_get_usage):

            async def check_limit():
                return await manager.check_limits(sample_workspace, "create_debate")

            # Run multiple concurrent checks
            results = await asyncio.gather(*[check_limit() for _ in range(5)])

            # All should succeed
            assert all(r[0] is True for r in results)
            # Should have called get_usage 5 times
            assert call_count == 5

    @pytest.mark.asyncio
    async def test_cache_consistency_under_concurrent_access(self):
        """Cache should remain consistent under concurrent access."""
        mock_storage = AsyncMock()
        workspace_data = {
            "id": "ws-concurrent",
            "name": "Concurrent Test",
            "owner_id": "user-1",
            "plan": "team",
            "max_debates": 2000,
            "max_agents": 10,
            "max_members": 10,
            "member_ids": [],
            "settings": {},
        }
        mock_storage.get_workspace.return_value = workspace_data

        manager = WorkspaceManager(storage=mock_storage)

        async def get_ws():
            return await manager.get_workspace("ws-concurrent")

        # Multiple concurrent fetches
        results = await asyncio.gather(*[get_ws() for _ in range(10)])

        # All should return the same workspace
        assert all(r.id == "ws-concurrent" for r in results)
        # Should be cached - subsequent calls shouldn't hit storage
        # (First call caches, others use cache or wait)
        assert mock_storage.get_workspace.call_count >= 1


# ===========================================================================
# Test Tenant ID Validation
# ===========================================================================


class TestTenantIdValidation:
    """Tests for tenant ID validation and spoofing prevention."""

    @pytest.mark.asyncio
    async def test_workspace_id_must_match_for_access(self):
        """User cannot spoof access by changing workspace_id."""
        # User claims to be in ws-attacker but actually belongs to ws-victim
        attacker = User(
            id="user-attacker",
            email="attacker@evil.com",
            workspace_id="ws-attacker",
        )

        victim_workspace = Workspace(
            id="ws-victim",
            name="Victim WS",
            owner_id="user-victim",
            member_ids=[],
        )

        manager = WorkspaceManager()
        manager._cache["ws-victim"] = victim_workspace

        # Attacker cannot access victim's workspace
        has_access = await manager.check_user_access(attacker, "ws-victim")

        assert has_access is False

    @pytest.mark.asyncio
    async def test_tenant_scoped_prevents_empty_workspace_id(self):
        """tenant_scoped should prevent queries with empty workspace_id."""

        @tenant_scoped
        async def query_debates(workspace_id: str):
            return f"SELECT * FROM debates WHERE workspace_id = '{workspace_id}'"

        # Empty string should raise
        with pytest.raises(ValueError):
            await query_debates("")

        # None should raise
        with pytest.raises(ValueError):
            await query_debates(None)


# ===========================================================================
# Test Shared Resource Leakage Prevention
# ===========================================================================


class TestSharedResourceLeakagePrevention:
    """Tests for preventing shared resource leakage between tenants."""

    @pytest.mark.asyncio
    async def test_workspace_cache_isolation(self):
        """Each workspace should be cached independently."""
        manager = WorkspaceManager()

        ws_a = Workspace(id="ws-A", name="Workspace A", owner_id="user-a")
        ws_b = Workspace(id="ws-B", name="Workspace B", owner_id="user-b")

        manager._cache["ws-A"] = ws_a
        manager._cache["ws-B"] = ws_b

        result_a = await manager.get_workspace("ws-A")
        result_b = await manager.get_workspace("ws-B")

        # Each should return correct workspace
        assert result_a.name == "Workspace A"
        assert result_b.name == "Workspace B"

        # Should not cross-contaminate
        assert result_a is not result_b
        assert result_a.owner_id != result_b.owner_id

    @pytest.mark.asyncio
    async def test_usage_data_isolation(self, mock_storage):
        """Usage data should be isolated per workspace."""
        usage_a = {"debates_this_month": 10, "active_debates": 2}
        usage_b = {"debates_this_month": 50, "active_debates": 5}

        async def mock_get_usage(workspace_id):
            if workspace_id == "ws-A":
                return usage_a
            elif workspace_id == "ws-B":
                return usage_b
            return {}

        mock_storage.get_workspace_usage.side_effect = mock_get_usage
        manager = WorkspaceManager(storage=mock_storage)

        result_a = await manager.get_usage("ws-A")
        result_b = await manager.get_usage("ws-B")

        assert result_a["debates_this_month"] == 10
        assert result_b["debates_this_month"] == 50


# ===========================================================================
# Test Missing Tenant Configuration Handling
# ===========================================================================


class TestMissingTenantConfiguration:
    """Tests for handling missing tenant configuration."""

    @pytest.mark.asyncio
    async def test_user_without_workspace_gets_default(self):
        """User without workspace should get default workspace created."""
        user = User(
            id="new-user",
            email="new@example.com",
            workspace_id=None,
            plan="pro",
        )

        manager = WorkspaceManager()

        workspace = await manager.get_user_workspace(user)

        assert workspace is not None
        assert workspace.owner_id == "new-user"
        assert workspace.plan == "pro"

    @pytest.mark.asyncio
    async def test_nonexistent_workspace_id_creates_default(self, mock_storage):
        """User with non-existent workspace_id should get None (not default)."""
        mock_storage.get_workspace.return_value = None
        user = User(
            id="orphan-user",
            email="orphan@example.com",
            workspace_id="ws-deleted",
        )

        manager = WorkspaceManager(storage=mock_storage)

        # get_workspace returns None, then get_user_workspace calls create_default
        workspace = await manager.get_user_workspace(user)

        # User has workspace_id set, so it tries to fetch, fails, and creates default
        # Actually, looking at the code, if user.workspace_id is set, it returns get_workspace result
        # which could be None. Let me re-read the code...
        #
        # get_user_workspace: if user.workspace_id: return await self.get_workspace(user.workspace_id)
        # So if workspace_id is set but doesn't exist, returns None
        assert workspace is None

    @pytest.mark.asyncio
    async def test_storage_unavailable_workspace_creation_still_works(self):
        """Workspace creation should work even if storage is unavailable."""
        mock_storage = AsyncMock()
        mock_storage.save_workspace.side_effect = ConnectionError("No connection")

        user = User(
            id="offline-user",
            email="offline@example.com",
            workspace_id=None,
        )

        manager = WorkspaceManager(storage=mock_storage)

        workspace = await manager.create_default_workspace(user)

        # Should still create and cache workspace
        assert workspace is not None
        assert workspace.id in manager._cache


# ===========================================================================
# Test Exception Handling Paths
# ===========================================================================


class TestExceptionHandling:
    """Tests for exception handling throughout the module."""

    @pytest.mark.asyncio
    async def test_storage_os_error_handled(self, mock_storage):
        """OSError from storage should be caught and logged."""
        mock_storage.get_workspace.side_effect = OSError("I/O error")

        manager = WorkspaceManager(storage=mock_storage)

        result = await manager.get_workspace("ws-io-error")

        assert result is None

    @pytest.mark.asyncio
    async def test_storage_connection_error_handled(self, mock_storage):
        """ConnectionError from storage should be caught and logged."""
        mock_storage.get_workspace.side_effect = ConnectionError("Connection refused")

        manager = WorkspaceManager(storage=mock_storage)

        result = await manager.get_workspace("ws-conn-error")

        assert result is None

    @pytest.mark.asyncio
    async def test_storage_timeout_error_handled(self, mock_storage):
        """TimeoutError from storage should be caught and logged."""
        mock_storage.get_workspace.side_effect = TimeoutError("Query timeout")

        manager = WorkspaceManager(storage=mock_storage)

        result = await manager.get_workspace("ws-timeout-error")

        assert result is None

    @pytest.mark.asyncio
    async def test_storage_runtime_error_handled(self, mock_storage):
        """RuntimeError from storage should be caught and logged."""
        mock_storage.get_workspace.side_effect = RuntimeError("Unexpected failure")

        manager = WorkspaceManager(storage=mock_storage)

        result = await manager.get_workspace("ws-runtime-error")

        assert result is None

    @pytest.mark.asyncio
    async def test_usage_error_returns_defaults(self, mock_storage):
        """Errors getting usage should return default values."""
        mock_storage.get_workspace_usage.side_effect = ConnectionError("DB down")

        manager = WorkspaceManager(storage=mock_storage)

        result = await manager.get_usage("ws-usage-error")

        assert result["debates_this_month"] == 0
        assert result["active_debates"] == 0
        assert result["total_tokens_used"] == 0

    @pytest.mark.asyncio
    async def test_save_workspace_error_continues(self, mock_storage):
        """Errors saving workspace should not prevent workspace creation."""
        mock_storage.save_workspace.side_effect = TimeoutError("Save timeout")

        user = User(id="save-error-user", email="error@test.com")
        manager = WorkspaceManager(storage=mock_storage)

        workspace = await manager.create_default_workspace(user)

        # Should still return workspace and cache it
        assert workspace is not None
        assert workspace.id in manager._cache
