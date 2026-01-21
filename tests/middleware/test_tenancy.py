"""
Tests for aragora.server.middleware.tenancy module.

Tests multi-tenant workspace isolation, plan limits, and access control.
"""

import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.middleware.user_auth import User, Workspace
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


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def free_user() -> User:
    """Create a free-tier user for testing."""
    return User(
        id="user-free-123",
        email="free@example.com",
        role="user",
        plan="free",
        workspace_id="ws-free-123",
    )


@pytest.fixture
def pro_user() -> User:
    """Create a pro-tier user for testing."""
    return User(
        id="user-pro-456",
        email="pro@example.com",
        role="user",
        plan="pro",
        workspace_id="ws-pro-456",
    )


@pytest.fixture
def team_user() -> User:
    """Create a team-tier user for testing."""
    return User(
        id="user-team-789",
        email="team@example.com",
        role="user",
        plan="team",
        workspace_id="ws-team-789",
    )


@pytest.fixture
def enterprise_user() -> User:
    """Create an enterprise-tier user for testing."""
    return User(
        id="user-ent-000",
        email="enterprise@example.com",
        role="user",
        plan="enterprise",
        workspace_id="ws-ent-000",
    )


@pytest.fixture
def free_workspace() -> Workspace:
    """Create a free-tier workspace for testing."""
    return Workspace(
        id="ws-free-123",
        name="Free User's Workspace",
        owner_id="user-free-123",
        plan="free",
        max_debates=50,
        max_agents=2,
        max_members=1,
        member_ids=[],
    )


@pytest.fixture
def pro_workspace() -> Workspace:
    """Create a pro-tier workspace for testing."""
    return Workspace(
        id="ws-pro-456",
        name="Pro User's Workspace",
        owner_id="user-pro-456",
        plan="pro",
        max_debates=500,
        max_agents=5,
        max_members=1,
        member_ids=[],
    )


@pytest.fixture
def team_workspace() -> Workspace:
    """Create a team-tier workspace with members for testing."""
    return Workspace(
        id="ws-team-789",
        name="Team Workspace",
        owner_id="user-team-789",
        plan="team",
        max_debates=2000,
        max_agents=10,
        max_members=10,
        member_ids=["member-1", "member-2", "member-3"],
    )


@pytest.fixture
def mock_storage() -> AsyncMock:
    """Create a mock storage backend."""
    storage = AsyncMock()
    storage.get_workspace.return_value = None
    storage.save_workspace.return_value = None
    storage.get_workspace_usage.return_value = {
        "debates_this_month": 0,
        "active_debates": 0,
        "total_tokens_used": 0,
    }
    return storage


@pytest.fixture
def mock_handler() -> MagicMock:
    """Create a mock HTTP request handler."""
    handler = MagicMock()
    handler.headers = {"Authorization": "Bearer test-token"}
    return handler


# =============================================================================
# Test Plan Limits
# =============================================================================


class TestPlanLimits:
    """Tests for plan limits configuration."""

    def test_plan_limits_exist_for_all_tiers(self):
        """All expected plan tiers have defined limits."""
        expected_tiers = ["free", "pro", "team", "enterprise"]
        for tier in expected_tiers:
            assert tier in PLAN_LIMITS, f"Missing plan tier: {tier}"

    def test_free_plan_limits(self):
        """Free plan has restrictive limits."""
        limits = PLAN_LIMITS["free"]

        assert limits["max_debates_per_month"] == 50
        assert limits["max_agents"] == 2
        assert limits["max_members"] == 1
        assert limits["max_concurrent_debates"] == 1
        assert limits["private_debates"] is False
        assert limits["api_access"] is False
        assert limits["priority_support"] is False

    def test_pro_plan_limits(self):
        """Pro plan has enhanced limits."""
        limits = PLAN_LIMITS["pro"]

        assert limits["max_debates_per_month"] == 500
        assert limits["max_agents"] == 5
        assert limits["max_members"] == 1
        assert limits["max_concurrent_debates"] == 3
        assert limits["private_debates"] is True
        assert limits["api_access"] is True
        assert limits["priority_support"] is False

    def test_team_plan_limits(self):
        """Team plan has collaboration-focused limits."""
        limits = PLAN_LIMITS["team"]

        assert limits["max_debates_per_month"] == 2000
        assert limits["max_agents"] == 10
        assert limits["max_members"] == 10
        assert limits["max_concurrent_debates"] == 10
        assert limits["private_debates"] is True
        assert limits["api_access"] is True
        assert limits["priority_support"] is True

    def test_enterprise_plan_unlimited(self):
        """Enterprise plan has unlimited capacity (-1)."""
        limits = PLAN_LIMITS["enterprise"]

        assert limits["max_debates_per_month"] == -1
        assert limits["max_agents"] == -1
        assert limits["max_members"] == -1
        assert limits["max_concurrent_debates"] == -1
        assert limits["private_debates"] is True
        assert limits["api_access"] is True
        assert limits["priority_support"] is True

    def test_get_plan_limits_valid_plan(self):
        """get_plan_limits returns correct limits for valid plans."""
        for tier in ["free", "pro", "team", "enterprise"]:
            limits = get_plan_limits(tier)
            assert limits == PLAN_LIMITS[tier]

    def test_get_plan_limits_invalid_plan_defaults_to_free(self):
        """get_plan_limits defaults to free tier for unknown plans."""
        limits = get_plan_limits("invalid_plan")
        assert limits == PLAN_LIMITS["free"]

        limits = get_plan_limits("")
        assert limits == PLAN_LIMITS["free"]

        limits = get_plan_limits(None)
        assert limits == PLAN_LIMITS["free"]


# =============================================================================
# Test WorkspaceManager
# =============================================================================


class TestWorkspaceManager:
    """Tests for WorkspaceManager class."""

    def test_manager_init_without_storage(self):
        """Manager initializes with empty cache when no storage provided."""
        manager = WorkspaceManager()

        assert manager._storage is None
        assert manager._cache == {}

    def test_manager_init_with_storage(self, mock_storage):
        """Manager initializes with provided storage backend."""
        manager = WorkspaceManager(storage=mock_storage)

        assert manager._storage is mock_storage
        assert manager._cache == {}

    @pytest.mark.asyncio
    async def test_get_workspace_from_cache(self, free_workspace):
        """get_workspace returns cached workspace."""
        manager = WorkspaceManager()
        manager._cache["ws-free-123"] = free_workspace

        result = await manager.get_workspace("ws-free-123")

        assert result is free_workspace
        assert result.id == "ws-free-123"

    @pytest.mark.asyncio
    async def test_get_workspace_cache_miss_no_storage(self):
        """get_workspace returns None for cache miss without storage."""
        manager = WorkspaceManager()

        result = await manager.get_workspace("nonexistent-ws")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_workspace_from_storage(self, mock_storage):
        """get_workspace queries storage on cache miss."""
        workspace_data = {
            "id": "ws-from-storage",
            "name": "Storage Workspace",
            "owner_id": "owner-1",
            "plan": "pro",
            "max_debates": 500,
            "max_agents": 5,
            "max_members": 1,
            "member_ids": [],
            "settings": {},
        }
        mock_storage.get_workspace.return_value = workspace_data

        manager = WorkspaceManager(storage=mock_storage)
        result = await manager.get_workspace("ws-from-storage")

        assert result is not None
        assert result.id == "ws-from-storage"
        assert result.plan == "pro"
        mock_storage.get_workspace.assert_called_once_with("ws-from-storage")

    @pytest.mark.asyncio
    async def test_get_workspace_caches_storage_result(self, mock_storage):
        """get_workspace caches results from storage."""
        workspace_data = {
            "id": "ws-cached",
            "name": "Cached Workspace",
            "owner_id": "owner-1",
            "plan": "free",
            "max_debates": 50,
            "max_agents": 2,
            "max_members": 1,
            "member_ids": [],
            "settings": {},
        }
        mock_storage.get_workspace.return_value = workspace_data

        manager = WorkspaceManager(storage=mock_storage)

        # First call
        await manager.get_workspace("ws-cached")
        # Second call should use cache
        result = await manager.get_workspace("ws-cached")

        assert result.id == "ws-cached"
        # Storage should only be called once
        mock_storage.get_workspace.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_workspace_handles_storage_error(self, mock_storage):
        """get_workspace handles storage errors gracefully."""
        mock_storage.get_workspace.side_effect = Exception("Database error")

        manager = WorkspaceManager(storage=mock_storage)
        result = await manager.get_workspace("ws-error")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_user_workspace_with_workspace_id(self, free_user, free_workspace):
        """get_user_workspace returns user's assigned workspace."""
        manager = WorkspaceManager()
        manager._cache["ws-free-123"] = free_workspace

        result = await manager.get_user_workspace(free_user)

        assert result is free_workspace

    @pytest.mark.asyncio
    async def test_get_user_workspace_creates_default(self):
        """get_user_workspace creates default workspace for user without one."""
        user = User(
            id="user-no-ws",
            email="noworkspace@example.com",
            role="user",
            plan="pro",
            workspace_id=None,
        )

        manager = WorkspaceManager()
        result = await manager.get_user_workspace(user)

        assert result is not None
        assert result.owner_id == "user-no-ws"
        assert result.name == "noworkspace@example.com's Workspace"
        assert result.plan == "pro"
        # Check that pro limits are applied
        assert result.max_debates == 500
        assert result.max_agents == 5

    @pytest.mark.asyncio
    async def test_create_default_workspace_uses_plan_limits(self):
        """create_default_workspace applies correct limits based on user's plan."""
        manager = WorkspaceManager()

        for plan, expected_limits in [
            ("free", {"max_debates": 50, "max_agents": 2, "max_members": 1}),
            ("pro", {"max_debates": 500, "max_agents": 5, "max_members": 1}),
            ("team", {"max_debates": 2000, "max_agents": 10, "max_members": 10}),
            ("enterprise", {"max_debates": -1, "max_agents": -1, "max_members": -1}),
        ]:
            user = User(id=f"user-{plan}", email=f"{plan}@example.com", plan=plan)
            workspace = await manager.create_default_workspace(user)

            assert workspace.max_debates == expected_limits["max_debates"]
            assert workspace.max_agents == expected_limits["max_agents"]
            assert workspace.max_members == expected_limits["max_members"]

    @pytest.mark.asyncio
    async def test_create_default_workspace_saves_to_storage(self, mock_storage):
        """create_default_workspace persists to storage when available."""
        user = User(id="user-persist", email="persist@example.com", plan="free")

        manager = WorkspaceManager(storage=mock_storage)
        workspace = await manager.create_default_workspace(user)

        mock_storage.save_workspace.assert_called_once_with(workspace)

    @pytest.mark.asyncio
    async def test_create_default_workspace_caches_result(self):
        """create_default_workspace caches the created workspace."""
        user = User(id="user-cache", email="cache@example.com", plan="free")

        manager = WorkspaceManager()
        workspace = await manager.create_default_workspace(user)

        assert workspace.id in manager._cache
        assert manager._cache[workspace.id] is workspace

    @pytest.mark.asyncio
    async def test_check_user_access_owner_has_access(self, free_user, free_workspace):
        """check_user_access grants access to workspace owner."""
        manager = WorkspaceManager()
        manager._cache["ws-free-123"] = free_workspace

        result = await manager.check_user_access(free_user, "ws-free-123")

        assert result is True

    @pytest.mark.asyncio
    async def test_check_user_access_member_has_access(self, team_workspace):
        """check_user_access grants access to workspace members."""
        member = User(id="member-1", email="member1@example.com")

        manager = WorkspaceManager()
        manager._cache["ws-team-789"] = team_workspace

        result = await manager.check_user_access(member, "ws-team-789")

        assert result is True

    @pytest.mark.asyncio
    async def test_check_user_access_non_member_denied(self, free_workspace):
        """check_user_access denies access to non-members."""
        outsider = User(id="outsider-999", email="outsider@example.com")

        manager = WorkspaceManager()
        manager._cache["ws-free-123"] = free_workspace

        result = await manager.check_user_access(outsider, "ws-free-123")

        assert result is False

    @pytest.mark.asyncio
    async def test_check_user_access_nonexistent_workspace(self, free_user):
        """check_user_access returns False for nonexistent workspace."""
        manager = WorkspaceManager()

        result = await manager.check_user_access(free_user, "nonexistent-ws")

        assert result is False


# =============================================================================
# Test Workspace Limits Checking
# =============================================================================


class TestWorkspaceLimits:
    """Tests for workspace limit enforcement."""

    @pytest.mark.asyncio
    async def test_check_limits_create_debate_allowed(self, free_workspace, mock_storage):
        """check_limits allows debate creation under limit."""
        mock_storage.get_workspace_usage.return_value = {
            "debates_this_month": 10,
            "active_debates": 0,
        }

        manager = WorkspaceManager(storage=mock_storage)
        allowed, message = await manager.check_limits(free_workspace, "create_debate")

        assert allowed is True
        assert message == ""

    @pytest.mark.asyncio
    async def test_check_limits_create_debate_monthly_limit_reached(
        self, free_workspace, mock_storage
    ):
        """check_limits blocks debate creation when monthly limit reached."""
        mock_storage.get_workspace_usage.return_value = {
            "debates_this_month": 50,  # At limit for free tier
            "active_debates": 0,
        }

        manager = WorkspaceManager(storage=mock_storage)
        allowed, message = await manager.check_limits(free_workspace, "create_debate")

        assert allowed is False
        assert "Monthly debate limit (50) reached" in message

    @pytest.mark.asyncio
    async def test_check_limits_create_debate_concurrent_limit_reached(
        self, free_workspace, mock_storage
    ):
        """check_limits blocks debate creation when concurrent limit reached."""
        mock_storage.get_workspace_usage.return_value = {
            "debates_this_month": 10,
            "active_debates": 1,  # At limit for free tier
        }

        manager = WorkspaceManager(storage=mock_storage)
        allowed, message = await manager.check_limits(free_workspace, "create_debate")

        assert allowed is False
        assert "Concurrent debate limit (1) reached" in message

    @pytest.mark.asyncio
    async def test_check_limits_add_member_allowed(self, team_workspace):
        """check_limits allows adding members under limit."""
        manager = WorkspaceManager()
        # team_workspace has 3 members, limit is 10
        allowed, message = await manager.check_limits(team_workspace, "add_member")

        assert allowed is True
        assert message == ""

    @pytest.mark.asyncio
    async def test_check_limits_add_member_limit_reached(self, free_workspace):
        """check_limits blocks adding members when limit reached."""
        # Add a member to free workspace (limit is 1)
        free_workspace.member_ids = ["existing-member"]

        manager = WorkspaceManager()
        allowed, message = await manager.check_limits(free_workspace, "add_member")

        assert allowed is False
        assert "Member limit (1) reached" in message

    @pytest.mark.asyncio
    async def test_check_limits_enterprise_unlimited(self, mock_storage):
        """check_limits always allows enterprise plan actions."""
        enterprise_workspace = Workspace(
            id="ws-ent",
            name="Enterprise Workspace",
            owner_id="owner-ent",
            plan="enterprise",
        )
        mock_storage.get_workspace_usage.return_value = {
            "debates_this_month": 999999,
            "active_debates": 999999,
        }

        manager = WorkspaceManager(storage=mock_storage)
        allowed, message = await manager.check_limits(enterprise_workspace, "create_debate")

        assert allowed is True

    @pytest.mark.asyncio
    async def test_get_usage_default_values(self):
        """get_usage returns zero defaults when no storage."""
        manager = WorkspaceManager()
        usage = await manager.get_usage("any-workspace")

        assert usage["debates_this_month"] == 0
        assert usage["active_debates"] == 0
        assert usage["total_tokens_used"] == 0

    @pytest.mark.asyncio
    async def test_get_usage_from_storage(self, mock_storage):
        """get_usage retrieves data from storage."""
        mock_storage.get_workspace_usage.return_value = {
            "debates_this_month": 42,
            "active_debates": 3,
            "total_tokens_used": 100000,
        }

        manager = WorkspaceManager(storage=mock_storage)
        usage = await manager.get_usage("ws-123")

        assert usage["debates_this_month"] == 42
        assert usage["active_debates"] == 3
        assert usage["total_tokens_used"] == 100000


# =============================================================================
# Test Global Workspace Manager
# =============================================================================


class TestGlobalWorkspaceManager:
    """Tests for global workspace manager singleton."""

    def test_get_workspace_manager_returns_singleton(self):
        """get_workspace_manager returns same instance on multiple calls."""
        import aragora.server.middleware.tenancy as tenancy_module

        # Reset global
        tenancy_module._workspace_manager = None

        manager1 = get_workspace_manager()
        manager2 = get_workspace_manager()

        assert manager1 is manager2
        assert isinstance(manager1, WorkspaceManager)

    def test_get_workspace_manager_creates_instance(self):
        """get_workspace_manager creates new instance if none exists."""
        import aragora.server.middleware.tenancy as tenancy_module

        tenancy_module._workspace_manager = None

        manager = get_workspace_manager()

        assert manager is not None
        assert isinstance(manager, WorkspaceManager)


# =============================================================================
# Test require_workspace Decorator
# =============================================================================


class TestRequireWorkspaceDecorator:
    """Tests for require_workspace decorator."""

    @pytest.mark.asyncio
    async def test_require_workspace_no_handler(self):
        """require_workspace returns 500 when no handler provided."""

        @require_workspace
        async def endpoint():
            return {"ok": True}

        result = await endpoint()

        assert result.status_code == 500
        assert b"No request handler" in result.body

    @pytest.mark.asyncio
    async def test_require_workspace_unauthenticated(self, mock_handler):
        """require_workspace returns 401 for unauthenticated requests."""

        @require_workspace
        async def endpoint(handler, user, workspace):
            return {"workspace_id": workspace.id}

        with patch("aragora.server.middleware.tenancy.get_current_user", return_value=None):
            result = await endpoint(handler=mock_handler)

        assert result.status_code == 401
        assert b"Authentication required" in result.body

    @pytest.mark.asyncio
    async def test_require_workspace_injects_user_and_workspace(
        self, mock_handler, free_user, free_workspace
    ):
        """require_workspace injects both user and workspace."""

        @require_workspace
        async def endpoint(handler, user, workspace):
            return {"user_id": user.id, "workspace_id": workspace.id}

        with patch("aragora.server.middleware.tenancy.get_current_user", return_value=free_user):
            with patch(
                "aragora.server.middleware.tenancy.get_workspace_manager"
            ) as mock_get_manager:
                mock_manager = AsyncMock()
                mock_manager.get_user_workspace.return_value = free_workspace
                mock_get_manager.return_value = mock_manager

                result = await endpoint(handler=mock_handler)

        assert result["user_id"] == "user-free-123"
        assert result["workspace_id"] == "ws-free-123"

    @pytest.mark.asyncio
    async def test_require_workspace_workspace_not_found(self, mock_handler, free_user):
        """require_workspace returns 404 when workspace not found."""

        @require_workspace
        async def endpoint(handler, user, workspace):
            return {"ok": True}

        with patch("aragora.server.middleware.tenancy.get_current_user", return_value=free_user):
            with patch(
                "aragora.server.middleware.tenancy.get_workspace_manager"
            ) as mock_get_manager:
                mock_manager = AsyncMock()
                mock_manager.get_user_workspace.return_value = None
                mock_get_manager.return_value = mock_manager

                result = await endpoint(handler=mock_handler)

        assert result.status_code == 404
        assert b"Workspace not found" in result.body

    @pytest.mark.asyncio
    async def test_require_workspace_with_sync_function(
        self, mock_handler, free_user, free_workspace
    ):
        """require_workspace works with sync functions."""

        @require_workspace
        def sync_endpoint(handler, user, workspace):
            return {"workspace_name": workspace.name}

        with patch("aragora.server.middleware.tenancy.get_current_user", return_value=free_user):
            with patch(
                "aragora.server.middleware.tenancy.get_workspace_manager"
            ) as mock_get_manager:
                mock_manager = AsyncMock()
                mock_manager.get_user_workspace.return_value = free_workspace
                mock_get_manager.return_value = mock_manager

                result = await sync_endpoint(handler=mock_handler)

        assert result["workspace_name"] == "Free User's Workspace"

    @pytest.mark.asyncio
    async def test_require_workspace_extracts_handler_from_args(
        self, mock_handler, free_user, free_workspace
    ):
        """require_workspace finds handler in positional args."""

        @require_workspace
        async def endpoint(handler, user=None, workspace=None):
            # handler passed as positional arg, user/workspace injected by decorator
            return {"user_id": user.id}

        with patch("aragora.server.middleware.tenancy.get_current_user", return_value=free_user):
            with patch(
                "aragora.server.middleware.tenancy.get_workspace_manager"
            ) as mock_get_manager:
                mock_manager = AsyncMock()
                mock_manager.get_user_workspace.return_value = free_workspace
                mock_get_manager.return_value = mock_manager

                # Pass handler as positional arg
                result = await endpoint(mock_handler)

        assert result["user_id"] == "user-free-123"


# =============================================================================
# Test check_limit Decorator
# =============================================================================


class TestCheckLimitDecorator:
    """Tests for check_limit decorator."""

    @pytest.mark.asyncio
    async def test_check_limit_no_workspace(self):
        """check_limit returns 400 when workspace not in kwargs."""

        @check_limit("create_debate")
        async def endpoint():
            return {"ok": True}

        result = await endpoint()

        assert result.status_code == 400
        assert b"Workspace required" in result.body

    @pytest.mark.asyncio
    async def test_check_limit_allows_under_limit(self, free_workspace, mock_storage):
        """check_limit allows action when under limit."""
        mock_storage.get_workspace_usage.return_value = {
            "debates_this_month": 10,
            "active_debates": 0,
        }

        @check_limit("create_debate")
        async def endpoint(workspace):
            return {"created": True}

        with patch("aragora.server.middleware.tenancy.get_workspace_manager") as mock_get_manager:
            mock_manager = WorkspaceManager(storage=mock_storage)
            mock_get_manager.return_value = mock_manager

            result = await endpoint(workspace=free_workspace)

        assert result == {"created": True}

    @pytest.mark.asyncio
    async def test_check_limit_blocks_at_limit(self, free_workspace, mock_storage):
        """check_limit returns 403 when limit reached."""
        mock_storage.get_workspace_usage.return_value = {
            "debates_this_month": 50,
            "active_debates": 0,
        }

        @check_limit("create_debate")
        async def endpoint(workspace):
            return {"created": True}

        with patch("aragora.server.middleware.tenancy.get_workspace_manager") as mock_get_manager:
            mock_manager = WorkspaceManager(storage=mock_storage)
            mock_get_manager.return_value = mock_manager

            result = await endpoint(workspace=free_workspace)

        assert result.status_code == 403
        assert b"Monthly debate limit" in result.body

    @pytest.mark.asyncio
    async def test_check_limit_with_sync_function(self, free_workspace):
        """check_limit works with sync functions."""

        @check_limit("add_member")
        def sync_endpoint(workspace):
            return {"added": True}

        with patch("aragora.server.middleware.tenancy.get_workspace_manager") as mock_get_manager:
            mock_manager = AsyncMock()
            mock_manager.check_limits.return_value = (True, "")
            mock_get_manager.return_value = mock_manager

            result = await sync_endpoint(workspace=free_workspace)

        assert result == {"added": True}


# =============================================================================
# Test tenant_scoped Decorator
# =============================================================================


class TestTenantScopedDecorator:
    """Tests for tenant_scoped decorator."""

    @pytest.mark.asyncio
    async def test_tenant_scoped_requires_workspace_id(self):
        """tenant_scoped raises ValueError for empty workspace_id."""

        @tenant_scoped
        async def get_debates(workspace_id: str):
            return [{"id": "debate-1"}]

        with pytest.raises(ValueError) as exc_info:
            await get_debates("")

        assert "workspace_id is required" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_tenant_scoped_passes_workspace_id(self):
        """tenant_scoped passes workspace_id to function."""

        @tenant_scoped
        async def get_debates(workspace_id: str, limit: int = 10):
            return {"workspace_id": workspace_id, "limit": limit}

        result = await get_debates("ws-123", limit=20)

        assert result["workspace_id"] == "ws-123"
        assert result["limit"] == 20

    @pytest.mark.asyncio
    async def test_tenant_scoped_with_sync_function(self):
        """tenant_scoped works with sync functions."""

        @tenant_scoped
        def get_debates_sync(workspace_id: str):
            return {"workspace_id": workspace_id}

        result = await get_debates_sync("ws-456")

        assert result["workspace_id"] == "ws-456"

    @pytest.mark.asyncio
    async def test_tenant_scoped_none_workspace_id(self):
        """tenant_scoped rejects None workspace_id."""

        @tenant_scoped
        async def get_data(workspace_id: str):
            return {"data": "value"}

        with pytest.raises(ValueError) as exc_info:
            await get_data(None)

        assert "workspace_id is required" in str(exc_info.value)


# =============================================================================
# Test scope_query Utility
# =============================================================================


class TestScopeQuery:
    """Tests for scope_query utility function."""

    def test_scope_query_with_filter_method(self):
        """scope_query adds filter_by for SQLAlchemy-style queries."""
        mock_query = MagicMock()
        mock_query.filter_by.return_value = mock_query

        result = scope_query(mock_query, "ws-123")

        mock_query.filter_by.assert_called_once_with(workspace_id="ws-123")
        assert result is mock_query

    def test_scope_query_with_where_method(self):
        """scope_query adds where clause for SQLite-style queries."""
        # Query without filter but with where
        mock_query = MagicMock(spec=["where"])
        mock_query.where.return_value = mock_query

        result = scope_query(mock_query, "ws-456")

        mock_query.where.assert_called_once_with("workspace_id = ?", ("ws-456",))
        assert result is mock_query

    def test_scope_query_unsupported_query_type(self):
        """scope_query returns query unchanged if no supported methods."""
        mock_query = MagicMock(spec=[])  # No filter or where method

        result = scope_query(mock_query, "ws-789")

        assert result is mock_query


# =============================================================================
# Test ensure_workspace_access Utility
# =============================================================================


class TestEnsureWorkspaceAccess:
    """Tests for ensure_workspace_access utility function."""

    @pytest.mark.asyncio
    async def test_ensure_workspace_access_grants_access(self, free_user, free_workspace):
        """ensure_workspace_access returns True for authorized access."""
        with patch("aragora.server.middleware.tenancy.get_workspace_manager") as mock_get_manager:
            mock_manager = AsyncMock()
            mock_manager.check_user_access.return_value = True
            mock_get_manager.return_value = mock_manager

            result = await ensure_workspace_access(free_user, "ws-free-123")

        assert result is True
        mock_manager.check_user_access.assert_called_once_with(free_user, "ws-free-123")

    @pytest.mark.asyncio
    async def test_ensure_workspace_access_raises_on_denial(self, free_user):
        """ensure_workspace_access raises PermissionError for unauthorized access."""
        with patch("aragora.server.middleware.tenancy.get_workspace_manager") as mock_get_manager:
            mock_manager = AsyncMock()
            mock_manager.check_user_access.return_value = False
            mock_get_manager.return_value = mock_manager

            with pytest.raises(PermissionError) as exc_info:
                await ensure_workspace_access(free_user, "other-workspace")

        assert "Access denied to workspace other-workspace" in str(exc_info.value)


# =============================================================================
# Test Cross-Tenant Access Prevention
# =============================================================================


class TestCrossTenantIsolation:
    """Tests for cross-tenant access prevention."""

    @pytest.mark.asyncio
    async def test_user_cannot_access_other_workspace(self, free_user, team_workspace):
        """User from one workspace cannot access another workspace."""
        manager = WorkspaceManager()
        manager._cache["ws-team-789"] = team_workspace

        result = await manager.check_user_access(free_user, "ws-team-789")

        assert result is False

    @pytest.mark.asyncio
    async def test_tenant_scoped_prevents_cross_workspace_queries(self):
        """tenant_scoped decorator prevents accessing other workspaces."""
        queries_made = []

        @tenant_scoped
        async def get_debates(workspace_id: str):
            queries_made.append(workspace_id)
            return [{"workspace_id": workspace_id}]

        # User A queries their workspace
        await get_debates("ws-user-a")

        # User B queries their workspace
        await get_debates("ws-user-b")

        # Each query is scoped to its own workspace
        assert queries_made == ["ws-user-a", "ws-user-b"]

    @pytest.mark.asyncio
    async def test_workspace_isolation_between_plans(
        self, free_user, pro_user, free_workspace, pro_workspace
    ):
        """Users from different plan tiers cannot access each other's workspaces."""
        manager = WorkspaceManager()
        manager._cache["ws-free-123"] = free_workspace
        manager._cache["ws-pro-456"] = pro_workspace

        # Free user cannot access pro workspace
        result1 = await manager.check_user_access(free_user, "ws-pro-456")
        assert result1 is False

        # Pro user cannot access free workspace
        result2 = await manager.check_user_access(pro_user, "ws-free-123")
        assert result2 is False


# =============================================================================
# Test Error Handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in tenancy middleware."""

    @pytest.mark.asyncio
    async def test_storage_error_in_get_workspace(self, mock_storage):
        """get_workspace handles storage errors gracefully."""
        mock_storage.get_workspace.side_effect = ConnectionError("Database down")

        manager = WorkspaceManager(storage=mock_storage)
        result = await manager.get_workspace("ws-error")

        assert result is None

    @pytest.mark.asyncio
    async def test_storage_error_in_save_workspace(self, mock_storage):
        """create_default_workspace handles storage save errors."""
        mock_storage.save_workspace.side_effect = Exception("Write error")

        user = User(id="user-save-error", email="error@example.com", plan="free")

        manager = WorkspaceManager(storage=mock_storage)
        # Should not raise, but log error
        workspace = await manager.create_default_workspace(user)

        # Workspace is still created in memory
        assert workspace is not None
        assert workspace.owner_id == "user-save-error"

    @pytest.mark.asyncio
    async def test_storage_error_in_get_usage(self, mock_storage):
        """get_usage returns defaults on storage error."""
        mock_storage.get_workspace_usage.side_effect = Exception("Query failed")

        manager = WorkspaceManager(storage=mock_storage)
        usage = await manager.get_usage("ws-error")

        assert usage["debates_this_month"] == 0
        assert usage["active_debates"] == 0


# =============================================================================
# Test Module Exports
# =============================================================================


class TestModuleExports:
    """Tests for module's __all__ exports."""

    def test_all_exports_importable(self):
        """All items in __all__ can be imported."""
        from aragora.server.middleware import tenancy

        for name in tenancy.__all__:
            assert hasattr(tenancy, name), f"Missing export: {name}"

    def test_exported_items(self):
        """Key items are exported in __all__."""
        from aragora.server.middleware.tenancy import __all__

        expected = [
            "PLAN_LIMITS",
            "get_plan_limits",
            "WorkspaceManager",
            "get_workspace_manager",
            "require_workspace",
            "check_limit",
            "tenant_scoped",
            "scope_query",
            "ensure_workspace_access",
        ]

        for item in expected:
            assert item in __all__, f"Expected {item} in __all__"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
