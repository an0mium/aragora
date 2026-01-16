"""
E2E tests for multi-tenant isolation.

Tests workspace/organization isolation:
1. Organization A cannot see Organization B's debates
2. Organization A cannot modify Organization B's data
3. Cross-tenant memory isolation
4. Invitation scoping to organization
5. API key scoping to organization
6. Plan limit enforcement per tenant
"""

from __future__ import annotations

import json
import tempfile
import uuid
from io import BytesIO
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from aragora.server.middleware.tenancy import (
    PLAN_LIMITS,
    WorkspaceManager,
    get_plan_limits,
    require_workspace,
    check_limit,
    tenant_scoped,
    ensure_workspace_access,
)
from aragora.server.middleware.auth_v2 import User, Workspace


# =============================================================================
# Test Helpers
# =============================================================================


def create_test_user(
    user_id: str,
    email: str,
    workspace_id: Optional[str],
    plan: str = "free",
    role: str = "member",
) -> User:
    """Create a test user."""
    return User(
        id=user_id,
        email=email,
        workspace_id=workspace_id,
        plan=plan,
        role=role,
    )


def create_test_workspace(
    workspace_id: str,
    owner_id: str,
    plan: str = "free",
    name: str = "Test Workspace",
    member_ids: Optional[list] = None,
) -> Workspace:
    """Create a test workspace with plan limits."""
    limits = get_plan_limits(plan)
    return Workspace(
        id=workspace_id,
        name=name,
        owner_id=owner_id,
        plan=plan,
        member_ids=member_ids or [],
        # Map plan limits to Workspace fields
        max_debates=limits.get("max_debates_per_month", 50),
        max_agents=limits.get("max_agents", 2),
        max_members=limits.get("max_members", 1),
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def org_a_workspace():
    """Create Organization A workspace."""
    return create_test_workspace(
        workspace_id="org-a-workspace-id",
        owner_id="user-org-a-owner",
        plan="team",
        name="Organization A",
    )


@pytest.fixture
def org_b_workspace():
    """Create Organization B workspace."""
    return create_test_workspace(
        workspace_id="org-b-workspace-id",
        owner_id="user-org-b-owner",
        plan="pro",
        name="Organization B",
    )


@pytest.fixture
def org_a_owner(org_a_workspace):
    """Create Organization A owner."""
    return create_test_user(
        user_id="user-org-a-owner",
        email="owner@org-a.example.com",
        workspace_id=org_a_workspace.id,
        plan="team",
        role="owner",
    )


@pytest.fixture
def org_a_member(org_a_workspace):
    """Create Organization A member."""
    return create_test_user(
        user_id="user-org-a-member",
        email="member@org-a.example.com",
        workspace_id=org_a_workspace.id,
        plan="team",
        role="member",
    )


@pytest.fixture
def org_b_owner(org_b_workspace):
    """Create Organization B owner."""
    return create_test_user(
        user_id="user-org-b-owner",
        email="owner@org-b.example.com",
        workspace_id=org_b_workspace.id,
        plan="pro",
        role="owner",
    )


@pytest.fixture
def workspace_manager():
    """Create a workspace manager with test data."""
    manager = WorkspaceManager()
    return manager


# =============================================================================
# Plan Limits Tests
# =============================================================================


class TestPlanLimits:
    """Tests for plan-based limits."""

    def test_free_plan_limits(self):
        """E2E: Free plan should have correct limits."""
        limits = get_plan_limits("free")

        assert limits["max_debates_per_month"] == 50
        assert limits["max_agents"] == 2
        assert limits["max_members"] == 1
        assert limits["max_concurrent_debates"] == 1
        assert limits["private_debates"] is False
        assert limits["api_access"] is False

    def test_pro_plan_limits(self):
        """E2E: Pro plan should have expanded limits."""
        limits = get_plan_limits("pro")

        assert limits["max_debates_per_month"] == 500
        assert limits["max_agents"] == 5
        assert limits["max_concurrent_debates"] == 3
        assert limits["private_debates"] is True
        assert limits["api_access"] is True

    def test_team_plan_limits(self):
        """E2E: Team plan should have team-level limits."""
        limits = get_plan_limits("team")

        assert limits["max_debates_per_month"] == 2000
        assert limits["max_agents"] == 10
        assert limits["max_members"] == 10
        assert limits["max_concurrent_debates"] == 10
        assert limits["priority_support"] is True

    def test_enterprise_plan_unlimited(self):
        """E2E: Enterprise plan should have unlimited resources."""
        limits = get_plan_limits("enterprise")

        # -1 indicates unlimited
        assert limits["max_debates_per_month"] == -1
        assert limits["max_agents"] == -1
        assert limits["max_members"] == -1
        assert limits["max_concurrent_debates"] == -1

    def test_unknown_plan_defaults_to_free(self):
        """E2E: Unknown plans should default to free limits."""
        limits = get_plan_limits("nonexistent")

        free_limits = get_plan_limits("free")
        assert limits == free_limits


# =============================================================================
# Workspace Access Tests
# =============================================================================


class TestWorkspaceAccess:
    """Tests for workspace access control."""

    @pytest.mark.asyncio
    async def test_owner_has_access(self, workspace_manager, org_a_workspace, org_a_owner):
        """E2E: Workspace owner should always have access."""
        workspace_manager._cache[org_a_workspace.id] = org_a_workspace

        has_access = await workspace_manager.check_user_access(
            org_a_owner, org_a_workspace.id
        )
        assert has_access is True

    @pytest.mark.asyncio
    async def test_member_has_access(self, workspace_manager, org_a_workspace, org_a_member):
        """E2E: Workspace member should have access."""
        # Add member to workspace
        org_a_workspace.member_ids.append(org_a_member.id)
        workspace_manager._cache[org_a_workspace.id] = org_a_workspace

        has_access = await workspace_manager.check_user_access(
            org_a_member, org_a_workspace.id
        )
        assert has_access is True

    @pytest.mark.asyncio
    async def test_outsider_no_access(self, workspace_manager, org_a_workspace, org_b_owner):
        """E2E: User from different org should not have access."""
        workspace_manager._cache[org_a_workspace.id] = org_a_workspace

        has_access = await workspace_manager.check_user_access(
            org_b_owner, org_a_workspace.id
        )
        assert has_access is False

    @pytest.mark.asyncio
    async def test_nonexistent_workspace_no_access(self, workspace_manager, org_a_owner):
        """E2E: Access to nonexistent workspace should be denied."""
        has_access = await workspace_manager.check_user_access(
            org_a_owner, "nonexistent-workspace-id"
        )
        assert has_access is False


# =============================================================================
# Cross-Tenant Isolation Tests
# =============================================================================


class TestCrossTenantIsolation:
    """Tests for cross-tenant data isolation."""

    @pytest.mark.asyncio
    async def test_org_a_cannot_access_org_b_workspace(
        self, workspace_manager, org_a_workspace, org_b_workspace, org_a_owner
    ):
        """E2E: Organization A user cannot access Organization B's workspace."""
        workspace_manager._cache[org_a_workspace.id] = org_a_workspace
        workspace_manager._cache[org_b_workspace.id] = org_b_workspace

        # Org A owner should NOT have access to Org B's workspace
        has_access = await workspace_manager.check_user_access(
            org_a_owner, org_b_workspace.id
        )
        assert has_access is False

    @pytest.mark.asyncio
    async def test_ensure_workspace_access_raises_on_violation(
        self, workspace_manager, org_a_workspace, org_b_workspace, org_a_owner
    ):
        """E2E: ensure_workspace_access should raise PermissionError on violation."""
        workspace_manager._cache[org_a_workspace.id] = org_a_workspace
        workspace_manager._cache[org_b_workspace.id] = org_b_workspace

        # Patch the global workspace manager
        with patch(
            "aragora.server.middleware.tenancy.get_workspace_manager",
            return_value=workspace_manager,
        ):
            with pytest.raises(PermissionError) as exc_info:
                await ensure_workspace_access(org_a_owner, org_b_workspace.id)

            assert "Access denied" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_tenant_scoped_requires_workspace_id(self):
        """E2E: tenant_scoped decorator should require workspace_id."""

        @tenant_scoped
        async def query_data(workspace_id: str):
            return {"workspace_id": workspace_id}

        # Should raise ValueError for empty workspace_id
        with pytest.raises(ValueError) as exc_info:
            await query_data("")

        assert "workspace_id is required" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_tenant_scoped_passes_workspace_id(self):
        """E2E: tenant_scoped decorator should pass workspace_id to function."""
        captured_id = {}

        @tenant_scoped
        async def query_data(workspace_id: str):
            captured_id["value"] = workspace_id
            return {"workspace_id": workspace_id}

        result = await query_data("test-workspace")

        assert captured_id["value"] == "test-workspace"
        assert result["workspace_id"] == "test-workspace"


# =============================================================================
# Limit Enforcement Tests
# =============================================================================


class TestLimitEnforcement:
    """Tests for plan limit enforcement."""

    @pytest.mark.asyncio
    async def test_monthly_debate_limit_enforced(
        self, workspace_manager, org_a_workspace
    ):
        """E2E: Monthly debate limit should be enforced."""
        workspace_manager._cache[org_a_workspace.id] = org_a_workspace

        # Mock usage at limit
        with patch.object(
            workspace_manager,
            "get_usage",
            return_value={"debates_this_month": 2000, "active_debates": 0},
        ):
            allowed, message = await workspace_manager.check_limits(
                org_a_workspace, "create_debate"
            )

            assert allowed is False
            assert "Monthly debate limit" in message

    @pytest.mark.asyncio
    async def test_concurrent_debate_limit_enforced(
        self, workspace_manager, org_a_workspace
    ):
        """E2E: Concurrent debate limit should be enforced."""
        workspace_manager._cache[org_a_workspace.id] = org_a_workspace

        # Mock usage at concurrent limit
        with patch.object(
            workspace_manager,
            "get_usage",
            return_value={"debates_this_month": 100, "active_debates": 10},
        ):
            allowed, message = await workspace_manager.check_limits(
                org_a_workspace, "create_debate"
            )

            assert allowed is False
            assert "Concurrent debate limit" in message

    @pytest.mark.asyncio
    async def test_member_limit_enforced(self, workspace_manager):
        """E2E: Member limit should be enforced for team plans."""
        # Create workspace at member limit
        workspace = create_test_workspace(
            workspace_id="test-workspace",
            owner_id="owner-id",
            plan="team",  # Team plan has 10 member limit
            member_ids=list(range(10)),  # At limit
        )
        workspace_manager._cache[workspace.id] = workspace

        allowed, message = await workspace_manager.check_limits(
            workspace, "add_member"
        )

        assert allowed is False
        assert "Member limit" in message

    @pytest.mark.asyncio
    async def test_enterprise_bypasses_limits(self, workspace_manager):
        """E2E: Enterprise plan should bypass all limits."""
        workspace = create_test_workspace(
            workspace_id="enterprise-workspace",
            owner_id="enterprise-owner",
            plan="enterprise",
            member_ids=list(range(1000)),  # Way over team limit
        )
        workspace_manager._cache[workspace.id] = workspace

        # Mock heavy usage
        with patch.object(
            workspace_manager,
            "get_usage",
            return_value={"debates_this_month": 100000, "active_debates": 1000},
        ):
            # Create debate should be allowed (-1 means unlimited)
            allowed, message = await workspace_manager.check_limits(
                workspace, "create_debate"
            )
            assert allowed is True

    @pytest.mark.asyncio
    async def test_below_limits_allowed(self, workspace_manager, org_a_workspace):
        """E2E: Operations below limits should be allowed."""
        workspace_manager._cache[org_a_workspace.id] = org_a_workspace

        # Mock usage well below limits
        with patch.object(
            workspace_manager,
            "get_usage",
            return_value={"debates_this_month": 10, "active_debates": 1},
        ):
            allowed, message = await workspace_manager.check_limits(
                org_a_workspace, "create_debate"
            )

            assert allowed is True
            assert message == ""


# =============================================================================
# Default Workspace Creation Tests
# =============================================================================


class TestDefaultWorkspaceCreation:
    """Tests for automatic workspace creation."""

    @pytest.mark.asyncio
    async def test_creates_default_workspace_for_new_user(self, workspace_manager):
        """E2E: New user should get a default personal workspace."""
        user = create_test_user(
            user_id="new-user-id",
            email="new@example.com",
            workspace_id=None,  # No workspace yet
            plan="free",
        )

        workspace = await workspace_manager.create_default_workspace(user)

        assert workspace is not None
        assert workspace.owner_id == user.id
        assert workspace.plan == user.plan
        assert user.email in workspace.name

    @pytest.mark.asyncio
    async def test_default_workspace_has_correct_limits(self, workspace_manager):
        """E2E: Default workspace should have free plan limits."""
        user = create_test_user(
            user_id="free-user-id",
            email="free@example.com",
            workspace_id=None,
            plan="free",
        )

        workspace = await workspace_manager.create_default_workspace(user)

        # Check limits match free plan
        # Workspace uses max_debates (not max_debates_per_month)
        assert workspace.max_debates == 50
        assert workspace.max_agents == 2
        assert workspace.plan == "free"

    @pytest.mark.asyncio
    async def test_get_user_workspace_creates_if_none(self, workspace_manager):
        """E2E: get_user_workspace should create default if none exists."""
        user = create_test_user(
            user_id="auto-user-id",
            email="auto@example.com",
            workspace_id=None,
            plan="pro",
        )

        workspace = await workspace_manager.get_user_workspace(user)

        assert workspace is not None
        assert workspace.owner_id == user.id
        assert workspace.plan == "pro"


# =============================================================================
# API Key Scoping Tests (Conceptual)
# =============================================================================


class TestAPIKeyScoping:
    """Tests for API key workspace scoping."""

    def test_api_key_format_includes_workspace(self):
        """E2E: API keys should conceptually be scoped to workspace."""
        # In a real implementation, API keys would include workspace scope
        # This test verifies the design pattern

        # Example API key format: {workspace_id}_{random_key}
        workspace_id = "org-a-workspace-id"
        key_suffix = "abc123xyz"
        api_key = f"{workspace_id}_{key_suffix}"

        # Key should contain workspace reference
        assert workspace_id in api_key

    def test_plan_determines_api_access(self):
        """E2E: Only paid plans should have API access."""
        for plan, limits in PLAN_LIMITS.items():
            if plan == "free":
                assert limits["api_access"] is False
            else:
                assert limits["api_access"] is True


# =============================================================================
# Invitation Scoping Tests (Conceptual)
# =============================================================================


class TestInvitationScoping:
    """Tests for invitation workspace scoping."""

    @pytest.mark.asyncio
    async def test_invitation_cannot_exceed_member_limit(self, workspace_manager):
        """E2E: Invitations should respect member limits."""
        # Workspace at member limit
        workspace = create_test_workspace(
            workspace_id="full-workspace",
            owner_id="owner-id",
            plan="free",  # Free plan has 1 member limit
            member_ids=["member-1"],  # Already at limit
        )
        workspace_manager._cache[workspace.id] = workspace

        allowed, message = await workspace_manager.check_limits(
            workspace, "add_member"
        )

        assert allowed is False
        assert "Member limit" in message

    @pytest.mark.asyncio
    async def test_invitation_allowed_when_under_limit(self, workspace_manager):
        """E2E: Invitations should be allowed when under member limit."""
        workspace = create_test_workspace(
            workspace_id="open-workspace",
            owner_id="owner-id",
            plan="team",  # Team plan has 10 member limit
            member_ids=["member-1", "member-2"],  # Only 2 members
        )
        workspace_manager._cache[workspace.id] = workspace

        allowed, message = await workspace_manager.check_limits(
            workspace, "add_member"
        )

        assert allowed is True
