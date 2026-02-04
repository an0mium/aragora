"""
Tests for aragora.server.handlers.workspace.policies module.

Tests cover:
1. WorkspacePoliciesMixin handler methods
2. Retention policy CRUD operations
3. Policy execution and dry-run mode
4. Cache behavior for policy lookups
5. Authentication and RBAC checks
6. Error handling and validation
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Mock Data Classes
# =============================================================================


@dataclass
class MockRetentionPolicy:
    """Mock retention policy for testing."""

    id: str = "pol_test123"
    name: str = "Test Policy"
    description: str = "Test description"
    retention_days: int = 90
    action: MagicMock = field(default_factory=lambda: MagicMock(value="delete"))
    enabled: bool = True
    applies_to: list[str] = field(default_factory=lambda: ["documents", "findings"])
    workspace_ids: list[str] | None = None
    grace_period_days: int = 7
    notify_before_days: int = 14
    exclude_sensitivity_levels: list[str] = field(default_factory=list)
    exclude_tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_run: datetime | None = None


@dataclass
class MockExecutionReport:
    """Mock policy execution report."""

    policy_id: str = "pol_test123"
    items_evaluated: int = 100
    items_deleted: int = 25
    items_archived: int = 0
    items_anonymized: int = 0
    items_notified: int = 0
    execution_time_ms: float = 150.5
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "items_evaluated": self.items_evaluated,
            "items_deleted": self.items_deleted,
            "items_archived": self.items_archived,
            "items_anonymized": self.items_anonymized,
            "items_notified": self.items_notified,
            "execution_time_ms": self.execution_time_ms,
            "errors": self.errors,
        }


@dataclass
class MockAuthContext:
    """Mock authorization context."""

    user_id: str = "user_123"
    is_authenticated: bool = True
    tenant_id: str = "tenant_123"


# =============================================================================
# Mock Module Helper
# =============================================================================


def create_mock_module():
    """Create a mock of the workspace_module with all required attributes."""
    mock_mod = MagicMock()
    mock_mod.PERM_RETENTION_READ = "retention:read"
    mock_mod.PERM_RETENTION_WRITE = "retention:write"
    mock_mod.PERM_RETENTION_DELETE = "retention:delete"
    mock_mod.PERM_RETENTION_EXECUTE = "retention:execute"
    mock_mod.RetentionAction = MagicMock(side_effect=lambda x: MagicMock(value=x))
    mock_mod.AuditAction = MagicMock()
    mock_mod.AuditAction.MODIFY_POLICY = "modify_policy"
    mock_mod.AuditAction.EXECUTE_RETENTION = "execute_retention"
    mock_mod.Actor = MagicMock()
    mock_mod.Resource = MagicMock()
    mock_mod.AuditOutcome = MagicMock()
    mock_mod.AuditOutcome.SUCCESS = "success"
    mock_mod._retention_policy_cache = MagicMock()
    mock_mod._retention_policy_cache.get = MagicMock(return_value=None)
    mock_mod._retention_policy_cache.set = MagicMock()
    mock_mod._invalidate_retention_cache = MagicMock()
    mock_mod.extract_user_from_request = MagicMock(return_value=MockAuthContext())
    mock_mod.error_response = MagicMock(
        side_effect=lambda msg, status: {"error": msg, "status": status}
    )
    mock_mod.json_response = MagicMock(
        side_effect=lambda data, status=200: {"data": data, "status": status}
    )
    return mock_mod


# =============================================================================
# Mock Handler Class
# =============================================================================


class MockWorkspaceHandler:
    """Mock handler class that includes the policies mixin."""

    def __init__(self):
        self._user_store = MagicMock()
        self._retention_manager = MagicMock()
        self._audit_log = MagicMock()
        self._audit_log.log = AsyncMock()
        self._rbac_error = None

    def _get_user_store(self):
        return self._user_store

    def _get_retention_manager(self):
        return self._retention_manager

    def _get_audit_log(self):
        return self._audit_log

    def _run_async(self, coro):
        """Run async coroutine synchronously."""
        if asyncio.iscoroutine(coro):
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()
        return coro

    def _check_rbac_permission(self, handler, perm, auth_ctx):
        return self._rbac_error

    def read_json_body(self, handler):
        return handler._json_body


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_handler():
    """Create a mock handler instance."""
    return MagicMock()


@pytest.fixture
def mock_workspace_handler():
    """Create a mock workspace handler with mixin methods."""
    handler = MockWorkspaceHandler()
    handler._retention_manager.list_policies = MagicMock(return_value=[MockRetentionPolicy()])
    handler._retention_manager.get_policy = MagicMock(return_value=MockRetentionPolicy())
    handler._retention_manager.create_policy = MagicMock(return_value=MockRetentionPolicy())
    handler._retention_manager.update_policy = MagicMock(return_value=MockRetentionPolicy())
    handler._retention_manager.delete_policy = MagicMock()
    handler._retention_manager.execute_policy = AsyncMock(return_value=MockExecutionReport())
    handler._retention_manager.check_expiring_soon = AsyncMock(return_value=[])
    return handler


# =============================================================================
# Test List Policies
# =============================================================================


class TestListPolicies:
    """Tests for _handle_list_policies."""

    def test_list_policies_success(self, mock_workspace_handler):
        """List policies returns policy list."""
        mock_mod = create_mock_module()
        http_handler = MagicMock()

        with patch("aragora.server.handlers.workspace.policies._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.policies import WorkspacePoliciesMixin

            # Create a combined handler
            class TestHandler(WorkspacePoliciesMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            result = handler._handle_list_policies(http_handler, {})

        # Check response structure
        assert "data" in result
        assert "policies" in result["data"]
        assert "total" in result["data"]

    def test_list_policies_not_authenticated(self, mock_workspace_handler):
        """List policies returns 401 when not authenticated."""
        mock_mod = create_mock_module()
        mock_mod.extract_user_from_request.return_value = MockAuthContext(is_authenticated=False)
        http_handler = MagicMock()

        with patch("aragora.server.handlers.workspace.policies._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.policies import WorkspacePoliciesMixin

            class TestHandler(WorkspacePoliciesMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            result = handler._handle_list_policies(http_handler, {})

        assert result["status"] == 401
        assert "Not authenticated" in result["error"]

    def test_list_policies_rbac_denied(self, mock_workspace_handler):
        """List policies returns error when RBAC denied."""
        mock_mod = create_mock_module()
        http_handler = MagicMock()

        with patch("aragora.server.handlers.workspace.policies._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.policies import WorkspacePoliciesMixin

            class TestHandler(WorkspacePoliciesMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            handler._rbac_error = {"error": "Permission denied", "status": 403}
            result = handler._handle_list_policies(http_handler, {})

        assert result["status"] == 403

    def test_list_policies_with_workspace_filter(self, mock_workspace_handler):
        """List policies filters by workspace_id."""
        mock_mod = create_mock_module()
        http_handler = MagicMock()

        with patch("aragora.server.handlers.workspace.policies._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.policies import WorkspacePoliciesMixin

            class TestHandler(WorkspacePoliciesMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            handler._handle_list_policies(http_handler, {"workspace_id": "ws_123"})

        # Verify filter was passed to manager
        handler._retention_manager.list_policies.assert_called_with(workspace_id="ws_123")

    def test_list_policies_cache_hit(self, mock_workspace_handler):
        """List policies returns cached result when available."""
        mock_mod = create_mock_module()
        cached_data = {"policies": [], "total": 0}
        mock_mod._retention_policy_cache.get.return_value = cached_data
        http_handler = MagicMock()

        with patch("aragora.server.handlers.workspace.policies._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.policies import WorkspacePoliciesMixin

            class TestHandler(WorkspacePoliciesMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            result = handler._handle_list_policies(http_handler, {})

        # Manager should not be called when cache hit
        handler._retention_manager.list_policies.assert_not_called()
        assert result["data"] == cached_data


# =============================================================================
# Test Create Policy
# =============================================================================


class TestCreatePolicy:
    """Tests for _handle_create_policy."""

    def test_create_policy_success(self, mock_workspace_handler):
        """Create policy returns created policy."""
        mock_mod = create_mock_module()
        http_handler = MagicMock()
        http_handler._json_body = {
            "name": "Test Policy",
            "retention_days": 90,
            "action": "delete",
        }

        with patch("aragora.server.handlers.workspace.policies._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.policies import WorkspacePoliciesMixin

            class TestHandler(WorkspacePoliciesMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            result = handler._handle_create_policy(http_handler)

        assert result["status"] == 201
        assert "policy" in result["data"]
        mock_mod._invalidate_retention_cache.assert_called_once()

    def test_create_policy_missing_name(self, mock_workspace_handler):
        """Create policy returns error when name missing."""
        mock_mod = create_mock_module()
        http_handler = MagicMock()
        http_handler._json_body = {"retention_days": 90}

        with patch("aragora.server.handlers.workspace.policies._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.policies import WorkspacePoliciesMixin

            class TestHandler(WorkspacePoliciesMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            result = handler._handle_create_policy(http_handler)

        assert result["status"] == 400
        assert "name is required" in result["error"]

    def test_create_policy_invalid_json(self, mock_workspace_handler):
        """Create policy returns error when JSON invalid."""
        mock_mod = create_mock_module()
        http_handler = MagicMock()
        http_handler._json_body = None

        with patch("aragora.server.handlers.workspace.policies._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.policies import WorkspacePoliciesMixin

            class TestHandler(WorkspacePoliciesMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            result = handler._handle_create_policy(http_handler)

        assert result["status"] == 400
        assert "Invalid JSON" in result["error"]

    def test_create_policy_invalid_action(self, mock_workspace_handler):
        """Create policy returns error when action invalid."""
        mock_mod = create_mock_module()
        mock_mod.RetentionAction.side_effect = ValueError("Invalid action")
        http_handler = MagicMock()
        http_handler._json_body = {
            "name": "Test",
            "action": "invalid_action",
        }

        with patch("aragora.server.handlers.workspace.policies._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.policies import WorkspacePoliciesMixin

            class TestHandler(WorkspacePoliciesMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            result = handler._handle_create_policy(http_handler)

        assert result["status"] == 400
        assert "Invalid action" in result["error"]

    def test_create_policy_not_authenticated(self, mock_workspace_handler):
        """Create policy returns 401 when not authenticated."""
        mock_mod = create_mock_module()
        mock_mod.extract_user_from_request.return_value = MockAuthContext(is_authenticated=False)
        http_handler = MagicMock()
        http_handler._json_body = {"name": "Test"}

        with patch("aragora.server.handlers.workspace.policies._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.policies import WorkspacePoliciesMixin

            class TestHandler(WorkspacePoliciesMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            result = handler._handle_create_policy(http_handler)

        assert result["status"] == 401


# =============================================================================
# Test Get Policy
# =============================================================================


class TestGetPolicy:
    """Tests for _handle_get_policy."""

    def test_get_policy_success(self, mock_workspace_handler):
        """Get policy returns policy details."""
        mock_mod = create_mock_module()
        http_handler = MagicMock()

        with patch("aragora.server.handlers.workspace.policies._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.policies import WorkspacePoliciesMixin

            class TestHandler(WorkspacePoliciesMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            result = handler._handle_get_policy(http_handler, "pol_123")

        assert "data" in result
        assert "policy" in result["data"]

    def test_get_policy_not_found(self, mock_workspace_handler):
        """Get policy returns 404 when policy not found."""
        mock_mod = create_mock_module()
        http_handler = MagicMock()

        with patch("aragora.server.handlers.workspace.policies._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.policies import WorkspacePoliciesMixin

            class TestHandler(WorkspacePoliciesMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            handler._retention_manager.get_policy.return_value = None
            result = handler._handle_get_policy(http_handler, "pol_nonexistent")

        assert result["status"] == 404
        assert "not found" in result["error"]

    def test_get_policy_cache_hit(self, mock_workspace_handler):
        """Get policy returns cached result when available."""
        mock_mod = create_mock_module()
        cached_data = {"policy": {"id": "pol_123", "name": "Cached Policy"}}
        mock_mod._retention_policy_cache.get.return_value = cached_data
        http_handler = MagicMock()

        with patch("aragora.server.handlers.workspace.policies._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.policies import WorkspacePoliciesMixin

            class TestHandler(WorkspacePoliciesMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            result = handler._handle_get_policy(http_handler, "pol_123")

        handler._retention_manager.get_policy.assert_not_called()
        assert result["data"] == cached_data


# =============================================================================
# Test Update Policy
# =============================================================================


class TestUpdatePolicy:
    """Tests for _handle_update_policy."""

    def test_update_policy_success(self, mock_workspace_handler):
        """Update policy returns updated policy."""
        mock_mod = create_mock_module()
        http_handler = MagicMock()
        http_handler._json_body = {"retention_days": 180}

        with patch("aragora.server.handlers.workspace.policies._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.policies import WorkspacePoliciesMixin

            class TestHandler(WorkspacePoliciesMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            result = handler._handle_update_policy(http_handler, "pol_123")

        assert result["status"] == 200
        assert "policy" in result["data"]
        mock_mod._invalidate_retention_cache.assert_called_with("pol_123")

    def test_update_policy_not_found(self, mock_workspace_handler):
        """Update policy returns 404 when policy not found."""
        mock_mod = create_mock_module()
        http_handler = MagicMock()
        http_handler._json_body = {"retention_days": 180}

        with patch("aragora.server.handlers.workspace.policies._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.policies import WorkspacePoliciesMixin

            class TestHandler(WorkspacePoliciesMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            handler._retention_manager.update_policy.side_effect = ValueError("Policy not found")
            result = handler._handle_update_policy(http_handler, "pol_nonexistent")

        assert result["status"] == 404

    def test_update_policy_invalid_json(self, mock_workspace_handler):
        """Update policy returns error when JSON invalid."""
        mock_mod = create_mock_module()
        http_handler = MagicMock()
        http_handler._json_body = None

        with patch("aragora.server.handlers.workspace.policies._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.policies import WorkspacePoliciesMixin

            class TestHandler(WorkspacePoliciesMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            result = handler._handle_update_policy(http_handler, "pol_123")

        assert result["status"] == 400


# =============================================================================
# Test Delete Policy
# =============================================================================


class TestDeletePolicy:
    """Tests for _handle_delete_policy."""

    def test_delete_policy_success(self, mock_workspace_handler):
        """Delete policy returns success message."""
        mock_mod = create_mock_module()
        http_handler = MagicMock()

        with patch("aragora.server.handlers.workspace.policies._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.policies import WorkspacePoliciesMixin

            class TestHandler(WorkspacePoliciesMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            result = handler._handle_delete_policy(http_handler, "pol_123")

        assert result["status"] == 200
        assert "deleted successfully" in result["data"]["message"]
        mock_mod._invalidate_retention_cache.assert_called_with("pol_123")

    def test_delete_policy_not_authenticated(self, mock_workspace_handler):
        """Delete policy returns 401 when not authenticated."""
        mock_mod = create_mock_module()
        mock_mod.extract_user_from_request.return_value = MockAuthContext(is_authenticated=False)
        http_handler = MagicMock()

        with patch("aragora.server.handlers.workspace.policies._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.policies import WorkspacePoliciesMixin

            class TestHandler(WorkspacePoliciesMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            result = handler._handle_delete_policy(http_handler, "pol_123")

        assert result["status"] == 401


# =============================================================================
# Test Execute Policy
# =============================================================================


class TestExecutePolicy:
    """Tests for _handle_execute_policy."""

    def test_execute_policy_success(self, mock_workspace_handler):
        """Execute policy returns execution report."""
        mock_mod = create_mock_module()
        http_handler = MagicMock()

        with patch("aragora.server.handlers.workspace.policies._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.policies import WorkspacePoliciesMixin

            class TestHandler(WorkspacePoliciesMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            result = handler._handle_execute_policy(http_handler, "pol_123", {})

        assert result["status"] == 200
        assert "report" in result["data"]
        assert result["data"]["dry_run"] is False

    def test_execute_policy_dry_run(self, mock_workspace_handler):
        """Execute policy with dry_run returns preview."""
        mock_mod = create_mock_module()
        http_handler = MagicMock()

        with patch("aragora.server.handlers.workspace.policies._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.policies import WorkspacePoliciesMixin

            class TestHandler(WorkspacePoliciesMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            result = handler._handle_execute_policy(http_handler, "pol_123", {"dry_run": "true"})

        assert result["data"]["dry_run"] is True

    def test_execute_policy_not_found(self, mock_workspace_handler):
        """Execute policy returns 404 when policy not found."""
        mock_mod = create_mock_module()
        http_handler = MagicMock()

        with patch("aragora.server.handlers.workspace.policies._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.policies import WorkspacePoliciesMixin

            class TestHandler(WorkspacePoliciesMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            handler._retention_manager.execute_policy = AsyncMock(
                side_effect=ValueError("Policy not found")
            )
            result = handler._handle_execute_policy(http_handler, "pol_nonexistent", {})

        assert result["status"] == 404


# =============================================================================
# Test Expiring Items
# =============================================================================


class TestExpiringItems:
    """Tests for _handle_expiring_items."""

    def test_expiring_items_success(self, mock_workspace_handler):
        """Get expiring items returns list."""
        mock_mod = create_mock_module()
        http_handler = MagicMock()

        with patch("aragora.server.handlers.workspace.policies._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.policies import WorkspacePoliciesMixin

            class TestHandler(WorkspacePoliciesMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            result = handler._handle_expiring_items(http_handler, {})

        assert result["status"] == 200
        assert "expiring" in result["data"]
        assert "total" in result["data"]
        assert "days_ahead" in result["data"]

    def test_expiring_items_custom_days(self, mock_workspace_handler):
        """Get expiring items respects days parameter."""
        mock_mod = create_mock_module()
        http_handler = MagicMock()

        with patch("aragora.server.handlers.workspace.policies._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.policies import WorkspacePoliciesMixin

            class TestHandler(WorkspacePoliciesMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            result = handler._handle_expiring_items(http_handler, {"days": "30"})

        assert result["data"]["days_ahead"] == 30

    def test_expiring_items_with_workspace_filter(self, mock_workspace_handler):
        """Get expiring items filters by workspace."""
        mock_mod = create_mock_module()
        http_handler = MagicMock()

        with patch("aragora.server.handlers.workspace.policies._mod", return_value=mock_mod):
            from aragora.server.handlers.workspace.policies import WorkspacePoliciesMixin

            class TestHandler(WorkspacePoliciesMixin, MockWorkspaceHandler):
                pass

            handler = TestHandler()
            handler._handle_expiring_items(http_handler, {"workspace_id": "ws_123"})

        # Verify filter was passed
        handler._retention_manager.check_expiring_soon.assert_called()


__all__ = [
    "TestListPolicies",
    "TestCreatePolicy",
    "TestGetPolicy",
    "TestUpdatePolicy",
    "TestDeletePolicy",
    "TestExecutePolicy",
    "TestExpiringItems",
]
