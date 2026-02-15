"""
Tests for AnalyticsMetricsHandler._validate_org_access method.

Tests cover:
1. Admin users can access any org
2. Platform admin role grants cross-org access
3. Non-admin users can only access their own org
4. Default to user's org when no org_id requested
5. Access denied when requesting other org
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aragora.server.handlers._analytics_metrics_impl import AnalyticsMetricsHandler


@pytest.fixture
def handler():
    """Create an AnalyticsMetricsHandler instance."""
    return AnalyticsMetricsHandler(ctx={})


def _make_auth_context(org_id=None, roles=None):
    """Create a mock auth context with given org_id and roles."""
    ctx = MagicMock()
    ctx.org_id = org_id
    ctx.roles = roles or []
    return ctx


class TestValidateOrgAccess:
    """Test _validate_org_access method."""

    def test_admin_can_access_any_org(self, handler):
        """Admin role should allow access to any organization."""
        auth = _make_auth_context(org_id="org-1", roles=["admin"])
        org_id, err = handler._validate_org_access(auth, "org-2")
        assert org_id == "org-2"
        assert err is None

    def test_platform_admin_can_access_any_org(self, handler):
        """Platform admin role should allow access to any organization."""
        auth = _make_auth_context(org_id="org-1", roles=["platform_admin"])
        org_id, err = handler._validate_org_access(auth, "org-999")
        assert org_id == "org-999"
        assert err is None

    def test_no_requested_org_returns_users_org(self, handler):
        """When no org_id is requested, use the user's own org."""
        auth = _make_auth_context(org_id="org-abc", roles=["member"])
        org_id, err = handler._validate_org_access(auth, None)
        assert org_id == "org-abc"
        assert err is None

    def test_user_can_access_own_org(self, handler):
        """User should be able to access their own organization."""
        auth = _make_auth_context(org_id="org-abc", roles=["member"])
        org_id, err = handler._validate_org_access(auth, "org-abc")
        assert org_id == "org-abc"
        assert err is None

    def test_user_cannot_access_other_org(self, handler):
        """Non-admin user should be denied access to other organizations."""
        auth = _make_auth_context(org_id="org-abc", roles=["member"])
        org_id, err = handler._validate_org_access(auth, "org-different")
        assert org_id is None
        assert err is not None
        # Verify it returns an error response (HandlerResult)
        assert err.status_code == 403

    def test_no_user_org_no_requested_org(self, handler):
        """When user has no org and none is requested, return None org_id."""
        auth = _make_auth_context(org_id=None, roles=["member"])
        org_id, err = handler._validate_org_access(auth, None)
        assert org_id is None
        assert err is None

    def test_no_user_org_with_requested_org(self, handler):
        """When user has no org_id but requests one, the check passes
        because the comparison user_org_id != requested_org_id only
        triggers when user_org_id is truthy."""
        auth = _make_auth_context(org_id=None, roles=["member"])
        org_id, err = handler._validate_org_access(auth, "org-xyz")
        assert org_id == "org-xyz"
        assert err is None

    def test_none_roles_treated_as_empty(self, handler):
        """When roles is None, should treat as empty list (no admin access)."""
        auth = _make_auth_context(org_id="org-1", roles=None)
        org_id, err = handler._validate_org_access(auth, "org-2")
        assert err is not None
        assert err.status_code == 403

    def test_admin_with_none_requested_org(self, handler):
        """Admin with no requested org should get their user org."""
        auth = _make_auth_context(org_id="admin-org", roles=["admin"])
        org_id, err = handler._validate_org_access(auth, None)
        assert org_id is None  # admin requesting None returns None (allow all)
        assert err is None

    def test_multiple_roles_including_admin(self, handler):
        """Admin among multiple roles should still grant cross-org access."""
        auth = _make_auth_context(org_id="org-1", roles=["member", "admin", "viewer"])
        org_id, err = handler._validate_org_access(auth, "org-other")
        assert org_id == "org-other"
        assert err is None

    def test_auth_context_without_org_id_attr(self, handler):
        """Auth context missing org_id attribute should use None."""
        auth = MagicMock(spec=[])  # No attributes
        org_id, err = handler._validate_org_access(auth, None)
        assert org_id is None
        assert err is None

    def test_auth_context_without_roles_attr(self, handler):
        """Auth context missing roles attribute should use empty list."""
        auth = MagicMock(spec=["org_id"])
        auth.org_id = "org-1"
        org_id, err = handler._validate_org_access(auth, "org-1")
        assert org_id == "org-1"
        assert err is None
