"""Security Tests for Multi-Tenancy - Cross-Tenant Access Prevention.

Tests to verify proper tenant isolation measures are in place.
"""

import json
import pytest
from datetime import datetime, timezone


class TestWorkspaceSecurityMeasures:
    """Tests for workspace handler security measures."""

    def test_workspace_handler_has_org_id_validation_in_create(self):
        """Test that create workspace has org_id validation logic."""
        import inspect
        from aragora.server.handlers.workspace import WorkspaceHandler

        # Get the source code of the method
        source = inspect.getsource(WorkspaceHandler._handle_create_workspace)

        # Verify security measures are present
        assert "auth_ctx.org_id" in source, "Should use authenticated org_id"
        assert (
            "Cannot create workspace in another organization" in source
            or "cross-tenant" in source.lower()
        ), "Should reject cross-tenant requests"

    def test_workspace_handler_has_org_id_validation_in_list(self):
        """Test that list workspaces has org_id validation logic."""
        import inspect
        from aragora.server.handlers.workspace import WorkspaceHandler

        source = inspect.getsource(WorkspaceHandler._handle_list_workspaces)

        # Verify security measures are present
        assert "auth_ctx.org_id" in source, "Should use authenticated org_id"
        assert (
            "Cannot list workspaces from another organization" in source
            or "cross-tenant" in source.lower()
        ), "Should reject cross-tenant requests"


class TestAnalyticsSecurityMeasures:
    """Tests for analytics handler security measures."""

    def test_analytics_handler_has_verify_org_access_method(self):
        """Test that analytics handler has org access verification."""
        from aragora.server.handlers.analytics_dashboard import AnalyticsDashboardHandler

        handler = AnalyticsDashboardHandler({})
        assert hasattr(handler, "_verify_org_access"), "Should have _verify_org_access method"
        assert callable(handler._verify_org_access), "Should be callable"

    def test_analytics_handler_has_verify_workspace_access_method(self):
        """Test that analytics handler has workspace access verification."""
        from aragora.server.handlers.analytics_dashboard import AnalyticsDashboardHandler

        handler = AnalyticsDashboardHandler({})
        assert hasattr(
            handler, "_verify_workspace_access_sync"
        ), "Should have workspace verification"
        assert callable(handler._verify_workspace_access_sync), "Should be callable"

    def test_verify_org_access_rejects_cross_tenant(self):
        """Test that _verify_org_access rejects cross-tenant requests."""
        from aragora.server.handlers.analytics_dashboard import AnalyticsDashboardHandler
        from unittest.mock import MagicMock

        handler = AnalyticsDashboardHandler({})

        # Create mock user from org_A
        mock_user = MagicMock()
        mock_user.org_id = "org_A"

        # Attempt to access org_B data - should return error
        result = handler._verify_org_access("org_B", mock_user)
        assert result is not None, "Should return error for cross-tenant access"
        assert result.status_code == 403, "Should return 403 Forbidden"

    def test_verify_org_access_allows_own_org(self):
        """Test that _verify_org_access allows access to user's own org."""
        from aragora.server.handlers.analytics_dashboard import AnalyticsDashboardHandler
        from unittest.mock import MagicMock

        handler = AnalyticsDashboardHandler({})

        mock_user = MagicMock()
        mock_user.org_id = "org_A"

        # Access own org data - should return None (no error)
        result = handler._verify_org_access("org_A", mock_user)
        assert result is None, "Should allow access to own org"


class TestImpersonationStoreSecurity:
    """Tests for impersonation store org_id scoping."""

    def test_session_record_has_org_id_field(self):
        """Test that SessionRecord has org_id field."""
        from aragora.storage.impersonation_store import SessionRecord

        session = SessionRecord(
            session_id="session_123",
            admin_user_id="admin_123",
            admin_email="admin@example.com",
            target_user_id="target_123",
            target_email="target@example.com",
            org_id="org_A",
            reason="Support request",
            started_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc),
            ip_address="127.0.0.1",
            user_agent="Test",
        )

        assert hasattr(session, "org_id"), "SessionRecord should have org_id field"
        assert session.org_id == "org_A"

    def test_session_record_to_dict_includes_org_id(self):
        """Test that SessionRecord.to_dict includes org_id."""
        from aragora.storage.impersonation_store import SessionRecord

        session = SessionRecord(
            session_id="session_123",
            admin_user_id="admin_123",
            admin_email="admin@example.com",
            target_user_id="target_123",
            target_email="target@example.com",
            org_id="org_A",
            reason="Support request",
            started_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc),
            ip_address="127.0.0.1",
            user_agent="Test",
        )

        session_dict = session.to_dict()
        assert "org_id" in session_dict, "to_dict should include org_id"
        assert session_dict["org_id"] == "org_A"

    def test_audit_record_has_org_id_field(self):
        """Test that AuditRecord has org_id field."""
        from aragora.storage.impersonation_store import AuditRecord

        record = AuditRecord(
            audit_id="audit_123",
            timestamp=datetime.now(timezone.utc),
            event_type="start",
            session_id="session_123",
            admin_user_id="admin_123",
            target_user_id="target_123",
            org_id="org_A",
            reason="Support request",
            action_details_json=None,
            ip_address="127.0.0.1",
            user_agent="Test",
            success=True,
        )

        assert hasattr(record, "org_id"), "AuditRecord should have org_id field"
        assert record.org_id == "org_A"

    def test_audit_record_to_dict_includes_org_id(self):
        """Test that AuditRecord.to_dict includes org_id."""
        from aragora.storage.impersonation_store import AuditRecord

        record = AuditRecord(
            audit_id="audit_123",
            timestamp=datetime.now(timezone.utc),
            event_type="start",
            session_id="session_123",
            admin_user_id="admin_123",
            target_user_id="target_123",
            org_id="org_A",
            reason="Support request",
            action_details_json=None,
            ip_address="127.0.0.1",
            user_agent="Test",
            success=True,
        )

        record_dict = record.to_dict()
        assert "org_id" in record_dict, "to_dict should include org_id"
        assert record_dict["org_id"] == "org_A"

    def test_impersonation_store_save_session_accepts_org_id(self):
        """Test that save_session method accepts org_id parameter."""
        import inspect
        from aragora.storage.impersonation_store import ImpersonationStore

        sig = inspect.signature(ImpersonationStore.save_session)
        params = list(sig.parameters.keys())
        assert "org_id" in params, "save_session should accept org_id parameter"

    def test_impersonation_store_save_audit_accepts_org_id(self):
        """Test that save_audit_entry method accepts org_id parameter."""
        import inspect
        from aragora.storage.impersonation_store import ImpersonationStore

        sig = inspect.signature(ImpersonationStore.save_audit_entry)
        params = list(sig.parameters.keys())
        assert "org_id" in params, "save_audit_entry should accept org_id parameter"


class TestEmailServicesSecurity:
    """Tests for email services tenant scoping."""

    def test_snoozed_emails_dict_supports_org_scoping(self):
        """Test that snoozed emails dict structure supports org_id scoping."""
        import aragora.server.handlers.email_services as email_services

        # Reset global state
        with email_services._snoozed_emails_lock:
            email_services._snoozed_emails.clear()

        # Verify the dict can store org-scoped data
        assert isinstance(email_services._snoozed_emails, dict)

        # Simulate org-scoped storage
        with email_services._snoozed_emails_lock:
            email_services._snoozed_emails["org_A"] = {
                "email_1": {"email_id": "email_1", "org_id": "org_A"}
            }
            email_services._snoozed_emails["org_B"] = {
                "email_2": {"email_id": "email_2", "org_id": "org_B"}
            }

        # Verify isolation
        org_a_emails = email_services._snoozed_emails.get("org_A", {})
        org_b_emails = email_services._snoozed_emails.get("org_B", {})

        assert "email_1" in org_a_emails
        assert "email_2" not in org_a_emails
        assert "email_2" in org_b_emails
        assert "email_1" not in org_b_emails

        # Cleanup
        with email_services._snoozed_emails_lock:
            email_services._snoozed_emails.clear()

    def test_email_handler_methods_accept_org_id(self):
        """Test that email handler methods accept org_id parameter."""
        import inspect
        import aragora.server.handlers.email_services as email_services

        # Check handle_apply_snooze
        sig = inspect.signature(email_services.handle_apply_snooze)
        params = list(sig.parameters.keys())
        assert "org_id" in params, "handle_apply_snooze should accept org_id"

        # Check handle_cancel_snooze
        sig = inspect.signature(email_services.handle_cancel_snooze)
        params = list(sig.parameters.keys())
        assert "org_id" in params, "handle_cancel_snooze should accept org_id"

        # Check handle_get_snoozed_emails
        sig = inspect.signature(email_services.handle_get_snoozed_emails)
        params = list(sig.parameters.keys())
        assert "org_id" in params, "handle_get_snoozed_emails should accept org_id"


class TestCreditsAdminSecurity:
    """Tests for credits admin handler permission decorators."""

    def test_credits_handler_methods_have_permission_decorators(self):
        """Test that credits handler methods have permission decorators."""
        import inspect
        from aragora.server.handlers.admin.credits import CreditsAdminHandler

        # Check each method has @require_permission in its definition
        methods_to_check = [
            "issue_credit",
            "get_credit_account",
            "list_transactions",
            "adjust_balance",
            "get_expiring_credits",
        ]

        for method_name in methods_to_check:
            method = getattr(CreditsAdminHandler, method_name, None)
            assert method is not None, f"Method {method_name} should exist"

            source = inspect.getsource(method)
            assert (
                "require_permission" in source
            ), f"Method {method_name} should have @require_permission decorator"

    def test_credits_handler_extends_secure_handler(self):
        """Test that CreditsAdminHandler extends SecureHandler."""
        from aragora.server.handlers.admin.credits import CreditsAdminHandler
        from aragora.server.handlers.secure import SecureHandler

        assert issubclass(
            CreditsAdminHandler, SecureHandler
        ), "CreditsAdminHandler should extend SecureHandler"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
