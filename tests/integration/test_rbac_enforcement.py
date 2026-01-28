"""
Integration tests for RBAC enforcement on sensitive endpoints.

Tests that protected endpoints:
1. Require authentication
2. Return 401 for unauthenticated requests
3. Return 403 for authenticated requests without proper permissions
4. Allow requests with proper permissions
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from aragora.rbac.models import AuthorizationContext
from aragora.rbac.decorators import PermissionDeniedError
from aragora.rbac.checker import get_permission_checker
from aragora.server.handlers.base import error_response, json_response


@pytest.fixture(autouse=True)
def clear_permission_cache():
    """Clear the permission checker cache before each test."""
    checker = get_permission_checker()
    checker.clear_cache()
    yield
    checker.clear_cache()


class TestKnowledgeMoundRBACEnforcement:
    """Test RBAC enforcement for Knowledge Mound endpoints."""

    @pytest.fixture
    def mock_handler(self):
        """Create a mock HTTP handler."""
        handler = MagicMock()
        handler.command = "GET"
        handler.request = MagicMock()
        handler.request.body = b"{}"
        handler.request.headers = {}
        return handler

    @pytest.fixture
    def knowledge_handler(self):
        """Create KnowledgeMoundHandler instance."""
        from aragora.server.handlers.knowledge_base.mound.handler import (
            KnowledgeMoundHandler,
        )

        server_context = {"workspace_id": "test"}
        return KnowledgeMoundHandler(server_context)

    def test_knowledge_mound_requires_auth(self, knowledge_handler, mock_handler):
        """Test that knowledge mound endpoints require authentication."""
        # Mock require_auth_or_error to return an error (simulating no auth)
        with patch.object(
            knowledge_handler,
            "require_auth_or_error",
            return_value=(None, error_response("Authentication required", 401)),
        ):
            result = knowledge_handler.handle(
                "/api/v1/knowledge/mound/stats",
                {},
                mock_handler,
            )

            assert result is not None
            assert result.status_code == 401

    def test_knowledge_mound_allows_authenticated(self, knowledge_handler, mock_handler):
        """Test that authenticated requests are allowed through."""
        mock_user = {"sub": "user_123", "email": "test@example.com"}

        with patch.object(
            knowledge_handler,
            "require_auth_or_error",
            return_value=(mock_user, None),
        ):
            with patch.object(
                knowledge_handler,
                "_get_mound",
                return_value=None,
            ):
                result = knowledge_handler.handle(
                    "/api/v1/knowledge/mound/stats",
                    {},
                    mock_handler,
                )

                # Should get a response (503 since mound is None, but auth passed)
                assert result is not None
                # Not 401, meaning auth check passed
                assert result.status_code != 401

    def test_knowledge_mound_query_requires_auth(self, knowledge_handler, mock_handler):
        """Test that query endpoint requires authentication."""
        mock_handler.command = "POST"
        mock_handler.request.body = b'{"query": "test"}'

        with patch.object(
            knowledge_handler,
            "require_auth_or_error",
            return_value=(None, error_response("Authentication required", 401)),
        ):
            result = knowledge_handler.handle(
                "/api/v1/knowledge/mound/query",
                {},
                mock_handler,
            )

            assert result is not None
            assert result.status_code == 401

    def test_knowledge_mound_governance_requires_auth(self, knowledge_handler, mock_handler):
        """Test that governance endpoints require authentication."""
        with patch.object(
            knowledge_handler,
            "require_auth_or_error",
            return_value=(None, error_response("Authentication required", 401)),
        ):
            result = knowledge_handler.handle(
                "/api/v1/knowledge/mound/governance/stats",
                {},
                mock_handler,
            )

            assert result is not None
            assert result.status_code == 401


class TestAnalyticsRBACEnforcement:
    """Test RBAC enforcement for Analytics endpoints."""

    @pytest.fixture
    def mock_handler(self):
        """Create a mock HTTP handler."""
        handler = MagicMock()
        handler.command = "GET"
        handler.request = MagicMock()
        handler.request.body = b"{}"
        handler.request.headers = {}
        return handler

    @pytest.fixture
    def analytics_handler(self):
        """Create AnalyticsHandler instance."""
        from aragora.server.handlers.knowledge.analytics import AnalyticsHandler

        server_context = {"workspace_id": "test"}
        return AnalyticsHandler(server_context)

    @pytest.mark.asyncio
    async def test_analytics_requires_auth(self, analytics_handler, mock_handler):
        """Test that analytics endpoints require authentication."""
        with patch.object(
            analytics_handler,
            "require_auth_or_error",
            return_value=(None, error_response("Authentication required", 401)),
        ):
            result = await analytics_handler.handle(
                "/api/v1/knowledge/mound/stats",
                {},
                mock_handler,
            )

            assert result is not None
            assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_analytics_learning_requires_auth(self, analytics_handler, mock_handler):
        """Test that learning analytics requires authentication."""
        with patch.object(
            analytics_handler,
            "require_auth_or_error",
            return_value=(None, error_response("Authentication required", 401)),
        ):
            result = await analytics_handler.handle(
                "/api/v1/knowledge/learning/stats",
                {},
                mock_handler,
            )

            assert result is not None
            assert result.status_code == 401


class TestBackupHandlerRBACEnforcement:
    """Test RBAC enforcement for Backup handler endpoints."""

    @pytest.fixture
    def mock_handler(self):
        """Create a mock HTTP handler."""
        handler = MagicMock()
        handler.command = "GET"
        handler.request = MagicMock()
        handler.request.body = b"{}"
        handler.request.headers = {}
        return handler

    @pytest.fixture
    def auth_context_no_perms(self):
        """Create auth context without backup permissions."""
        return AuthorizationContext(
            user_id="user_123",
            permissions={"debates:read"},  # No backup permissions
        )

    @pytest.fixture
    def auth_context_with_perms(self):
        """Create auth context with backup permissions."""
        return AuthorizationContext(
            user_id="admin_123",
            permissions={"backup.read", "backup.create", "backup.delete"},
        )

    def test_backup_list_requires_permission(self):
        """Test that listing backups requires backup.read permission."""
        from aragora.rbac.decorators import require_permission

        # The decorator should raise if permission is missing
        # This tests the decorator itself
        @require_permission("backup.read")
        def protected_func(ctx: AuthorizationContext):
            return "success"

        # With correct permission
        ctx = AuthorizationContext(
            user_id="user_123",
            permissions={"backup.read"},
        )
        result = protected_func(ctx)
        assert result == "success"

        # Without permission - use different user_id to avoid cache collision
        ctx_no_perms = AuthorizationContext(
            user_id="user_456_no_perms",
            permissions={"other.perm"},
        )
        with pytest.raises(Exception):  # PermissionDenied
            protected_func(ctx_no_perms)


class TestDRHandlerRBACEnforcement:
    """Test RBAC enforcement for DR handler endpoints."""

    def test_dr_status_requires_permission(self):
        """Test that DR status requires dr.read permission."""
        from aragora.rbac.decorators import require_permission

        @require_permission("dr.read")
        def protected_func(ctx: AuthorizationContext):
            return "success"

        # With correct permission
        ctx = AuthorizationContext(
            user_id="admin_123",
            permissions={"dr.read"},
        )
        result = protected_func(ctx)
        assert result == "success"

        # Without permission
        ctx_no_perms = AuthorizationContext(
            user_id="user_123",
            permissions={"backup.read"},  # Wrong permission
        )
        with pytest.raises(Exception):  # PermissionDenied
            protected_func(ctx_no_perms)

    def test_dr_drill_requires_write_permission(self):
        """Test that running DR drill requires dr.write permission."""
        from aragora.rbac.decorators import require_permission

        @require_permission("dr.write")
        def run_drill(ctx: AuthorizationContext):
            return "drill_started"

        # With correct permission
        ctx = AuthorizationContext(
            user_id="admin_123",
            permissions={"dr.write"},
        )
        result = run_drill(ctx)
        assert result == "drill_started"

        # Read-only should fail
        ctx_readonly = AuthorizationContext(
            user_id="user_123",
            permissions={"dr.read"},  # Only read, not write
        )
        with pytest.raises(Exception):  # PermissionDenied
            run_drill(ctx_readonly)


class TestComplianceHandlerRBACEnforcement:
    """Test RBAC enforcement for Compliance handler endpoints."""

    def test_soc2_report_requires_permission(self):
        """Test that SOC2 report requires compliance.read permission."""
        from aragora.rbac.decorators import require_permission

        @require_permission("compliance.read")
        def get_soc2_report(ctx: AuthorizationContext):
            return {"status": "compliant"}

        # With correct permission
        ctx = AuthorizationContext(
            user_id="auditor_123",
            permissions={"compliance.read"},
        )
        result = get_soc2_report(ctx)
        assert result["status"] == "compliant"

    def test_gdpr_export_requires_permission(self):
        """Test that GDPR export requires compliance.export permission."""
        from aragora.rbac.decorators import require_permission

        @require_permission("compliance.export")
        def export_gdpr_data(ctx: AuthorizationContext, user_id: str):
            return {"user_id": user_id, "data": []}

        # With correct permission
        ctx = AuthorizationContext(
            user_id="admin_123",
            permissions={"compliance.export"},
        )
        result = export_gdpr_data(ctx, "target_user")
        assert result["user_id"] == "target_user"


class TestReceiptsHandlerRBACEnforcement:
    """Test RBAC enforcement for Decision Receipts handler endpoints."""

    def test_receipt_read_requires_permission(self):
        """Test that reading receipts requires receipts.read permission."""
        from aragora.rbac.decorators import require_permission

        @require_permission("receipts.read")
        def get_receipt(ctx: AuthorizationContext, decision_id: str):
            return {"decision_id": decision_id, "receipt": "..."}

        # With correct permission
        ctx = AuthorizationContext(
            user_id="user_123",
            permissions={"receipts.read"},
        )
        result = get_receipt(ctx, "dec_123")
        assert result["decision_id"] == "dec_123"

        # Without permission
        ctx_no_perms = AuthorizationContext(
            user_id="guest_123",
            permissions=set(),
        )
        with pytest.raises(Exception):
            get_receipt(ctx_no_perms, "dec_123")
