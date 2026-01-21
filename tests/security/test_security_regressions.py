"""
Security Regression Tests.

Run these tests on every PR to prevent security regressions.
Covers the key security fixes implemented:
1. Header trust vulnerability - X-User-ID/X-User-Roles cannot be spoofed
2. JWT authentication required for sensitive operations
3. SecureHandler base class provides proper auth/audit/RBAC
4. Encryption service properly encrypts sensitive fields

Run with: pytest tests/security/test_security_regressions.py -v
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest


class TestHeaderTrustPrevention:
    """Verify that header-based identity spoofing is prevented."""

    def test_workflows_handler_ignores_spoofed_headers(self):
        """WorkflowHandler must not trust X-User-ID headers."""
        from aragora.server.handlers.workflows import WorkflowHandler, RBAC_AVAILABLE

        if not RBAC_AVAILABLE:
            pytest.skip("RBAC not available")

        handler = WorkflowHandler({})
        request = MagicMock()
        request.headers = {
            "X-User-ID": "attacker-spoofed-admin",
            "X-User-Roles": "admin,owner",
        }

        with patch("aragora.server.handlers.workflows.extract_user_from_request") as mock_extract:
            mock_jwt_context = MagicMock()
            mock_jwt_context.authenticated = False
            mock_jwt_context.user_id = None
            mock_extract.return_value = mock_jwt_context

            context = handler._get_auth_context(request)

            # MUST return unauthenticated sentinel, NOT use spoofed headers
            assert (
                context == "unauthenticated"
            ), "SECURITY REGRESSION: Handler is trusting X-User-ID headers!"

    def test_finding_workflow_ignores_spoofed_headers(self):
        """FindingWorkflowHandler must not trust X-User-ID headers."""
        from aragora.server.handlers.features.finding_workflow import FindingWorkflowHandler

        handler = FindingWorkflowHandler({})
        request = MagicMock()
        request.headers = {
            "X-User-ID": "attacker-spoofed-admin",
            "X-User-Roles": "admin,owner",
        }

        with patch(
            "aragora.server.handlers.features.finding_workflow.extract_user_from_request"
        ) as mock_extract:
            mock_jwt_context = MagicMock()
            mock_jwt_context.authenticated = False
            mock_jwt_context.user_id = None
            mock_extract.return_value = mock_jwt_context

            context = handler._get_auth_context(request)

            # MUST return None, NOT use spoofed headers
            assert context is None, "SECURITY REGRESSION: Handler is trusting X-User-ID headers!"

    def test_utils_auth_ignores_spoofed_headers(self):
        """_extract_user_from_headers must not fall back to X-User-ID."""
        from aragora.server.handlers.utils.auth import _extract_user_from_headers

        handler = MagicMock()
        handler.headers = {
            "X-User-ID": "attacker-spoofed-admin",
            "X-User-Name": "Attacker",
        }

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_jwt_context = MagicMock()
            mock_jwt_context.is_authenticated = False
            mock_extract.return_value = mock_jwt_context

            user_id, user_name = _extract_user_from_headers(handler)

            # MUST return anonymous, NOT use spoofed headers
            assert (
                user_id == "anonymous"
            ), "SECURITY REGRESSION: _extract_user_from_headers trusts headers!"
            assert "attacker" not in user_id.lower()


class TestSecureHandlerInheritance:
    """Verify critical handlers extend SecureHandler."""

    def test_billing_handler_extends_secure_handler(self):
        """BillingHandler must extend SecureHandler."""
        from aragora.server.handlers.admin.billing import BillingHandler
        from aragora.server.handlers.secure import SecureHandler

        assert issubclass(
            BillingHandler, SecureHandler
        ), "SECURITY REGRESSION: BillingHandler must extend SecureHandler"
        assert hasattr(BillingHandler, "RESOURCE_TYPE")

    def test_privacy_handler_extends_secure_handler(self):
        """PrivacyHandler must extend SecureHandler."""
        from aragora.server.handlers.privacy import PrivacyHandler
        from aragora.server.handlers.secure import SecureHandler

        assert issubclass(
            PrivacyHandler, SecureHandler
        ), "SECURITY REGRESSION: PrivacyHandler must extend SecureHandler"

    def test_auth_handler_extends_secure_handler(self):
        """AuthHandler must extend SecureHandler."""
        from aragora.server.handlers.auth.handler import AuthHandler
        from aragora.server.handlers.secure import SecureHandler

        assert issubclass(
            AuthHandler, SecureHandler
        ), "SECURITY REGRESSION: AuthHandler must extend SecureHandler"

    def test_organizations_handler_extends_secure_handler(self):
        """OrganizationsHandler must extend SecureHandler."""
        from aragora.server.handlers.organizations import OrganizationsHandler
        from aragora.server.handlers.secure import SecureHandler

        assert issubclass(
            OrganizationsHandler, SecureHandler
        ), "SECURITY REGRESSION: OrganizationsHandler must extend SecureHandler"

    def test_webhook_handler_extends_secure_handler(self):
        """WebhookHandler must extend SecureHandler."""
        from aragora.server.handlers.webhooks import WebhookHandler
        from aragora.server.handlers.secure import SecureHandler

        assert issubclass(
            WebhookHandler, SecureHandler
        ), "SECURITY REGRESSION: WebhookHandler must extend SecureHandler"

    def test_sso_handler_extends_secure_handler(self):
        """SSOHandler must extend SecureHandler."""
        from aragora.server.handlers.sso import SSOHandler
        from aragora.server.handlers.secure import SecureHandler

        assert issubclass(
            SSOHandler, SecureHandler
        ), "SECURITY REGRESSION: SSOHandler must extend SecureHandler"

    def test_connectors_handler_extends_secure_handler(self):
        """ConnectorsHandler must extend SecureHandler."""
        from aragora.server.handlers.features.connectors import ConnectorsHandler
        from aragora.server.handlers.secure import SecureHandler

        assert issubclass(
            ConnectorsHandler, SecureHandler
        ), "SECURITY REGRESSION: ConnectorsHandler must extend SecureHandler"

    def test_oauth_handler_extends_secure_handler(self):
        """OAuthHandler must extend SecureHandler."""
        from aragora.server.handlers.oauth import OAuthHandler
        from aragora.server.handlers.secure import SecureHandler

        assert issubclass(
            OAuthHandler, SecureHandler
        ), "SECURITY REGRESSION: OAuthHandler must extend SecureHandler"

    def test_workspace_handler_extends_secure_handler(self):
        """WorkspaceHandler must extend SecureHandler."""
        from aragora.server.handlers.workspace import WorkspaceHandler
        from aragora.server.handlers.secure import SecureHandler

        assert issubclass(
            WorkspaceHandler, SecureHandler
        ), "SECURITY REGRESSION: WorkspaceHandler must extend SecureHandler"


class TestEncryptionService:
    """Verify encryption service works correctly."""

    @pytest.fixture
    def encryption_key(self):
        """Provide a test encryption key."""
        import secrets

        key = secrets.token_hex(32)
        with patch.dict(os.environ, {"ARAGORA_ENCRYPTION_KEY": key}):
            # Reset singleton
            import aragora.security.encryption as enc_module

            enc_module._service = None
            yield key
            enc_module._service = None

    def test_encryption_round_trip(self, encryption_key):
        """Test that encryption/decryption works correctly."""
        from aragora.security.encryption import get_encryption_service

        service = get_encryption_service()

        data = {
            "api_key": "sk-secret-key-12345",
            "name": "Test",
        }

        encrypted = service.encrypt_fields(data.copy(), ["api_key"])

        # Verify encryption happened
        assert (
            encrypted["api_key"] != data["api_key"]
        ), "SECURITY REGRESSION: Field was not encrypted"
        assert encrypted["api_key"].get("_encrypted") is True

        # Verify decryption works
        decrypted = service.decrypt_fields(encrypted, ["api_key"])
        assert decrypted["api_key"] == data["api_key"], "SECURITY REGRESSION: Decryption failed"

    def test_encryption_with_aad(self, encryption_key):
        """Test that AAD (Associated Authenticated Data) prevents tampering."""
        from aragora.security.encryption import get_encryption_service

        service = get_encryption_service()

        data = {"api_key": "secret123"}

        # Encrypt with record ID
        encrypted = service.encrypt_fields(data.copy(), ["api_key"], associated_data="record_1")

        # Must fail with wrong record ID
        with pytest.raises(Exception):
            service.decrypt_fields(encrypted, ["api_key"], associated_data="record_2")


class TestRBACEnforcement:
    """Verify RBAC is properly enforced."""

    def test_rbac_module_available(self):
        """RBAC module must be available."""
        try:
            from aragora.rbac import AuthorizationContext, check_permission

            assert AuthorizationContext is not None
            assert check_permission is not None
        except ImportError:
            pytest.skip("RBAC module not installed")

    def test_admin_permissions(self):
        """Admin role must have elevated permissions."""
        try:
            from aragora.rbac import AuthorizationContext, check_permission
        except ImportError:
            pytest.skip("RBAC not available")

        ctx = AuthorizationContext(
            user_id="admin_user",
            roles={"admin"},
            org_id="org_123",
        )

        # Admin should have update permission
        decision = check_permission(ctx, "organization.update")
        assert decision.allowed, "Admin should have organization.update"

    def test_viewer_restrictions(self):
        """Viewer role must be restricted from write operations."""
        try:
            from aragora.rbac import AuthorizationContext, check_permission
        except ImportError:
            pytest.skip("RBAC not available")

        ctx = AuthorizationContext(
            user_id="viewer_user",
            roles={"viewer"},
            org_id="org_123",
        )

        # Viewer should NOT have update permission
        decision = check_permission(ctx, "organization.update")
        assert not decision.allowed, "SECURITY REGRESSION: Viewer should NOT have update permission"


class TestApprovalPersistence:
    """Verify approval persistence is implemented."""

    def test_governance_store_available(self):
        """GovernanceStore must be available for approval persistence."""
        try:
            from aragora.storage.governance_store import get_governance_store

            store = get_governance_store()
            assert store is not None, "SECURITY REGRESSION: GovernanceStore not available"
        except ImportError:
            pytest.skip("GovernanceStore not available")

    def test_approval_recovery_function_exists(self):
        """recover_pending_approvals must exist."""
        from aragora.workflow.nodes.human_checkpoint import recover_pending_approvals

        assert callable(
            recover_pending_approvals
        ), "SECURITY REGRESSION: recover_pending_approvals not callable"


# Quick sanity check that can run fast
class TestQuickSecuritySanity:
    """Quick sanity checks for CI/CD."""

    def test_secure_handler_exists(self):
        """SecureHandler must exist."""
        from aragora.server.handlers.secure import SecureHandler

        assert SecureHandler is not None

    def test_encryption_module_exists(self):
        """Encryption module must exist."""
        from aragora.security.encryption import get_encryption_service

        assert get_encryption_service is not None

    def test_jwt_auth_module_exists(self):
        """JWT auth module must exist."""
        from aragora.billing.jwt_auth import extract_user_from_request

        assert extract_user_from_request is not None
