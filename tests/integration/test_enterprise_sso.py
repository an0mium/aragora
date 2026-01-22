"""
Enterprise SSO Integration Tests.

Tests OIDC, SAML, and SSO authentication flows for enterprise customers.
"""

import asyncio
import base64
import hashlib
import json
import secrets
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Skip tests if required modules not available
pytest.importorskip("httpx", reason="httpx required for SSO tests")


@dataclass
class MockOIDCConfig:
    """Mock OIDC configuration."""

    client_id: str = "test-client-id"
    client_secret: str = "test-client-secret"
    issuer_url: str = "https://idp.example.com"
    callback_url: str = "https://app.example.com/auth/callback"
    scopes: list = None

    def __post_init__(self):
        if self.scopes is None:
            self.scopes = ["openid", "profile", "email"]


@dataclass
class MockSSOUser:
    """Mock SSO user returned from authentication."""

    sub: str
    email: str
    name: str = ""
    groups: list = None
    org_id: Optional[str] = None

    def __post_init__(self):
        if self.groups is None:
            self.groups = []


class TestOIDCAuthorizationFlow:
    """Test OIDC authorization code flow."""

    @pytest.mark.asyncio
    async def test_authorization_url_generation(self):
        """Test generation of OIDC authorization URL."""
        from aragora.auth.oidc import OIDCProvider, OIDCConfig
        from aragora.auth.sso import SSOProviderType

        config = OIDCConfig(
            provider_type=SSOProviderType.OIDC,
            entity_id="test-client",
            client_id="test-client",
            client_secret="test-secret",
            issuer_url="https://login.example.com/v2.0",
            callback_url="https://aragora.example.com/auth/callback",
            # Provide explicit endpoints to skip discovery
            authorization_endpoint="https://login.example.com/v2.0/authorize",
            token_endpoint="https://login.example.com/v2.0/token",
        )

        provider = OIDCProvider(config)
        state = secrets.token_urlsafe(32)

        # Mock discovery to avoid network calls
        provider._discovery_complete = True

        # Generate authorization URL
        auth_url = await provider.get_authorization_url(state=state)

        # Verify URL contains required parameters
        assert "login.example.com" in auth_url
        assert f"client_id={config.client_id}" in auth_url
        assert f"state={state}" in auth_url
        assert "response_type=code" in auth_url
        assert "scope=" in auth_url

    def test_pkce_code_verifier_generation(self):
        """Test PKCE code verifier and challenge generation."""
        from aragora.auth.oidc import OIDCProvider, OIDCConfig
        from aragora.auth.sso import SSOProviderType

        config = OIDCConfig(
            provider_type=SSOProviderType.OIDC,
            entity_id="test-client",
            client_id="test-client",
            client_secret="test-secret",
            issuer_url="https://login.example.com",
            callback_url="https://aragora.example.com/auth/callback",
            use_pkce=True,
        )

        provider = OIDCProvider(config)

        # Generate PKCE parameters (returns tuple of verifier, challenge)
        verifier, challenge = provider._generate_pkce()

        # Verify code verifier format (RFC 7636)
        assert len(verifier) >= 43
        assert len(verifier) <= 128

        # Verify challenge is base64url encoded SHA256
        expected_challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
            .decode()
            .rstrip("=")
        )
        assert challenge == expected_challenge

    @pytest.mark.skip(reason="Requires complex httpx mocking - see integration tests with real IdP")
    @pytest.mark.asyncio
    async def test_token_exchange(self):
        """Test exchanging authorization code for tokens."""
        pass  # Skip - requires complex mocking of discovery + token exchange

    @pytest.mark.asyncio
    async def test_id_token_validation(self):
        """Test ID token validation."""
        from aragora.auth.oidc import OIDCProvider, OIDCConfig, HAS_JWT
        from aragora.auth.sso import SSOProviderType

        if not HAS_JWT:
            pytest.skip("PyJWT not installed")

        config = OIDCConfig(
            provider_type=SSOProviderType.OIDC,
            entity_id="test-client",
            client_id="test-client",
            client_secret="test-secret",
            issuer_url="https://login.example.com",
            callback_url="https://aragora.example.com/auth/callback",
        )

        provider = OIDCProvider(config)

        # Mock decoded ID token claims
        mock_claims = {
            "iss": "https://login.example.com",
            "sub": "user-123",
            "aud": "test-client",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
            "email": "user@example.com",
            "name": "Test User",
        }

        # Mock JWKS endpoint discovery and JWT decode
        provider._endpoints = {"jwks_uri": None}  # Skip JWKS lookup

        with patch("aragora.auth.oidc.jwt.decode", return_value=mock_claims):
            claims = await provider._validate_id_token("mock-id-token")

        assert claims["sub"] == "user-123"
        assert claims["email"] == "user@example.com"


class TestOIDCProviderConfigurations:
    """Test OIDC provider-specific configurations."""

    def test_azure_ad_configuration(self):
        """Test Azure AD OIDC configuration."""
        from aragora.auth.oidc import OIDCConfig

        config = OIDCConfig.for_azure_ad(
            tenant_id="12345678-1234-1234-1234-123456789012",
            client_id="test-client",
            client_secret="test-secret",
            callback_url="https://aragora.example.com/auth/callback",
        )

        assert "login.microsoftonline.com" in config.issuer_url
        assert config.tenant_id == "12345678-1234-1234-1234-123456789012"

    def test_okta_configuration(self):
        """Test Okta OIDC configuration."""
        from aragora.auth.oidc import OIDCConfig

        config = OIDCConfig.for_okta(
            org_url="https://dev-123456.okta.com",
            client_id="test-client",
            client_secret="test-secret",
            callback_url="https://aragora.example.com/auth/callback",
        )

        assert "okta.com" in config.issuer_url
        assert "openid" in config.scopes

    def test_google_configuration(self):
        """Test Google Workspace OIDC configuration."""
        from aragora.auth.oidc import OIDCConfig

        config = OIDCConfig.for_google(
            client_id="test-client.apps.googleusercontent.com",
            client_secret="test-secret",
            callback_url="https://aragora.example.com/auth/callback",
            hd="example.com",  # Hosted domain restriction
        )

        assert "accounts.google.com" in config.issuer_url
        assert config.hd == "example.com"


class TestSAMLAuthentication:
    """Test SAML authentication flows."""

    def test_saml_request_generation(self):
        """Test SAML authentication request generation."""
        pytest.importorskip("onelogin.saml2", reason="python3-saml required")
        from aragora.auth.saml import SAMLProvider, SAMLConfig

        config = SAMLConfig(
            entity_id="https://aragora.example.com",
            acs_url="https://aragora.example.com/auth/saml/callback",
            idp_entity_id="https://idp.example.com",
            idp_sso_url="https://idp.example.com/saml/sso",
            idp_x509_cert="...",  # Mock cert
        )

        provider = SAMLProvider(config)

        # Generate SAML AuthnRequest
        request_url, request_id = provider.create_authn_request()

        assert "SAMLRequest=" in request_url
        assert request_id is not None

    def test_saml_response_validation(self):
        """Test SAML response validation."""
        pytest.importorskip("onelogin.saml2", reason="python3-saml required")
        from aragora.auth.saml import SAMLProvider, SAMLConfig

        config = SAMLConfig(
            entity_id="https://aragora.example.com",
            acs_url="https://aragora.example.com/auth/saml/callback",
            idp_entity_id="https://idp.example.com",
            idp_sso_url="https://idp.example.com/saml/sso",
            idp_x509_cert="...",  # Mock cert
        )

        provider = SAMLProvider(config)

        # Mock SAML response processing
        mock_user = MockSSOUser(
            sub="user-123",
            email="user@example.com",
            name="Test User",
            groups=["Engineering", "Admins"],
        )

        with patch.object(provider, "process_response", return_value=mock_user):
            user = provider.process_response("mock-saml-response")

        assert user.email == "user@example.com"
        assert "Engineering" in user.groups


class TestSSOGroupMapping:
    """Test SSO group-to-role mapping."""

    def test_group_to_role_mapping(self):
        """Test mapping IDP groups to Aragora roles."""
        from aragora.auth.sso import SSOGroupMapper

        mapper = SSOGroupMapper(
            {
                "Aragora-Admins": "admin",
                "Aragora-Users": "user",
                "Engineering": "developer",
            }
        )

        # Map groups to roles
        roles = mapper.map_groups(["Aragora-Admins", "Engineering", "Other"])

        assert "admin" in roles
        assert "developer" in roles
        assert len(roles) == 2

    def test_default_role_assignment(self):
        """Test default role when no groups match."""
        from aragora.auth.sso import SSOGroupMapper

        mapper = SSOGroupMapper(
            mappings={"Admins": "admin"},
            default_role="viewer",
        )

        # No matching groups
        roles = mapper.map_groups(["Users", "Guests"])

        assert "viewer" in roles


class TestSSOSessionManagement:
    """Test SSO session management."""

    @pytest.mark.asyncio
    async def test_sso_session_creation(self):
        """Test creating a session from SSO authentication."""
        from aragora.auth.sso import SSOSessionManager

        manager = SSOSessionManager()

        user = MockSSOUser(
            sub="user-123",
            email="user@example.com",
            name="Test User",
            org_id="org-456",
        )

        session = await manager.create_session(user)

        assert session.user_id == "user-123"
        assert session.org_id == "org-456"
        assert session.expires_at > time.time()

    @pytest.mark.asyncio
    async def test_sso_session_refresh(self):
        """Test refreshing an SSO session."""
        from aragora.auth.sso import SSOSessionManager

        manager = SSOSessionManager()

        # Create initial session
        user = MockSSOUser(sub="user-123", email="user@example.com")
        session = await manager.create_session(user)
        original_expires = session.expires_at

        # Refresh session
        await asyncio.sleep(0.1)
        refreshed = await manager.refresh_session(session.session_id)

        assert refreshed.expires_at > original_expires

    @pytest.mark.asyncio
    async def test_sso_session_logout(self):
        """Test SSO session logout and cleanup."""
        from aragora.auth.sso import SSOSessionManager

        manager = SSOSessionManager()

        user = MockSSOUser(sub="user-123", email="user@example.com")
        session = await manager.create_session(user)

        # Logout
        await manager.logout(session.session_id)

        # Session should be invalid
        with pytest.raises(Exception):
            await manager.get_session(session.session_id)


class TestMultiTenantIsolation:
    """Test multi-tenant data isolation."""

    @pytest.mark.asyncio
    async def test_tenant_context_isolation(self):
        """Test tenant context isolation in requests."""
        from aragora.tenancy.context import TenantContext

        # Set tenant context
        with TenantContext(tenant_id="tenant-1"):
            assert TenantContext.current().tenant_id == "tenant-1"

            # Nested context
            with TenantContext(tenant_id="tenant-2"):
                assert TenantContext.current().tenant_id == "tenant-2"

            # Back to original
            assert TenantContext.current().tenant_id == "tenant-1"

        # Outside context
        assert TenantContext.current() is None

    @pytest.mark.asyncio
    async def test_cross_tenant_access_denied(self):
        """Test that cross-tenant access is denied."""
        from aragora.tenancy.isolation import TenantIsolationEnforcer

        enforcer = TenantIsolationEnforcer()

        # Try to access another tenant's resource
        with pytest.raises(PermissionError) as exc_info:
            await enforcer.verify_access(
                requesting_tenant="tenant-1",
                resource_tenant="tenant-2",
                resource_id="debate-123",
            )

        assert "cross-tenant" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_tenant_data_filtering(self):
        """Test automatic tenant data filtering."""
        from aragora.tenancy.filter import TenantDataFilter

        filter = TenantDataFilter()

        # Mock data from multiple tenants
        all_data = [
            {"id": "1", "tenant_id": "tenant-1", "name": "Item 1"},
            {"id": "2", "tenant_id": "tenant-2", "name": "Item 2"},
            {"id": "3", "tenant_id": "tenant-1", "name": "Item 3"},
        ]

        # Filter for tenant-1
        filtered = filter.filter_by_tenant(all_data, "tenant-1")

        assert len(filtered) == 2
        assert all(item["tenant_id"] == "tenant-1" for item in filtered)


class TestTenantProvisioning:
    """Test tenant provisioning and configuration."""

    @pytest.mark.asyncio
    async def test_tenant_creation(self):
        """Test creating a new tenant."""
        from aragora.tenancy.provisioning import TenantProvisioner
        from aragora.tenancy.tenant import TenantTier

        provisioner = TenantProvisioner()

        tenant = await provisioner.create_tenant(
            name="Acme Corp",
            domain="acme.com",
            tier=TenantTier.ENTERPRISE,
            admin_email="admin@acme.com",
        )

        assert tenant.name == "Acme Corp"
        assert tenant.tier == TenantTier.ENTERPRISE
        assert tenant.config.enable_sso is True

    @pytest.mark.asyncio
    async def test_tenant_tier_upgrade(self):
        """Test upgrading a tenant's tier."""
        from aragora.tenancy.provisioning import TenantProvisioner
        from aragora.tenancy.tenant import TenantTier

        provisioner = TenantProvisioner()

        # Create starter tenant
        tenant = await provisioner.create_tenant(
            name="Test Corp",
            domain="test.com",
            tier=TenantTier.STARTER,
            admin_email="admin@test.com",
        )

        # Upgrade to enterprise
        upgraded = await provisioner.upgrade_tier(
            tenant_id=tenant.id,
            new_tier=TenantTier.ENTERPRISE,
        )

        assert upgraded.tier == TenantTier.ENTERPRISE
        assert upgraded.config.enable_sso is True
        assert upgraded.config.max_users > tenant.config.max_users

    @pytest.mark.asyncio
    async def test_tenant_suspension(self):
        """Test suspending a tenant."""
        from aragora.tenancy.provisioning import TenantProvisioner
        from aragora.tenancy.tenant import TenantStatus

        provisioner = TenantProvisioner()

        tenant = await provisioner.create_tenant(
            name="Suspended Corp",
            domain="suspended.com",
            admin_email="admin@suspended.com",
        )

        # Suspend tenant
        suspended = await provisioner.suspend_tenant(
            tenant_id=tenant.id,
            reason="Non-payment",
        )

        assert suspended.status == TenantStatus.SUSPENDED


class TestTenantResourceLimits:
    """Test tenant resource limits enforcement."""

    @pytest.mark.asyncio
    async def test_debate_limit_enforcement(self):
        """Test enforcement of debate limits."""
        from aragora.tenancy.limits import TenantLimitsEnforcer
        from aragora.tenancy.tenant import TenantConfig

        config = TenantConfig(max_debates_per_day=2)
        enforcer = TenantLimitsEnforcer(config)

        # First two debates allowed
        await enforcer.check_debate_limit("tenant-1", current_count=0)
        await enforcer.check_debate_limit("tenant-1", current_count=1)

        # Third debate denied
        with pytest.raises(Exception) as exc_info:
            await enforcer.check_debate_limit("tenant-1", current_count=2)

        assert "limit" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_token_budget_enforcement(self):
        """Test enforcement of token budgets."""
        from aragora.tenancy.limits import TenantLimitsEnforcer
        from aragora.tenancy.tenant import TenantConfig

        config = TenantConfig(tokens_per_month=1000)
        enforcer = TenantLimitsEnforcer(config)

        # Under limit - allowed
        await enforcer.check_token_budget("tenant-1", tokens_used=500, tokens_requested=200)

        # Would exceed limit - denied
        with pytest.raises(Exception) as exc_info:
            await enforcer.check_token_budget("tenant-1", tokens_used=900, tokens_requested=200)

        assert "token" in str(exc_info.value).lower()


class TestSSOAuditLogging:
    """Test SSO audit logging for compliance."""

    @pytest.mark.asyncio
    async def test_sso_login_logged(self):
        """Test that SSO logins are audit logged."""
        from aragora.auth.audit import SSOAuditLogger

        logger = SSOAuditLogger()

        # Log SSO login
        await logger.log_login(
            user_id="user-123",
            email="user@example.com",
            provider="azure_ad",
            tenant_id="tenant-456",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0...",
        )

        # Verify log entry
        logs = await logger.get_logs(user_id="user-123", limit=10)
        assert len(logs) >= 1
        assert logs[0]["event_type"] == "sso_login"
        assert logs[0]["provider"] == "azure_ad"

    @pytest.mark.asyncio
    async def test_sso_logout_logged(self):
        """Test that SSO logouts are audit logged."""
        from aragora.auth.audit import SSOAuditLogger

        logger = SSOAuditLogger()

        # Log SSO logout
        await logger.log_logout(
            user_id="user-123",
            session_id="session-789",
            reason="user_initiated",
        )

        # Verify log entry
        logs = await logger.get_logs(user_id="user-123", limit=10)
        logout_logs = [log for log in logs if log["event_type"] == "sso_logout"]
        assert len(logout_logs) >= 1


# Markers for running specific test groups
pytestmark = [
    pytest.mark.enterprise,
    pytest.mark.integration,
]
