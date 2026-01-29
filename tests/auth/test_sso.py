"""
Tests for SSO Provider Abstraction (aragora/auth/sso.py).

Tests cover:
- SSOProviderType enum
- SSOError and subclasses
- SSOUser dataclass
- SSOConfig validation
- SSOProvider base class methods
- SSOGroupMapper
- SSOSession and SSOSessionManager
- SSOAuditEntry and SSOAuditLogger
- Global provider management functions
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.auth.sso import (
    SSOAuditEntry,
    SSOAuditLogger,
    SSOAuthenticationError,
    SSOConfig,
    SSOConfigurationError,
    SSOError,
    SSOGroupMapper,
    SSOProvider,
    SSOProviderType,
    SSOSession,
    SSOSessionManager,
    SSOUser,
    get_sso_provider,
    reset_sso_provider,
)


# ============================================================================
# SSOProviderType Tests
# ============================================================================


class TestSSOProviderType:
    """Tests for SSOProviderType enum."""

    def test_provider_type_saml(self):
        """Test SAML provider type."""
        assert SSOProviderType.SAML == "saml"
        assert SSOProviderType.SAML.value == "saml"

    def test_provider_type_oidc(self):
        """Test OIDC provider type."""
        assert SSOProviderType.OIDC == "oidc"
        assert SSOProviderType.OIDC.value == "oidc"

    def test_provider_type_azure_ad(self):
        """Test Azure AD provider type."""
        assert SSOProviderType.AZURE_AD == "azure_ad"

    def test_provider_type_okta(self):
        """Test Okta provider type."""
        assert SSOProviderType.OKTA == "okta"

    def test_provider_type_google(self):
        """Test Google provider type."""
        assert SSOProviderType.GOOGLE == "google"

    def test_provider_type_github(self):
        """Test GitHub provider type."""
        assert SSOProviderType.GITHUB == "github"

    def test_provider_type_is_str(self):
        """Test that provider types are strings."""
        for provider_type in SSOProviderType:
            assert isinstance(provider_type, str)


# ============================================================================
# SSOError Tests
# ============================================================================


class TestSSOError:
    """Tests for SSOError and subclasses."""

    def test_sso_error_creation(self):
        """Test creating basic SSOError."""
        error = SSOError("Test error message")

        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.code == "SSO_ERROR"
        assert error.details == {}

    def test_sso_error_with_code(self):
        """Test SSOError with custom code."""
        error = SSOError("Error", code="CUSTOM_CODE")

        assert error.code == "CUSTOM_CODE"

    def test_sso_error_with_details(self):
        """Test SSOError with details."""
        details = {"key": "value", "count": 42}
        error = SSOError("Error", details=details)

        assert error.details == details
        assert error.details["key"] == "value"
        assert error.details["count"] == 42

    def test_sso_error_is_exception(self):
        """Test SSOError inherits from Exception."""
        error = SSOError("Test")
        assert isinstance(error, Exception)

    def test_sso_authentication_error(self):
        """Test SSOAuthenticationError."""
        error = SSOAuthenticationError("Auth failed")

        assert error.code == "SSO_AUTH_FAILED"
        assert error.message == "Auth failed"
        assert isinstance(error, SSOError)

    def test_sso_authentication_error_with_details(self):
        """Test SSOAuthenticationError with details."""
        error = SSOAuthenticationError("Auth failed", {"reason": "invalid_token"})

        assert error.details["reason"] == "invalid_token"

    def test_sso_configuration_error(self):
        """Test SSOConfigurationError."""
        error = SSOConfigurationError("Config invalid")

        assert error.code == "SSO_CONFIG_ERROR"
        assert error.message == "Config invalid"
        assert isinstance(error, SSOError)

    def test_sso_configuration_error_with_details(self):
        """Test SSOConfigurationError with details."""
        error = SSOConfigurationError("Config invalid", {"field": "client_id"})

        assert error.details["field"] == "client_id"


# ============================================================================
# SSOUser Tests
# ============================================================================


class TestSSOUser:
    """Tests for SSOUser dataclass."""

    def test_user_minimal_creation(self):
        """Test creating SSOUser with minimal fields."""
        user = SSOUser(id="user123", email="user@example.com")

        assert user.id == "user123"
        assert user.email == "user@example.com"
        assert user.name == ""
        assert user.first_name == ""
        assert user.last_name == ""
        assert user.roles == []
        assert user.groups == []

    def test_user_full_creation(self):
        """Test creating SSOUser with all fields."""
        user = SSOUser(
            id="user123",
            email="user@example.com",
            name="Test User",
            first_name="Test",
            last_name="User",
            display_name="Test U.",
            username="testuser",
            organization_id="org123",
            organization_name="Test Org",
            tenant_id="tenant123",
            azure_object_id="azure-oid",
            azure_tenant_id="azure-tid",
            roles=["admin", "developer"],
            groups=["engineering", "leadership"],
            provider_type="oidc",
            provider_id="https://login.example.com",
            access_token="access-token-123",
            refresh_token="refresh-token-456",
            id_token="id-token-789",
            token_expires_at=time.time() + 3600,
            raw_claims={"custom": "value"},
        )

        assert user.first_name == "Test"
        assert user.last_name == "User"
        assert user.organization_id == "org123"
        assert user.azure_object_id == "azure-oid"
        assert "admin" in user.roles
        assert "engineering" in user.groups
        assert user.access_token == "access-token-123"
        assert user.raw_claims["custom"] == "value"

    def test_user_is_admin_with_admin_role(self):
        """Test is_admin property returns True for admin role."""
        user = SSOUser(id="1", email="admin@example.com", roles=["admin"])
        assert user.is_admin is True

    def test_user_is_admin_with_administrator_role(self):
        """Test is_admin property returns True for administrator role."""
        user = SSOUser(id="1", email="admin@example.com", roles=["administrator"])
        assert user.is_admin is True

    def test_user_is_admin_with_superadmin_role(self):
        """Test is_admin property returns True for superadmin role."""
        user = SSOUser(id="1", email="admin@example.com", roles=["superadmin"])
        assert user.is_admin is True

    def test_user_is_admin_with_owner_role(self):
        """Test is_admin property returns True for owner role."""
        user = SSOUser(id="1", email="admin@example.com", roles=["owner"])
        assert user.is_admin is True

    def test_user_is_admin_case_insensitive(self):
        """Test is_admin property is case insensitive."""
        user = SSOUser(id="1", email="admin@example.com", roles=["ADMIN"])
        assert user.is_admin is True

    def test_user_is_not_admin(self):
        """Test is_admin property returns False for non-admin roles."""
        user = SSOUser(id="1", email="user@example.com", roles=["user", "developer"])
        assert user.is_admin is False

    def test_user_is_not_admin_empty_roles(self):
        """Test is_admin property returns False for empty roles."""
        user = SSOUser(id="1", email="user@example.com", roles=[])
        assert user.is_admin is False

    def test_user_full_name_from_first_last(self):
        """Test full_name combines first and last name."""
        user = SSOUser(
            id="1",
            email="user@example.com",
            first_name="John",
            last_name="Doe",
        )
        assert user.full_name == "John Doe"

    def test_user_full_name_falls_back_to_name(self):
        """Test full_name falls back to name field."""
        user = SSOUser(
            id="1",
            email="user@example.com",
            name="John Doe",
        )
        assert user.full_name == "John Doe"

    def test_user_full_name_falls_back_to_display_name(self):
        """Test full_name falls back to display_name."""
        user = SSOUser(
            id="1",
            email="user@example.com",
            display_name="J. Doe",
        )
        assert user.full_name == "J. Doe"

    def test_user_full_name_falls_back_to_email_local(self):
        """Test full_name falls back to email local part."""
        user = SSOUser(
            id="1",
            email="johndoe@example.com",
        )
        assert user.full_name == "johndoe"

    def test_user_to_dict(self):
        """Test to_dict returns correct dictionary."""
        user = SSOUser(
            id="user123",
            email="user@example.com",
            name="Test User",
            organization_id="org123",
            organization_name="Test Org",
            roles=["admin"],
            groups=["engineering"],
            provider_type="oidc",
        )

        result = user.to_dict()

        assert result["id"] == "user123"
        assert result["email"] == "user@example.com"
        assert result["name"] == "Test User"
        assert result["organization_id"] == "org123"
        assert result["organization_name"] == "Test Org"
        assert result["roles"] == ["admin"]
        assert result["groups"] == ["engineering"]
        assert result["provider_type"] == "oidc"
        assert result["is_admin"] is True
        assert "authenticated_at" in result

    def test_user_to_dict_includes_azure_fields_when_present(self):
        """Test to_dict includes Azure fields when present."""
        user = SSOUser(
            id="user123",
            email="user@example.com",
            azure_object_id="azure-oid",
            azure_tenant_id="azure-tid",
        )

        result = user.to_dict()

        assert result["azure_object_id"] == "azure-oid"
        assert result["azure_tenant_id"] == "azure-tid"

    def test_user_to_dict_excludes_azure_fields_when_none(self):
        """Test to_dict excludes Azure fields when None."""
        user = SSOUser(
            id="user123",
            email="user@example.com",
        )

        result = user.to_dict()

        assert "azure_object_id" not in result
        assert "azure_tenant_id" not in result

    def test_user_to_dict_username_fallback(self):
        """Test to_dict uses email local part for username fallback."""
        user = SSOUser(
            id="user123",
            email="johndoe@example.com",
        )

        result = user.to_dict()

        assert result["username"] == "johndoe"

    def test_user_authenticated_at_default(self):
        """Test authenticated_at is set to current time by default."""
        before = time.time()
        user = SSOUser(id="1", email="user@example.com")
        after = time.time()

        assert before <= user.authenticated_at <= after


# ============================================================================
# SSOConfig Tests
# ============================================================================


class TestSSOConfig:
    """Tests for SSOConfig dataclass."""

    def test_config_minimal_creation(self):
        """Test creating SSOConfig with minimal fields."""
        config = SSOConfig(provider_type=SSOProviderType.OIDC)

        assert config.provider_type == SSOProviderType.OIDC
        assert config.enabled is False
        assert config.callback_url == ""
        assert config.entity_id == ""

    def test_config_full_creation(self):
        """Test creating SSOConfig with all fields."""
        config = SSOConfig(
            provider_type=SSOProviderType.SAML,
            provider_id="custom-provider",
            enabled=True,
            callback_url="https://example.com/callback",
            entity_id="https://example.com",
            logout_url="https://example.com/logout",
            post_logout_redirect_url="https://example.com/logged-out",
            session_duration_seconds=7200,
            allowed_domains=["example.com", "company.com"],
            role_mapping={"IdPAdmin": "admin"},
            group_mapping={"IdPEng": "engineering"},
            auto_provision=True,
            default_role="viewer",
        )

        assert config.enabled is True
        assert config.session_duration_seconds == 7200
        assert "example.com" in config.allowed_domains
        assert config.role_mapping["IdPAdmin"] == "admin"
        assert config.default_role == "viewer"

    def test_config_default_session_duration(self):
        """Test default session duration is 8 hours."""
        config = SSOConfig(provider_type=SSOProviderType.OIDC)

        assert config.session_duration_seconds == 3600 * 8

    def test_config_default_auto_provision(self):
        """Test auto_provision defaults to True."""
        config = SSOConfig(provider_type=SSOProviderType.OIDC)

        assert config.auto_provision is True

    def test_config_default_role(self):
        """Test default_role defaults to 'user'."""
        config = SSOConfig(provider_type=SSOProviderType.OIDC)

        assert config.default_role == "user"

    def test_config_validate_missing_entity_id(self):
        """Test validation fails without entity_id."""
        config = SSOConfig(
            provider_type=SSOProviderType.OIDC,
            entity_id="",
            callback_url="https://example.com/callback",
        )

        errors = config.validate()

        assert any("entity_id" in e for e in errors)

    def test_config_validate_missing_callback_url(self):
        """Test validation fails without callback_url."""
        config = SSOConfig(
            provider_type=SSOProviderType.OIDC,
            entity_id="https://example.com",
            callback_url="",
        )

        errors = config.validate()

        assert any("callback_url" in e for e in errors)

    def test_config_validate_success(self):
        """Test validation succeeds with required fields."""
        config = SSOConfig(
            provider_type=SSOProviderType.OIDC,
            entity_id="https://example.com",
            callback_url="https://example.com/callback",
        )

        errors = config.validate()

        # Only entity_id and callback_url validation in base config
        assert not any("entity_id" in e for e in errors)
        assert not any("callback_url" in e for e in errors)


# ============================================================================
# SSOProvider Base Class Tests
# ============================================================================


class ConcreteSSOProvider(SSOProvider):
    """Concrete implementation for testing SSOProvider base class."""

    @property
    def provider_type(self) -> SSOProviderType:
        return SSOProviderType.OIDC

    async def get_authorization_url(
        self,
        state: str | None = None,
        redirect_uri: str | None = None,
        **kwargs,
    ) -> str:
        return f"https://auth.example.com/authorize?state={state or 'default'}"

    async def authenticate(
        self,
        code: str | None = None,
        saml_response: str | None = None,
        **kwargs,
    ) -> SSOUser:
        if code:
            return SSOUser(id="test", email="test@example.com")
        raise SSOAuthenticationError("No code provided")


class TestSSOProviderBase:
    """Tests for SSOProvider base class."""

    @pytest.fixture
    def config(self) -> SSOConfig:
        """Create a valid config for testing."""
        return SSOConfig(
            provider_type=SSOProviderType.OIDC,
            entity_id="https://example.com",
            callback_url="https://example.com/callback",
            role_mapping={"IdPAdmin": "admin", "IdPDev": "developer"},
            group_mapping={"IdPEngineering": "engineering"},
            allowed_domains=["example.com", "company.com"],
            default_role="viewer",
        )

    @pytest.fixture
    def provider(self, config: SSOConfig) -> ConcreteSSOProvider:
        """Create a provider instance for testing."""
        return ConcreteSSOProvider(config)

    def test_provider_init(self, provider: ConcreteSSOProvider, config: SSOConfig):
        """Test provider initialization."""
        assert provider.config == config
        assert provider._state_store == {}

    def test_generate_state(self, provider: ConcreteSSOProvider):
        """Test state generation."""
        state = provider.generate_state()

        assert state is not None
        assert len(state) > 20  # Should be reasonably long
        assert state in provider._state_store

    def test_generate_state_unique(self, provider: ConcreteSSOProvider):
        """Test that generated states are unique."""
        states = [provider.generate_state() for _ in range(10)]

        assert len(states) == len(set(states))  # All unique

    def test_validate_state_success(self, provider: ConcreteSSOProvider):
        """Test successful state validation."""
        state = provider.generate_state()

        result = provider.validate_state(state)

        assert result is True
        assert state not in provider._state_store  # Consumed

    def test_validate_state_unknown(self, provider: ConcreteSSOProvider):
        """Test validation fails for unknown state."""
        result = provider.validate_state("unknown-state")

        assert result is False

    def test_validate_state_expired(self, provider: ConcreteSSOProvider):
        """Test validation fails for expired state."""
        state = provider.generate_state()
        # Backdate the state by 11 minutes (beyond 10 minute window)
        provider._state_store[state] = time.time() - 660

        result = provider.validate_state(state)

        assert result is False

    def test_validate_state_just_before_expiry(self, provider: ConcreteSSOProvider):
        """Test validation succeeds just before expiry."""
        state = provider.generate_state()
        # Backdate the state by 9 minutes (within 10 minute window)
        provider._state_store[state] = time.time() - 540

        result = provider.validate_state(state)

        assert result is True

    def test_cleanup_expired_states(self, provider: ConcreteSSOProvider):
        """Test cleanup of expired states."""
        # Generate some states
        valid_state = provider.generate_state()
        expired_state = provider.generate_state()

        # Make one expired
        provider._state_store[expired_state] = time.time() - 660

        cleaned = provider.cleanup_expired_states()

        assert cleaned == 1
        assert valid_state in provider._state_store
        assert expired_state not in provider._state_store

    def test_cleanup_expired_states_none_expired(self, provider: ConcreteSSOProvider):
        """Test cleanup when no states are expired."""
        provider.generate_state()
        provider.generate_state()

        cleaned = provider.cleanup_expired_states()

        assert cleaned == 0
        assert len(provider._state_store) == 2

    def test_map_roles_with_mapping(self, provider: ConcreteSSOProvider):
        """Test role mapping with configured mappings."""
        idp_roles = ["IdPAdmin", "IdPDev", "IdPOther"]

        result = provider.map_roles(idp_roles)

        assert "admin" in result
        assert "developer" in result
        assert "IdPOther" in result  # Unmapped role passed through

    def test_map_roles_without_mapping(self):
        """Test role mapping without configured mappings."""
        config = SSOConfig(
            provider_type=SSOProviderType.OIDC,
            entity_id="test",
            callback_url="test",
            role_mapping={},
            default_role="user",
        )
        provider = ConcreteSSOProvider(config)

        result = provider.map_roles(["role1", "role2"])

        assert "role1" in result
        assert "role2" in result

    def test_map_roles_empty_uses_default(self, provider: ConcreteSSOProvider):
        """Test that empty roles get default role."""
        result = provider.map_roles([])

        assert provider.config.default_role in result

    def test_map_roles_deduplicates(self, provider: ConcreteSSOProvider):
        """Test that mapped roles are deduplicated."""
        idp_roles = ["IdPAdmin", "IdPAdmin", "admin"]  # admin maps to admin

        result = provider.map_roles(idp_roles)

        # Should not have duplicates
        assert len(result) == len(set(result))

    def test_map_groups_with_mapping(self, provider: ConcreteSSOProvider):
        """Test group mapping with configured mappings."""
        idp_groups = ["IdPEngineering", "IdPOther"]

        result = provider.map_groups(idp_groups)

        assert "engineering" in result
        assert "IdPOther" in result

    def test_map_groups_without_mapping(self):
        """Test group mapping without configured mappings."""
        config = SSOConfig(
            provider_type=SSOProviderType.OIDC,
            entity_id="test",
            callback_url="test",
            group_mapping={},
        )
        provider = ConcreteSSOProvider(config)

        result = provider.map_groups(["group1", "group2"])

        assert "group1" in result
        assert "group2" in result

    def test_map_groups_deduplicates(self, provider: ConcreteSSOProvider):
        """Test that mapped groups are deduplicated."""
        idp_groups = ["IdPEngineering", "engineering"]

        result = provider.map_groups(idp_groups)

        assert len(result) == len(set(result))

    def test_is_domain_allowed_no_restrictions(self):
        """Test domain check with no restrictions."""
        config = SSOConfig(
            provider_type=SSOProviderType.OIDC,
            entity_id="test",
            callback_url="test",
            allowed_domains=[],
        )
        provider = ConcreteSSOProvider(config)

        assert provider.is_domain_allowed("user@anything.com") is True

    def test_is_domain_allowed_in_list(self, provider: ConcreteSSOProvider):
        """Test domain check when domain is allowed."""
        assert provider.is_domain_allowed("user@example.com") is True
        assert provider.is_domain_allowed("user@company.com") is True

    def test_is_domain_allowed_not_in_list(self, provider: ConcreteSSOProvider):
        """Test domain check when domain is not allowed."""
        assert provider.is_domain_allowed("user@other.com") is False

    def test_is_domain_allowed_case_insensitive(self, provider: ConcreteSSOProvider):
        """Test domain check is case insensitive."""
        assert provider.is_domain_allowed("user@EXAMPLE.COM") is True
        assert provider.is_domain_allowed("user@Example.Com") is True

    @pytest.mark.asyncio
    async def test_logout_returns_logout_url(self, provider: ConcreteSSOProvider):
        """Test logout returns configured logout URL."""
        provider.config.logout_url = "https://example.com/logout"
        user = SSOUser(id="1", email="user@example.com")

        result = await provider.logout(user)

        assert result == "https://example.com/logout"

    @pytest.mark.asyncio
    async def test_logout_returns_none_when_no_url(self, provider: ConcreteSSOProvider):
        """Test logout returns None when no logout URL configured."""
        provider.config.logout_url = ""
        user = SSOUser(id="1", email="user@example.com")

        result = await provider.logout(user)

        assert result is None

    @pytest.mark.asyncio
    async def test_refresh_token_default_not_supported(self, provider: ConcreteSSOProvider):
        """Test default refresh_token returns None."""
        user = SSOUser(id="1", email="user@example.com", refresh_token="token")

        result = await provider.refresh_token(user)

        assert result is None


# ============================================================================
# SSOGroupMapper Tests
# ============================================================================


class TestSSOGroupMapper:
    """Tests for SSOGroupMapper utility class."""

    def test_mapper_initialization(self):
        """Test mapper initialization."""
        mappings = {"Aragora-Admins": "admin", "Engineering": "developer"}
        mapper = SSOGroupMapper(mappings, default_role="user")

        assert mapper.mappings == mappings
        assert mapper.default_role == "user"

    def test_map_groups_with_matches(self):
        """Test mapping groups that match."""
        mapper = SSOGroupMapper(
            {"Aragora-Admins": "admin", "Engineering": "developer"},
        )

        result = mapper.map_groups(["Aragora-Admins", "Engineering"])

        assert "admin" in result
        assert "developer" in result

    def test_map_groups_partial_matches(self):
        """Test mapping groups with some matches."""
        mapper = SSOGroupMapper(
            {"Aragora-Admins": "admin"},
        )

        result = mapper.map_groups(["Aragora-Admins", "Unknown"])

        assert "admin" in result
        assert "Unknown" not in result  # Not in mappings, not passed through

    def test_map_groups_no_matches_with_default(self):
        """Test mapping with no matches uses default role."""
        mapper = SSOGroupMapper(
            {"Aragora-Admins": "admin"},
            default_role="user",
        )

        result = mapper.map_groups(["Unknown", "Other"])

        assert result == ["user"]

    def test_map_groups_no_matches_no_default(self):
        """Test mapping with no matches and no default returns empty."""
        mapper = SSOGroupMapper(
            {"Aragora-Admins": "admin"},
            default_role=None,
        )

        result = mapper.map_groups(["Unknown", "Other"])

        assert result == []

    def test_map_groups_empty_input(self):
        """Test mapping empty group list."""
        mapper = SSOGroupMapper(
            {"Aragora-Admins": "admin"},
            default_role="user",
        )

        result = mapper.map_groups([])

        assert result == ["user"]

    def test_map_groups_deduplicates(self):
        """Test mapping deduplicates results."""
        mapper = SSOGroupMapper(
            {"Group1": "admin", "Group2": "admin"},
        )

        result = mapper.map_groups(["Group1", "Group2"])

        assert result == ["admin"]


# ============================================================================
# SSOSession Tests
# ============================================================================


class TestSSOSession:
    """Tests for SSOSession dataclass."""

    def test_session_minimal_creation(self):
        """Test creating session with minimal fields."""
        session = SSOSession(
            session_id="sess123",
            user_id="user123",
            email="user@example.com",
        )

        assert session.session_id == "sess123"
        assert session.user_id == "user123"
        assert session.email == "user@example.com"
        assert session.org_id is None

    def test_session_full_creation(self):
        """Test creating session with all fields."""
        now = time.time()
        session = SSOSession(
            session_id="sess123",
            user_id="user123",
            email="user@example.com",
            org_id="org123",
            created_at=now,
            expires_at=now + 3600,
            metadata={"custom": "data"},
        )

        assert session.org_id == "org123"
        assert session.created_at == now
        assert session.expires_at == now + 3600
        assert session.metadata["custom"] == "data"

    def test_session_default_timestamps(self):
        """Test session has sensible default timestamps."""
        before = time.time()
        session = SSOSession(
            session_id="sess123",
            user_id="user123",
            email="user@example.com",
        )
        after = time.time()

        # created_at should be around now
        assert before <= session.created_at <= after

        # expires_at should be 8 hours from now (default)
        expected_expiry = session.created_at + 3600 * 8
        assert abs(session.expires_at - expected_expiry) < 1


# ============================================================================
# SSOSessionManager Tests
# ============================================================================


class TestSSOSessionManager:
    """Tests for SSOSessionManager."""

    @pytest.fixture
    def manager(self) -> SSOSessionManager:
        """Create a session manager for testing."""
        return SSOSessionManager(session_duration=3600)

    @pytest.fixture
    def user(self) -> SSOUser:
        """Create a test user."""
        return SSOUser(
            id="user123",
            email="user@example.com",
            organization_id="org123",
        )

    @pytest.mark.asyncio
    async def test_create_session(self, manager: SSOSessionManager, user: SSOUser):
        """Test creating a session."""
        session = await manager.create_session(user)

        assert session.user_id == user.id
        assert session.email == user.email
        assert session.org_id == user.organization_id
        assert session.session_id is not None

    @pytest.mark.asyncio
    async def test_create_session_uses_tenant_id_fallback(self, manager: SSOSessionManager):
        """Test session uses tenant_id if organization_id is None."""
        user = SSOUser(
            id="user123",
            email="user@example.com",
            organization_id=None,
            tenant_id="tenant123",
        )

        session = await manager.create_session(user)

        assert session.org_id == "tenant123"

    @pytest.mark.asyncio
    async def test_create_session_expiry(self, manager: SSOSessionManager, user: SSOUser):
        """Test session has correct expiry time."""
        before = time.time()
        session = await manager.create_session(user)
        after = time.time()

        expected_expiry = before + manager.session_duration
        assert session.expires_at >= expected_expiry
        assert session.expires_at <= after + manager.session_duration

    @pytest.mark.asyncio
    async def test_get_session_success(self, manager: SSOSessionManager, user: SSOUser):
        """Test getting an existing session."""
        session = await manager.create_session(user)

        retrieved = await manager.get_session(session.session_id)

        assert retrieved.session_id == session.session_id
        assert retrieved.user_id == user.id

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, manager: SSOSessionManager):
        """Test getting a non-existent session raises KeyError."""
        with pytest.raises(KeyError) as exc:
            await manager.get_session("unknown-session")

        assert "not found" in str(exc.value).lower()

    @pytest.mark.asyncio
    async def test_get_session_expired(self, manager: SSOSessionManager, user: SSOUser):
        """Test getting an expired session raises KeyError."""
        session = await manager.create_session(user)
        # Expire the session
        manager._sessions[session.session_id].expires_at = time.time() - 1

        with pytest.raises(KeyError) as exc:
            await manager.get_session(session.session_id)

        assert "expired" in str(exc.value).lower()

    @pytest.mark.asyncio
    async def test_get_session_removes_expired(self, manager: SSOSessionManager, user: SSOUser):
        """Test that getting an expired session removes it from store."""
        session = await manager.create_session(user)
        manager._sessions[session.session_id].expires_at = time.time() - 1

        with pytest.raises(KeyError):
            await manager.get_session(session.session_id)

        assert session.session_id not in manager._sessions

    @pytest.mark.asyncio
    async def test_refresh_session(self, manager: SSOSessionManager, user: SSOUser):
        """Test refreshing a session extends its expiry."""
        session = await manager.create_session(user)
        original_expiry = session.expires_at

        # Wait a tiny bit
        await manager.refresh_session(session.session_id)

        refreshed = await manager.get_session(session.session_id)
        assert refreshed.expires_at >= original_expiry

    @pytest.mark.asyncio
    async def test_logout_removes_session(self, manager: SSOSessionManager, user: SSOUser):
        """Test logout removes the session."""
        session = await manager.create_session(user)

        await manager.logout(session.session_id)

        with pytest.raises(KeyError):
            await manager.get_session(session.session_id)

    @pytest.mark.asyncio
    async def test_logout_nonexistent_session(self, manager: SSOSessionManager):
        """Test logout handles non-existent session gracefully."""
        # Should not raise
        await manager.logout("unknown-session")

    def test_custom_session_duration(self):
        """Test manager with custom session duration."""
        manager = SSOSessionManager(session_duration=7200)

        assert manager.session_duration == 7200


# ============================================================================
# SSOAuditEntry Tests
# ============================================================================


class TestSSOAuditEntry:
    """Tests for SSOAuditEntry dataclass."""

    def test_entry_minimal_creation(self):
        """Test creating entry with minimal fields."""
        entry = SSOAuditEntry(
            timestamp=time.time(),
            event_type="sso_login",
            user_id="user123",
        )

        assert entry.event_type == "sso_login"
        assert entry.user_id == "user123"
        assert entry.email is None
        assert entry.metadata == {}

    def test_entry_full_creation(self):
        """Test creating entry with all fields."""
        entry = SSOAuditEntry(
            timestamp=time.time(),
            event_type="sso_login",
            user_id="user123",
            email="user@example.com",
            provider="azure_ad",
            tenant_id="tenant123",
            session_id="sess123",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            reason="user_initiated",
            metadata={"mfa": True},
        )

        assert entry.email == "user@example.com"
        assert entry.provider == "azure_ad"
        assert entry.ip_address == "192.168.1.1"
        assert entry.metadata["mfa"] is True


# ============================================================================
# SSOAuditLogger Tests
# ============================================================================


class TestSSOAuditLogger:
    """Tests for SSOAuditLogger."""

    @pytest.fixture
    def logger(self) -> SSOAuditLogger:
        """Create an audit logger for testing."""
        return SSOAuditLogger()

    @pytest.mark.asyncio
    async def test_log_login(self, logger: SSOAuditLogger):
        """Test logging a login event."""
        await logger.log_login(
            user_id="user123",
            email="user@example.com",
            provider="azure_ad",
            tenant_id="tenant123",
            ip_address="192.168.1.1",
        )

        logs = await logger.get_logs()

        assert len(logs) == 1
        assert logs[0]["event_type"] == "sso_login"
        assert logs[0]["user_id"] == "user123"
        assert logs[0]["email"] == "user@example.com"
        assert logs[0]["provider"] == "azure_ad"

    @pytest.mark.asyncio
    async def test_log_login_with_metadata(self, logger: SSOAuditLogger):
        """Test logging a login with extra metadata."""
        await logger.log_login(
            user_id="user123",
            email="user@example.com",
            provider="okta",
            mfa_used=True,
            auth_method="passkey",
        )

        logs = await logger.get_logs()

        assert logs[0]["mfa_used"] is True
        assert logs[0]["auth_method"] == "passkey"

    @pytest.mark.asyncio
    async def test_log_logout(self, logger: SSOAuditLogger):
        """Test logging a logout event."""
        await logger.log_logout(
            user_id="user123",
            session_id="sess123",
            reason="user_initiated",
        )

        logs = await logger.get_logs()

        assert len(logs) == 1
        assert logs[0]["event_type"] == "sso_logout"
        assert logs[0]["user_id"] == "user123"
        assert logs[0]["session_id"] == "sess123"
        assert logs[0]["reason"] == "user_initiated"

    @pytest.mark.asyncio
    async def test_log_logout_default_reason(self, logger: SSOAuditLogger):
        """Test logout default reason."""
        await logger.log_logout(user_id="user123")

        logs = await logger.get_logs()

        assert logs[0]["reason"] == "user_initiated"

    @pytest.mark.asyncio
    async def test_get_logs_filter_by_user(self, logger: SSOAuditLogger):
        """Test filtering logs by user_id."""
        await logger.log_login(user_id="user1", email="u1@example.com", provider="okta")
        await logger.log_login(user_id="user2", email="u2@example.com", provider="okta")
        await logger.log_login(user_id="user1", email="u1@example.com", provider="okta")

        logs = await logger.get_logs(user_id="user1")

        assert len(logs) == 2
        assert all(log["user_id"] == "user1" for log in logs)

    @pytest.mark.asyncio
    async def test_get_logs_filter_by_event_type(self, logger: SSOAuditLogger):
        """Test filtering logs by event_type."""
        await logger.log_login(user_id="user1", email="u1@example.com", provider="okta")
        await logger.log_logout(user_id="user1")
        await logger.log_login(user_id="user2", email="u2@example.com", provider="okta")

        logs = await logger.get_logs(event_type="sso_login")

        assert len(logs) == 2
        assert all(log["event_type"] == "sso_login" for log in logs)

    @pytest.mark.asyncio
    async def test_get_logs_limit(self, logger: SSOAuditLogger):
        """Test limiting log results."""
        for i in range(10):
            await logger.log_login(user_id=f"user{i}", email=f"u{i}@example.com", provider="okta")

        logs = await logger.get_logs(limit=5)

        assert len(logs) == 5

    @pytest.mark.asyncio
    async def test_get_logs_returns_most_recent(self, logger: SSOAuditLogger):
        """Test that get_logs returns most recent entries."""
        for i in range(10):
            await logger.log_login(user_id=f"user{i}", email=f"u{i}@example.com", provider="okta")

        logs = await logger.get_logs(limit=3)

        # Should be the last 3 entries (most recent)
        assert logs[0]["user_id"] == "user7"
        assert logs[1]["user_id"] == "user8"
        assert logs[2]["user_id"] == "user9"

    @pytest.mark.asyncio
    async def test_get_logs_combined_filters(self, logger: SSOAuditLogger):
        """Test combining multiple filters."""
        await logger.log_login(user_id="user1", email="u1@example.com", provider="okta")
        await logger.log_logout(user_id="user1")
        await logger.log_login(user_id="user1", email="u1@example.com", provider="okta")
        await logger.log_logout(user_id="user2")

        logs = await logger.get_logs(user_id="user1", event_type="sso_logout")

        assert len(logs) == 1
        assert logs[0]["user_id"] == "user1"
        assert logs[0]["event_type"] == "sso_logout"


# ============================================================================
# Global Provider Management Tests
# ============================================================================


class TestGlobalProviderManagement:
    """Tests for get_sso_provider and reset_sso_provider."""

    def setup_method(self):
        """Reset provider state before each test."""
        reset_sso_provider()

    def teardown_method(self):
        """Reset provider state after each test."""
        reset_sso_provider()

    def test_reset_sso_provider(self):
        """Test reset_sso_provider clears state."""
        # Just verify it doesn't raise
        reset_sso_provider()

    def test_get_sso_provider_returns_none_when_not_configured(self):
        """Test get_sso_provider returns None when not configured."""
        with patch("aragora.config.settings.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(sso=MagicMock(enabled=False))
            result = get_sso_provider()

        assert result is None

    def test_get_sso_provider_returns_none_when_no_sso_settings(self):
        """Test get_sso_provider returns None when SSO settings don't exist."""
        with patch("aragora.config.settings.get_settings") as mock_settings:
            settings = MagicMock(spec=[])  # No sso attribute
            mock_settings.return_value = settings
            result = get_sso_provider()

        assert result is None

    def test_get_sso_provider_caches_result(self):
        """Test get_sso_provider caches the result."""
        with patch("aragora.config.settings.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(sso=MagicMock(enabled=False))

            result1 = get_sso_provider()
            result2 = get_sso_provider()

        # Should only call get_settings once due to caching
        assert mock_settings.call_count == 1
        assert result1 is result2

    def test_get_sso_provider_handles_import_error(self):
        """Test get_sso_provider handles errors gracefully."""
        reset_sso_provider()

        with patch("aragora.config.settings.get_settings") as mock_settings:
            mock_settings.side_effect = ImportError("Config not available")
            result = get_sso_provider()

        assert result is None


# ============================================================================
# Integration Tests
# ============================================================================


class TestSSOIntegration:
    """Integration tests for SSO components."""

    @pytest.mark.asyncio
    async def test_full_sso_flow(self):
        """Test a complete SSO authentication flow."""
        # Setup
        config = SSOConfig(
            provider_type=SSOProviderType.OIDC,
            entity_id="https://example.com",
            callback_url="https://example.com/callback",
            allowed_domains=["example.com"],
            role_mapping={"IdPAdmin": "admin"},
        )
        provider = ConcreteSSOProvider(config)
        session_manager = SSOSessionManager()
        audit_logger = SSOAuditLogger()

        # Step 1: Generate authorization URL with state
        state = provider.generate_state()
        auth_url = await provider.get_authorization_url(state=state)
        assert "state=" in auth_url

        # Step 2: Validate state (simulating callback)
        assert provider.validate_state(state) is True

        # Step 3: Authenticate user
        user = await provider.authenticate(code="auth-code")
        assert user.email == "test@example.com"

        # Step 4: Create session
        session = await session_manager.create_session(user)
        assert session.user_id == user.id

        # Step 5: Log the authentication
        await audit_logger.log_login(
            user_id=user.id,
            email=user.email,
            provider=config.provider_type.value,
        )

        logs = await audit_logger.get_logs()
        assert len(logs) == 1

        # Step 6: Logout
        await session_manager.logout(session.session_id)
        await audit_logger.log_logout(user_id=user.id, session_id=session.session_id)

        # Verify session is gone
        with pytest.raises(KeyError):
            await session_manager.get_session(session.session_id)

        # Verify audit log has both events
        logs = await audit_logger.get_logs()
        assert len(logs) == 2

    def test_role_and_group_mapping_integration(self):
        """Test role and group mapping work together."""
        config = SSOConfig(
            provider_type=SSOProviderType.OIDC,
            entity_id="test",
            callback_url="test",
            role_mapping={
                "Aragora-Admins": "admin",
                "Aragora-Developers": "developer",
            },
            group_mapping={
                "Engineering": "engineering",
                "Leadership": "leadership",
            },
            default_role="viewer",
        )
        provider = ConcreteSSOProvider(config)

        # Map roles
        idp_roles = ["Aragora-Admins", "Unknown-Role"]
        mapped_roles = provider.map_roles(idp_roles)
        assert "admin" in mapped_roles
        assert "Unknown-Role" in mapped_roles

        # Map groups
        idp_groups = ["Engineering", "Unknown-Group"]
        mapped_groups = provider.map_groups(idp_groups)
        assert "engineering" in mapped_groups
        assert "Unknown-Group" in mapped_groups

    def test_domain_restriction_integration(self):
        """Test domain restrictions work correctly."""
        config = SSOConfig(
            provider_type=SSOProviderType.OIDC,
            entity_id="test",
            callback_url="test",
            allowed_domains=["company.com", "subsidiary.com"],
        )
        provider = ConcreteSSOProvider(config)

        # Allowed domains
        assert provider.is_domain_allowed("user@company.com") is True
        assert provider.is_domain_allowed("user@subsidiary.com") is True

        # Disallowed domains
        assert provider.is_domain_allowed("user@competitor.com") is False
        assert provider.is_domain_allowed("user@personal.com") is False
