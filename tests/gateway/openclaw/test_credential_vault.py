"""
Comprehensive tests for OpenClaw credential vault.

Tests cover:
- CredentialType and CredentialFramework enums
- CredentialMetadata dataclass
- RotationPolicy dataclass and policy presets
- StoredCredential dataclass
- CredentialVault CRUD operations
- Encryption/decryption with cryptography library
- Security behavior (no fallback to unencrypted storage)
- Expired credentials handling
- Tenant isolation
- Agent and capability restrictions
- Rate limiting
- RBAC-based access control
- Audit logging
- Credential rotation
- Factory functions and singleton management
"""

from __future__ import annotations

import asyncio
import os
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.gateway.openclaw.credential_vault import (
    # Main classes
    CredentialVault,
    CredentialRateLimiter,
    # Dataclasses
    StoredCredential,
    CredentialMetadata,
    RotationPolicy,
    # Enums
    CredentialType,
    CredentialFramework,
    CredentialAuditEvent,
    # Exceptions
    CredentialVaultError,
    CredentialNotFoundError,
    CredentialAccessDeniedError,
    CredentialExpiredError,
    CredentialRateLimitedError,
    TenantIsolationError,
    EncryptionError,
    # Factory functions
    get_credential_vault,
    reset_credential_vault,
    init_credential_vault,
)


# =============================================================================
# CredentialType Tests
# =============================================================================


class TestCredentialType:
    """Tests for CredentialType enum."""

    def test_api_key_value(self) -> None:
        """Test API key value."""
        assert CredentialType.API_KEY.value == "api_key"

    def test_oauth_token_value(self) -> None:
        """Test OAuth token value."""
        assert CredentialType.OAUTH_TOKEN.value == "oauth_token"

    def test_oauth_secret_value(self) -> None:
        """Test OAuth secret value."""
        assert CredentialType.OAUTH_SECRET.value == "oauth_secret"

    def test_oauth_refresh_token_value(self) -> None:
        """Test OAuth refresh token value."""
        assert CredentialType.OAUTH_REFRESH_TOKEN.value == "oauth_refresh_token"

    def test_service_account_value(self) -> None:
        """Test service account value."""
        assert CredentialType.SERVICE_ACCOUNT.value == "service_account"

    def test_certificate_value(self) -> None:
        """Test certificate value."""
        assert CredentialType.CERTIFICATE.value == "certificate"

    def test_password_value(self) -> None:
        """Test password value."""
        assert CredentialType.PASSWORD.value == "password"

    def test_bearer_token_value(self) -> None:
        """Test bearer token value."""
        assert CredentialType.BEARER_TOKEN.value == "bearer_token"

    def test_webhook_secret_value(self) -> None:
        """Test webhook secret value."""
        assert CredentialType.WEBHOOK_SECRET.value == "webhook_secret"

    def test_encryption_key_value(self) -> None:
        """Test encryption key value."""
        assert CredentialType.ENCRYPTION_KEY.value == "encryption_key"

    def test_all_credential_types_unique(self) -> None:
        """Test all credential types are unique."""
        values = [t.value for t in CredentialType]
        assert len(values) == len(set(values))


# =============================================================================
# CredentialFramework Tests
# =============================================================================


class TestCredentialFramework:
    """Tests for CredentialFramework enum."""

    def test_openai_value(self) -> None:
        """Test OpenAI value."""
        assert CredentialFramework.OPENAI.value == "openai"

    def test_anthropic_value(self) -> None:
        """Test Anthropic value."""
        assert CredentialFramework.ANTHROPIC.value == "anthropic"

    def test_google_value(self) -> None:
        """Test Google value."""
        assert CredentialFramework.GOOGLE.value == "google"

    def test_azure_value(self) -> None:
        """Test Azure value."""
        assert CredentialFramework.AZURE.value == "azure"

    def test_aws_value(self) -> None:
        """Test AWS value."""
        assert CredentialFramework.AWS.value == "aws"

    def test_huggingface_value(self) -> None:
        """Test HuggingFace value."""
        assert CredentialFramework.HUGGINGFACE.value == "huggingface"

    def test_langchain_value(self) -> None:
        """Test LangChain value."""
        assert CredentialFramework.LANGCHAIN.value == "langchain"

    def test_crewai_value(self) -> None:
        """Test CrewAI value."""
        assert CredentialFramework.CREWAI.value == "crewai"

    def test_autogen_value(self) -> None:
        """Test AutoGen value."""
        assert CredentialFramework.AUTOGEN.value == "autogen"

    def test_openclaw_value(self) -> None:
        """Test OpenClaw value."""
        assert CredentialFramework.OPENCLAW.value == "openclaw"

    def test_custom_value(self) -> None:
        """Test custom value."""
        assert CredentialFramework.CUSTOM.value == "custom"

    def test_all_frameworks_unique(self) -> None:
        """Test all frameworks are unique."""
        values = [f.value for f in CredentialFramework]
        assert len(values) == len(set(values))


# =============================================================================
# CredentialMetadata Tests
# =============================================================================


class TestCredentialMetadata:
    """Tests for CredentialMetadata dataclass."""

    def test_metadata_creation_defaults(self) -> None:
        """Test metadata creation with defaults."""
        metadata = CredentialMetadata()
        assert metadata.created_at is not None
        assert metadata.created_by is None
        assert metadata.rotated_at is None
        assert metadata.access_count == 0
        assert metadata.version == 1
        assert metadata.tags == []

    def test_metadata_creation_full(self) -> None:
        """Test metadata creation with all fields."""
        now = datetime.now(timezone.utc)
        metadata = CredentialMetadata(
            created_at=now,
            created_by="user-123",
            rotated_at=now - timedelta(days=1),
            rotated_by="user-456",
            expires_at=now + timedelta(days=30),
            access_count=100,
            last_accessed_at=now - timedelta(hours=1),
            last_accessed_by="user-789",
            version=5,
            tags=["production", "critical"],
            description="Test credential",
        )
        assert metadata.created_by == "user-123"
        assert metadata.rotated_by == "user-456"
        assert metadata.access_count == 100
        assert metadata.version == 5
        assert "production" in metadata.tags

    def test_metadata_is_expired_no_expiry(self) -> None:
        """Test is_expired when no expiry set."""
        metadata = CredentialMetadata(expires_at=None)
        assert metadata.is_expired is False

    def test_metadata_is_expired_future(self) -> None:
        """Test is_expired when expiry in future."""
        metadata = CredentialMetadata(expires_at=datetime.now(timezone.utc) + timedelta(days=1))
        assert metadata.is_expired is False

    def test_metadata_is_expired_past(self) -> None:
        """Test is_expired when expiry in past."""
        metadata = CredentialMetadata(expires_at=datetime.now(timezone.utc) - timedelta(hours=1))
        assert metadata.is_expired is True

    def test_metadata_days_until_expiry_none(self) -> None:
        """Test days_until_expiry when no expiry."""
        metadata = CredentialMetadata(expires_at=None)
        assert metadata.days_until_expiry is None

    def test_metadata_days_until_expiry_future(self) -> None:
        """Test days_until_expiry when expiry in future."""
        metadata = CredentialMetadata(
            expires_at=datetime.now(timezone.utc) + timedelta(days=10, hours=1)
        )
        # Allow for minor timing variations - should be 10 days
        assert metadata.days_until_expiry in (9, 10)

    def test_metadata_days_until_expiry_past(self) -> None:
        """Test days_until_expiry when expired."""
        metadata = CredentialMetadata(expires_at=datetime.now(timezone.utc) - timedelta(days=5))
        assert metadata.days_until_expiry == 0

    def test_metadata_to_dict(self) -> None:
        """Test metadata serialization."""
        metadata = CredentialMetadata(
            description="Test",
            tags=["tag1", "tag2"],
        )
        result = metadata.to_dict()
        assert "created_at" in result
        assert "description" in result
        assert result["description"] == "Test"
        assert result["tags"] == ["tag1", "tag2"]
        assert "is_expired" in result
        assert "days_until_expiry" in result

    def test_metadata_from_dict(self) -> None:
        """Test metadata deserialization."""
        data = {
            "created_at": "2024-01-01T00:00:00+00:00",
            "created_by": "user-123",
            "version": 3,
            "tags": ["test"],
            "description": "Test credential",
        }
        metadata = CredentialMetadata.from_dict(data)
        assert metadata.created_by == "user-123"
        assert metadata.version == 3
        assert metadata.tags == ["test"]

    def test_metadata_from_dict_with_z_suffix(self) -> None:
        """Test metadata deserialization with Z timezone suffix."""
        data = {
            "created_at": "2024-01-01T00:00:00Z",
        }
        metadata = CredentialMetadata.from_dict(data)
        assert metadata.created_at is not None


# =============================================================================
# RotationPolicy Tests
# =============================================================================


class TestRotationPolicy:
    """Tests for RotationPolicy dataclass."""

    def test_rotation_policy_defaults(self) -> None:
        """Test rotation policy defaults."""
        policy = RotationPolicy()
        assert policy.interval_days == 90
        assert policy.notify_before_days == 14
        assert policy.auto_rotate is False
        assert policy.on_access_count == 0
        assert policy.on_compromise is True
        assert policy.require_approval is False

    def test_rotation_policy_strict(self) -> None:
        """Test strict rotation policy preset."""
        policy = RotationPolicy.strict()
        assert policy.interval_days == 30
        assert policy.notify_before_days == 7
        assert policy.require_approval is True
        assert policy.on_access_count == 1000

    def test_rotation_policy_standard(self) -> None:
        """Test standard rotation policy preset."""
        policy = RotationPolicy.standard()
        assert policy.interval_days == 90
        assert policy.auto_rotate is True

    def test_rotation_policy_relaxed(self) -> None:
        """Test relaxed rotation policy preset."""
        policy = RotationPolicy.relaxed()
        assert policy.interval_days == 365
        assert policy.notify_before_days == 30

    def test_is_rotation_due_no_interval(self) -> None:
        """Test rotation not due when interval is 0."""
        policy = RotationPolicy(interval_days=0)
        metadata = CredentialMetadata()
        assert policy.is_rotation_due(metadata) is False

    def test_is_rotation_due_time_based(self) -> None:
        """Test time-based rotation detection."""
        policy = RotationPolicy(interval_days=30)
        metadata = CredentialMetadata(created_at=datetime.now(timezone.utc) - timedelta(days=31))
        assert policy.is_rotation_due(metadata) is True

    def test_is_rotation_due_time_based_not_due(self) -> None:
        """Test time-based rotation not due."""
        policy = RotationPolicy(interval_days=30)
        metadata = CredentialMetadata(created_at=datetime.now(timezone.utc) - timedelta(days=10))
        assert policy.is_rotation_due(metadata) is False

    def test_is_rotation_due_access_count(self) -> None:
        """Test access-count based rotation detection."""
        policy = RotationPolicy(interval_days=365, on_access_count=100)
        metadata = CredentialMetadata(access_count=150)
        assert policy.is_rotation_due(metadata) is True

    def test_needs_expiry_alert_no_notify(self) -> None:
        """Test no expiry alert when notify days is 0."""
        policy = RotationPolicy(notify_before_days=0)
        metadata = CredentialMetadata(expires_at=datetime.now(timezone.utc) + timedelta(days=5))
        assert policy.needs_expiry_alert(metadata) is False

    def test_needs_expiry_alert_within_window(self) -> None:
        """Test expiry alert within notification window."""
        policy = RotationPolicy(notify_before_days=14)
        metadata = CredentialMetadata(expires_at=datetime.now(timezone.utc) + timedelta(days=10))
        assert policy.needs_expiry_alert(metadata) is True

    def test_needs_expiry_alert_outside_window(self) -> None:
        """Test no expiry alert outside notification window."""
        policy = RotationPolicy(notify_before_days=14)
        metadata = CredentialMetadata(expires_at=datetime.now(timezone.utc) + timedelta(days=30))
        assert policy.needs_expiry_alert(metadata) is False

    def test_rotation_policy_to_dict(self) -> None:
        """Test rotation policy serialization."""
        policy = RotationPolicy(interval_days=60, auto_rotate=True)
        result = policy.to_dict()
        assert result["interval_days"] == 60
        assert result["auto_rotate"] is True

    def test_rotation_policy_from_dict(self) -> None:
        """Test rotation policy deserialization."""
        data = {"interval_days": 45, "notify_before_days": 7}
        policy = RotationPolicy.from_dict(data)
        assert policy.interval_days == 45
        assert policy.notify_before_days == 7


# =============================================================================
# StoredCredential Tests
# =============================================================================


class TestStoredCredential:
    """Tests for StoredCredential dataclass."""

    def test_stored_credential_creation(self) -> None:
        """Test stored credential creation."""
        credential = StoredCredential(
            credential_id="cred-123",
            tenant_id="tenant-abc",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            encrypted_value=b"encrypted-data",
            encryption_key_id="key-456",
        )
        assert credential.credential_id == "cred-123"
        assert credential.tenant_id == "tenant-abc"
        assert credential.framework == "openai"
        assert credential.credential_type == CredentialType.API_KEY

    def test_stored_credential_is_expired(self) -> None:
        """Test stored credential is_expired property."""
        credential = StoredCredential(
            credential_id="cred",
            tenant_id="tenant",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            encrypted_value=b"data",
            encryption_key_id="key",
            metadata=CredentialMetadata(expires_at=datetime.now(timezone.utc) - timedelta(hours=1)),
        )
        assert credential.is_expired is True

    def test_stored_credential_needs_rotation(self) -> None:
        """Test stored credential needs_rotation property."""
        credential = StoredCredential(
            credential_id="cred",
            tenant_id="tenant",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            encrypted_value=b"data",
            encryption_key_id="key",
            metadata=CredentialMetadata(
                created_at=datetime.now(timezone.utc) - timedelta(days=100)
            ),
            rotation_policy=RotationPolicy(interval_days=90),
        )
        assert credential.needs_rotation is True

    def test_stored_credential_to_dict(self) -> None:
        """Test stored credential serialization."""
        credential = StoredCredential(
            credential_id="cred-123",
            tenant_id="tenant",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            encrypted_value=b"data",
            encryption_key_id="key",
            allowed_agents=["agent1"],
        )
        result = credential.to_dict()
        assert result["credential_id"] == "cred-123"
        assert "encrypted_value" not in result  # Excluded by default
        assert result["allowed_agents"] == ["agent1"]

    def test_stored_credential_to_dict_with_encrypted(self) -> None:
        """Test stored credential serialization with encrypted value."""
        credential = StoredCredential(
            credential_id="cred",
            tenant_id="tenant",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            encrypted_value=b"data",
            encryption_key_id="key",
        )
        result = credential.to_dict(include_encrypted=True)
        assert "encrypted_value" in result

    def test_stored_credential_from_dict(self) -> None:
        """Test stored credential deserialization."""
        data = {
            "credential_id": "cred-123",
            "tenant_id": "tenant-abc",
            "framework": "anthropic",
            "credential_type": "api_key",
            "encrypted_value": "ZGF0YQ==",  # base64 encoded "data"
            "encryption_key_id": "key-456",
        }
        credential = StoredCredential.from_dict(data)
        assert credential.credential_id == "cred-123"
        assert credential.framework == "anthropic"


# =============================================================================
# CredentialRateLimiter Tests
# =============================================================================


class TestCredentialRateLimiter:
    """Tests for CredentialRateLimiter."""

    def test_rate_limiter_creation(self) -> None:
        """Test rate limiter creation."""
        limiter = CredentialRateLimiter(
            max_per_minute=10,
            max_per_hour=50,
            lockout_duration_seconds=120,
        )
        assert limiter.max_per_minute == 10
        assert limiter.max_per_hour == 50
        assert limiter.lockout_duration == 120

    def test_rate_limiter_allows_requests(self) -> None:
        """Test rate limiter allows requests within limits."""
        limiter = CredentialRateLimiter(max_per_minute=10, max_per_hour=100)

        for _ in range(5):
            allowed, retry_after = limiter.check_rate_limit("user-123", "tenant-abc")
            assert allowed is True
            assert retry_after == 0

    def test_rate_limiter_blocks_over_minute_limit(self) -> None:
        """Test rate limiter blocks when minute limit exceeded."""
        limiter = CredentialRateLimiter(max_per_minute=5, max_per_hour=100)

        # Consume all minute quota
        for _ in range(5):
            limiter.check_rate_limit("user-123")

        # Next request should be blocked
        allowed, retry_after = limiter.check_rate_limit("user-123")
        assert allowed is False
        assert retry_after > 0

    def test_rate_limiter_blocks_over_hour_limit(self) -> None:
        """Test rate limiter blocks when hour limit exceeded."""
        limiter = CredentialRateLimiter(max_per_minute=1000, max_per_hour=5)

        # Consume all hour quota
        for _ in range(5):
            limiter.check_rate_limit("user-123")

        # Next request should be blocked
        allowed, retry_after = limiter.check_rate_limit("user-123")
        assert allowed is False
        assert retry_after > 0

    def test_rate_limiter_different_users(self) -> None:
        """Test rate limiter tracks users independently."""
        limiter = CredentialRateLimiter(max_per_minute=2, max_per_hour=10)

        # User 1 consumes quota
        limiter.check_rate_limit("user-1")
        limiter.check_rate_limit("user-1")
        allowed1, _ = limiter.check_rate_limit("user-1")

        # User 2 should still be allowed
        allowed2, _ = limiter.check_rate_limit("user-2")

        assert allowed1 is False
        assert allowed2 is True

    def test_rate_limiter_clear_user(self) -> None:
        """Test clearing rate limit state for user."""
        limiter = CredentialRateLimiter(max_per_minute=2)

        # Consume quota
        limiter.check_rate_limit("user-123")
        limiter.check_rate_limit("user-123")

        # Clear state
        limiter.clear_user("user-123")

        # Should be allowed again
        allowed, _ = limiter.check_rate_limit("user-123")
        assert allowed is True

    def test_rate_limiter_tenant_isolation(self) -> None:
        """Test rate limiter tracks tenants independently."""
        limiter = CredentialRateLimiter(max_per_minute=2)

        # Same user, different tenants
        limiter.check_rate_limit("user-1", "tenant-A")
        limiter.check_rate_limit("user-1", "tenant-A")

        # Should be blocked for tenant A
        allowed_a, _ = limiter.check_rate_limit("user-1", "tenant-A")

        # Should be allowed for tenant B
        allowed_b, _ = limiter.check_rate_limit("user-1", "tenant-B")

        assert allowed_a is False
        assert allowed_b is True


# =============================================================================
# Mock Authorization Context
# =============================================================================


class MockAuthContext:
    """Mock authorization context for testing."""

    def __init__(
        self,
        user_id: str = "test-user",
        org_id: str | None = None,
        roles: set[str] | None = None,
        permissions: set[str] | None = None,
    ):
        self.user_id = user_id
        self.org_id = org_id
        self.roles = roles or set()
        self.permissions = permissions or set()

    def has_permission(self, permission_key: str) -> bool:
        return permission_key in self.permissions

    def has_role(self, role_name: str) -> bool:
        return role_name in self.roles


# =============================================================================
# CredentialVault Tests
# =============================================================================


class TestCredentialVaultCreation:
    """Tests for CredentialVault creation."""

    def test_vault_creation_default(self) -> None:
        """Test vault creation with defaults."""
        vault = CredentialVault()
        assert vault._rate_limiter is not None

    def test_vault_creation_with_kms_provider(self) -> None:
        """Test vault creation with KMS provider."""
        mock_kms = MagicMock()
        vault = CredentialVault(kms_provider=mock_kms)
        assert vault._kms_provider == mock_kms

    def test_vault_creation_with_audit_logger(self) -> None:
        """Test vault creation with audit logger."""
        mock_logger = AsyncMock()
        vault = CredentialVault(audit_logger=mock_logger)
        assert vault._audit_logger == mock_logger

    def test_vault_creation_with_rate_limiter(self) -> None:
        """Test vault creation with custom rate limiter."""
        custom_limiter = CredentialRateLimiter(max_per_minute=5)
        vault = CredentialVault(rate_limiter=custom_limiter)
        assert vault._rate_limiter == custom_limiter


# =============================================================================
# CredentialVault Encryption Tests
# =============================================================================


class TestCredentialVaultEncryption:
    """Tests for CredentialVault encryption/decryption."""

    @pytest.mark.asyncio
    async def test_encrypt_decrypt_roundtrip(self) -> None:
        """Test encryption and decryption roundtrip."""
        vault = CredentialVault()
        key = await vault._get_tenant_key("test-tenant")
        original = "super-secret-api-key-12345"

        encrypted = vault._encrypt(original, key)
        decrypted = vault._decrypt(encrypted, key)

        assert decrypted == original

    @pytest.mark.asyncio
    async def test_encrypt_produces_different_ciphertext(self) -> None:
        """Test encryption produces different ciphertext each time."""
        vault = CredentialVault()
        key = await vault._get_tenant_key("test-tenant")
        original = "test-value"

        encrypted1 = vault._encrypt(original, key)
        encrypted2 = vault._encrypt(original, key)

        assert encrypted1 != encrypted2

    @pytest.mark.asyncio
    async def test_tenant_keys_are_isolated(self) -> None:
        """Test different tenants get different keys."""
        vault = CredentialVault()

        key_a = await vault._get_tenant_key("tenant-A")
        key_b = await vault._get_tenant_key("tenant-B")

        assert key_a != key_b

    @pytest.mark.asyncio
    async def test_tenant_key_is_cached(self) -> None:
        """Test tenant key is cached."""
        vault = CredentialVault()

        key1 = await vault._get_tenant_key("tenant-X")
        key2 = await vault._get_tenant_key("tenant-X")

        assert key1 == key2


# =============================================================================
# CredentialVault Security - No Fallback Tests
# =============================================================================


class TestCredentialVaultSecurityNoFallback:
    """Tests for security behavior: encryption MUST fail without cryptography."""

    @pytest.mark.asyncio
    async def test_encrypt_raises_without_crypto(self) -> None:
        """Test encryption raises EncryptionError without cryptography."""
        vault = CredentialVault()
        key = await vault._get_tenant_key("test-tenant")

        with patch("aragora.gateway.openclaw.credential_vault.CRYPTO_AVAILABLE", False):
            with pytest.raises(EncryptionError) as exc_info:
                vault._encrypt("test-value", key)
            assert "cryptography library is required" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_decrypt_raises_without_crypto(self) -> None:
        """Test decryption raises EncryptionError without cryptography."""
        vault = CredentialVault()
        key = await vault._get_tenant_key("test-tenant")

        with patch("aragora.gateway.openclaw.credential_vault.CRYPTO_AVAILABLE", False):
            with pytest.raises(EncryptionError) as exc_info:
                vault._decrypt(b"encrypted-data", key)
            assert "cryptography library is required" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_store_fails_without_crypto(self) -> None:
        """Test storing credential fails without cryptography."""
        vault = CredentialVault()

        with patch("aragora.gateway.openclaw.credential_vault.CRYPTO_AVAILABLE", False):
            with pytest.raises(EncryptionError):
                await vault.store_credential(
                    tenant_id="tenant",
                    framework="openai",
                    credential_type=CredentialType.API_KEY,
                    value="secret",
                )


# =============================================================================
# CredentialVault Store Tests
# =============================================================================


class TestCredentialVaultStore:
    """Tests for CredentialVault.store_credential method."""

    @pytest.mark.asyncio
    async def test_store_credential_success(self) -> None:
        """Test storing credential successfully."""
        vault = CredentialVault()
        auth_ctx = MockAuthContext(
            permissions={"credentials:create"},
            org_id="tenant-123",
        )

        cred_id = await vault.store_credential(
            tenant_id="tenant-123",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value="sk-test-key",
            auth_context=auth_ctx,
        )

        assert cred_id is not None
        assert cred_id.startswith("cred_")
        assert cred_id in vault._credentials

    @pytest.mark.asyncio
    async def test_store_credential_with_custom_id(self) -> None:
        """Test storing credential with custom ID."""
        vault = CredentialVault()

        cred_id = await vault.store_credential(
            tenant_id="tenant",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value="secret",
            credential_id="my-custom-id",
        )

        assert cred_id == "my-custom-id"

    @pytest.mark.asyncio
    async def test_store_credential_with_expiry(self) -> None:
        """Test storing credential with expiry."""
        vault = CredentialVault()

        cred_id = await vault.store_credential(
            tenant_id="tenant",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value="secret",
            expires_in=timedelta(hours=24),
        )

        credential = vault._credentials[cred_id]
        assert credential.metadata.expires_at is not None

    @pytest.mark.asyncio
    async def test_store_credential_permission_denied(self) -> None:
        """Test storing credential with insufficient permissions."""
        vault = CredentialVault()
        auth_ctx = MockAuthContext(permissions=set())  # No permissions

        with pytest.raises(CredentialAccessDeniedError) as exc_info:
            await vault.store_credential(
                tenant_id="tenant",
                framework="openai",
                credential_type=CredentialType.API_KEY,
                value="secret",
                auth_context=auth_ctx,
            )
        assert "credentials:create required" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_store_credential_tenant_isolation(self) -> None:
        """Test storing credential respects tenant isolation."""
        vault = CredentialVault()
        auth_ctx = MockAuthContext(
            permissions={"credentials:create"},
            org_id="tenant-A",
        )

        with pytest.raises(TenantIsolationError):
            await vault.store_credential(
                tenant_id="tenant-B",  # Different from auth context
                framework="openai",
                credential_type=CredentialType.API_KEY,
                value="secret",
                auth_context=auth_ctx,
            )


# =============================================================================
# CredentialVault Get Tests
# =============================================================================


class TestCredentialVaultGet:
    """Tests for CredentialVault get methods."""

    @pytest.mark.asyncio
    async def test_get_credential_success(self) -> None:
        """Test getting credential metadata."""
        vault = CredentialVault()

        cred_id = await vault.store_credential(
            tenant_id="tenant",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value="secret",
        )

        auth_ctx = MockAuthContext(
            permissions={"credentials:read"},
            org_id="tenant",
        )
        credential = await vault.get_credential(cred_id, auth_ctx)

        assert credential.credential_id == cred_id
        assert credential.framework == "openai"

    @pytest.mark.asyncio
    async def test_get_credential_not_found(self) -> None:
        """Test getting non-existent credential."""
        vault = CredentialVault()

        with pytest.raises(CredentialNotFoundError):
            await vault.get_credential("nonexistent-id")

    @pytest.mark.asyncio
    async def test_get_credential_value_success(self) -> None:
        """Test getting decrypted credential value."""
        vault = CredentialVault()

        cred_id = await vault.store_credential(
            tenant_id="tenant",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value="sk-test-secret-key",
        )

        auth_ctx = MockAuthContext(
            permissions={"credentials:read"},
            org_id="tenant",
        )
        value = await vault.get_credential_value(cred_id, auth_ctx)

        assert value == "sk-test-secret-key"

    @pytest.mark.asyncio
    async def test_get_credential_value_expired(self) -> None:
        """Test getting expired credential raises error."""
        vault = CredentialVault()

        cred_id = await vault.store_credential(
            tenant_id="tenant",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value="secret",
            expires_in=timedelta(seconds=-1),  # Already expired
        )

        auth_ctx = MockAuthContext(
            permissions={"credentials:read"},
            org_id="tenant",
        )

        with pytest.raises(CredentialExpiredError):
            await vault.get_credential_value(cred_id, auth_ctx)

    @pytest.mark.asyncio
    async def test_get_credential_value_rate_limited(self) -> None:
        """Test getting credential when rate limited."""
        limiter = CredentialRateLimiter(max_per_minute=1)
        vault = CredentialVault(rate_limiter=limiter)

        cred_id = await vault.store_credential(
            tenant_id="tenant",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value="secret",
        )

        auth_ctx = MockAuthContext(
            permissions={"credentials:read"},
            org_id="tenant",
        )

        # First request succeeds
        await vault.get_credential_value(cred_id, auth_ctx)

        # Second request should be rate limited
        with pytest.raises(CredentialRateLimitedError) as exc_info:
            await vault.get_credential_value(cred_id, auth_ctx)
        assert exc_info.value.retry_after_seconds > 0

    @pytest.mark.asyncio
    async def test_get_credential_value_agent_not_allowed(self) -> None:
        """Test getting credential with unauthorized agent."""
        vault = CredentialVault()

        cred_id = await vault.store_credential(
            tenant_id="tenant",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value="secret",
            allowed_agents=["allowed-agent"],
        )

        auth_ctx = MockAuthContext(
            permissions={"credentials:read"},
            org_id="tenant",
        )

        with pytest.raises(CredentialAccessDeniedError) as exc_info:
            await vault.get_credential_value(cred_id, auth_ctx, agent_name="other-agent")
        assert exc_info.value.reason == "agent_not_allowed"

    @pytest.mark.asyncio
    async def test_get_credential_value_capability_not_allowed(self) -> None:
        """Test getting credential with unauthorized capability."""
        vault = CredentialVault()

        cred_id = await vault.store_credential(
            tenant_id="tenant",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value="secret",
            allowed_capabilities=["text_generation"],
        )

        auth_ctx = MockAuthContext(
            permissions={"credentials:read"},
            org_id="tenant",
        )

        with pytest.raises(CredentialAccessDeniedError) as exc_info:
            await vault.get_credential_value(cred_id, auth_ctx, capability="code_execution")
        assert exc_info.value.reason == "capability_not_allowed"


# =============================================================================
# CredentialVault Update Tests
# =============================================================================


class TestCredentialVaultUpdate:
    """Tests for CredentialVault.update_credential method."""

    @pytest.mark.asyncio
    async def test_update_credential_success(self) -> None:
        """Test updating credential metadata."""
        vault = CredentialVault()

        cred_id = await vault.store_credential(
            tenant_id="tenant",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value="secret",
        )

        auth_ctx = MockAuthContext(
            permissions={"credentials:read", "credentials:update"},
            org_id="tenant",
        )

        updated = await vault.update_credential(
            cred_id,
            auth_ctx,
            description="Updated description",
            tags=["production"],
        )

        assert updated.metadata.description == "Updated description"
        assert "production" in updated.metadata.tags

    @pytest.mark.asyncio
    async def test_update_credential_permission_denied(self) -> None:
        """Test updating credential without permission."""
        vault = CredentialVault()

        cred_id = await vault.store_credential(
            tenant_id="tenant",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value="secret",
        )

        auth_ctx = MockAuthContext(
            permissions={"credentials:read"},  # No update permission
            org_id="tenant",
        )

        with pytest.raises(CredentialAccessDeniedError):
            await vault.update_credential(cred_id, auth_ctx, description="New")


# =============================================================================
# CredentialVault Delete Tests
# =============================================================================


class TestCredentialVaultDelete:
    """Tests for CredentialVault.delete_credential method."""

    @pytest.mark.asyncio
    async def test_delete_credential_success(self) -> None:
        """Test deleting credential."""
        vault = CredentialVault()

        cred_id = await vault.store_credential(
            tenant_id="tenant",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value="secret",
        )

        auth_ctx = MockAuthContext(
            permissions={"credentials:read", "credentials:delete"},
            org_id="tenant",
        )

        result = await vault.delete_credential(cred_id, auth_ctx)

        assert result is True
        assert cred_id not in vault._credentials

    @pytest.mark.asyncio
    async def test_delete_credential_permission_denied(self) -> None:
        """Test deleting credential without permission."""
        vault = CredentialVault()

        cred_id = await vault.store_credential(
            tenant_id="tenant",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value="secret",
        )

        auth_ctx = MockAuthContext(
            permissions={"credentials:read"},  # No delete permission
            org_id="tenant",
        )

        with pytest.raises(CredentialAccessDeniedError):
            await vault.delete_credential(cred_id, auth_ctx)


# =============================================================================
# CredentialVault Rotation Tests
# =============================================================================


class TestCredentialVaultRotation:
    """Tests for CredentialVault rotation methods."""

    @pytest.mark.asyncio
    async def test_rotate_credential_success(self) -> None:
        """Test rotating credential."""
        vault = CredentialVault()

        cred_id = await vault.store_credential(
            tenant_id="tenant",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value="old-secret",
        )

        auth_ctx = MockAuthContext(
            permissions={"credentials:read", "credentials:rotate"},
            org_id="tenant",
        )

        updated = await vault.rotate_credential(cred_id, "new-secret", auth_ctx)

        assert updated.metadata.version == 2
        assert updated.metadata.rotated_at is not None
        assert updated.metadata.access_count == 0

        # Verify new value
        value = await vault.get_credential_value(cred_id)
        assert value == "new-secret"

    @pytest.mark.asyncio
    async def test_rotate_credential_permission_denied(self) -> None:
        """Test rotating credential without permission."""
        vault = CredentialVault()

        cred_id = await vault.store_credential(
            tenant_id="tenant",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value="secret",
        )

        auth_ctx = MockAuthContext(
            permissions={"credentials:read"},  # No rotate permission
            org_id="tenant",
        )

        with pytest.raises(CredentialAccessDeniedError):
            await vault.rotate_credential(cred_id, "new-secret", auth_ctx)

    @pytest.mark.asyncio
    async def test_get_credentials_needing_rotation(self) -> None:
        """Test getting credentials that need rotation."""
        vault = CredentialVault()

        # Create credential with short rotation interval
        await vault.store_credential(
            tenant_id="tenant",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value="secret",
            rotation_policy=RotationPolicy(interval_days=0),  # Always needs rotation
        )

        auth_ctx = MockAuthContext(
            permissions={"credentials:list"},
            org_id="tenant",
        )

        # Manually set created_at to past
        cred = list(vault._credentials.values())[0]
        cred.metadata.created_at = datetime.now(timezone.utc) - timedelta(days=1)
        cred.rotation_policy = RotationPolicy(interval_days=1)

        needing_rotation = await vault.get_credentials_needing_rotation(
            tenant_id="tenant",
            auth_context=auth_ctx,
        )

        assert len(needing_rotation) == 1


# =============================================================================
# CredentialVault List Tests
# =============================================================================


class TestCredentialVaultList:
    """Tests for CredentialVault.list_credentials method."""

    @pytest.mark.asyncio
    async def test_list_credentials_success(self) -> None:
        """Test listing credentials."""
        vault = CredentialVault()

        await vault.store_credential(
            tenant_id="tenant",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value="secret1",
        )
        await vault.store_credential(
            tenant_id="tenant",
            framework="anthropic",
            credential_type=CredentialType.API_KEY,
            value="secret2",
        )

        auth_ctx = MockAuthContext(
            permissions={"credentials:list"},
            org_id="tenant",
        )

        credentials = await vault.list_credentials(auth_context=auth_ctx)

        assert len(credentials) == 2

    @pytest.mark.asyncio
    async def test_list_credentials_filter_by_framework(self) -> None:
        """Test listing credentials filtered by framework."""
        vault = CredentialVault()

        await vault.store_credential(
            tenant_id="tenant",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value="secret1",
        )
        await vault.store_credential(
            tenant_id="tenant",
            framework="anthropic",
            credential_type=CredentialType.API_KEY,
            value="secret2",
        )

        auth_ctx = MockAuthContext(
            permissions={"credentials:list"},
            org_id="tenant",
        )

        credentials = await vault.list_credentials(
            framework="openai",
            auth_context=auth_ctx,
        )

        assert len(credentials) == 1
        assert credentials[0].framework == "openai"

    @pytest.mark.asyncio
    async def test_list_credentials_excludes_expired(self) -> None:
        """Test listing excludes expired credentials by default."""
        vault = CredentialVault()

        await vault.store_credential(
            tenant_id="tenant",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value="valid",
        )
        await vault.store_credential(
            tenant_id="tenant",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value="expired",
            expires_in=timedelta(seconds=-1),
        )

        auth_ctx = MockAuthContext(
            permissions={"credentials:list"},
            org_id="tenant",
        )

        credentials = await vault.list_credentials(auth_context=auth_ctx)

        assert len(credentials) == 1

    @pytest.mark.asyncio
    async def test_list_credentials_include_expired(self) -> None:
        """Test listing includes expired when requested."""
        vault = CredentialVault()

        await vault.store_credential(
            tenant_id="tenant",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value="valid",
        )
        await vault.store_credential(
            tenant_id="tenant",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value="expired",
            expires_in=timedelta(seconds=-1),
        )

        auth_ctx = MockAuthContext(
            permissions={"credentials:list"},
            org_id="tenant",
        )

        credentials = await vault.list_credentials(
            auth_context=auth_ctx,
            include_expired=True,
        )

        assert len(credentials) == 2


# =============================================================================
# CredentialVault Execution Tests
# =============================================================================


class TestCredentialVaultExecution:
    """Tests for CredentialVault.get_credentials_for_execution method."""

    @pytest.mark.asyncio
    async def test_get_credentials_for_execution(self) -> None:
        """Test getting credentials for execution."""
        vault = CredentialVault()

        await vault.store_credential(
            tenant_id="tenant",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value="sk-openai",
        )
        await vault.store_credential(
            tenant_id="tenant",
            framework="anthropic",
            credential_type=CredentialType.API_KEY,
            value="sk-anthropic",
        )

        auth_ctx = MockAuthContext(
            permissions={"credentials:list", "credentials:read"},
            org_id="tenant",
        )

        credentials = await vault.get_credentials_for_execution(
            tenant_id="tenant",
            auth_context=auth_ctx,
        )

        assert "openai" in credentials
        assert "anthropic" in credentials
        assert credentials["openai"] == "sk-openai"

    @pytest.mark.asyncio
    async def test_get_credentials_for_execution_filter_frameworks(self) -> None:
        """Test getting credentials filtered by frameworks."""
        vault = CredentialVault()

        await vault.store_credential(
            tenant_id="tenant",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value="sk-openai",
        )
        await vault.store_credential(
            tenant_id="tenant",
            framework="anthropic",
            credential_type=CredentialType.API_KEY,
            value="sk-anthropic",
        )

        auth_ctx = MockAuthContext(
            permissions={"credentials:list", "credentials:read"},
            org_id="tenant",
        )

        credentials = await vault.get_credentials_for_execution(
            tenant_id="tenant",
            frameworks=["openai"],
            auth_context=auth_ctx,
        )

        assert "openai" in credentials
        assert "anthropic" not in credentials


# =============================================================================
# CredentialVault Cleanup Tests
# =============================================================================


class TestCredentialVaultCleanup:
    """Tests for CredentialVault.cleanup_expired method."""

    @pytest.mark.asyncio
    async def test_cleanup_expired(self) -> None:
        """Test cleaning up expired credentials."""
        vault = CredentialVault()

        await vault.store_credential(
            tenant_id="tenant",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value="valid",
        )
        await vault.store_credential(
            tenant_id="tenant",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value="expired1",
            expires_in=timedelta(seconds=-1),
        )
        await vault.store_credential(
            tenant_id="tenant",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value="expired2",
            expires_in=timedelta(seconds=-100),
        )

        auth_ctx = MockAuthContext(
            permissions={"credentials:admin"},
        )

        count = await vault.cleanup_expired(auth_ctx)

        assert count == 2
        assert len(vault._credentials) == 1


# =============================================================================
# CredentialVault Stats Tests
# =============================================================================


class TestCredentialVaultStats:
    """Tests for CredentialVault.get_stats method."""

    @pytest.mark.asyncio
    async def test_get_stats(self) -> None:
        """Test getting vault statistics."""
        vault = CredentialVault()

        await vault.store_credential(
            tenant_id="tenant-A",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value="secret1",
        )
        await vault.store_credential(
            tenant_id="tenant-A",
            framework="anthropic",
            credential_type=CredentialType.API_KEY,
            value="secret2",
        )
        await vault.store_credential(
            tenant_id="tenant-B",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value="secret3",
        )

        stats = vault.get_stats()

        assert stats["total_credentials"] == 3
        assert stats["by_framework"]["openai"] == 2
        assert stats["by_framework"]["anthropic"] == 1
        assert stats["by_tenant"]["tenant-A"] == 2
        assert stats["by_tenant"]["tenant-B"] == 1


# =============================================================================
# Factory Functions Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_get_credential_vault_singleton(self) -> None:
        """Test get_credential_vault returns singleton."""
        reset_credential_vault()

        vault1 = get_credential_vault()
        vault2 = get_credential_vault()

        assert vault1 is vault2

    def test_reset_credential_vault(self) -> None:
        """Test reset_credential_vault clears singleton."""
        vault1 = get_credential_vault()
        reset_credential_vault()
        vault2 = get_credential_vault()

        assert vault1 is not vault2

    def test_init_credential_vault(self) -> None:
        """Test init_credential_vault creates new instance."""
        reset_credential_vault()

        mock_kms = MagicMock()
        vault = init_credential_vault(kms_provider=mock_kms)

        assert vault._kms_provider == mock_kms
        assert get_credential_vault() is vault


# =============================================================================
# CredentialAuditEvent Tests
# =============================================================================


class TestCredentialAuditEvent:
    """Tests for CredentialAuditEvent enum."""

    def test_audit_event_values(self) -> None:
        """Test audit event values."""
        assert CredentialAuditEvent.CREDENTIAL_CREATED.value == "credential_created"
        assert CredentialAuditEvent.CREDENTIAL_ACCESSED.value == "credential_accessed"
        assert CredentialAuditEvent.CREDENTIAL_UPDATED.value == "credential_updated"
        assert CredentialAuditEvent.CREDENTIAL_ROTATED.value == "credential_rotated"
        assert CredentialAuditEvent.CREDENTIAL_DELETED.value == "credential_deleted"
        assert CredentialAuditEvent.CREDENTIAL_EXPIRED.value == "credential_expired"
        assert CredentialAuditEvent.CREDENTIAL_ACCESS_DENIED.value == "credential_access_denied"
        assert CredentialAuditEvent.CREDENTIAL_RATE_LIMITED.value == "credential_rate_limited"


# =============================================================================
# Exception Tests
# =============================================================================


class TestExceptions:
    """Tests for exception classes."""

    def test_credential_vault_error(self) -> None:
        """Test base CredentialVaultError."""
        error = CredentialVaultError("Test error")
        assert str(error) == "Test error"

    def test_credential_not_found_error(self) -> None:
        """Test CredentialNotFoundError."""
        error = CredentialNotFoundError("Credential not found")
        assert isinstance(error, CredentialVaultError)

    def test_credential_access_denied_error(self) -> None:
        """Test CredentialAccessDeniedError."""
        error = CredentialAccessDeniedError(
            "Access denied",
            credential_id="cred-123",
            user_id="user-456",
            reason="permission_denied",
        )
        assert error.credential_id == "cred-123"
        assert error.user_id == "user-456"
        assert error.reason == "permission_denied"

    def test_credential_expired_error(self) -> None:
        """Test CredentialExpiredError."""
        error = CredentialExpiredError("Credential expired")
        assert isinstance(error, CredentialVaultError)

    def test_credential_rate_limited_error(self) -> None:
        """Test CredentialRateLimitedError."""
        error = CredentialRateLimitedError("Rate limited", retry_after_seconds=60)
        assert error.retry_after_seconds == 60

    def test_tenant_isolation_error(self) -> None:
        """Test TenantIsolationError."""
        error = TenantIsolationError("Cross-tenant access denied")
        assert isinstance(error, CredentialVaultError)

    def test_encryption_error(self) -> None:
        """Test EncryptionError."""
        error = EncryptionError("Encryption failed")
        assert isinstance(error, CredentialVaultError)


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Edge cases and error handling tests."""

    @pytest.mark.asyncio
    async def test_store_with_all_options(self) -> None:
        """Test storing credential with all options."""
        vault = CredentialVault()

        cred_id = await vault.store_credential(
            tenant_id="tenant",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value="secret",
            rotation_policy=RotationPolicy.strict(),
            expires_in=timedelta(days=30),
            allowed_agents=["agent1", "agent2"],
            allowed_capabilities=["text_generation"],
            description="Test credential",
            tags=["production", "critical"],
        )

        credential = vault._credentials[cred_id]
        assert credential.allowed_agents == ["agent1", "agent2"]
        assert credential.allowed_capabilities == ["text_generation"]
        assert credential.metadata.description == "Test credential"
        assert credential.rotation_policy.interval_days == 30

    @pytest.mark.asyncio
    async def test_concurrent_store_operations(self) -> None:
        """Test concurrent store operations."""
        vault = CredentialVault()

        async def store_credential(i: int):
            return await vault.store_credential(
                tenant_id="tenant",
                framework="openai",
                credential_type=CredentialType.API_KEY,
                value=f"secret-{i}",
            )

        tasks = [store_credential(i) for i in range(10)]
        cred_ids = await asyncio.gather(*tasks)

        # All should be unique
        assert len(set(cred_ids)) == 10
        assert len(vault._credentials) == 10

    @pytest.mark.asyncio
    async def test_admin_has_full_access(self) -> None:
        """Test admin role has cross-tenant access."""
        vault = CredentialVault()

        cred_id = await vault.store_credential(
            tenant_id="tenant-A",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value="secret",
        )

        # Admin can access any tenant
        auth_ctx = MockAuthContext(
            roles={"admin"},
            permissions={"credentials:read"},
            org_id="tenant-B",  # Different tenant
        )

        credential = await vault.get_credential(cred_id, auth_ctx)
        assert credential.credential_id == cred_id

    @pytest.mark.asyncio
    async def test_credentials_admin_permission_grants_all(self) -> None:
        """Test credentials:admin permission grants all operations."""
        vault = CredentialVault()

        cred_id = await vault.store_credential(
            tenant_id="tenant",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value="secret",
        )

        auth_ctx = MockAuthContext(
            permissions={"credentials:admin"},
            org_id="tenant",
        )

        # Should be able to read
        credential = await vault.get_credential(cred_id, auth_ctx)
        assert credential is not None

        # Should be able to get value
        value = await vault.get_credential_value(cred_id, auth_ctx)
        assert value == "secret"

    @pytest.mark.asyncio
    async def test_special_characters_in_credential(self) -> None:
        """Test special characters in credential value."""
        vault = CredentialVault()

        special_value = 'key="value"&token=abc123!@#$%^&*()\n\t\r'

        cred_id = await vault.store_credential(
            tenant_id="tenant",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value=special_value,
        )

        value = await vault.get_credential_value(cred_id)
        assert value == special_value

    @pytest.mark.asyncio
    async def test_unicode_in_credential(self) -> None:
        """Test unicode characters in credential value."""
        vault = CredentialVault()

        unicode_value = "secret-key-\u00e9\u00e8\u00ea-\u4e2d\u6587"

        cred_id = await vault.store_credential(
            tenant_id="tenant",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value=unicode_value,
        )

        value = await vault.get_credential_value(cred_id)
        assert value == unicode_value
