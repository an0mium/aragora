"""
Tests for scheduler rotation handlers.

Tests base handler, API key, database, encryption, and OAuth handlers.
"""

import base64
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.scheduler.rotation_handlers.base import (
    RotationError,
    RotationHandler,
    RotationResult,
    RotationStatus,
)
from aragora.scheduler.rotation_handlers.api_key import APIKeyRotationHandler
from aragora.scheduler.rotation_handlers.database import DatabaseRotationHandler
from aragora.scheduler.rotation_handlers.encryption import EncryptionKeyRotationHandler
from aragora.scheduler.rotation_handlers.oauth import OAuthRotationHandler


# =============================================================================
# Base Classes Tests
# =============================================================================


class TestRotationStatus:
    """Test RotationStatus enum."""

    def test_all_statuses(self):
        assert RotationStatus.SUCCESS.value == "success"
        assert RotationStatus.FAILED.value == "failed"
        assert RotationStatus.PARTIAL.value == "partial"
        assert RotationStatus.SKIPPED.value == "skipped"
        assert RotationStatus.PENDING.value == "pending"

    def test_status_count(self):
        assert len(RotationStatus) == 5


class TestRotationResult:
    """Test RotationResult dataclass."""

    def test_success_result(self):
        result = RotationResult(
            status=RotationStatus.SUCCESS,
            secret_id="test-key",
            secret_type="api_key",
            old_version="v1",
            new_version="v2",
        )
        assert result.status == RotationStatus.SUCCESS
        assert result.secret_id == "test-key"
        assert result.error_message is None

    def test_failed_result(self):
        result = RotationResult(
            status=RotationStatus.FAILED,
            secret_id="test-key",
            secret_type="api_key",
            error_message="Validation failed",
        )
        assert result.status == RotationStatus.FAILED
        assert result.error_message == "Validation failed"

    def test_to_dict(self):
        result = RotationResult(
            status=RotationStatus.SUCCESS,
            secret_id="test-key",
            secret_type="api_key",
            old_version="v1",
            new_version="v2",
        )
        d = result.to_dict()
        assert d["status"] == "success"
        assert d["secret_id"] == "test-key"
        assert d["secret_type"] == "api_key"
        assert d["old_version"] == "v1"
        assert d["new_version"] == "v2"
        assert "rotated_at" in d

    def test_to_dict_with_grace_period(self):
        grace = datetime(2024, 1, 15, 12, 0, 0)
        result = RotationResult(
            status=RotationStatus.SUCCESS,
            secret_id="test-key",
            secret_type="api_key",
            grace_period_ends=grace,
        )
        d = result.to_dict()
        assert d["grace_period_ends"] == grace.isoformat()

    def test_to_dict_without_grace_period(self):
        result = RotationResult(
            status=RotationStatus.SUCCESS,
            secret_id="test-key",
            secret_type="api_key",
        )
        d = result.to_dict()
        assert d["grace_period_ends"] is None

    def test_default_metadata(self):
        result = RotationResult(
            status=RotationStatus.SUCCESS,
            secret_id="test-key",
            secret_type="api_key",
        )
        assert result.metadata == {}


class TestRotationError:
    """Test RotationError exception."""

    def test_basic_error(self):
        err = RotationError("test error")
        assert str(err) == "test error"
        assert err.secret_id is None

    def test_error_with_secret_id(self):
        err = RotationError("test error", secret_id="key-123")
        assert str(err) == "test error"
        assert err.secret_id == "key-123"


# =============================================================================
# Concrete Test Handler
# =============================================================================


class SimpleHandler(RotationHandler):
    """Simple concrete handler for testing base class behavior."""

    @property
    def secret_type(self):
        return "test"

    async def generate_new_credentials(self, secret_id, metadata):
        return "new-credential-value", {**metadata, "version": "v2"}

    async def validate_credentials(self, secret_id, secret_value, metadata):
        return secret_value == "new-credential-value"

    async def revoke_old_credentials(self, secret_id, old_value, metadata):
        return True


class TestRotationHandler:
    """Test RotationHandler base class lifecycle."""

    @pytest.fixture
    def handler(self):
        return SimpleHandler(grace_period_hours=24, max_retries=3)

    @pytest.mark.asyncio
    async def test_rotate_success(self, handler):
        result = await handler.rotate("secret-1", "old-value", {"version": "v1"})
        assert result.status == RotationStatus.SUCCESS
        assert result.secret_id == "secret-1"
        assert result.secret_type == "test"
        assert result.old_version == "v1"
        assert result.grace_period_ends is not None

    @pytest.mark.asyncio
    async def test_rotate_validation_failure(self, handler):
        # Override validate to return False
        handler.validate_credentials = AsyncMock(return_value=False)

        result = await handler.rotate("secret-1", "old-value")
        assert result.status == RotationStatus.FAILED
        assert "validation failed" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_rotate_generation_error(self, handler):
        handler.generate_new_credentials = AsyncMock(
            side_effect=RotationError("generation failed", "secret-1")
        )

        result = await handler.rotate("secret-1", "old-value")
        assert result.status == RotationStatus.FAILED
        assert "generation failed" in result.error_message

    @pytest.mark.asyncio
    async def test_rotate_none_metadata(self, handler):
        result = await handler.rotate("secret-1", "old-value", None)
        assert result.status == RotationStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_cleanup_expired_success(self, handler):
        result = await handler.cleanup_expired("secret-1", "old-value")
        assert result.status == RotationStatus.SUCCESS
        assert result.metadata["action"] == "revocation"

    @pytest.mark.asyncio
    async def test_cleanup_expired_partial(self, handler):
        handler.revoke_old_credentials = AsyncMock(return_value=False)

        result = await handler.cleanup_expired("secret-1", "old-value")
        assert result.status == RotationStatus.PARTIAL

    @pytest.mark.asyncio
    async def test_cleanup_expired_error(self, handler):
        handler.revoke_old_credentials = AsyncMock(side_effect=RuntimeError("revocation error"))

        result = await handler.cleanup_expired("secret-1", "old-value")
        assert result.status == RotationStatus.FAILED
        assert "revocation error" in result.error_message

    def test_handler_properties(self, handler):
        assert handler.grace_period_hours == 24
        assert handler.max_retries == 3
        assert handler.secret_type == "test"


# =============================================================================
# APIKeyRotationHandler Tests
# =============================================================================


class TestAPIKeyRotationHandler:
    """Test API key rotation handler."""

    @pytest.fixture
    def handler(self):
        return APIKeyRotationHandler()

    def test_secret_type(self, handler):
        assert handler.secret_type == "api_key"

    def test_default_grace_period(self, handler):
        assert handler.grace_period_hours == 48

    @pytest.mark.asyncio
    async def test_manual_rotation_with_new_key(self, handler):
        result = await handler.rotate(
            "ANTHROPIC_API_KEY",
            "old-key",
            {"provider": "anthropic", "new_key": "sk-ant-new-key"},
        )
        assert result.status == RotationStatus.SUCCESS
        assert result.metadata["new_value"] == "sk-ant-new-key"
        assert result.metadata["rotation_method"] == "manual"

    @pytest.mark.asyncio
    async def test_manual_rotation_requires_notification(self, handler):
        """Providers without programmatic rotation should raise RotationError."""
        result = await handler.rotate(
            "ANTHROPIC_API_KEY",
            "old-key",
            {"provider": "anthropic"},
        )
        # Without a new_key, Anthropic requires manual rotation
        assert result.status == RotationStatus.FAILED
        assert "manual" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_notification_callback_called(self):
        callback = AsyncMock()
        handler = APIKeyRotationHandler(notification_callback=callback)

        await handler.rotate(
            "MISTRAL_API_KEY",
            "old-key",
            {"provider": "mistral"},
        )
        callback.assert_called_once()
        assert callback.call_args.kwargs["secret_id"] == "MISTRAL_API_KEY"

    @pytest.mark.asyncio
    async def test_validate_unknown_provider(self, handler):
        result = await handler.validate_credentials(
            "CUSTOM_KEY", "some-key", {"provider": "custom_provider"}
        )
        # Unknown providers return True (no validator)
        assert result is True

    @pytest.mark.asyncio
    async def test_revoke_non_openai(self, handler):
        result = await handler.revoke_old_credentials(
            "ANTHROPIC_KEY", "old-key", {"provider": "anthropic"}
        )
        # Non-OpenAI providers return True (no programmatic revocation)
        assert result is True

    @pytest.mark.asyncio
    async def test_rotate_with_new_key_convenience(self, handler):
        result = await handler.rotate_with_new_key(
            "API_KEY",
            "new-key-123",
            current_value="old-key",
            metadata={"provider": "anthropic"},
        )
        assert result.status == RotationStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_github_requires_manual(self, handler):
        result = await handler.rotate(
            "GITHUB_TOKEN",
            "old-token",
            {"provider": "github"},
        )
        assert result.status == RotationStatus.FAILED
        assert "manual" in result.error_message.lower()


# =============================================================================
# DatabaseRotationHandler Tests
# =============================================================================


class TestDatabaseRotationHandler:
    """Test database rotation handler."""

    @pytest.fixture
    def handler(self):
        return DatabaseRotationHandler()

    def test_secret_type(self, handler):
        assert handler.secret_type == "database"

    def test_default_password_length(self, handler):
        assert handler.password_length == 32

    def test_generate_password(self, handler):
        pw1 = handler._generate_password()
        pw2 = handler._generate_password()
        assert len(pw1) == 32
        assert pw1 != pw2  # Random, should be unique

    def test_custom_password_length(self):
        handler = DatabaseRotationHandler(password_length=64)
        pw = handler._generate_password()
        assert len(pw) == 64

    @pytest.mark.asyncio
    async def test_generate_requires_username(self, handler):
        with pytest.raises(RotationError, match="Username required"):
            await handler.generate_new_credentials("db-cred", {"db_type": "postgresql"})

    @pytest.mark.asyncio
    async def test_unsupported_db_type(self, handler):
        with pytest.raises(RotationError, match="Unsupported database type"):
            await handler.generate_new_credentials(
                "db-cred",
                {"db_type": "mongodb", "username": "testuser"},
            )

    @pytest.mark.asyncio
    async def test_revoke_is_noop(self, handler):
        result = await handler.revoke_old_credentials("db-cred", "old-pw", {})
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_without_username(self, handler):
        result = await handler.validate_credentials("db-cred", "pw", {})
        assert result is False


# =============================================================================
# EncryptionKeyRotationHandler Tests
# =============================================================================


class TestEncryptionKeyRotationHandler:
    """Test encryption key rotation handler."""

    @pytest.fixture
    def handler(self):
        return EncryptionKeyRotationHandler()

    def test_secret_type(self, handler):
        assert handler.secret_type == "encryption_key"

    def test_default_key_length(self, handler):
        assert handler.key_length == 32

    def test_default_grace_period(self, handler):
        assert handler.grace_period_hours == 168  # 7 days

    def test_generate_key(self, handler):
        key1 = handler._generate_key()
        key2 = handler._generate_key()
        assert len(key1) == 32
        assert key1 != key2

    @pytest.mark.asyncio
    async def test_generate_aes256_key(self, handler):
        value, meta = await handler.generate_new_credentials("ENC_KEY", {"key_type": "aes256"})
        # Should be base64 encoded
        key_bytes = base64.b64decode(value)
        assert len(key_bytes) == 32
        assert meta["key_type"] == "aes256"
        assert meta["key_length"] == 32

    @pytest.mark.asyncio
    async def test_generate_aes128_key(self, handler):
        value, meta = await handler.generate_new_credentials("ENC_KEY", {"key_type": "aes128"})
        key_bytes = base64.b64decode(value)
        assert len(key_bytes) == 16

    @pytest.mark.asyncio
    async def test_generate_hmac_key(self, handler):
        value, meta = await handler.generate_new_credentials(
            "HMAC_KEY", {"key_type": "hmac", "key_length": 64}
        )
        key_bytes = base64.b64decode(value)
        assert len(key_bytes) == 64

    @pytest.mark.asyncio
    async def test_generate_jwt_key(self, handler):
        value, meta = await handler.generate_new_credentials(
            "JWT_KEY", {"key_type": "jwt", "jwt_algorithm": "HS256"}
        )
        key_bytes = base64.b64decode(value)
        assert len(key_bytes) == 64  # 512 bits for HMAC JWT

    @pytest.mark.asyncio
    async def test_validate_valid_key(self, handler):
        value, meta = await handler.generate_new_credentials("ENC_KEY", {"key_type": "aes256"})
        is_valid = await handler.validate_credentials("ENC_KEY", value, meta)
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_validate_wrong_length(self, handler):
        short_key = base64.b64encode(b"too-short").decode()
        is_valid = await handler.validate_credentials("ENC_KEY", short_key, {"key_length": 32})
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_validate_invalid_base64(self, handler):
        is_valid = await handler.validate_credentials(
            "ENC_KEY", "not-valid-base64!!!", {"key_length": 32}
        )
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_validate_hmac_key(self, handler):
        value, meta = await handler.generate_new_credentials(
            "HMAC_KEY", {"key_type": "hmac", "key_length": 32}
        )
        is_valid = await handler.validate_credentials("HMAC_KEY", value, meta)
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_revoke_archives_key(self, handler):
        result = await handler.revoke_old_credentials("ENC_KEY", "old-key", {"key_id": "kid-123"})
        assert result is True

    @pytest.mark.asyncio
    async def test_trigger_reencryption(self, handler):
        result = await handler.trigger_reencryption(
            "ENC_KEY",
            "old-key-b64",
            "new-key-b64",
            {"data_stores": ["postgres", "redis"]},
        )
        assert result["secret_id"] == "ENC_KEY"
        assert "postgres" in result["stores"]
        assert "redis" in result["stores"]

    @pytest.mark.asyncio
    async def test_trigger_reencryption_empty_stores(self, handler):
        result = await handler.trigger_reencryption("ENC_KEY", "old", "new", {"data_stores": []})
        assert result["stores"] == {}

    @pytest.mark.asyncio
    async def test_full_rotation_lifecycle(self, handler):
        result = await handler.rotate(
            "ENC_KEY_PRIMARY",
            "old-value",
            {"key_type": "aes256"},
        )
        assert result.status == RotationStatus.SUCCESS
        assert result.grace_period_ends is not None


# =============================================================================
# OAuthRotationHandler Tests
# =============================================================================


class TestOAuthRotationHandler:
    """Test OAuth rotation handler."""

    @pytest.fixture
    def handler(self):
        return OAuthRotationHandler()

    def test_secret_type(self, handler):
        assert handler.secret_type == "oauth"

    def test_default_grace_period(self, handler):
        assert handler.grace_period_hours == 1

    def test_token_urls(self, handler):
        assert "googleapis.com" in handler._get_token_url("google", {})
        assert "microsoftonline.com" in handler._get_token_url("microsoft", {})
        assert "github.com" in handler._get_token_url("github", {})
        assert "slack.com" in handler._get_token_url("slack", {})
        assert "discord.com" in handler._get_token_url("discord", {})

    def test_custom_token_url(self, handler):
        url = handler._get_token_url("custom", {"token_url": "https://auth.example.com/token"})
        assert url == "https://auth.example.com/token"

    def test_microsoft_tenant_url(self, handler):
        url = handler._get_token_url("microsoft", {"tenant": "my-tenant"})
        assert "my-tenant" in url

    def test_validation_urls(self, handler):
        assert handler._get_validation_url("google", {}) is not None
        assert handler._get_validation_url("microsoft", {}) is not None
        assert handler._get_validation_url("github", {}) is not None
        assert handler._get_validation_url("slack", {}) is not None

    def test_custom_validation_url(self, handler):
        url = handler._get_validation_url(
            "custom", {"validation_url": "https://api.example.com/me"}
        )
        assert url == "https://api.example.com/me"

    def test_revoke_urls(self, handler):
        assert handler._get_revoke_url("google", {}) is not None
        assert handler._get_revoke_url("github", {}) is None  # GitHub doesn't support
        assert handler._get_revoke_url("slack", {}) is not None

    def test_custom_revoke_url(self, handler):
        url = handler._get_revoke_url("custom", {"revoke_url": "https://api.example.com/revoke"})
        assert url == "https://api.example.com/revoke"

    @pytest.mark.asyncio
    async def test_generate_requires_refresh_token(self, handler):
        with pytest.raises(RotationError, match="Refresh token required"):
            await handler.generate_new_credentials("oauth-1", {"provider": "google"})

    @pytest.mark.asyncio
    async def test_generate_requires_client_credentials(self, handler):
        with pytest.raises(RotationError, match="Client credentials required"):
            await handler.generate_new_credentials(
                "oauth-1",
                {"provider": "google", "refresh_token": "rt-123"},
            )

    @pytest.mark.asyncio
    async def test_validate_no_validation_url(self, handler):
        # Unknown provider with no validation URL returns True
        result = await handler.validate_credentials("oauth-1", "token", {"provider": "unknown"})
        assert result is True

    @pytest.mark.asyncio
    async def test_revoke_no_revoke_url(self, handler):
        # Provider without revoke URL returns True
        result = await handler.revoke_old_credentials(
            "oauth-1", "old-token", {"provider": "github"}
        )
        assert result is True


# =============================================================================
# Integration Tests
# =============================================================================


class TestRotationHandlerIntegration:
    """Test cross-handler patterns."""

    def test_all_handlers_have_secret_type(self):
        api = APIKeyRotationHandler()
        db = DatabaseRotationHandler()
        enc = EncryptionKeyRotationHandler()
        oauth = OAuthRotationHandler()

        types = {api.secret_type, db.secret_type, enc.secret_type, oauth.secret_type}
        assert types == {"api_key", "database", "encryption_key", "oauth"}

    @pytest.mark.asyncio
    async def test_encryption_full_lifecycle(self):
        handler = EncryptionKeyRotationHandler(grace_period_hours=1)

        # Rotate
        result = await handler.rotate("ENC_KEY", None, {"key_type": "aes256"})
        assert result.status == RotationStatus.SUCCESS
        new_key = result.metadata["new_value"]

        # Validate
        is_valid = await handler.validate_credentials(
            "ENC_KEY",
            new_key,
            result.metadata,
        )
        assert is_valid is True

        # Cleanup
        cleanup = await handler.cleanup_expired("ENC_KEY", "old-key-b64", {})
        assert cleanup.status == RotationStatus.SUCCESS
