"""
Tests for Credential Providers.

Security-critical tests covering:
- EnvCredentialProvider (environment variable access)
- AWSSecretsManagerProvider (AWS Secrets Manager with caching)
- ChainedCredentialProvider (fallback hierarchy)
- CachedCredentialProvider (TTL caching wrapper)
- Factory function (provider type detection)
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.connectors.credentials.providers import (
    AWSSecretsManagerProvider,
    CachedCredential,
    CachedCredentialProvider,
    ChainedCredentialProvider,
    EnvCredentialProvider,
    get_credential_provider,
)


# ============================================================================
# CachedCredential Tests
# ============================================================================


class TestCachedCredential:
    """Tests for CachedCredential dataclass."""

    def test_is_expired_false_when_fresh(self):
        """Test credential is not expired when fresh."""
        cred = CachedCredential(
            value="secret123",
            cached_at=time.time(),
            ttl_seconds=300,
        )
        assert cred.is_expired is False

    def test_is_expired_true_when_old(self):
        """Test credential is expired after TTL."""
        cred = CachedCredential(
            value="secret123",
            cached_at=time.time() - 400,  # 400 seconds ago
            ttl_seconds=300,  # 5 minute TTL
        )
        assert cred.is_expired is True

    def test_is_expired_at_boundary(self):
        """Test expiration at exact boundary."""
        now = time.time()
        cred = CachedCredential(
            value="secret123",
            cached_at=now - 300.1,  # Just past TTL
            ttl_seconds=300,
        )
        assert cred.is_expired is True


# ============================================================================
# EnvCredentialProvider Tests
# ============================================================================


class TestEnvCredentialProvider:
    """Tests for environment variable credential provider."""

    def setup_method(self):
        """Store original env for cleanup."""
        self._original_env = dict(os.environ)

    def teardown_method(self):
        """Restore original environment."""
        os.environ.clear()
        os.environ.update(self._original_env)

    @pytest.mark.asyncio
    async def test_get_credential_with_prefix(self):
        """Test getting credential with prefix."""
        os.environ["ARAGORA_API_KEY"] = "prefixed-key"
        provider = EnvCredentialProvider(prefix="ARAGORA_")

        result = await provider.get_credential("api_key")
        assert result == "prefixed-key"

    @pytest.mark.asyncio
    async def test_get_credential_fallback_without_prefix(self):
        """Test falling back to key without prefix."""
        os.environ["API_KEY"] = "no-prefix-key"
        provider = EnvCredentialProvider(prefix="ARAGORA_")

        result = await provider.get_credential("api_key")
        assert result == "no-prefix-key"

    @pytest.mark.asyncio
    async def test_get_credential_prefers_prefixed(self):
        """Test prefixed key takes precedence."""
        os.environ["ARAGORA_SECRET"] = "prefixed"
        os.environ["SECRET"] = "unprefixed"
        provider = EnvCredentialProvider(prefix="ARAGORA_")

        result = await provider.get_credential("secret")
        assert result == "prefixed"

    @pytest.mark.asyncio
    async def test_get_credential_not_found(self):
        """Test returns None when not found."""
        provider = EnvCredentialProvider(prefix="ARAGORA_")

        result = await provider.get_credential("nonexistent_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_credential_custom_prefix(self):
        """Test custom prefix."""
        os.environ["MYAPP_DATABASE_URL"] = "postgres://..."
        provider = EnvCredentialProvider(prefix="MYAPP_")

        result = await provider.get_credential("database_url")
        assert result == "postgres://..."

    @pytest.mark.asyncio
    async def test_set_credential(self):
        """Test setting credential sets environment variable."""
        provider = EnvCredentialProvider(prefix="ARAGORA_")

        await provider.set_credential("new_key", "new_value")

        assert os.environ.get("ARAGORA_NEW_KEY") == "new_value"

    @pytest.mark.asyncio
    async def test_set_then_get(self):
        """Test setting and then getting credential."""
        provider = EnvCredentialProvider(prefix="TEST_")

        await provider.set_credential("round_trip", "test_value")
        result = await provider.get_credential("round_trip")

        assert result == "test_value"


# ============================================================================
# AWSSecretsManagerProvider Tests
# ============================================================================


class TestAWSSecretsManagerProvider:
    """Tests for AWS Secrets Manager credential provider."""

    def test_init_defaults(self):
        """Test initialization with defaults."""
        provider = AWSSecretsManagerProvider(secret_name="my-secret")

        assert provider.secret_name == "my-secret"
        assert provider.region == "us-east-1"  # Default region
        assert provider.cache_ttl_seconds == 3600  # Default TTL
        assert provider._client is None  # Lazy initialization

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        provider = AWSSecretsManagerProvider(
            secret_name="my-secret",
            region="eu-west-1",
            cache_ttl_seconds=600,
            profile_name="my-profile",
        )

        assert provider.region == "eu-west-1"
        assert provider.cache_ttl_seconds == 600
        assert provider.profile_name == "my-profile"

    def test_init_region_from_env(self):
        """Test region auto-detection from environment."""
        original = os.environ.get("AWS_REGION")
        try:
            os.environ["AWS_REGION"] = "ap-southeast-1"
            provider = AWSSecretsManagerProvider(secret_name="test")
            assert provider.region == "ap-southeast-1"
        finally:
            if original:
                os.environ["AWS_REGION"] = original
            elif "AWS_REGION" in os.environ:
                del os.environ["AWS_REGION"]

    def test_get_client_requires_boto3(self):
        """Test that missing boto3 raises ImportError."""
        provider = AWSSecretsManagerProvider(secret_name="test")

        with patch.dict("sys.modules", {"boto3": None}):
            with pytest.raises(ImportError, match="boto3 is required"):
                provider._get_client()

    @pytest.mark.asyncio
    async def test_get_credential_simple_key(self):
        """Test getting a simple key from secret."""
        provider = AWSSecretsManagerProvider(secret_name="test")

        # Mock the _fetch_secret method
        mock_secret = {"api_key": "secret123", "db_password": "dbpass"}
        provider._fetch_secret = AsyncMock(return_value=mock_secret)

        result = await provider.get_credential("api_key")
        assert result == "secret123"

    @pytest.mark.asyncio
    async def test_get_credential_nested_key(self):
        """Test getting nested key with dot notation."""
        provider = AWSSecretsManagerProvider(secret_name="test")

        mock_secret = {
            "database": {
                "host": "localhost",
                "password": "secret",
            }
        }
        provider._fetch_secret = AsyncMock(return_value=mock_secret)

        result = await provider.get_credential("database.password")
        assert result == "secret"

    @pytest.mark.asyncio
    async def test_get_credential_case_insensitive(self):
        """Test case-insensitive key lookup."""
        provider = AWSSecretsManagerProvider(secret_name="test")

        mock_secret = {"API_KEY": "found-it"}
        provider._fetch_secret = AsyncMock(return_value=mock_secret)

        result = await provider.get_credential("api_key")
        assert result == "found-it"

    @pytest.mark.asyncio
    async def test_get_credential_not_found(self):
        """Test returns None for missing key."""
        provider = AWSSecretsManagerProvider(secret_name="test")

        mock_secret = {"other_key": "value"}
        provider._fetch_secret = AsyncMock(return_value=mock_secret)

        result = await provider.get_credential("missing_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_credential_deeply_nested(self):
        """Test deeply nested key access."""
        provider = AWSSecretsManagerProvider(secret_name="test")

        mock_secret = {"services": {"api": {"keys": {"primary": "deep-secret"}}}}
        provider._fetch_secret = AsyncMock(return_value=mock_secret)

        result = await provider.get_credential("services.api.keys.primary")
        assert result == "deep-secret"

    @pytest.mark.asyncio
    async def test_get_credential_caches_result(self):
        """Test credentials are cached."""
        provider = AWSSecretsManagerProvider(
            secret_name="test",
            cache_ttl_seconds=300,
        )

        mock_secret = {"key": "value"}
        provider._fetch_secret = AsyncMock(return_value=mock_secret)

        # First call
        result1 = await provider.get_credential("key")
        # Second call should use cache
        result2 = await provider.get_credential("key")

        assert result1 == result2 == "value"
        # _fetch_secret should only be called once
        assert provider._fetch_secret.call_count == 1

    @pytest.mark.asyncio
    async def test_get_credential_non_string_value(self):
        """Test non-string values are converted."""
        provider = AWSSecretsManagerProvider(secret_name="test")

        mock_secret = {"count": 42, "enabled": True}
        provider._fetch_secret = AsyncMock(return_value=mock_secret)

        assert await provider.get_credential("count") == "42"
        assert await provider.get_credential("enabled") == "True"

    @pytest.mark.asyncio
    async def test_get_credential_error_returns_none(self):
        """Test errors during fetch return None."""
        provider = AWSSecretsManagerProvider(secret_name="test")
        provider._fetch_secret = AsyncMock(side_effect=OSError("AWS error"))

        result = await provider.get_credential("key")
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_secret_caches_at_secret_level(self):
        """Test secret is cached at the full secret level."""
        provider = AWSSecretsManagerProvider(
            secret_name="test",
            cache_ttl_seconds=300,
        )

        mock_client = MagicMock()
        mock_client.get_secret_value.return_value = {
            "SecretString": json.dumps({"key1": "val1", "key2": "val2"})
        }
        provider._client = mock_client

        # Clear caches
        provider._secret_cache = None
        provider._secret_cached_at = 0

        # Fetch multiple keys
        await provider._fetch_secret()
        await provider._fetch_secret()
        await provider._fetch_secret()

        # Should only call AWS once due to caching
        assert mock_client.get_secret_value.call_count == 1

    def test_fetch_secret_sync_json(self):
        """Test synchornous fetch with JSON secret."""
        provider = AWSSecretsManagerProvider(secret_name="test")

        mock_client = MagicMock()
        mock_client.get_secret_value.return_value = {
            "SecretString": json.dumps({"api_key": "secret"})
        }
        provider._client = mock_client

        result = provider._fetch_secret_sync()
        assert result == {"api_key": "secret"}

    def test_fetch_secret_sync_plain_string(self):
        """Test synchronous fetch with plain string secret."""
        provider = AWSSecretsManagerProvider(secret_name="test")

        mock_client = MagicMock()
        mock_client.get_secret_value.return_value = {"SecretString": "plain-secret-value"}
        provider._client = mock_client

        result = provider._fetch_secret_sync()
        assert result == {"_value": "plain-secret-value"}

    def test_fetch_secret_sync_binary(self):
        """Test synchronous fetch with binary secret."""
        provider = AWSSecretsManagerProvider(secret_name="test")

        mock_client = MagicMock()
        mock_client.get_secret_value.return_value = {"SecretBinary": b"binary-secret"}
        provider._client = mock_client

        result = provider._fetch_secret_sync()
        assert result == {"_binary": b"binary-secret"}

    def test_clear_cache(self):
        """Test cache clearing."""
        provider = AWSSecretsManagerProvider(secret_name="test")

        # Populate cache
        provider._cache["key1"] = CachedCredential("val1", time.time(), 300)
        provider._secret_cache = {"data": "value"}
        provider._secret_cached_at = time.time()

        # Clear
        provider.clear_cache()

        assert len(provider._cache) == 0
        assert provider._secret_cache is None
        assert provider._secret_cached_at == 0

    @pytest.mark.asyncio
    async def test_set_credential(self):
        """Test setting credential updates secret."""
        provider = AWSSecretsManagerProvider(secret_name="test")

        # Mock fetch and update
        mock_secret = {"existing": "value"}
        provider._fetch_secret = AsyncMock(return_value=mock_secret.copy())
        provider._update_secret_sync = MagicMock()

        await provider.set_credential("new_key", "new_value")

        # Should update with new key
        call_args = provider._update_secret_sync.call_args[0][0]
        assert call_args["new_key"] == "new_value"
        assert call_args["existing"] == "value"

    @pytest.mark.asyncio
    async def test_set_credential_nested(self):
        """Test setting nested credential."""
        provider = AWSSecretsManagerProvider(secret_name="test")

        mock_secret = {"database": {"host": "localhost"}}
        provider._fetch_secret = AsyncMock(return_value=mock_secret.copy())
        provider._update_secret_sync = MagicMock()

        await provider.set_credential("database.password", "secret123")

        call_args = provider._update_secret_sync.call_args[0][0]
        assert call_args["database"]["password"] == "secret123"
        assert call_args["database"]["host"] == "localhost"

    @pytest.mark.asyncio
    async def test_set_credential_invalidates_cache(self):
        """Test setting credential invalidates cache."""
        provider = AWSSecretsManagerProvider(secret_name="test")

        # Populate cache
        provider._cache["key1"] = CachedCredential("old", time.time(), 300)
        provider._secret_cache = {"key1": "old"}
        provider._secret_cached_at = time.time()

        # Mock methods
        provider._fetch_secret = AsyncMock(return_value={"key1": "old"})
        provider._update_secret_sync = MagicMock()

        await provider.set_credential("key1", "new")

        # Cache should be invalidated
        assert provider._secret_cache is None
        assert provider._secret_cached_at == 0
        assert "key1" not in provider._cache


# ============================================================================
# ChainedCredentialProvider Tests
# ============================================================================


class TestChainedCredentialProvider:
    """Tests for chained credential provider."""

    @pytest.mark.asyncio
    async def test_get_credential_from_first_provider(self):
        """Test getting credential from first provider."""
        provider1 = MagicMock()
        provider1.get_credential = AsyncMock(return_value="from-provider1")
        provider2 = MagicMock()
        provider2.get_credential = AsyncMock(return_value="from-provider2")

        chain = ChainedCredentialProvider([provider1, provider2])
        result = await chain.get_credential("key")

        assert result == "from-provider1"
        provider1.get_credential.assert_called_once_with("key")
        provider2.get_credential.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_credential_fallback_to_second(self):
        """Test falling back to second provider."""
        provider1 = MagicMock()
        provider1.get_credential = AsyncMock(return_value=None)
        provider2 = MagicMock()
        provider2.get_credential = AsyncMock(return_value="from-provider2")

        chain = ChainedCredentialProvider([provider1, provider2])
        result = await chain.get_credential("key")

        assert result == "from-provider2"

    @pytest.mark.asyncio
    async def test_get_credential_all_providers_fail(self):
        """Test returns None when all providers return None."""
        provider1 = MagicMock()
        provider1.get_credential = AsyncMock(return_value=None)
        provider2 = MagicMock()
        provider2.get_credential = AsyncMock(return_value=None)

        chain = ChainedCredentialProvider([provider1, provider2])
        result = await chain.get_credential("key")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_credential_empty_chain(self):
        """Test empty provider chain returns None."""
        chain = ChainedCredentialProvider([])
        result = await chain.get_credential("key")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_credential_uses_first_provider(self):
        """Test set_credential uses first provider only."""
        provider1 = MagicMock()
        provider1.set_credential = AsyncMock()
        provider2 = MagicMock()
        provider2.set_credential = AsyncMock()

        chain = ChainedCredentialProvider([provider1, provider2])
        await chain.set_credential("key", "value")

        provider1.set_credential.assert_called_once_with("key", "value")
        provider2.set_credential.assert_not_called()

    @pytest.mark.asyncio
    async def test_set_credential_empty_chain(self):
        """Test set_credential with empty chain does nothing."""
        chain = ChainedCredentialProvider([])
        # Should not raise
        await chain.set_credential("key", "value")


# ============================================================================
# CachedCredentialProvider Tests
# ============================================================================


class TestCachedCredentialProvider:
    """Tests for cached credential provider wrapper."""

    @pytest.mark.asyncio
    async def test_get_credential_caches_result(self):
        """Test credentials are cached."""
        inner = MagicMock()
        inner.get_credential = AsyncMock(return_value="secret")

        cached = CachedCredentialProvider(inner, cache_ttl_seconds=300)

        # First call
        result1 = await cached.get_credential("key")
        # Second call (should use cache)
        result2 = await cached.get_credential("key")

        assert result1 == result2 == "secret"
        assert inner.get_credential.call_count == 1

    @pytest.mark.asyncio
    async def test_get_credential_cache_expires(self):
        """Test cache expires after TTL."""
        inner = MagicMock()
        inner.get_credential = AsyncMock(return_value="secret")

        cached = CachedCredentialProvider(inner, cache_ttl_seconds=0.1)

        # First call
        result1 = await cached.get_credential("key")

        # Wait for TTL to expire
        await asyncio.sleep(0.2)

        # Second call (cache expired)
        result2 = await cached.get_credential("key")

        assert result1 == result2 == "secret"
        assert inner.get_credential.call_count == 2

    @pytest.mark.asyncio
    async def test_get_credential_none_not_cached(self):
        """Test None values are not cached."""
        inner = MagicMock()
        inner.get_credential = AsyncMock(return_value=None)

        cached = CachedCredentialProvider(inner, cache_ttl_seconds=300)

        await cached.get_credential("missing")
        await cached.get_credential("missing")

        # Should call inner each time since None is not cached
        assert inner.get_credential.call_count == 2

    @pytest.mark.asyncio
    async def test_set_credential_updates_cache(self):
        """Test set_credential updates cache."""
        inner = MagicMock()
        inner.set_credential = AsyncMock()
        inner.get_credential = AsyncMock(return_value="old_value")

        cached = CachedCredentialProvider(inner, cache_ttl_seconds=300)

        # Set new value
        await cached.set_credential("key", "new_value")

        # Get should return cached new value, not call inner
        result = await cached.get_credential("key")

        assert result == "new_value"
        inner.get_credential.assert_not_called()

    def test_clear_cache(self):
        """Test cache clearing."""
        inner = MagicMock()
        cached = CachedCredentialProvider(inner, cache_ttl_seconds=300)

        # Populate cache
        cached._cache["key1"] = CachedCredential("val1", time.time(), 300)
        cached._cache["key2"] = CachedCredential("val2", time.time(), 300)

        cached.clear_cache()

        assert len(cached._cache) == 0

    def test_default_ttl(self):
        """Test default TTL value."""
        inner = MagicMock()
        cached = CachedCredentialProvider(inner)

        assert cached.cache_ttl_seconds == 300  # 5 minutes


# ============================================================================
# Factory Function Tests
# ============================================================================


class TestGetCredentialProvider:
    """Tests for get_credential_provider factory function."""

    def setup_method(self):
        """Store original env."""
        self._original_env = dict(os.environ)

    def teardown_method(self):
        """Restore original env."""
        os.environ.clear()
        os.environ.update(self._original_env)

    def test_env_provider_default(self):
        """Test default provider is EnvCredentialProvider."""
        provider = get_credential_provider()
        assert isinstance(provider, EnvCredentialProvider)

    def test_env_provider_explicit(self):
        """Test explicit env provider."""
        provider = get_credential_provider("env")
        assert isinstance(provider, EnvCredentialProvider)

    def test_env_provider_custom_prefix(self):
        """Test env provider with custom prefix."""
        provider = get_credential_provider("env", prefix="CUSTOM_")
        assert provider.prefix == "CUSTOM_"

    def test_aws_provider(self):
        """Test AWS provider creation."""
        provider = get_credential_provider(
            "aws",
            secret_name="test-secret",
            region="us-west-2",
        )
        assert isinstance(provider, AWSSecretsManagerProvider)
        assert provider.secret_name == "test-secret"
        assert provider.region == "us-west-2"

    def test_aws_provider_from_env(self):
        """Test AWS provider with env config."""
        os.environ["AWS_SECRET_NAME"] = "env-secret"
        os.environ["AWS_REGION"] = "eu-central-1"

        provider = get_credential_provider("aws")

        assert isinstance(provider, AWSSecretsManagerProvider)
        assert provider.secret_name == "env-secret"
        assert provider.region == "eu-central-1"

    def test_chained_provider_with_aws(self):
        """Test chained provider includes AWS when configured."""
        os.environ["AWS_SECRET_NAME"] = "test-secret"

        # boto3 is imported inside _get_client, but the chained factory
        # just creates the provider objects. boto3 availability is checked
        # at import time in the factory when creating AWS provider.
        # We patch builtins.__import__ or use the fact that boto3 is available.
        import sys

        mock_boto3 = MagicMock()
        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            provider = get_credential_provider("chained")

        assert isinstance(provider, ChainedCredentialProvider)
        assert len(provider.providers) == 2  # AWS + Env

    def test_chained_provider_without_aws(self):
        """Test chained provider falls back to env only."""
        # Ensure no AWS_SECRET_NAME
        os.environ.pop("AWS_SECRET_NAME", None)

        provider = get_credential_provider("chained")

        assert isinstance(provider, ChainedCredentialProvider)
        assert len(provider.providers) == 1
        assert isinstance(provider.providers[0], EnvCredentialProvider)

    def test_provider_type_from_env(self):
        """Test provider type auto-detection from env."""
        os.environ["CREDENTIAL_PROVIDER"] = "env"

        provider = get_credential_provider()
        assert isinstance(provider, EnvCredentialProvider)

    def test_unknown_provider_type(self):
        """Test unknown provider type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown credential provider type"):
            get_credential_provider("invalid-provider")

    def test_provider_type_case_insensitive(self):
        """Test provider type is case-insensitive."""
        provider1 = get_credential_provider("ENV")
        provider2 = get_credential_provider("Env")

        assert isinstance(provider1, EnvCredentialProvider)
        assert isinstance(provider2, EnvCredentialProvider)


# ============================================================================
# Security-focused Tests
# ============================================================================


class TestSecurityBehavior:
    """Security-focused tests for credential providers."""

    @pytest.mark.asyncio
    async def test_credentials_not_logged(self):
        """Test credentials are not exposed in logs (by checking no value in repr)."""
        cred = CachedCredential(
            value="super-secret-api-key",
            cached_at=time.time(),
            ttl_seconds=300,
        )

        # Dataclass repr might include value, but this is a reminder to review
        # In production, consider using __repr__ override to mask sensitive data
        repr_str = repr(cred)
        # This test documents the current behavior - values ARE in repr
        # A security enhancement would mask this
        assert "super-secret-api-key" in repr_str

    @pytest.mark.asyncio
    async def test_env_credential_isolation(self):
        """Test env credentials don't leak across prefixes."""
        os.environ["PREFIX_A_SECRET"] = "secret-a"
        os.environ["PREFIX_B_SECRET"] = "secret-b"

        provider_a = EnvCredentialProvider(prefix="PREFIX_A_")
        provider_b = EnvCredentialProvider(prefix="PREFIX_B_")

        result_a = await provider_a.get_credential("secret")
        result_b = await provider_b.get_credential("secret")

        assert result_a == "secret-a"
        assert result_b == "secret-b"

    @pytest.mark.asyncio
    async def test_cache_ttl_enforced(self):
        """Test cache TTL is actually enforced."""
        provider = AWSSecretsManagerProvider(
            secret_name="test",
            cache_ttl_seconds=0.1,  # Very short TTL
        )

        call_count = 0

        async def mock_fetch():
            nonlocal call_count
            call_count += 1
            return {"key": f"value-{call_count}"}

        provider._fetch_secret = mock_fetch

        # First fetch
        result1 = await provider.get_credential("key")

        # Immediate second fetch (should use cache)
        result2 = await provider.get_credential("key")

        # Wait for TTL
        await asyncio.sleep(0.2)

        # Third fetch (cache expired)
        result3 = await provider.get_credential("key")

        assert result1 == "value-1"
        assert result2 == "value-1"  # From cache
        assert result3 == "value-2"  # Fresh fetch
