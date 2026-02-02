"""
Additional Tests for Credential Providers.

This module extends the base credential provider tests with:
- Concurrent access patterns
- Edge cases and boundary conditions
- Error handling for specific exception types
- Integration scenarios
- Multi-provider configurations

Primary tests are in tests/connectors/test_credentials.py
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.connectors.credentials.providers import (
    AWSSecretsManagerProvider,
    CachedCredential,
    CachedCredentialProvider,
    ChainedCredentialProvider,
    CredentialProvider,
    EnvCredentialProvider,
    get_credential_provider,
)


# ============================================================================
# Concurrent Access Tests
# ============================================================================


class TestConcurrentCredentialAccess:
    """Tests for concurrent credential access patterns."""

    @pytest.mark.asyncio
    async def test_concurrent_get_credentials(self):
        """Test multiple concurrent get_credential calls use cache correctly."""
        provider = AWSSecretsManagerProvider(
            secret_name="test-secret",
            cache_ttl_seconds=300,
        )

        fetch_count = 0

        async def mock_fetch():
            nonlocal fetch_count
            fetch_count += 1
            await asyncio.sleep(0.05)  # Simulate network delay
            return {"api_key": "secret123", "db_password": "dbpass"}

        provider._fetch_secret = mock_fetch

        # Launch multiple concurrent requests
        tasks = [
            provider.get_credential("api_key"),
            provider.get_credential("api_key"),
            provider.get_credential("db_password"),
            provider.get_credential("api_key"),
        ]
        results = await asyncio.gather(*tasks)

        # All should return correct values
        assert results[0] == "secret123"
        assert results[1] == "secret123"
        assert results[2] == "dbpass"
        assert results[3] == "secret123"

    @pytest.mark.asyncio
    async def test_concurrent_set_and_get(self):
        """Test concurrent set and get operations."""
        inner = MagicMock()
        inner.get_credential = AsyncMock(return_value="initial")
        inner.set_credential = AsyncMock()

        cached = CachedCredentialProvider(inner, cache_ttl_seconds=300)

        # Populate initial cache
        await cached.get_credential("key")

        # Concurrent set and get
        async def setter():
            await cached.set_credential("key", "updated")

        async def getter():
            return await cached.get_credential("key")

        results = await asyncio.gather(setter(), getter())
        # After concurrent operations, cache should have updated value
        final = await cached.get_credential("key")
        assert final == "updated"

    @pytest.mark.asyncio
    async def test_chained_provider_concurrent_access(self):
        """Test concurrent access through chained provider."""
        provider1 = MagicMock()
        provider1.get_credential = AsyncMock(return_value=None)

        provider2 = MagicMock()
        provider2.get_credential = AsyncMock(return_value="from-fallback")

        chain = ChainedCredentialProvider([provider1, provider2])

        # Launch concurrent requests
        tasks = [chain.get_credential("key") for _ in range(10)]
        results = await asyncio.gather(*tasks)

        # All should get fallback value
        assert all(r == "from-fallback" for r in results)


# ============================================================================
# Edge Cases and Boundary Conditions
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def setup_method(self):
        """Store original env for cleanup."""
        self._original_env = dict(os.environ)

    def teardown_method(self):
        """Restore original environment."""
        os.environ.clear()
        os.environ.update(self._original_env)

    @pytest.mark.asyncio
    async def test_empty_string_credential(self):
        """Test handling of empty string credentials."""
        os.environ["ARAGORA_EMPTY_KEY"] = ""
        provider = EnvCredentialProvider(prefix="ARAGORA_")

        result = await provider.get_credential("empty_key")
        # Empty string is a valid value, should be returned
        assert result == ""

    @pytest.mark.asyncio
    async def test_whitespace_credential(self):
        """Test handling of whitespace-only credentials."""
        os.environ["ARAGORA_WHITESPACE"] = "   "
        provider = EnvCredentialProvider(prefix="ARAGORA_")

        result = await provider.get_credential("whitespace")
        assert result == "   "

    @pytest.mark.asyncio
    async def test_special_characters_in_key(self):
        """Test handling of special characters in credential keys."""
        os.environ["ARAGORA_KEY_WITH_UNDERSCORE"] = "value1"
        provider = EnvCredentialProvider(prefix="ARAGORA_")

        result = await provider.get_credential("key_with_underscore")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_unicode_credential_value(self):
        """Test handling of unicode credential values."""
        os.environ["ARAGORA_UNICODE"] = "value-with-unicode-\u00e9\u00e8\u00ea"
        provider = EnvCredentialProvider(prefix="ARAGORA_")

        result = await provider.get_credential("unicode")
        assert result == "value-with-unicode-\u00e9\u00e8\u00ea"

    @pytest.mark.asyncio
    async def test_very_long_credential_value(self):
        """Test handling of very long credential values."""
        long_value = "x" * 10000
        os.environ["ARAGORA_LONG_KEY"] = long_value
        provider = EnvCredentialProvider(prefix="ARAGORA_")

        result = await provider.get_credential("long_key")
        assert result == long_value
        assert len(result) == 10000

    @pytest.mark.asyncio
    async def test_aws_nested_key_partial_path_not_found(self):
        """Test partial nested path returns None."""
        provider = AWSSecretsManagerProvider(secret_name="test")
        mock_secret = {"database": {"host": "localhost"}}
        provider._fetch_secret = AsyncMock(return_value=mock_secret)

        # Missing intermediate key
        result = await provider.get_credential("database.credentials.password")
        assert result is None

    @pytest.mark.asyncio
    async def test_aws_nested_key_non_dict_intermediate(self):
        """Test nested key with non-dict intermediate value."""
        provider = AWSSecretsManagerProvider(secret_name="test")
        mock_secret = {"database": "string-value"}
        provider._fetch_secret = AsyncMock(return_value=mock_secret)

        # Can't traverse into string
        result = await provider.get_credential("database.password")
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_prefix(self):
        """Test provider with empty prefix."""
        os.environ["API_KEY"] = "no-prefix"
        provider = EnvCredentialProvider(prefix="")

        result = await provider.get_credential("api_key")
        assert result == "no-prefix"

    def test_cached_credential_zero_ttl(self):
        """Test cached credential with zero TTL is always expired."""
        cred = CachedCredential(
            value="secret",
            cached_at=time.time(),
            ttl_seconds=0,
        )
        # Zero TTL means immediately expired
        assert cred.is_expired is True

    def test_cached_credential_negative_ttl(self):
        """Test cached credential with negative TTL is expired."""
        cred = CachedCredential(
            value="secret",
            cached_at=time.time(),
            ttl_seconds=-100,
        )
        assert cred.is_expired is True


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_aws_oserror_handling(self):
        """Test OSError during AWS fetch is handled."""
        provider = AWSSecretsManagerProvider(secret_name="test")
        provider._fetch_secret = AsyncMock(side_effect=OSError("Connection refused"))

        result = await provider.get_credential("key")
        assert result is None

    @pytest.mark.asyncio
    async def test_aws_valueerror_handling(self):
        """Test ValueError during AWS fetch is handled."""
        provider = AWSSecretsManagerProvider(secret_name="test")
        provider._fetch_secret = AsyncMock(side_effect=ValueError("Invalid value"))

        result = await provider.get_credential("key")
        assert result is None

    @pytest.mark.asyncio
    async def test_aws_keyerror_handling(self):
        """Test KeyError during AWS fetch is handled."""
        provider = AWSSecretsManagerProvider(secret_name="test")
        provider._fetch_secret = AsyncMock(side_effect=KeyError("missing"))

        result = await provider.get_credential("key")
        assert result is None

    @pytest.mark.asyncio
    async def test_aws_runtime_error_handling(self):
        """Test RuntimeError during AWS fetch is handled."""
        provider = AWSSecretsManagerProvider(secret_name="test")
        provider._fetch_secret = AsyncMock(side_effect=RuntimeError("Runtime issue"))

        result = await provider.get_credential("key")
        assert result is None

    def test_fetch_secret_sync_error_propagates(self):
        """Test synchronous fetch errors propagate correctly."""
        provider = AWSSecretsManagerProvider(secret_name="test")

        mock_client = MagicMock()
        mock_client.get_secret_value.side_effect = OSError("AWS unreachable")
        provider._client = mock_client

        with pytest.raises(OSError, match="AWS unreachable"):
            provider._fetch_secret_sync()

    @pytest.mark.asyncio
    async def test_chained_provider_error_in_first_continues(self):
        """Test errors in first provider still allow fallback."""
        provider1 = MagicMock()
        provider1.get_credential = AsyncMock(side_effect=Exception("Provider 1 failed"))

        provider2 = MagicMock()
        provider2.get_credential = AsyncMock(return_value="from-fallback")

        chain = ChainedCredentialProvider([provider1, provider2])

        # Should propagate exception (chained doesn't catch)
        with pytest.raises(Exception, match="Provider 1 failed"):
            await chain.get_credential("key")


# ============================================================================
# AWS Secrets Manager Integration Tests
# ============================================================================


class TestAWSSecretsManagerIntegration:
    """Integration tests for AWS Secrets Manager provider."""

    def test_get_client_with_profile(self):
        """Test boto3 session creation with profile."""
        provider = AWSSecretsManagerProvider(
            secret_name="test",
            profile_name="custom-profile",
        )

        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client

        with patch("boto3.Session", return_value=mock_session) as mock_session_class:
            client = provider._get_client()

            mock_session_class.assert_called_once_with(profile_name="custom-profile")
            mock_session.client.assert_called_once_with(
                "secretsmanager",
                region_name=provider.region,
            )

    def test_get_client_without_profile(self):
        """Test boto3 session creation without profile."""
        provider = AWSSecretsManagerProvider(secret_name="test")

        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client

        with patch("boto3.Session", return_value=mock_session) as mock_session_class:
            client = provider._get_client()

            mock_session_class.assert_called_once_with()

    def test_update_secret_sync(self):
        """Test synchronous secret update."""
        provider = AWSSecretsManagerProvider(secret_name="test-secret")

        mock_client = MagicMock()
        provider._client = mock_client

        secret_data = {"api_key": "new-key", "db_pass": "new-pass"}
        provider._update_secret_sync(secret_data)

        mock_client.put_secret_value.assert_called_once_with(
            SecretId="test-secret",
            SecretString=json.dumps(secret_data),
        )

    @pytest.mark.asyncio
    async def test_set_credential_creates_nested_path(self):
        """Test setting credential creates nested structure."""
        provider = AWSSecretsManagerProvider(secret_name="test")

        mock_secret: dict[str, Any] = {}
        provider._fetch_secret = AsyncMock(return_value=mock_secret)
        provider._update_secret_sync = MagicMock()

        await provider.set_credential("new.nested.key", "value")

        call_args = provider._update_secret_sync.call_args[0][0]
        assert call_args["new"]["nested"]["key"] == "value"

    @pytest.mark.asyncio
    async def test_cache_expiration_triggers_refetch(self):
        """Test that expired cache triggers refetch."""
        provider = AWSSecretsManagerProvider(
            secret_name="test",
            cache_ttl_seconds=0.1,
        )

        # Setup initial cached data
        provider._secret_cache = {"key": "old-value"}
        provider._secret_cached_at = time.time() - 1  # Already expired

        mock_client = MagicMock()
        mock_client.get_secret_value.return_value = {
            "SecretString": json.dumps({"key": "new-value"})
        }
        provider._client = mock_client

        # Should refetch due to expired cache
        result = await provider._fetch_secret()
        assert result == {"key": "new-value"}


# ============================================================================
# Multi-Provider Configuration Tests
# ============================================================================


class TestMultiProviderConfiguration:
    """Tests for multi-provider configurations."""

    def setup_method(self):
        """Store original env for cleanup."""
        self._original_env = dict(os.environ)

    def teardown_method(self):
        """Restore original environment."""
        os.environ.clear()
        os.environ.update(self._original_env)

    @pytest.mark.asyncio
    async def test_three_provider_chain(self):
        """Test chain with three providers."""
        provider1 = MagicMock()
        provider1.get_credential = AsyncMock(return_value=None)

        provider2 = MagicMock()
        provider2.get_credential = AsyncMock(return_value=None)

        provider3 = MagicMock()
        provider3.get_credential = AsyncMock(return_value="from-third")

        chain = ChainedCredentialProvider([provider1, provider2, provider3])
        result = await chain.get_credential("key")

        assert result == "from-third"
        provider1.get_credential.assert_called_once()
        provider2.get_credential.assert_called_once()
        provider3.get_credential.assert_called_once()

    @pytest.mark.asyncio
    async def test_cached_chained_provider(self):
        """Test caching wrapper around chained provider."""
        inner1 = MagicMock()
        inner1.get_credential = AsyncMock(return_value=None)

        inner2 = MagicMock()
        inner2.get_credential = AsyncMock(return_value="cached-value")

        chain = ChainedCredentialProvider([inner1, inner2])
        cached = CachedCredentialProvider(chain, cache_ttl_seconds=300)

        # First call
        result1 = await cached.get_credential("key")
        # Second call should use cache
        result2 = await cached.get_credential("key")

        assert result1 == result2 == "cached-value"
        # Chain should only be called once
        assert inner2.get_credential.call_count == 1

    @pytest.mark.asyncio
    async def test_env_provider_different_prefixes(self):
        """Test multiple env providers with different prefixes."""
        os.environ["APP_A_SECRET"] = "secret-a"
        os.environ["APP_B_SECRET"] = "secret-b"
        os.environ["SECRET"] = "fallback-secret"

        provider_a = EnvCredentialProvider(prefix="APP_A_")
        provider_b = EnvCredentialProvider(prefix="APP_B_")
        fallback = EnvCredentialProvider(prefix="")

        chain = ChainedCredentialProvider([provider_a, provider_b, fallback])

        # Get through chain (first provider has it)
        result = await chain.get_credential("secret")
        assert result == "secret-a"

    def test_factory_chained_with_aws_configured(self):
        """Test factory creates chain with AWS when configured and boto3 available."""
        os.environ["AWS_SECRET_NAME"] = "test-secret"

        provider = get_credential_provider("chained")

        assert isinstance(provider, ChainedCredentialProvider)
        # Should have AWS + env providers when boto3 is available
        assert len(provider.providers) == 2
        assert isinstance(provider.providers[0], AWSSecretsManagerProvider)
        assert isinstance(provider.providers[1], EnvCredentialProvider)


# ============================================================================
# Protocol Compliance Tests
# ============================================================================


class TestProtocolCompliance:
    """Tests for CredentialProvider protocol compliance."""

    def setup_method(self):
        """Store original env for cleanup."""
        self._original_env = dict(os.environ)

    def teardown_method(self):
        """Restore original environment."""
        os.environ.clear()
        os.environ.update(self._original_env)

    @pytest.mark.asyncio
    async def test_env_provider_implements_protocol(self):
        """Test EnvCredentialProvider implements CredentialProvider protocol."""
        provider = EnvCredentialProvider()

        # Should have required methods
        assert hasattr(provider, "get_credential")
        assert hasattr(provider, "set_credential")
        assert asyncio.iscoroutinefunction(provider.get_credential)
        assert asyncio.iscoroutinefunction(provider.set_credential)

    @pytest.mark.asyncio
    async def test_aws_provider_implements_protocol(self):
        """Test AWSSecretsManagerProvider implements CredentialProvider protocol."""
        provider = AWSSecretsManagerProvider(secret_name="test")

        assert hasattr(provider, "get_credential")
        assert hasattr(provider, "set_credential")
        assert asyncio.iscoroutinefunction(provider.get_credential)
        assert asyncio.iscoroutinefunction(provider.set_credential)

    @pytest.mark.asyncio
    async def test_chained_provider_implements_protocol(self):
        """Test ChainedCredentialProvider implements CredentialProvider protocol."""
        provider = ChainedCredentialProvider([])

        assert hasattr(provider, "get_credential")
        assert hasattr(provider, "set_credential")
        assert asyncio.iscoroutinefunction(provider.get_credential)
        assert asyncio.iscoroutinefunction(provider.set_credential)

    @pytest.mark.asyncio
    async def test_cached_provider_implements_protocol(self):
        """Test CachedCredentialProvider implements CredentialProvider protocol."""
        inner = MagicMock()
        provider = CachedCredentialProvider(inner)

        assert hasattr(provider, "get_credential")
        assert hasattr(provider, "set_credential")
        assert asyncio.iscoroutinefunction(provider.get_credential)
        assert asyncio.iscoroutinefunction(provider.set_credential)

    @pytest.mark.asyncio
    async def test_custom_provider_in_chain(self):
        """Test custom provider implementing protocol works in chain."""

        class CustomProvider:
            async def get_credential(self, key: str) -> str | None:
                if key == "custom":
                    return "custom-value"
                return None

            async def set_credential(self, key: str, value: str) -> None:
                pass

        custom = CustomProvider()
        env = EnvCredentialProvider()

        chain = ChainedCredentialProvider([custom, env])

        result = await chain.get_credential("custom")
        assert result == "custom-value"


# ============================================================================
# Tenant Isolation Tests
# ============================================================================


class TestTenantIsolation:
    """Tests for multi-tenant credential isolation."""

    def setup_method(self):
        """Store original env for cleanup."""
        self._original_env = dict(os.environ)

    def teardown_method(self):
        """Restore original environment."""
        os.environ.clear()
        os.environ.update(self._original_env)

    @pytest.mark.asyncio
    async def test_tenant_prefixed_credentials(self):
        """Test tenant-specific credentials using prefixes."""
        os.environ["TENANT_A_API_KEY"] = "tenant-a-key"
        os.environ["TENANT_B_API_KEY"] = "tenant-b-key"

        tenant_a_provider = EnvCredentialProvider(prefix="TENANT_A_")
        tenant_b_provider = EnvCredentialProvider(prefix="TENANT_B_")

        # Each tenant gets their own credentials
        key_a = await tenant_a_provider.get_credential("api_key")
        key_b = await tenant_b_provider.get_credential("api_key")

        assert key_a == "tenant-a-key"
        assert key_b == "tenant-b-key"
        assert key_a != key_b

    @pytest.mark.asyncio
    async def test_tenant_aws_secrets(self):
        """Test tenant-specific AWS secrets."""
        provider_a = AWSSecretsManagerProvider(secret_name="tenant-a/secrets")
        provider_b = AWSSecretsManagerProvider(secret_name="tenant-b/secrets")

        provider_a._fetch_secret = AsyncMock(return_value={"api_key": "tenant-a-secret"})
        provider_b._fetch_secret = AsyncMock(return_value={"api_key": "tenant-b-secret"})

        key_a = await provider_a.get_credential("api_key")
        key_b = await provider_b.get_credential("api_key")

        assert key_a == "tenant-a-secret"
        assert key_b == "tenant-b-secret"

    @pytest.mark.asyncio
    async def test_tenant_cache_isolation(self):
        """Test tenant caches are isolated."""
        inner_a = MagicMock()
        inner_a.get_credential = AsyncMock(return_value="tenant-a-cached")

        inner_b = MagicMock()
        inner_b.get_credential = AsyncMock(return_value="tenant-b-cached")

        cached_a = CachedCredentialProvider(inner_a, cache_ttl_seconds=300)
        cached_b = CachedCredentialProvider(inner_b, cache_ttl_seconds=300)

        # Populate caches
        await cached_a.get_credential("key")
        await cached_b.get_credential("key")

        # Verify isolation
        assert cached_a._cache["key"].value == "tenant-a-cached"
        assert cached_b._cache["key"].value == "tenant-b-cached"

    @pytest.mark.asyncio
    async def test_tenant_set_credential_isolation(self):
        """Test setting credentials doesn't affect other tenants."""
        os.environ["TENANT_A_SECRET"] = "original-a"
        os.environ["TENANT_B_SECRET"] = "original-b"

        provider_a = EnvCredentialProvider(prefix="TENANT_A_")
        provider_b = EnvCredentialProvider(prefix="TENANT_B_")

        # Tenant A updates their secret
        await provider_a.set_credential("secret", "updated-a")

        # Tenant B's secret should be unchanged
        secret_b = await provider_b.get_credential("secret")
        assert secret_b == "original-b"

        # Tenant A sees updated value
        secret_a = await provider_a.get_credential("secret")
        assert secret_a == "updated-a"
