"""
Credential Provider Implementations.

Provides credential management with support for:
- Environment variables (default)
- AWS Secrets Manager with automatic refresh
- Chained providers (fallback hierarchy)
- Caching with TTL
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


class CredentialProvider(Protocol):
    """Protocol for credential providers."""

    async def get_credential(self, key: str) -> Optional[str]:
        """Get a credential by key."""
        ...

    async def set_credential(self, key: str, value: str) -> None:
        """Set a credential."""
        ...


class EnvCredentialProvider:
    """
    Credential provider using environment variables.

    Simple provider that reads credentials from environment variables.
    Useful for local development and container deployments.
    """

    def __init__(self, prefix: str = "ARAGORA_"):
        """
        Initialize with optional prefix.

        Args:
            prefix: Prefix for environment variable names
        """
        self.prefix = prefix

    async def get_credential(self, key: str) -> Optional[str]:
        """Get credential from environment variable."""
        # Try with prefix first
        env_key = f"{self.prefix}{key.upper()}"
        value = os.environ.get(env_key)

        # Fall back to key without prefix
        if value is None:
            value = os.environ.get(key.upper())

        return value

    async def set_credential(self, key: str, value: str) -> None:
        """Set credential as environment variable (in-memory only)."""
        env_key = f"{self.prefix}{key.upper()}"
        os.environ[env_key] = value


@dataclass
class CachedCredential:
    """A cached credential with TTL."""

    value: str
    cached_at: float
    ttl_seconds: float

    @property
    def is_expired(self) -> bool:
        """Check if the cached credential has expired."""
        return time.time() - self.cached_at > self.ttl_seconds


class AWSSecretsManagerProvider:
    """
    Credential provider using AWS Secrets Manager.

    Features:
    - Automatic credential refresh with configurable TTL
    - Support for JSON secrets with nested keys
    - Automatic rotation handling
    - Region auto-detection

    Requires boto3: pip install boto3
    """

    DEFAULT_TTL_SECONDS = 3600  # 1 hour cache
    DEFAULT_REGION = "us-east-1"

    def __init__(
        self,
        secret_name: str,
        region: Optional[str] = None,
        cache_ttl_seconds: float = DEFAULT_TTL_SECONDS,
        profile_name: Optional[str] = None,
    ):
        """
        Initialize AWS Secrets Manager provider.

        Args:
            secret_name: Name or ARN of the secret
            region: AWS region (auto-detected if not provided)
            cache_ttl_seconds: How long to cache credentials
            profile_name: AWS profile name (optional)
        """
        self.secret_name = secret_name
        self.region = region or os.environ.get("AWS_REGION", self.DEFAULT_REGION)
        self.cache_ttl_seconds = cache_ttl_seconds
        self.profile_name = profile_name

        # Cache storage
        self._cache: Dict[str, CachedCredential] = {}
        self._secret_cache: Optional[Dict[str, Any]] = None
        self._secret_cached_at: float = 0

        # Client (lazy initialized)
        self._client = None

    def _get_client(self):
        """Get or create Secrets Manager client."""
        if self._client is None:
            try:
                import boto3
            except ImportError:
                raise ImportError(
                    "boto3 is required for AWS Secrets Manager. Install with: pip install boto3"
                )

            session_kwargs = {}
            if self.profile_name:
                session_kwargs["profile_name"] = self.profile_name

            session = boto3.Session(**session_kwargs)
            self._client = session.client(
                "secretsmanager",
                region_name=self.region,
            )

        return self._client

    async def _fetch_secret(self) -> Dict[str, Any]:
        """Fetch secret from AWS Secrets Manager."""
        # Check cache first
        now = time.time()
        if self._secret_cache is not None and now - self._secret_cached_at < self.cache_ttl_seconds:
            return self._secret_cache

        # Fetch from AWS (run in thread pool for async compatibility)
        loop = asyncio.get_event_loop()
        secret_value = await loop.run_in_executor(
            None,
            self._fetch_secret_sync,
        )

        # Cache the result
        self._secret_cache = secret_value
        self._secret_cached_at = now

        return secret_value

    def _fetch_secret_sync(self) -> Dict[str, Any]:
        """Synchronous secret fetch."""
        client = self._get_client()

        try:
            response = client.get_secret_value(SecretId=self.secret_name)

            # Parse secret value
            if "SecretString" in response:
                secret_str = response["SecretString"]
                # Try to parse as JSON
                try:
                    return json.loads(secret_str)
                except json.JSONDecodeError:
                    # Not JSON, return as single value
                    return {"_value": secret_str}
            else:
                # Binary secret
                return {"_binary": response["SecretBinary"]}

        except Exception as e:
            logger.error(f"Failed to fetch secret {self.secret_name}: {e}")
            raise

    async def get_credential(self, key: str) -> Optional[str]:
        """
        Get a credential by key.

        Supports nested keys using dot notation:
        - "api_key" -> secret["api_key"]
        - "database.password" -> secret["database"]["password"]

        Args:
            key: Credential key (supports dot notation for nested)

        Returns:
            Credential value or None if not found
        """
        # Check cache
        if key in self._cache:
            cached = self._cache[key]
            if not cached.is_expired:
                return cached.value

        # Fetch secret
        try:
            secret_data = await self._fetch_secret()
        except Exception as e:
            logger.error(f"Failed to fetch credentials: {e}")
            return None

        # Navigate nested keys
        value = secret_data
        for part in key.split("."):
            if isinstance(value, dict):
                # Try exact match first, then case-insensitive
                if part in value:
                    value = value[part]
                else:
                    # Case-insensitive lookup
                    lower_part = part.lower()
                    found = False
                    for k, v in value.items():
                        if k.lower() == lower_part:
                            value = v
                            found = True
                            break
                    if not found:
                        return None
            else:
                return None

        # Convert to string if needed
        str_value = value if isinstance(value, str) else str(value)

        # Cache the result
        self._cache[key] = CachedCredential(
            value=str_value,
            cached_at=time.time(),
            ttl_seconds=self.cache_ttl_seconds,
        )

        return str_value

    async def set_credential(self, key: str, value: str) -> None:
        """
        Set a credential in AWS Secrets Manager.

        Note: This updates the cached secret and pushes to AWS.
        Requires secretsmanager:PutSecretValue permission.
        """
        # Fetch current secret
        secret_data = await self._fetch_secret()

        # Update the key
        parts = key.split(".")
        target = secret_data
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        target[parts[-1]] = value

        # Push to AWS
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._update_secret_sync,
            secret_data,
        )

        # Invalidate cache
        self._secret_cache = None
        self._secret_cached_at = 0
        if key in self._cache:
            del self._cache[key]

    def _update_secret_sync(self, secret_data: Dict[str, Any]) -> None:
        """Synchronous secret update."""
        client = self._get_client()
        client.put_secret_value(
            SecretId=self.secret_name,
            SecretString=json.dumps(secret_data),
        )

    def clear_cache(self) -> None:
        """Clear the credential cache."""
        self._cache.clear()
        self._secret_cache = None
        self._secret_cached_at = 0


class ChainedCredentialProvider:
    """
    Credential provider that chains multiple providers.

    Tries providers in order until one returns a value.
    Useful for fallback hierarchies (e.g., AWS SM -> Env vars).
    """

    def __init__(self, providers: List[CredentialProvider]):
        """
        Initialize with ordered list of providers.

        Args:
            providers: List of providers to try in order
        """
        self.providers = providers

    async def get_credential(self, key: str) -> Optional[str]:
        """Get credential from first provider that has it."""
        for provider in self.providers:
            value = await provider.get_credential(key)
            if value is not None:
                return value
        return None

    async def set_credential(self, key: str, value: str) -> None:
        """Set credential in the first provider only."""
        if self.providers:
            await self.providers[0].set_credential(key, value)


class CachedCredentialProvider:
    """
    Wraps any credential provider with caching.

    Adds TTL-based caching to any provider implementation.
    """

    DEFAULT_TTL_SECONDS = 300  # 5 minutes

    def __init__(
        self,
        provider: CredentialProvider,
        cache_ttl_seconds: float = DEFAULT_TTL_SECONDS,
    ):
        """
        Initialize with underlying provider.

        Args:
            provider: The credential provider to wrap
            cache_ttl_seconds: Cache TTL in seconds
        """
        self.provider = provider
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cache: Dict[str, CachedCredential] = {}

    async def get_credential(self, key: str) -> Optional[str]:
        """Get credential with caching."""
        # Check cache
        if key in self._cache:
            cached = self._cache[key]
            if not cached.is_expired:
                return cached.value

        # Fetch from underlying provider
        value = await self.provider.get_credential(key)
        if value is not None:
            self._cache[key] = CachedCredential(
                value=value,
                cached_at=time.time(),
                ttl_seconds=self.cache_ttl_seconds,
            )

        return value

    async def set_credential(self, key: str, value: str) -> None:
        """Set credential and update cache."""
        await self.provider.set_credential(key, value)
        self._cache[key] = CachedCredential(
            value=value,
            cached_at=time.time(),
            ttl_seconds=self.cache_ttl_seconds,
        )

    def clear_cache(self) -> None:
        """Clear the credential cache."""
        self._cache.clear()


def get_credential_provider(
    provider_type: Optional[str] = None,
    **kwargs,
) -> CredentialProvider:
    """
    Factory function to get the appropriate credential provider.

    Auto-detects provider type from environment if not specified.

    Args:
        provider_type: Provider type (env, aws, chained). Auto-detected if None.
        **kwargs: Provider-specific configuration

    Environment Variables:
        CREDENTIAL_PROVIDER: Provider type (env, aws, chained)
        AWS_SECRET_NAME: Secret name for AWS provider
        AWS_REGION: Region for AWS provider

    Returns:
        Configured credential provider

    Example:
        # Auto-detect from environment
        provider = get_credential_provider()

        # Explicitly use AWS
        provider = get_credential_provider(
            "aws",
            secret_name="aragora/production/api-keys",
        )

        # Chain AWS -> Env fallback
        provider = get_credential_provider("chained")
    """
    if provider_type is None:
        provider_type = os.environ.get("CREDENTIAL_PROVIDER", "env")

    provider_type = provider_type.lower()

    if provider_type == "env":
        prefix = kwargs.get("prefix", "ARAGORA_")
        return EnvCredentialProvider(prefix=prefix)

    elif provider_type == "aws":
        secret_name = kwargs.get(
            "secret_name",
            os.environ.get("AWS_SECRET_NAME", "aragora/production/credentials"),
        )
        region = kwargs.get("region", os.environ.get("AWS_REGION"))
        cache_ttl = kwargs.get(
            "cache_ttl_seconds",
            AWSSecretsManagerProvider.DEFAULT_TTL_SECONDS,
        )

        return AWSSecretsManagerProvider(
            secret_name=secret_name,
            region=region,
            cache_ttl_seconds=cache_ttl,
        )

    elif provider_type == "chained":
        # Default chain: AWS -> Env
        secret_name = kwargs.get(
            "secret_name",
            os.environ.get("AWS_SECRET_NAME"),
        )

        providers: List[CredentialProvider] = []

        # Add AWS if configured
        if secret_name:
            try:
                providers.append(
                    AWSSecretsManagerProvider(
                        secret_name=secret_name,
                        region=kwargs.get("region", os.environ.get("AWS_REGION")),
                    )
                )
            except ImportError:
                logger.warning("boto3 not installed, skipping AWS provider")

        # Always add env as fallback
        providers.append(EnvCredentialProvider())

        return ChainedCredentialProvider(providers=providers)

    else:
        raise ValueError(f"Unknown credential provider type: {provider_type}")
