"""
AWS Secrets Manager integration for Aragora.

This module provides secure secret management with multiple fallback strategies:
1. AWS Secrets Manager (production)
2. Environment variables (local development)
3. Default values (for non-sensitive config)

Usage:
    from aragora.config.secrets import get_secret, SecretManager

    # Get individual secrets
    jwt_secret = get_secret("JWT_SECRET_KEY")
    stripe_key = get_secret("STRIPE_SECRET_KEY")

    # Or use the manager for batch loading
    manager = SecretManager()
    secrets = manager.get_secrets(["JWT_SECRET_KEY", "STRIPE_SECRET_KEY"])
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

# Secret names that should be loaded from Secrets Manager
MANAGED_SECRETS = frozenset({
    # Authentication
    "JWT_SECRET_KEY",
    "JWT_REFRESH_SECRET",
    # OAuth
    "GOOGLE_OAUTH_CLIENT_ID",
    "GOOGLE_OAUTH_CLIENT_SECRET",
    "GITHUB_OAUTH_CLIENT_ID",
    "GITHUB_OAUTH_CLIENT_SECRET",
    # Stripe billing
    "STRIPE_SECRET_KEY",
    "STRIPE_WEBHOOK_SECRET",
    "STRIPE_PRICE_STARTER",
    "STRIPE_PRICE_PROFESSIONAL",
    "STRIPE_PRICE_ENTERPRISE",
    # Database
    "DATABASE_URL",
    "SUPABASE_URL",
    "SUPABASE_KEY",
    # Redis
    "REDIS_URL",
    "REDIS_PASSWORD",
    # API Keys (sensitive)
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "XAI_API_KEY",
    "OPENROUTER_API_KEY",
    "MISTRAL_API_KEY",
    # Monitoring
    "SENTRY_DSN",
})


@dataclass
class SecretsConfig:
    """Configuration for secrets management."""

    # AWS Secrets Manager settings
    aws_region: str = "us-east-1"
    secret_name: str = "aragora/production"
    use_aws: bool = False

    # Cache settings
    cache_ttl_seconds: int = 300

    @classmethod
    def from_env(cls) -> "SecretsConfig":
        """Load config from environment."""
        return cls(
            aws_region=os.environ.get("AWS_REGION", "us-east-1"),
            secret_name=os.environ.get("ARAGORA_SECRET_NAME", "aragora/production"),
            use_aws=os.environ.get("ARAGORA_USE_SECRETS_MANAGER", "").lower() in ("true", "1", "yes"),
        )


class SecretManager:
    """
    Manages secrets from multiple sources with fallback.

    Priority order:
    1. AWS Secrets Manager (if enabled)
    2. Environment variables
    3. Default values (for non-sensitive config)
    """

    def __init__(self, config: SecretsConfig | None = None):
        self.config = config or SecretsConfig.from_env()
        self._aws_client: Any = None
        self._cached_secrets: dict[str, str] = {}
        self._initialized = False

    def _get_aws_client(self) -> Any:
        """Lazily initialize AWS Secrets Manager client."""
        if self._aws_client is not None:
            return self._aws_client

        try:
            import boto3

            self._aws_client = boto3.client(
                "secretsmanager",
                region_name=self.config.aws_region,
            )
            return self._aws_client
        except ImportError:
            logger.debug("boto3 not installed, AWS Secrets Manager unavailable")
            return None
        except Exception as e:
            logger.warning(f"Failed to initialize AWS client: {e}")
            return None

    def _load_from_aws(self) -> dict[str, str]:
        """Load secrets from AWS Secrets Manager."""
        if not self.config.use_aws:
            return {}

        client = self._get_aws_client()
        if client is None:
            return {}

        try:
            from botocore.exceptions import ClientError

            response = client.get_secret_value(SecretId=self.config.secret_name)
            secret_string = response.get("SecretString", "{}")
            secrets: dict[str, str] = json.loads(secret_string)
            logger.info(f"Loaded {len(secrets)} secrets from AWS Secrets Manager")
            return secrets
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "ResourceNotFoundException":
                logger.warning(f"Secret '{self.config.secret_name}' not found in AWS")
            elif error_code == "AccessDeniedException":
                logger.warning("Access denied to AWS Secrets Manager")
            else:
                logger.error(f"AWS Secrets Manager error: {e}")
            return {}
        except json.JSONDecodeError:
            logger.error("Failed to parse secrets JSON from AWS")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error loading secrets: {e}")
            return {}

    def _initialize(self) -> None:
        """Initialize the secret manager (load from AWS if enabled)."""
        if self._initialized:
            return

        if self.config.use_aws:
            self._cached_secrets = self._load_from_aws()

        self._initialized = True

    def get(self, name: str, default: str | None = None) -> str | None:
        """
        Get a secret value.

        Args:
            name: Secret name (e.g., "JWT_SECRET_KEY")
            default: Default value if not found

        Returns:
            Secret value or default
        """
        self._initialize()

        # 1. Check AWS cache first
        if name in self._cached_secrets:
            return self._cached_secrets[name]

        # 2. Fall back to environment variable
        env_value = os.environ.get(name)
        if env_value is not None:
            return env_value

        # 3. Return default
        return default

    def get_required(self, name: str) -> str:
        """
        Get a required secret value.

        Args:
            name: Secret name

        Returns:
            Secret value

        Raises:
            ValueError: If secret is not found
        """
        value = self.get(name)
        if value is None:
            raise ValueError(f"Required secret '{name}' not found")
        return value

    def get_secrets(self, names: list[str]) -> dict[str, str | None]:
        """
        Get multiple secrets at once.

        Args:
            names: List of secret names

        Returns:
            Dictionary of secret name -> value (or None if not found)
        """
        return {name: self.get(name) for name in names}

    def is_configured(self, name: str) -> bool:
        """Check if a secret is configured (has a value)."""
        return self.get(name) is not None

    def get_auth_secrets(self) -> dict[str, str | None]:
        """Get all authentication-related secrets."""
        auth_secrets = [
            "JWT_SECRET_KEY",
            "JWT_REFRESH_SECRET",
            "GOOGLE_OAUTH_CLIENT_ID",
            "GOOGLE_OAUTH_CLIENT_SECRET",
            "GITHUB_OAUTH_CLIENT_ID",
            "GITHUB_OAUTH_CLIENT_SECRET",
        ]
        return self.get_secrets(auth_secrets)

    def get_billing_secrets(self) -> dict[str, str | None]:
        """Get all billing-related secrets."""
        billing_secrets = [
            "STRIPE_SECRET_KEY",
            "STRIPE_WEBHOOK_SECRET",
            "STRIPE_PRICE_STARTER",
            "STRIPE_PRICE_PROFESSIONAL",
            "STRIPE_PRICE_ENTERPRISE",
        ]
        return self.get_secrets(billing_secrets)


# Global singleton instance
_manager: SecretManager | None = None


def get_secret_manager() -> SecretManager:
    """Get the global secret manager instance."""
    global _manager
    if _manager is None:
        _manager = SecretManager()
    return _manager


def reset_secret_manager() -> None:
    """Reset the global secret manager (for testing)."""
    global _manager
    _manager = None


@lru_cache(maxsize=128)
def get_secret(name: str, default: str | None = None) -> str | None:
    """
    Get a secret value (cached).

    This is the main entry point for getting secrets throughout the application.

    Args:
        name: Secret name (e.g., "JWT_SECRET_KEY")
        default: Default value if not found

    Returns:
        Secret value or default

    Example:
        jwt_secret = get_secret("JWT_SECRET_KEY")
        stripe_key = get_secret("STRIPE_SECRET_KEY", "")
    """
    return get_secret_manager().get(name, default)


def get_required_secret(name: str) -> str:
    """
    Get a required secret value.

    Args:
        name: Secret name

    Returns:
        Secret value

    Raises:
        ValueError: If secret is not found
    """
    return get_secret_manager().get_required(name)


def clear_secret_cache() -> None:
    """Clear the secret cache (for testing or secret rotation)."""
    get_secret.cache_clear()
