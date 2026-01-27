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
import threading
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Secret names that should be loaded from Secrets Manager
MANAGED_SECRETS = frozenset(
    {
        # Authentication
        "JWT_SECRET_KEY",
        "JWT_REFRESH_SECRET",
        "ARAGORA_JWT_SECRET",
        # Encryption
        "ARAGORA_ENCRYPTION_KEY",
        # OAuth
        "GOOGLE_OAUTH_CLIENT_ID",
        "GOOGLE_OAUTH_CLIENT_SECRET",
        "GITHUB_OAUTH_CLIENT_ID",
        "GITHUB_OAUTH_CLIENT_SECRET",
        # Gmail OAuth (for inbox integration)
        "GMAIL_CLIENT_ID",
        "GMAIL_CLIENT_SECRET",
        # Stripe billing
        "STRIPE_SECRET_KEY",
        "STRIPE_WEBHOOK_SECRET",
        "STRIPE_PRICE_STARTER",
        "STRIPE_PRICE_PROFESSIONAL",
        "STRIPE_PRICE_ENTERPRISE",
        # Database (Supabase PostgreSQL)
        "DATABASE_URL",
        "ARAGORA_POSTGRES_DSN",
        "SUPABASE_URL",
        "SUPABASE_KEY",
        "SUPABASE_DB_PASSWORD",
        "SUPABASE_POSTGRES_DSN",
        "SUPABASE_SERVICE_ROLE_KEY",
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
        "DEEPSEEK_API_KEY",
        "KIMI_API_KEY",
        "ELEVENLABS_API_KEY",
        "AZURE_CLIENT_SECRET",
        "SUPABASE_PROJECT_ID",
        # Monitoring
        "SENTRY_DSN",
    }
)


@dataclass
class SecretsConfig:
    """Configuration for secrets management."""

    # AWS Secrets Manager settings
    aws_region: str = "us-east-1"
    aws_regions: list[str] = field(default_factory=list)
    secret_name: str = "aragora/production"
    use_aws: bool = False

    # Cache settings
    cache_ttl_seconds: int = 300

    @classmethod
    def from_env(cls) -> "SecretsConfig":
        """Load config from environment."""
        use_flag = os.environ.get("ARAGORA_USE_SECRETS_MANAGER", "")
        if use_flag:
            use_aws = use_flag.lower() in ("true", "1", "yes")
        else:
            env_value = os.environ.get("ARAGORA_ENV", "").lower()
            use_aws = env_value in ("production", "prod", "staging", "stage")
            if not use_aws:
                use_aws = bool(
                    os.environ.get("ARAGORA_SECRET_NAME")
                    and (os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"))
                )

        primary_region = (
            os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
        )
        raw_regions = os.environ.get("ARAGORA_SECRET_REGIONS", "")
        explicit_regions = [r.strip() for r in raw_regions.split(",") if r.strip()]
        if explicit_regions:
            regions = []
            for region in [primary_region, *explicit_regions]:
                if region and region not in regions:
                    regions.append(region)
        else:
            regions = [primary_region]
            if primary_region != "us-east-2":
                regions.append("us-east-2")
            if primary_region != "us-east-1":
                regions.append("us-east-1")
        return cls(
            aws_region=primary_region,
            aws_regions=regions,
            secret_name=os.environ.get("ARAGORA_SECRET_NAME", "aragora/production"),
            use_aws=use_aws,
        )


class SecretManager:
    """
    Manages secrets from multiple sources with fallback.

    Priority order:
    1. AWS Secrets Manager (if enabled)
    2. Environment variables
    3. Default values (for non-sensitive config)

    Features:
    - Automatic cache expiration based on TTL
    - Audit logging for secret access (SOC 2 compliance)
    - Thread-safe secret access
    """

    def __init__(self, config: SecretsConfig | None = None):
        self.config = config or SecretsConfig.from_env()
        self._aws_clients: dict[str, Any] = {}
        self._cached_secrets: dict[str, str] = {}
        self._cache_timestamp: float = 0.0
        self._initialized = False
        self._lock = threading.Lock()
        self._access_log: list[dict[str, Any]] = []
        self._max_access_log_size = 1000

    def _is_cache_expired(self) -> bool:
        """Check if the secret cache has expired."""
        import time

        if self._cache_timestamp == 0.0:
            return True
        elapsed = time.time() - self._cache_timestamp
        return elapsed > self.config.cache_ttl_seconds

    def _log_access(self, secret_name: str, source: str, success: bool) -> None:
        """Log secret access for audit purposes (SOC 2 compliance)."""
        import time

        entry = {
            "timestamp": time.time(),
            "secret_name": secret_name,
            "source": source,  # "aws", "env", "default"
            "success": success,
        }
        with self._lock:
            self._access_log.append(entry)
            # Trim log if too large
            if len(self._access_log) > self._max_access_log_size:
                self._access_log = self._access_log[-self._max_access_log_size // 2 :]

    def get_access_log(self) -> list[dict[str, Any]]:
        """Get the access log for audit purposes."""
        with self._lock:
            return list(self._access_log)

    def _get_aws_client(self, region: str) -> Any:
        """Lazily initialize AWS Secrets Manager client for a region."""
        if region in self._aws_clients:
            return self._aws_clients[region]

        try:
            import boto3

            client = boto3.client("secretsmanager", region_name=region)
            self._aws_clients[region] = client
            return client
        except ImportError:
            logger.debug("boto3 not installed, AWS Secrets Manager unavailable")
            return None
        except Exception as e:
            logger.warning(f"Failed to initialize AWS client ({region}): {e}")
            return None

    def _load_from_aws(self) -> dict[str, str]:
        """Load secrets from AWS Secrets Manager."""
        if not self.config.use_aws:
            return {}

        regions = self.config.aws_regions or [self.config.aws_region]
        if not regions:
            return {}

        last_error: Exception | None = None
        for region in regions:
            client = self._get_aws_client(region)
            if client is None:
                continue
            try:
                response = client.get_secret_value(SecretId=self.config.secret_name)
                secret_string = response.get("SecretString", "{}")
                secrets: dict[str, str] = json.loads(secret_string)
                logger.info(
                    "Loaded %d secrets from AWS Secrets Manager (region=%s)",
                    len(secrets),
                    region,
                )
                return secrets
            except json.JSONDecodeError:
                logger.error("Failed to parse secrets JSON from AWS (region=%s)", region)
                return {}
            except Exception as e:
                last_error = e
                if type(e).__name__ == "ClientError" and hasattr(e, "response"):
                    error_code = e.response.get("Error", {}).get("Code", "")
                    if error_code == "ResourceNotFoundException":
                        logger.warning(
                            "Secret '%s' not found in AWS (region=%s)",
                            self.config.secret_name,
                            region,
                        )
                        continue
                    if error_code == "AccessDeniedException":
                        logger.warning("Access denied to AWS Secrets Manager (region=%s)", region)
                        continue
                    logger.error("AWS Secrets Manager error (region=%s): %s", region, e)
                    continue
                logger.error("Unexpected error loading secrets (region=%s): %s", region, e)
                continue

        if last_error:
            logger.warning("Failed to load secrets from AWS Secrets Manager in all regions")
        return {}

    def _initialize(self, force_refresh: bool = False) -> None:
        """Initialize the secret manager (load from AWS if enabled).

        Args:
            force_refresh: Force reload from AWS even if cache is valid
        """
        import time

        with self._lock:
            # First initialization
            if not self._initialized:
                if self.config.use_aws:
                    self._cached_secrets = self._load_from_aws()
                    self._cache_timestamp = time.time()
                    logger.debug(
                        f"Secrets cache initialized, TTL: {self.config.cache_ttl_seconds}s"
                    )
                self._initialized = True
                return

            # Already initialized - check if refresh needed (only for AWS)
            if not self.config.use_aws:
                return  # No AWS, no refresh needed

            needs_refresh = force_refresh or self._is_cache_expired()
            if not needs_refresh:
                return

            self._cached_secrets = self._load_from_aws()
            self._cache_timestamp = time.time()
            logger.debug(f"Secrets cache refreshed, TTL: {self.config.cache_ttl_seconds}s")

    def refresh(self) -> None:
        """Force refresh secrets from AWS (for manual rotation)."""
        self._initialize(force_refresh=True)
        logger.info("Secrets manually refreshed")

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
            self._log_access(name, "aws", True)
            return self._cached_secrets[name]

        # 2. Fall back to environment variable
        env_value = os.environ.get(name)
        if env_value is not None:
            self._log_access(name, "env", True)
            return env_value

        # 3. Return default
        if default is not None:
            self._log_access(name, "default", True)
        else:
            self._log_access(name, "not_found", False)
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


# Global singleton instance with thread-safe initialization
_manager: SecretManager | None = None
_manager_lock = threading.Lock()


def get_secret_manager() -> SecretManager:
    """Get the global secret manager instance (thread-safe)."""
    global _manager
    if _manager is None:
        with _manager_lock:
            # Double-checked locking pattern
            if _manager is None:
                _manager = SecretManager()
    return _manager


def reset_secret_manager() -> None:
    """Reset the global secret manager (for testing)."""
    global _manager
    _manager = None


def get_secret(name: str, default: str | None = None) -> str | None:
    """
    Get a secret value.

    This is the main entry point for getting secrets throughout the application.
    Caching happens inside SecretManager (AWS secrets are loaded once on first access).

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
    reset_secret_manager()


def refresh_secrets() -> None:
    """Force refresh secrets from AWS Secrets Manager.

    Call this after rotating secrets in AWS to ensure the application
    picks up the new values immediately.
    """
    get_secret_manager().refresh()


def get_secret_access_log() -> list[dict[str, Any]]:
    """Get the secret access log for audit purposes (SOC 2 compliance).

    Returns:
        List of access log entries with timestamp, secret_name, source, and success.
    """
    return get_secret_manager().get_access_log()
