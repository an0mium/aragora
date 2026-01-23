"""
Production Guards for Storage Backends.

Enforces fail-closed semantics for storage backends in production environments.
When ARAGORA_REQUIRE_DISTRIBUTED=true (default in production), stores that
fall back to SQLite will raise errors instead of silently degrading.

This prevents multi-instance deployments from diverging due to per-node state.

Usage:
    from aragora.storage.production_guards import (
        require_distributed_store,
        is_production_mode,
        get_storage_mode,
    )

    # In store initialization:
    require_distributed_store("integration_store")

Environment Variables:
    ARAGORA_ENV: Environment name (production, staging, development)
    ARAGORA_REQUIRE_DISTRIBUTED: Force distributed storage requirement (true/false)
    ARAGORA_STORAGE_MODE: Override storage mode (postgres, redis, sqlite, file)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class StorageMode(str, Enum):
    """Storage backend modes."""

    POSTGRES = "postgres"
    REDIS = "redis"
    SQLITE = "sqlite"
    FILE = "file"
    MEMORY = "memory"


class EnvironmentMode(str, Enum):
    """Deployment environment modes."""

    PRODUCTION = "production"
    STAGING = "staging"
    DEVELOPMENT = "development"
    TEST = "test"


@dataclass
class StorageGuardConfig:
    """Configuration for storage guards."""

    require_distributed: bool = True
    allowed_fallback_envs: set[EnvironmentMode] = None
    fail_open_stores: set[str] = None

    def __post_init__(self):
        if self.allowed_fallback_envs is None:
            self.allowed_fallback_envs = {
                EnvironmentMode.DEVELOPMENT,
                EnvironmentMode.TEST,
            }
        if self.fail_open_stores is None:
            # Stores that can safely fall back to local storage
            self.fail_open_stores = {
                "cache_store",  # Ephemeral cache
                "session_store",  # Short-lived sessions
                "workflow_store",  # Workflow definitions (static config, not dynamic state)
            }


# Global configuration
_config: Optional[StorageGuardConfig] = None


def get_config() -> StorageGuardConfig:
    """Get or create storage guard configuration."""
    global _config
    if _config is None:
        _config = StorageGuardConfig(
            require_distributed=os.environ.get("ARAGORA_REQUIRE_DISTRIBUTED", "true").lower()
            == "true",
        )
    return _config


def get_environment() -> EnvironmentMode:
    """Get the current environment mode."""
    env = os.environ.get("ARAGORA_ENV", "development").lower()

    if env in ("prod", "production"):
        return EnvironmentMode.PRODUCTION
    elif env in ("stage", "staging"):
        return EnvironmentMode.STAGING
    elif env in ("dev", "development"):
        return EnvironmentMode.DEVELOPMENT
    elif env in ("test", "testing"):
        return EnvironmentMode.TEST
    else:
        logger.warning(f"Unknown environment '{env}', treating as development")
        return EnvironmentMode.DEVELOPMENT


def is_production_mode() -> bool:
    """Check if running in production mode."""
    return get_environment() in (
        EnvironmentMode.PRODUCTION,
        EnvironmentMode.STAGING,
    )


def get_storage_mode() -> Optional[StorageMode]:
    """Get the explicitly configured storage mode, if any."""
    mode = os.environ.get("ARAGORA_STORAGE_MODE", "").lower()
    if not mode:
        return None

    try:
        return StorageMode(mode)
    except ValueError:
        logger.warning(f"Unknown storage mode '{mode}'")
        return None


class DistributedStateError(Exception):
    """
    Raised when a store falls back to local storage in production.

    This indicates a configuration or infrastructure issue that must be
    resolved before the service can safely handle requests.
    """

    def __init__(self, store_name: str, reason: str):
        self.store_name = store_name
        self.reason = reason
        super().__init__(
            f"Distributed storage required for '{store_name}' in production mode. "
            f"Reason: {reason}. "
            f"Set ARAGORA_REQUIRE_DISTRIBUTED=false to allow fallback (NOT recommended for production)."
        )


def require_distributed_store(
    store_name: str,
    current_mode: StorageMode,
    reason: str = "Store initialized with local backend",
) -> None:
    """
    Enforce distributed storage requirement in production.

    Call this when a store falls back to SQLite, file, or memory storage.
    In production mode (unless explicitly disabled), this will raise an error.

    Args:
        store_name: Name of the store (for error messages)
        current_mode: The storage mode the store is using
        reason: Explanation of why fallback occurred

    Raises:
        DistributedStateError: If distributed storage is required but not available

    Example:
        if not postgres_available:
            require_distributed_store(
                "integration_store",
                StorageMode.SQLITE,
                "PostgreSQL connection failed"
            )
            # Will raise in production, continue in development
            self._backend = SQLiteBackend()
    """
    config = get_config()
    env = get_environment()

    # Allow fallback in development/test
    if env in config.allowed_fallback_envs:
        logger.info(
            f"Store '{store_name}' using {current_mode.value} backend "
            f"(allowed in {env.value} environment)"
        )
        return

    # Allow fallback for explicitly excluded stores
    if store_name in config.fail_open_stores:
        logger.info(
            f"Store '{store_name}' using {current_mode.value} backend " f"(fail-open store)"
        )
        return

    # Allow fallback if explicitly disabled
    if not config.require_distributed:
        logger.warning(
            f"Store '{store_name}' using {current_mode.value} backend "
            f"in {env.value} environment. "
            f"Distributed state enforcement is DISABLED. "
            f"This is NOT recommended for multi-instance deployments."
        )
        return

    # Local storage modes that require distributed enforcement
    local_modes = {StorageMode.SQLITE, StorageMode.FILE, StorageMode.MEMORY}

    if current_mode in local_modes:
        raise DistributedStateError(store_name, reason)

    # Using distributed backend, all good
    logger.debug(f"Store '{store_name}' using distributed backend: {current_mode.value}")


def validate_store_config(
    store_name: str,
    postgres_url: Optional[str] = None,
    redis_url: Optional[str] = None,
    fallback_mode: StorageMode = StorageMode.SQLITE,
) -> StorageMode:
    """
    Validate and determine storage mode for a store.

    In production, this ensures a distributed backend is available.
    In development, falls back gracefully to local storage.

    Args:
        store_name: Name of the store
        postgres_url: PostgreSQL connection URL (if available)
        redis_url: Redis connection URL (if available)
        fallback_mode: Mode to use if distributed storage unavailable

    Returns:
        The storage mode to use

    Raises:
        DistributedStateError: If distributed storage required but unavailable
    """
    # Check for explicit override
    explicit_mode = get_storage_mode()
    if explicit_mode:
        logger.info(f"Store '{store_name}' using explicit mode: {explicit_mode.value}")
        require_distributed_store(
            store_name,
            explicit_mode,
            "Explicit mode configured via ARAGORA_STORAGE_MODE",
        )
        return explicit_mode

    # Try PostgreSQL first
    if postgres_url:
        logger.info(f"Store '{store_name}' using PostgreSQL backend")
        return StorageMode.POSTGRES

    # Try Redis
    if redis_url:
        logger.info(f"Store '{store_name}' using Redis backend")
        return StorageMode.REDIS

    # Fall back to local storage
    require_distributed_store(
        store_name,
        fallback_mode,
        "No PostgreSQL or Redis URL configured (DATABASE_URL, ARAGORA_REDIS_URL)",
    )

    logger.info(f"Store '{store_name}' using {fallback_mode.value} backend (fallback)")
    return fallback_mode


def check_multi_instance_readiness() -> dict[str, bool]:
    """
    Check if all critical stores are configured for multi-instance deployment.

    Returns:
        Dict mapping store names to readiness status
    """
    from aragora.storage import (  # type: ignore[attr-defined]
        get_integration_store,
        get_webhook_config_store,
        get_gmail_token_store,
    )
    from aragora.queue.job_queue import get_job_queue_store

    stores = {
        "integration_store": False,
        "webhook_config_store": False,
        "gmail_token_store": False,
        "job_queue_store": False,
        "workflow_store": False,
        "checkpoint_store": False,
    }

    # Check each store's backend
    try:
        int_store = get_integration_store()
        stores["integration_store"] = getattr(int_store, "_uses_distributed", False)
    except Exception as e:
        logger.debug(f"Could not check integration_store: {e}")

    try:
        webhook_store = get_webhook_config_store()
        stores["webhook_config_store"] = getattr(webhook_store, "_uses_distributed", False)
    except Exception as e:
        logger.debug(f"Could not check webhook_config_store: {e}")

    try:
        gmail_store = get_gmail_token_store()
        stores["gmail_token_store"] = getattr(gmail_store, "_uses_distributed", False)
    except Exception as e:
        logger.debug(f"Could not check gmail_token_store: {e}")

    try:
        job_store = get_job_queue_store()
        stores["job_queue_store"] = getattr(job_store, "_uses_distributed", False)
    except Exception as e:
        logger.debug(f"Could not check job_queue_store: {e}")

    return stores


__all__ = [
    "StorageMode",
    "EnvironmentMode",
    "StorageGuardConfig",
    "DistributedStateError",
    "get_config",
    "get_environment",
    "is_production_mode",
    "get_storage_mode",
    "require_distributed_store",
    "validate_store_config",
    "check_multi_instance_readiness",
]
