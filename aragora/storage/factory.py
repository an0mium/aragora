"""
Storage Factory for Aragora.

Provides a unified interface for creating storage backends based on configuration.
Supports Supabase PostgreSQL (preferred), self-hosted PostgreSQL, and SQLite fallback.

Backend Selection:
1. Explicit: ARAGORA_DB_BACKEND=supabase|postgres|sqlite
2. Auto: Supabase if configured, else ARAGORA_POSTGRES_DSN/DATABASE_URL
3. Production: Distributed storage required by default
4. Fallback: SQLite for local development

Usage:
    from aragora.storage.factory import get_storage_backend, StorageBackend

    # Get configured backend
    backend = get_storage_backend()

    if backend == StorageBackend.POSTGRES:
        # Use async PostgreSQL stores
        pool = await get_postgres_pool()
        store = MyPostgresStore(pool)
    else:
        # Use SQLite stores (development fallback)
        store = MySQLiteStore(db_path)

    # Validate configuration at startup
    validate_storage_config()  # Warns/errors on misconfiguration
"""

import logging
import os
from enum import Enum
from pathlib import Path
from typing import Optional, Union

from aragora.storage.connection_factory import (
    get_selfhosted_postgres_dsn,
    get_supabase_postgres_dsn,
)

logger = logging.getLogger(__name__)

# Production environment indicators
_PRODUCTION_ENVS = {"production", "prod", "live"}


class StorageBackend(Enum):
    """Available storage backend types."""

    SQLITE = "sqlite"
    POSTGRES = "postgres"
    SUPABASE = "supabase"


def is_production_environment() -> bool:
    """Check if running in a production environment."""
    env = os.environ.get("ARAGORA_ENV", "development").lower()
    return env in _PRODUCTION_ENVS


def get_storage_backend() -> StorageBackend:
    """
    Determine which storage backend to use based on configuration.

    Priority order:
    1. Explicit ARAGORA_DB_BACKEND environment variable ("supabase", "postgres", "sqlite")
    2. Auto-detect: Supabase if configured, else PostgreSQL DSN
    3. Production enforcement: In production, distributed storage is preferred
    4. Fallback: SQLite for local development

    Returns:
        StorageBackend enum value
    """
    is_prod = is_production_environment()

    # Check for explicit backend setting
    explicit_backend = os.environ.get("ARAGORA_DB_BACKEND")
    if explicit_backend:
        backend = explicit_backend.lower()
        if backend == "supabase":
            if not get_supabase_postgres_dsn():
                logger.warning("ARAGORA_DB_BACKEND=supabase set but Supabase is not configured.")
            return StorageBackend.SUPABASE
        elif backend == "postgres" or backend == "postgresql":
            if not get_selfhosted_postgres_dsn():
                logger.warning("ARAGORA_DB_BACKEND=postgres set but no PostgreSQL DSN configured.")
            return StorageBackend.POSTGRES
        elif backend == "auto":
            pass
        elif backend == "sqlite":
            if is_prod:
                logger.warning(
                    "SQLite explicitly configured in production environment. "
                    "This is not recommended for multi-user deployments. "
                    "Configure Supabase or PostgreSQL for distributed storage."
                )
            return StorageBackend.SQLITE
        else:
            logger.warning(
                f"Unknown ARAGORA_DB_BACKEND value: {backend}, falling back to auto-detect"
            )

    # Auto-detect: prefer Supabase, then self-hosted PostgreSQL
    supabase_dsn = get_supabase_postgres_dsn()
    if supabase_dsn:
        logger.debug("Supabase configuration detected, using Supabase backend")
        return StorageBackend.SUPABASE

    postgres_dsn = get_selfhosted_postgres_dsn()
    if postgres_dsn:
        logger.debug("PostgreSQL DSN detected, using PostgreSQL backend")
        return StorageBackend.POSTGRES

    # Production warning if no distributed storage configured
    if is_prod:
        logger.warning(
            "No Supabase or PostgreSQL DSN configured in production environment. "
            "SQLite is being used as fallback, which is NOT safe for multi-user deployments. "
            "Configure SUPABASE_URL + SUPABASE_DB_PASSWORD or ARAGORA_POSTGRES_DSN."
        )

    # Fallback: SQLite for local development
    logger.debug("No Supabase or PostgreSQL DSN configured, using SQLite backend")
    return StorageBackend.SQLITE


def is_postgres_configured() -> bool:
    """
    Check if a PostgreSQL connection is configured (Supabase or self-hosted).

    Returns:
        True if a PostgreSQL DSN is available and backend is Postgres-based
    """
    backend = get_storage_backend()
    if backend == StorageBackend.SUPABASE:
        return bool(get_supabase_postgres_dsn())
    if backend == StorageBackend.POSTGRES:
        return bool(get_selfhosted_postgres_dsn())
    return False


def get_default_db_path(name: str, nomic_dir: Optional[Union[str, Path]] = None) -> Path:
    """
    Get the default SQLite database path for a given store name.

    Args:
        name: Store name (e.g., "debates", "elo", "memory")
        nomic_dir: Optional nomic directory path. If not provided,
                   uses ARAGORA_NOMIC_DIR or ~/.nomic/aragora/

    Returns:
        Path to the SQLite database file
    """
    if nomic_dir is None:
        nomic_dir = os.environ.get("ARAGORA_NOMIC_DIR")

    if nomic_dir is None:
        nomic_dir = Path.home() / ".nomic" / "aragora"
    else:
        nomic_dir = Path(nomic_dir)

    # Ensure directory exists
    db_dir = nomic_dir / "db"
    db_dir.mkdir(parents=True, exist_ok=True)

    return db_dir / f"{name}.db"


# =========================================================================
# Store Registry
# =========================================================================


_store_registry: dict[str, type] = {}


def register_store(name: str, store_class: type) -> None:
    """
    Register a store class for factory creation.

    Args:
        name: Store name for lookup
        store_class: Store class to register
    """
    _store_registry[name] = store_class
    logger.debug(f"Registered store: {name} -> {store_class.__name__}")


def get_registered_stores() -> dict[str, type]:
    """
    Get all registered store classes.

    Returns:
        Dictionary mapping store names to classes
    """
    return _store_registry.copy()


# =========================================================================
# Convenience Functions
# =========================================================================


def storage_info() -> dict[str, object]:
    """
    Get information about current storage configuration.

    Returns:
        Dictionary with storage backend info
    """
    backend = get_storage_backend()
    info: dict[str, object] = {
        "backend": backend.value,
        "is_postgres": backend in (StorageBackend.POSTGRES, StorageBackend.SUPABASE),
        "is_supabase": backend == StorageBackend.SUPABASE,
        "postgres_configured": is_postgres_configured(),
        "is_production": is_production_environment(),
    }

    if backend in (StorageBackend.POSTGRES, StorageBackend.SUPABASE):
        dsn = (
            get_supabase_postgres_dsn()
            if backend == StorageBackend.SUPABASE
            else get_selfhosted_postgres_dsn()
        )
        dsn = dsn or ""
        # Redact password from DSN for logging
        if dsn and "@" in dsn:
            parts = dsn.split("@")
            user_pass = parts[0].rsplit(":", 1)
            if len(user_pass) == 2:
                dsn = f"{user_pass[0]}:***@{parts[1]}"
        info["dsn_redacted"] = dsn
    else:
        default_dir = os.environ.get("ARAGORA_NOMIC_DIR") or str(Path.home() / ".nomic" / "aragora")
        info["default_db_dir"] = str(Path(default_dir) / "db")

    return info


def validate_storage_config(strict: bool = False) -> dict:
    """
    Validate storage configuration at startup.

    Call this at server startup to ensure storage is properly configured.
    In production, this will error if distributed storage is not configured.

    Args:
        strict: If True, raise an error on critical misconfigurations

    Returns:
        Dict with validation results: {"valid": bool, "errors": list, "warnings": list}

    Raises:
        RuntimeError: If strict=True and critical errors are found
    """
    errors: list[str] = []
    warnings_list: list[str] = []

    is_prod = is_production_environment()
    backend = get_storage_backend()
    supabase_dsn = get_supabase_postgres_dsn()
    postgres_dsn = get_selfhosted_postgres_dsn()

    # Check distributed configuration in production
    if is_prod and backend == StorageBackend.SQLITE:
        errors.append(
            "Production environment requires distributed storage. "
            "Configure Supabase (SUPABASE_URL + SUPABASE_DB_PASSWORD) or "
            "set ARAGORA_DB_BACKEND=postgres with ARAGORA_POSTGRES_DSN."
        )
    elif is_prod and backend == StorageBackend.POSTGRES and not postgres_dsn:
        errors.append(
            "PostgreSQL backend selected but no DSN configured. "
            "Set ARAGORA_POSTGRES_DSN or DATABASE_URL."
        )
    elif is_prod and backend == StorageBackend.SUPABASE and not supabase_dsn:
        errors.append(
            "Supabase backend selected but no Supabase DSN configured. "
            "Set SUPABASE_URL + SUPABASE_DB_PASSWORD or SUPABASE_POSTGRES_DSN."
        )

    # Check for DSN without backend being set to the matching backend
    if postgres_dsn and backend not in (StorageBackend.POSTGRES, StorageBackend.SUPABASE):
        warnings_list.append(
            f"PostgreSQL DSN is configured but backend is set to {backend.value}. "
            "Consider setting ARAGORA_DB_BACKEND=postgres."
        )
    if supabase_dsn and backend != StorageBackend.SUPABASE:
        warnings_list.append(
            f"Supabase is configured but backend is set to {backend.value}. "
            "Consider setting ARAGORA_DB_BACKEND=supabase."
        )

    # Warn about SQLite in non-development environments
    if backend == StorageBackend.SQLITE and not is_prod:
        env = os.environ.get("ARAGORA_ENV", "development")
        if env not in ("development", "dev", "local", "test"):
            warnings_list.append(
                f"SQLite is being used in {env} environment. "
                "Consider using PostgreSQL for better concurrency."
            )

    # Log warnings
    for warning in warnings_list:
        logger.warning(warning)

    # Handle errors based on strict mode
    if errors:
        for error in errors:
            logger.error(f"Storage configuration error: {error}")
        if strict:
            raise RuntimeError(f"Storage configuration failed: {'; '.join(errors)}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings_list,
        "backend": backend.value,
        "is_production": is_prod,
        "postgres_dsn_configured": bool(postgres_dsn or supabase_dsn),
        "supabase_dsn_configured": bool(supabase_dsn),
    }


__all__ = [
    "StorageBackend",
    "get_storage_backend",
    "is_postgres_configured",
    "is_production_environment",
    "get_default_db_path",
    "register_store",
    "get_registered_stores",
    "storage_info",
    "validate_storage_config",
]
