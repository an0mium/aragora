"""
Centralized CORS configuration for all Aragora server components.

This module provides a single source of truth for allowed origins,
preventing configuration drift across auth, api, stream, and unified_server.

PRODUCTION NOTE: Set ARAGORA_ALLOWED_ORIGINS to your domain(s) in production.
The defaults include localhost for development convenience but should be
overridden with explicit production domains.
"""

import logging
import os
from typing import Set

logger = logging.getLogger(__name__)

# Check if we're in production mode
_IS_PRODUCTION = os.environ.get("ARAGORA_ENV", "").lower() == "production"

# Development origins (included by default in dev mode only)
_DEV_ORIGINS: Set[str] = {
    "http://localhost:3000",
    "http://localhost:8080",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8080",
}

# Production origins (included by default in production)
_PROD_ORIGINS: Set[str] = {
    "https://aragora.ai",
    "https://www.aragora.ai",
    "https://live.aragora.ai",
    "https://api.aragora.ai",
}

# Default allowed origins based on environment
if _IS_PRODUCTION:
    DEFAULT_ORIGINS: Set[str] = _PROD_ORIGINS.copy()
else:
    DEFAULT_ORIGINS = _DEV_ORIGINS | _PROD_ORIGINS


class CORSConfig:
    """Centralized CORS configuration with environment variable support."""

    def __init__(self) -> None:
        """Initialize CORS config from environment or defaults."""
        env_origins = os.getenv("ARAGORA_ALLOWED_ORIGINS", "").strip()
        if env_origins:
            # Parse comma-separated origins from environment
            self.allowed_origins: Set[str] = {
                o.strip() for o in env_origins.split(",") if o.strip()
            }
            self._using_env_config = True
        else:
            self.allowed_origins = DEFAULT_ORIGINS.copy()
            self._using_env_config = False

            # Warn in production if not explicitly configured
            if _IS_PRODUCTION:
                logger.warning(
                    "[CORS] ARAGORA_ALLOWED_ORIGINS not set in production! "
                    "Using default production origins. For custom domains, "
                    "set ARAGORA_ALLOWED_ORIGINS explicitly."
                )
            else:
                logger.debug(
                    "[CORS] Using default origins (dev mode). "
                    "Set ARAGORA_ALLOWED_ORIGINS to customize."
                )

        # Security: Reject wildcard origins which bypass CORS protection
        if "*" in self.allowed_origins:
            raise ValueError(
                "Wildcard origin '*' is not allowed for security. "
                "Specify explicit origins in ARAGORA_ALLOWED_ORIGINS."
            )

        # Log configured origins at debug level
        logger.debug(f"[CORS] Allowed origins: {self.allowed_origins}")

    def is_origin_allowed(self, origin: str) -> bool:
        """Check if an origin is in the allowlist."""
        return origin in self.allowed_origins

    def get_origins_list(self) -> list[str]:
        """Return allowed origins as a list (for compatibility)."""
        return list(self.allowed_origins)

    def add_origin(self, origin: str) -> None:
        """Add an origin to the allowlist at runtime."""
        self.allowed_origins.add(origin)

    def remove_origin(self, origin: str) -> None:
        """Remove an origin from the allowlist at runtime."""
        self.allowed_origins.discard(origin)


# Singleton instance for import
cors_config = CORSConfig()

# Convenience exports for backwards compatibility
ALLOWED_ORIGINS = cors_config.get_origins_list()
WS_ALLOWED_ORIGINS = ALLOWED_ORIGINS  # Alias for stream.py compatibility
