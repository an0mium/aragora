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
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Environment modes considered non-production (wildcards and HTTP allowed with warnings)
_DEV_MODES = {"development", "dev", "local", "test"}

# Check if we're in production mode
# Production = ARAGORA_ENV is set and NOT one of the dev modes
_ENV_MODE = os.environ.get("ARAGORA_ENV", "").lower()
_IS_PRODUCTION = _ENV_MODE not in _DEV_MODES and _ENV_MODE != ""

# Development origins (included by default in dev mode only)
_DEV_ORIGINS: set[str] = {
    "http://localhost:3000",
    "http://localhost:8080",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8080",
}

# Production origins (included by default in production)
_PROD_ORIGINS: set[str] = {
    "https://aragora.ai",
    "https://www.aragora.ai",
    "https://live.aragora.ai",
    "https://api.aragora.ai",
    "https://aragora.pages.dev",  # Cloudflare Pages deployment
}

# Default allowed origins based on environment
if _IS_PRODUCTION:
    DEFAULT_ORIGINS: set[str] = _PROD_ORIGINS.copy()
else:
    DEFAULT_ORIGINS = _DEV_ORIGINS | _PROD_ORIGINS


def _is_production_mode(env_mode_override: str | None = None) -> bool:
    """Determine if we are in production mode.

    Args:
        env_mode_override: Override for ARAGORA_ENV (used in testing only).
                           When None, uses the module-level _IS_PRODUCTION detection.

    Returns:
        True if running in production mode.
    """
    if env_mode_override is not None:
        mode = env_mode_override.lower()
        return mode not in _DEV_MODES and mode != ""
    return _IS_PRODUCTION


class CORSConfig:
    """Centralized CORS configuration with environment variable support."""

    def __init__(self, *, _env_mode: str | None = None) -> None:
        """Initialize CORS config from environment or defaults.

        Args:
            _env_mode: Override for ARAGORA_ENV (used in testing only).
                       When None, uses the module-level _IS_PRODUCTION detection.
        """
        is_production = _is_production_mode(_env_mode)

        env_origins = os.getenv("ARAGORA_ALLOWED_ORIGINS", "").strip()
        if env_origins:
            # Parse comma-separated origins from environment
            self.allowed_origins: set[str] = {
                o.strip() for o in env_origins.split(",") if o.strip()
            }
            self._using_env_config = True
        else:
            self.allowed_origins = DEFAULT_ORIGINS.copy()
            self._using_env_config = False

            # Warn in production if not explicitly configured
            if is_production:
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

        # --- Wildcard validation (environment-aware) ---
        if "*" in self.allowed_origins:
            if is_production:
                raise ValueError(
                    "Wildcard origin '*' is not allowed in production. "
                    "Specify explicit origins in ARAGORA_ALLOWED_ORIGINS."
                )
            else:
                logger.warning(
                    "[CORS] Wildcard origin '*' detected in development mode. "
                    "This disables CORS protection. Do NOT use in production."
                )
                # Remove wildcard from the set so downstream validation
                # doesn't try to parse it as a URL. The wildcard flag is
                # stored separately for is_origin_allowed() to honour.
                self.allowed_origins.discard("*")
                self._allow_all = True
        else:
            self._allow_all = False

        # --- Strip trailing slashes for consistency ---
        self.allowed_origins = {origin.rstrip("/") for origin in self.allowed_origins}

        # --- Validate each origin ---
        for origin in self.allowed_origins:
            parsed = urlparse(origin)
            if not parsed.scheme or not parsed.hostname:
                raise ValueError(
                    f"Invalid CORS origin '{origin}': must include scheme "
                    f"(e.g. https://example.com). Check ARAGORA_ALLOWED_ORIGINS."
                )
            if parsed.scheme not in ("http", "https"):
                raise ValueError(
                    f"Invalid CORS origin '{origin}': scheme must be "
                    f"http or https. Check ARAGORA_ALLOWED_ORIGINS."
                )
            # Warn about non-HTTPS origins in production
            if is_production and parsed.scheme != "https":
                logger.warning(
                    f"[CORS] Non-HTTPS origin '{origin}' configured in production. "
                    "HTTPS is strongly recommended for all production origins."
                )

        # Log configured origins at debug level
        logger.debug(f"[CORS] Allowed origins: {self.allowed_origins}")

    def is_origin_allowed(self, origin: str) -> bool:
        """Check if an origin is in the allowlist."""
        if self._allow_all:
            return True
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
