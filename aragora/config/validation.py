"""
Input Validation Configuration.

Centralized configuration for request validation middleware, providing:
- Environment-based mode selection (blocking vs warn-only)
- Configurable limits for body size, JSON depth, etc.
- Production-safe defaults with opt-out for migration

Usage:
    from aragora.config.validation import get_validation_config, ValidationMode

    config = get_validation_config()

    if config.mode == ValidationMode.BLOCKING:
        # Reject invalid requests
        return error_response(400, "Invalid input")
    else:
        # Log but allow through (migration mode)
        logger.warning("Invalid input detected")

Environment Variables:
    ARAGORA_VALIDATION_MODE: "blocking" (default), "warn", or "disabled"
    ARAGORA_VALIDATION_BLOCKING: legacy alias, "true"/"false"
    ARAGORA_VALIDATION_MAX_BODY_SIZE: max body size in bytes (default 10MB)
    ARAGORA_VALIDATION_MAX_JSON_DEPTH: max nesting depth (default 10)
    ARAGORA_VALIDATION_MAX_ARRAY_ITEMS: max array items (default 1000)
    ARAGORA_VALIDATION_MAX_OBJECT_KEYS: max object keys (default 500)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)


class ValidationMode(Enum):
    """Validation enforcement mode.

    BLOCKING: Invalid requests are rejected with 400 error (production default)
    WARN: Invalid requests are logged but allowed through (migration mode)
    DISABLED: No validation performed (not recommended)
    """

    BLOCKING = "blocking"
    WARN = "warn"
    DISABLED = "disabled"


@dataclass(frozen=True)
class ValidationConfig:
    """Configuration for request validation.

    Attributes:
        mode: Enforcement mode (blocking, warn, disabled)
        max_body_size: Maximum request body size in bytes
        max_json_depth: Maximum JSON nesting depth
        max_array_items: Maximum items per JSON array
        max_object_keys: Maximum keys per JSON object
        log_all_requests: Log all validation attempts (verbose)
        strict_schemas: Require all fields match schema exactly
    """

    mode: ValidationMode = ValidationMode.BLOCKING
    max_body_size: int = 10_485_760  # 10MB
    max_json_depth: int = 10
    max_array_items: int = 1000
    max_object_keys: int = 500
    log_all_requests: bool = False
    strict_schemas: bool = False

    @property
    def is_blocking(self) -> bool:
        """Check if validation should block invalid requests."""
        return self.mode == ValidationMode.BLOCKING

    @property
    def is_enabled(self) -> bool:
        """Check if validation is enabled."""
        return self.mode != ValidationMode.DISABLED

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "mode": self.mode.value,
            "is_blocking": self.is_blocking,
            "is_enabled": self.is_enabled,
            "max_body_size": self.max_body_size,
            "max_json_depth": self.max_json_depth,
            "max_array_items": self.max_array_items,
            "max_object_keys": self.max_object_keys,
            "log_all_requests": self.log_all_requests,
            "strict_schemas": self.strict_schemas,
        }


def _parse_bool(value: str) -> bool:
    """Parse boolean from environment variable."""
    return value.lower() in ("true", "1", "yes", "on")


def _parse_int(value: str, default: int) -> int:
    """Parse integer from environment variable with fallback."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _determine_mode() -> ValidationMode:
    """Determine validation mode from environment.

    Priority:
    1. ARAGORA_VALIDATION_MODE (explicit mode)
    2. ARAGORA_VALIDATION_BLOCKING (legacy boolean)
    3. Default to BLOCKING for security
    """
    # New-style explicit mode
    mode_str = os.environ.get("ARAGORA_VALIDATION_MODE", "").lower()
    if mode_str:
        try:
            return ValidationMode(mode_str)
        except ValueError:
            logger.warning(
                "Unknown validation mode '%s', defaulting to blocking. Valid modes: %s",
                mode_str,
                [m.value for m in ValidationMode],
            )
            return ValidationMode.BLOCKING

    # Legacy boolean setting
    blocking_str = os.environ.get("ARAGORA_VALIDATION_BLOCKING", "")
    if blocking_str:
        if _parse_bool(blocking_str):
            return ValidationMode.BLOCKING
        else:
            return ValidationMode.WARN

    # Default to blocking for security
    return ValidationMode.BLOCKING


@lru_cache(maxsize=1)
def get_validation_config() -> ValidationConfig:
    """Get validation configuration from environment.

    This function is cached - configuration is loaded once at startup.
    Use `clear_validation_config_cache()` to reload.

    Returns:
        ValidationConfig with settings from environment
    """
    mode = _determine_mode()

    config = ValidationConfig(
        mode=mode,
        max_body_size=_parse_int(
            os.environ.get("ARAGORA_VALIDATION_MAX_BODY_SIZE", ""),
            10_485_760,
        ),
        max_json_depth=_parse_int(
            os.environ.get("ARAGORA_VALIDATION_MAX_JSON_DEPTH", ""),
            10,
        ),
        max_array_items=_parse_int(
            os.environ.get("ARAGORA_VALIDATION_MAX_ARRAY_ITEMS", ""),
            1000,
        ),
        max_object_keys=_parse_int(
            os.environ.get("ARAGORA_VALIDATION_MAX_OBJECT_KEYS", ""),
            500,
        ),
        log_all_requests=_parse_bool(os.environ.get("ARAGORA_VALIDATION_LOG_ALL", "false")),
        strict_schemas=_parse_bool(os.environ.get("ARAGORA_VALIDATION_STRICT", "false")),
    )

    # Log mode on startup
    if mode == ValidationMode.WARN:
        logger.warning(
            "Request validation running in WARN-ONLY mode. "
            "Invalid requests will be logged but not rejected. "
            "Set ARAGORA_VALIDATION_MODE=blocking for production."
        )
    elif mode == ValidationMode.DISABLED:
        logger.warning(
            "Request validation is DISABLED. This is not recommended. "
            "Set ARAGORA_VALIDATION_MODE=blocking for production."
        )
    else:
        logger.info(
            "Request validation enabled in BLOCKING mode. Max body: %sMB, max depth: %s",
            config.max_body_size // 1024 // 1024,
            config.max_json_depth,
        )

    return config


def clear_validation_config_cache() -> None:
    """Clear the cached validation config.

    Call this to reload configuration from environment,
    primarily useful for testing.
    """
    get_validation_config.cache_clear()


def create_validation_response(
    errors: list[str],
    config: ValidationConfig | None = None,
) -> dict[str, Any]:
    """Create a standardized validation error response.

    Args:
        errors: List of validation error messages
        config: Optional config (uses default if not provided)

    Returns:
        Response dict suitable for JSONResponse
    """
    config = config or get_validation_config()

    if not config.is_blocking:
        # In warn mode, return None to signal "don't block"
        return {}

    return {
        "error": "Validation failed",
        "code": "validation_error",
        "details": {
            "errors": errors,
            "count": len(errors),
        },
    }


# Route-specific overrides for validation limits
@dataclass
class RouteValidationOverride:
    """Per-route validation limit overrides.

    Use for routes that legitimately need higher limits,
    like batch operations or large document uploads.
    """

    path_pattern: str
    max_body_size: int | None = None
    max_json_depth: int | None = None
    max_array_items: int | None = None


# Default route overrides for known high-volume endpoints
DEFAULT_ROUTE_OVERRIDES: list[RouteValidationOverride] = [
    # Batch endpoints allow larger payloads
    RouteValidationOverride(
        path_pattern=r"^/api/(v1/)?batch",
        max_body_size=50_000_000,  # 50MB
        max_array_items=10000,
    ),
    # Knowledge/document uploads
    RouteValidationOverride(
        path_pattern=r"^/api/(v1/)?knowledge",
        max_body_size=100_000_000,  # 100MB
    ),
    # Large debate contexts
    RouteValidationOverride(
        path_pattern=r"^/api/(v1/)?debates?$",
        max_body_size=10_000_000,  # 10MB
        max_json_depth=15,
    ),
]


__all__ = [
    "ValidationMode",
    "ValidationConfig",
    "RouteValidationOverride",
    "get_validation_config",
    "clear_validation_config_cache",
    "create_validation_response",
    "DEFAULT_ROUTE_OVERRIDES",
]
