"""
Configuration Validator for Aragora.

Provides comprehensive configuration validation at startup, including
production-specific security checks.

Usage:
    from aragora.config.validator import validate_all, validate_production

    # Basic validation
    result = validate_all()
    if not result["valid"]:
        for error in result["errors"]:
            print(f"ERROR: {error}")

    # Production mode (strict)
    result = validate_production()  # Raises ConfigurationError on failure
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration validation fails in strict mode."""

    pass


def validate_all(strict: bool = False) -> Dict[str, Any]:
    """
    Validate all required configuration.

    Combines basic validation from legacy.py with additional security
    and production checks.

    Args:
        strict: If True, raise ConfigurationError on failures

    Returns:
        Dict with validation results:
        {
            "valid": True/False,
            "errors": [...],
            "warnings": [...],
            "config_summary": {...}
        }
    """
    errors: List[str] = []
    warnings: List[str] = []

    # Run basic validation from legacy module
    try:
        from aragora.config.legacy import validate_configuration

        basic_result = validate_configuration(strict=False)
        errors.extend(basic_result.get("errors", []))
        warnings.extend(basic_result.get("warnings", []))
        config_summary = basic_result.get("config_summary", {})
    except Exception as e:
        errors.append(f"Basic configuration validation failed: {e}")
        config_summary = {}

    # Additional security checks
    env = os.environ.get("ARAGORA_ENV", "development").lower()
    is_production = env in ("production", "prod", "live")

    # Check encryption key in production
    if is_production:
        if not os.environ.get("ARAGORA_ENCRYPTION_KEY"):
            errors.append("ARAGORA_ENCRYPTION_KEY required in production for secrets encryption")

        # Check for debug mode in production
        if os.environ.get("ARAGORA_DEBUG", "").lower() in ("true", "1", "yes"):
            warnings.append(
                "ARAGORA_DEBUG is enabled in production - this may expose sensitive info"
            )

        # Check for secure cookies
        if os.environ.get("ARAGORA_SECURE_COOKIES", "").lower() not in ("true", "1", "yes"):
            warnings.append("ARAGORA_SECURE_COOKIES should be enabled in production")

        # Check for HTTPS
        base_url = os.environ.get("ARAGORA_BASE_URL", "")
        if base_url and not base_url.startswith("https://"):
            warnings.append("ARAGORA_BASE_URL should use HTTPS in production")

    # Check Redis configuration for horizontal scaling
    try:
        from aragora.control_plane.leader import is_distributed_state_required

        distributed_required = is_distributed_state_required()
    except ImportError:
        distributed_required = False

    state_backend = os.environ.get("ARAGORA_STATE_BACKEND", "")
    redis_url = os.environ.get("ARAGORA_REDIS_URL", "") or os.environ.get("REDIS_URL", "")

    if state_backend == "redis" and not redis_url:
        errors.append("ARAGORA_STATE_BACKEND=redis but no Redis URL configured")

    if distributed_required and not redis_url:
        errors.append(
            "Distributed state required (multi-instance or production) but REDIS_URL not configured. "
            "Set ARAGORA_SINGLE_INSTANCE=true for single-node deployments."
        )

    if redis_url:
        config_summary["redis_configured"] = True
        config_summary["state_backend"] = state_backend or "hybrid"
    else:
        config_summary["redis_configured"] = False
        config_summary["state_backend"] = "sqlite"

    # Check database configuration
    db_backend = os.environ.get("ARAGORA_DB_BACKEND", "sqlite").lower()
    config_summary["db_backend"] = db_backend

    if db_backend in ("postgres", "postgresql"):
        pg_dsn = os.environ.get("ARAGORA_POSTGRES_DSN") or os.environ.get("DATABASE_URL")
        if not pg_dsn:
            errors.append(
                "ARAGORA_DB_BACKEND=postgres but no PostgreSQL DSN configured. "
                "Set ARAGORA_POSTGRES_DSN or DATABASE_URL"
            )

    # Check JWT secret for user auth
    jwt_secret = os.environ.get("SUPABASE_JWT_SECRET") or os.environ.get("ARAGORA_JWT_SECRET")
    if not jwt_secret:
        warnings.append(
            "No JWT secret configured (SUPABASE_JWT_SECRET or ARAGORA_JWT_SECRET). "
            "User authentication will be limited."
        )

    # Check Supabase configuration
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_ANON_KEY")
    if supabase_url and not supabase_key:
        warnings.append("SUPABASE_URL set but SUPABASE_KEY not configured")

    config_summary["supabase_configured"] = bool(supabase_url and supabase_key)

    # Update config summary
    config_summary["environment"] = env
    config_summary["is_production"] = is_production
    config_summary["encryption_configured"] = bool(os.environ.get("ARAGORA_ENCRYPTION_KEY"))

    # Build result
    is_valid = len(errors) == 0
    result = {
        "valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "config_summary": config_summary,
    }

    # Log results
    if is_valid:
        logger.info("Configuration validation passed")
    else:
        for error in errors:
            logger.error(f"Configuration error: {error}")
    for warning in warnings:
        logger.warning(f"Configuration warning: {warning}")

    if strict and errors:
        raise ConfigurationError(f"Configuration validation failed: {'; '.join(errors)}")

    return result


def validate_production() -> Dict[str, Any]:
    """
    Validate configuration for production deployment.

    This is a strict validation that raises ConfigurationError on any failures.
    Call this at server startup in production environments.

    Returns:
        Validation result dict

    Raises:
        ConfigurationError: If validation fails
    """
    return validate_all(strict=True)


def get_missing_required_keys() -> List[str]:
    """
    Get list of missing required environment variables.

    Returns:
        List of missing variable names
    """
    missing = []

    # At least one AI provider
    ai_providers = [
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "XAI_API_KEY",
        "OPENROUTER_API_KEY",
    ]
    if not any(os.environ.get(key) for key in ai_providers):
        missing.append("AI_PROVIDER_KEY (one of: " + ", ".join(ai_providers) + ")")

    # Check production-only requirements
    env = os.environ.get("ARAGORA_ENV", "development").lower()
    if env in ("production", "prod", "live"):
        if not os.environ.get("ARAGORA_ENCRYPTION_KEY"):
            missing.append("ARAGORA_ENCRYPTION_KEY")
        if not os.environ.get("ARAGORA_API_TOKEN"):
            missing.append("ARAGORA_API_TOKEN")

    return missing


def print_config_status() -> None:
    """Print a formatted configuration status report to stdout."""
    result = validate_all(strict=False)

    summary = result.get("config_summary", {})

    summary.get("api_providers", [])

    if result["errors"]:
        for error in result["errors"]:
            pass

    if result["warnings"]:
        for warning in result["warnings"]:
            pass

    "PASS" if result["valid"] else "FAIL"


__all__ = [
    "ConfigurationError",
    "validate_all",
    "validate_production",
    "get_missing_required_keys",
    "print_config_status",
]
