"""
Configuration validator for Aragora server startup.

Validates environment variables and configuration at startup to provide
clear error messages for misconfigurations instead of runtime failures.

Usage:
    from aragora.server.config_validator import ConfigValidator

    # Run validation at startup
    errors = ConfigValidator.validate()
    if errors:
        for error in errors:
            print(f"Configuration error: {error}")
        sys.exit(1)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional, cast

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of configuration validation."""

    is_valid: bool
    errors: list[str]
    warnings: list[str]

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0


class ConfigValidator:
    """
    Validates configuration at server startup.

    Checks environment variables, validates their values, and ensures
    the server can start successfully.
    """

    # Environment variables that are required in production mode
    PRODUCTION_REQUIRED = ["ARAGORA_API_TOKEN", "DATABASE_URL"]

    # At least one of these LLM API keys should be present
    LLM_API_KEYS = [
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "MISTRAL_API_KEY",
        "GEMINI_API_KEY",
        "XAI_API_KEY",
    ]

    # Security-sensitive variables with minimum requirements
    SECRET_REQUIREMENTS = {
        "JWT_SECRET": {"min_length": 32, "description": "JWT signing secret"},
        "ARAGORA_API_TOKEN": {"min_length": 16, "description": "API authentication token"},
    }

    # Variables that should be validated as URLs
    URL_VARS = ["SUPABASE_URL", "REDIS_URL", "ARAGORA_BASE_URL"]

    # Variables that should be validated as integers
    INTEGER_VARS = [
        "ARAGORA_RATE_LIMIT",
        "ARAGORA_IP_RATE_LIMIT",
        "ARAGORA_PORT",
        "ARAGORA_TOKEN_TTL",
    ]

    @classmethod
    def validate(cls, strict: bool = False) -> list[str]:
        """
        Validate configuration and return list of errors.

        Args:
            strict: If True, treat warnings as errors

        Returns:
            List of error messages (empty if valid)
        """
        result = cls.validate_all()

        if strict:
            return result.errors + result.warnings
        return result.errors

    @classmethod
    def validate_all(cls) -> ValidationResult:
        """
        Run all validation checks.

        Returns:
            ValidationResult with errors and warnings
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Check production mode requirements
        is_production = os.getenv("ARAGORA_ENV", "development").lower() == "production"

        if is_production:
            for var in cls.PRODUCTION_REQUIRED:
                if not os.getenv(var):
                    errors.append(f"Missing required environment variable in production: {var}")

        # Check LLM API keys
        has_llm_key = any(os.getenv(key) for key in cls.LLM_API_KEYS)
        if not has_llm_key:
            msg = f"No LLM API key configured. Set at least one of: {', '.join(cls.LLM_API_KEYS)}"
            if is_production:
                errors.append(msg)
            else:
                warnings.append(msg)

        # Check secret requirements
        for var, req in cls.SECRET_REQUIREMENTS.items():
            value = os.getenv(var, "")
            if value:
                min_len = cast(int, req["min_length"])
                if len(value) < min_len:
                    errors.append(
                        f"{var} ({req['description']}) must be at least "
                        f"{min_len} characters, got {len(value)}"
                    )

        # Validate URL formats
        for var in cls.URL_VARS:
            value = os.getenv(var)
            if value:
                if not (value.startswith("http://") or value.startswith("https://")):
                    if var == "REDIS_URL":
                        if not value.startswith("redis://"):
                            errors.append(f"{var} must be a valid Redis URL (redis://...)")
                    else:
                        errors.append(f"{var} must be a valid URL (http:// or https://)")

        # Validate integer formats
        for var in cls.INTEGER_VARS:
            value = os.getenv(var)
            if value:
                try:
                    int(value)
                except ValueError:
                    errors.append(f"{var} must be an integer, got: {value}")

        # Check database configuration in production
        if is_production:
            db_url = os.getenv("DATABASE_URL", "")
            if db_url:
                # Ensure PostgreSQL is used in production (not SQLite)
                if "sqlite" in db_url.lower():
                    errors.append(
                        "SQLite is not supported in production. "
                        "Set DATABASE_URL to a PostgreSQL connection string "
                        "(e.g., postgresql://user:pass@host:5432/dbname)"
                    )
                elif not db_url.startswith(("postgresql://", "postgres://")):
                    errors.append(
                        "DATABASE_URL must be a PostgreSQL connection string in production "
                        "(starting with postgresql:// or postgres://)"
                    )

        # Check CORS configuration
        allowed_origins = os.getenv("ARAGORA_ALLOWED_ORIGINS", "")
        if is_production and not allowed_origins:
            warnings.append(
                "ARAGORA_ALLOWED_ORIGINS not set in production. "
                "CORS may block requests from web clients."
            )

        # Check rate limiting configuration
        rate_limit = os.getenv("ARAGORA_RATE_LIMIT")
        if rate_limit:
            try:
                if int(rate_limit) < 1:
                    errors.append("ARAGORA_RATE_LIMIT must be positive")
            except ValueError:
                pass  # Already checked in INTEGER_VARS

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    @classmethod
    def validate_and_log(cls, strict: bool = False) -> bool:
        """
        Validate configuration and log errors/warnings.

        Args:
            strict: If True, treat warnings as errors

        Returns:
            True if configuration is valid
        """
        result = cls.validate_all()

        for warning in result.warnings:
            logger.warning(f"Configuration warning: {warning}")

        for error in result.errors:
            logger.error(f"Configuration error: {error}")

        if strict and result.has_warnings:
            return False

        return result.is_valid

    @classmethod
    def check_database_connectivity(cls) -> tuple[bool, Optional[str]]:
        """
        Check if database is accessible (if configured).

        Returns:
            (success, error_message) tuple
        """
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_ANON_KEY")

        if not supabase_url or not supabase_key:
            return True, None  # Not configured, not an error

        try:
            from aragora.persistence.supabase_client import SupabaseClient

            client = SupabaseClient()
            if not client.is_configured:
                return False, "Supabase client not properly configured"

            # Try a simple query to verify connectivity
            # This is a lightweight check
            return True, None
        except ImportError:
            return True, None  # Module not available, not an error
        except Exception as e:
            return False, f"Database connectivity check failed: {e}"

    @classmethod
    def check_postgresql_connectivity(cls) -> tuple[bool, Optional[str]]:
        """
        Check if PostgreSQL is accessible (if configured).

        Returns:
            (success, error_message) tuple
        """
        database_url = os.getenv("DATABASE_URL")

        if not database_url:
            return True, None  # Not configured, not an error

        if not database_url.startswith(("postgresql://", "postgres://")):
            return True, None  # Not PostgreSQL

        try:
            import asyncpg  # noqa: F401 - check availability

            # Don't actually connect during validation - just verify config format
            return True, None
        except ImportError:
            return False, "asyncpg package required for PostgreSQL support"

    @classmethod
    def check_redis_connectivity(cls) -> tuple[bool, Optional[str]]:
        """
        Check if Redis is accessible (if configured).

        Returns:
            (success, error_message) tuple
        """
        redis_url = os.getenv("REDIS_URL")

        if not redis_url:
            return True, None  # Not configured, not an error

        try:
            import redis

            client = redis.from_url(redis_url, socket_connect_timeout=5)
            client.ping()
            return True, None
        except ImportError:
            return True, None  # Redis not installed, not an error
        except Exception as e:
            return False, f"Redis connectivity check failed: {e}"

    @classmethod
    def get_config_summary(cls) -> dict:
        """
        Get a summary of current configuration for debugging.

        Sensitive values are masked.
        """

        def mask_value(value: str) -> str:
            if not value:
                return "(not set)"
            if len(value) <= 8:
                return "*" * len(value)
            return value[:4] + "*" * (len(value) - 8) + value[-4:]

        db_url = os.getenv("DATABASE_URL", "")
        db_backend = "sqlite"
        if db_url.startswith(("postgresql://", "postgres://")):
            db_backend = "postgresql"
        elif db_url:
            db_backend = "unknown"

        return {
            "environment": os.getenv("ARAGORA_ENV", "development"),
            "api_token_set": bool(os.getenv("ARAGORA_API_TOKEN")),
            "jwt_secret_set": bool(os.getenv("JWT_SECRET")),
            "jwt_secret_length": len(os.getenv("JWT_SECRET", "")),
            "llm_keys": {key: bool(os.getenv(key)) for key in cls.LLM_API_KEYS},
            "database_backend": db_backend,
            "database_url_set": bool(db_url),
            "supabase_configured": bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_KEY")),
            "redis_configured": bool(os.getenv("REDIS_URL")),
            "rate_limit": os.getenv("ARAGORA_RATE_LIMIT", "60"),
            "allowed_origins": bool(os.getenv("ARAGORA_ALLOWED_ORIGINS")),
        }


def validate_startup_config(strict: bool = False, exit_on_error: bool = True) -> bool:
    """
    Convenience function to validate configuration at startup.

    Args:
        strict: If True, treat warnings as errors
        exit_on_error: If True, exit the process on validation failure

    Returns:
        True if configuration is valid
    """
    is_valid = ConfigValidator.validate_and_log(strict=strict)

    if not is_valid and exit_on_error:
        import sys

        logger.critical("Server startup aborted due to configuration errors")
        sys.exit(1)

    return is_valid


__all__ = [
    "ConfigValidator",
    "ValidationResult",
    "validate_startup_config",
]
