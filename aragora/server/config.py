"""
Centralized server configuration module.

Provides dataclasses for server-layer configuration with dependency injection
support. Replaces scattered os.getenv() calls with centralized, typed configuration.

Usage:
    from aragora.server.config import (
        ServerEnvironmentConfig,
        SSOConfig,
        OAuthConfig,
        PagerDutyConfig,
        ResearchConfig,
        OpenRouterConfig,
        SentryConfig,
        LoggingConfig,
        SecurityConfig,
        TimeoutConfig,
        AuthConfig,
        AuditConfig,
        RateLimitConfig,
        XSSProtectionConfig,
        get_server_environment_config,
        get_sso_config,
        # ... etc
    )

    # Factory function pattern
    config = get_server_environment_config()

    # Or pass config directly for testing
    config = ServerEnvironmentConfig(env="test", is_production=False)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


# =============================================================================
# Core Server Environment Config
# =============================================================================


@dataclass(frozen=True)
class ServerEnvironmentConfig:
    """Core server environment configuration.

    Controls environment mode and production-specific behaviors.
    """

    env: str = field(default_factory=lambda: os.getenv("ARAGORA_ENV", "development"))
    is_production: bool = field(default=False)

    def __post_init__(self):
        # Use object.__setattr__ for frozen dataclass
        object.__setattr__(self, "is_production", self.env.lower() == "production")


def get_server_environment_config() -> ServerEnvironmentConfig:
    """Factory function for ServerEnvironmentConfig."""
    return ServerEnvironmentConfig()


# =============================================================================
# SSO Configuration
# =============================================================================


@dataclass(frozen=True)
class SSOConfig:
    """SSO (Single Sign-On) configuration.

    Controls redirect host validation and production security requirements.
    """

    env: str = field(default_factory=lambda: os.getenv("ARAGORA_ENV", "development"))
    allowed_redirect_hosts: str = field(
        default_factory=lambda: os.getenv("ARAGORA_SSO_ALLOWED_REDIRECT_HOSTS", "")
    )

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.env.lower() == "production"

    @property
    def allowed_hosts_list(self) -> list[str]:
        """Get list of allowed redirect hosts."""
        if not self.allowed_redirect_hosts:
            return []
        return [h.strip().lower() for h in self.allowed_redirect_hosts.split(",")]


def get_sso_config() -> SSOConfig:
    """Factory function for SSOConfig."""
    return SSOConfig()


# =============================================================================
# OAuth Configuration
# =============================================================================


@dataclass(frozen=True)
class OAuthConfig:
    """OAuth configuration for social media connectors.

    Controls allowed OAuth redirect hosts for CSRF protection.
    """

    env: str = field(default_factory=lambda: os.getenv("ARAGORA_ENV", "development"))
    allowed_oauth_hosts: str | None = field(
        default_factory=lambda: os.getenv("ARAGORA_ALLOWED_OAUTH_HOSTS")
    )

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.env.lower() == "production"

    @property
    def allowed_hosts(self) -> frozenset[str]:
        """Get frozenset of allowed OAuth hosts."""
        if self.allowed_oauth_hosts:
            return frozenset(h.strip() for h in self.allowed_oauth_hosts.split(","))
        if self.is_production:
            return frozenset()
        # Dev mode: use localhost fallbacks
        return frozenset(["localhost:8080", "127.0.0.1:8080"])


def get_oauth_config() -> OAuthConfig:
    """Factory function for OAuthConfig."""
    return OAuthConfig()


# =============================================================================
# PagerDuty / DevOps Configuration
# =============================================================================


@dataclass(frozen=True)
class PagerDutyConfig:
    """PagerDuty incident management configuration."""

    api_key: str | None = field(default_factory=lambda: os.getenv("PAGERDUTY_API_KEY"))
    email: str | None = field(default_factory=lambda: os.getenv("PAGERDUTY_EMAIL"))
    webhook_secret: str | None = field(
        default_factory=lambda: os.getenv("PAGERDUTY_WEBHOOK_SECRET")
    )

    @property
    def is_configured(self) -> bool:
        """Check if PagerDuty is properly configured."""
        return bool(self.api_key and self.email)


def get_pagerduty_config() -> PagerDutyConfig:
    """Factory function for PagerDutyConfig."""
    return PagerDutyConfig()


# =============================================================================
# Research Phase Configuration
# =============================================================================


@dataclass(frozen=True)
class ResearchConfig:
    """Pre-debate research phase configuration.

    API keys for external search services.
    """

    brave_api_key: str | None = field(default_factory=lambda: os.getenv("BRAVE_API_KEY"))
    serper_api_key: str | None = field(default_factory=lambda: os.getenv("SERPER_API_KEY"))

    @property
    def has_brave(self) -> bool:
        """Check if Brave Search API is configured."""
        return bool(self.brave_api_key)

    @property
    def has_serper(self) -> bool:
        """Check if Serper API is configured."""
        return bool(self.serper_api_key)

    @property
    def has_any_search_api(self) -> bool:
        """Check if any external search API is configured."""
        return self.has_brave or self.has_serper


def get_research_config() -> ResearchConfig:
    """Factory function for ResearchConfig."""
    return ResearchConfig()


# =============================================================================
# OpenRouter Configuration
# =============================================================================


@dataclass(frozen=True)
class OpenRouterConfig:
    """OpenRouter API configuration for LLM fallback."""

    api_key: str | None = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY"))

    @property
    def is_available(self) -> bool:
        """Check if OpenRouter API key is configured."""
        return bool(self.api_key and self.api_key.strip())


def get_openrouter_config() -> OpenRouterConfig:
    """Factory function for OpenRouterConfig."""
    return OpenRouterConfig()


# =============================================================================
# Sentry Error Monitoring Configuration
# =============================================================================


@dataclass(frozen=True)
class SentryConfig:
    """Sentry error monitoring configuration."""

    dsn: str | None = field(default_factory=lambda: os.getenv("SENTRY_DSN"))
    environment: str = field(default_factory=lambda: os.getenv("SENTRY_ENVIRONMENT", "development"))
    traces_sample_rate: float = field(
        default_factory=lambda: float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.1"))
    )
    profiles_sample_rate: float = field(
        default_factory=lambda: float(os.getenv("SENTRY_PROFILES_SAMPLE_RATE", "0.1"))
    )
    server_name: str | None = field(default_factory=lambda: os.getenv("SENTRY_SERVER_NAME"))

    def __post_init__(self) -> None:
        """Validate Sentry configuration values."""
        if not 0 <= self.traces_sample_rate <= 1:
            raise ValueError(
                f"traces_sample_rate must be between 0 and 1, got {self.traces_sample_rate}"
            )
        if not 0 <= self.profiles_sample_rate <= 1:
            raise ValueError(
                f"profiles_sample_rate must be between 0 and 1, got {self.profiles_sample_rate}"
            )

    @property
    def is_configured(self) -> bool:
        """Check if Sentry DSN is configured."""
        return bool(self.dsn)


def get_sentry_config() -> SentryConfig:
    """Factory function for SentryConfig."""
    return SentryConfig()


# =============================================================================
# Logging Configuration
# =============================================================================


@dataclass(frozen=True)
class LoggingConfig:
    """Structured logging configuration."""

    log_level: str = field(default_factory=lambda: os.getenv("ARAGORA_LOG_LEVEL", "INFO").upper())
    log_format: str = field(default_factory=lambda: os.getenv("ARAGORA_LOG_FORMAT", "json"))
    include_timestamp: bool = field(
        default_factory=lambda: os.getenv("ARAGORA_LOG_TIMESTAMP", "true").lower() == "true"
    )


def get_logging_config() -> LoggingConfig:
    """Factory function for LoggingConfig."""
    return LoggingConfig()


# =============================================================================
# Security Middleware Configuration
# =============================================================================


@dataclass(frozen=True)
class SecurityConfig:
    """Security middleware configuration (CSP, CORS, etc)."""

    env: str = field(default_factory=lambda: os.getenv("ARAGORA_ENV", "development"))
    trusted_proxies: str = field(
        default_factory=lambda: os.getenv("ARAGORA_TRUSTED_PROXIES", "127.0.0.1,::1,localhost")
    )
    enable_csp: bool = field(
        default_factory=lambda: os.getenv("ARAGORA_ENABLE_CSP", "true").lower() == "true"
    )
    csp_mode: str = field(default_factory=lambda: os.getenv("ARAGORA_CSP_MODE", "standard"))
    csp_report_uri: str | None = field(default_factory=lambda: os.getenv("ARAGORA_CSP_REPORT_URI"))

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.env.lower() == "production"

    @property
    def trusted_proxies_list(self) -> list[str]:
        """Get list of trusted proxy addresses."""
        return [p.strip() for p in self.trusted_proxies.split(",") if p.strip()]

    @property
    def allow_unsafe_inline(self) -> bool:
        """Check if unsafe-inline should be allowed (dev mode only)."""
        return not self.is_production


def get_security_config() -> SecurityConfig:
    """Factory function for SecurityConfig."""
    return SecurityConfig()


# =============================================================================
# Timeout Configuration
# =============================================================================


@dataclass(frozen=True)
class TimeoutConfig:
    """Request timeout configuration."""

    default_timeout: float = field(
        default_factory=lambda: float(os.getenv("ARAGORA_REQUEST_TIMEOUT", "30"))
    )
    slow_timeout: float = field(
        default_factory=lambda: float(os.getenv("ARAGORA_SLOW_REQUEST_TIMEOUT", "120"))
    )
    max_timeout: float = field(
        default_factory=lambda: float(os.getenv("ARAGORA_MAX_REQUEST_TIMEOUT", "600"))
    )
    timeout_workers: int = field(
        default_factory=lambda: int(os.getenv("ARAGORA_TIMEOUT_WORKERS", "10"))
    )

    def __post_init__(self) -> None:
        """Validate timeout configuration values."""
        if self.default_timeout <= 0:
            raise ValueError(f"default_timeout must be positive, got {self.default_timeout}")
        if self.max_timeout < self.default_timeout:
            raise ValueError(
                f"max_timeout ({self.max_timeout}) must be >= default_timeout ({self.default_timeout})"
            )
        if self.timeout_workers < 1:
            raise ValueError(f"timeout_workers must be >= 1, got {self.timeout_workers}")


def get_timeout_config() -> TimeoutConfig:
    """Factory function for TimeoutConfig."""
    return TimeoutConfig()


# =============================================================================
# User Auth Configuration
# =============================================================================


@dataclass(frozen=True)
class UserAuthConfig:
    """User authentication middleware configuration."""

    jwt_secret: str | None = field(default_factory=lambda: os.getenv("SUPABASE_JWT_SECRET"))
    supabase_url: str | None = field(default_factory=lambda: os.getenv("SUPABASE_URL"))
    env: str = field(default_factory=lambda: os.getenv("ARAGORA_ENV", "development"))
    environment: str = field(
        default_factory=lambda: os.getenv("ARAGORA_ENVIRONMENT", "development")
    )

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.env.lower() == "production"

    @property
    def is_configured(self) -> bool:
        """Check if auth is properly configured."""
        return bool(self.jwt_secret and self.supabase_url)


def get_user_auth_config() -> UserAuthConfig:
    """Factory function for UserAuthConfig."""
    return UserAuthConfig()


# =============================================================================
# Audit Logger Configuration
# =============================================================================


@dataclass(frozen=True)
class AuditConfig:
    """Audit logging configuration."""

    log_dir: str = field(default_factory=lambda: os.getenv("AUDIT_LOG_DIR", "logs/audit"))
    testing: bool = field(default_factory=lambda: os.getenv("TESTING", "") == "1")


def get_audit_config() -> AuditConfig:
    """Factory function for AuditConfig."""
    return AuditConfig()


# =============================================================================
# Rate Limit Configuration
# =============================================================================


@dataclass(frozen=True)
class RateLimitConfig:
    """Rate limiting configuration."""

    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379"))
    fail_open: bool = field(
        default_factory=lambda: os.getenv("ARAGORA_RATE_LIMIT_FAIL_OPEN", "false").lower() == "true"
    )
    redis_failure_threshold: int = field(
        default_factory=lambda: int(os.getenv("ARAGORA_REDIS_FAILURE_THRESHOLD", "3"))
    )
    circuit_breaker_enabled: bool = field(
        default_factory=lambda: os.getenv("ARAGORA_RATE_LIMIT_CIRCUIT_BREAKER", "true").lower()
        == "true"
    )
    distributed_metrics_enabled: bool = field(
        default_factory=lambda: os.getenv("ARAGORA_RATE_LIMIT_DISTRIBUTED_METRICS", "true").lower()
        == "true"
    )
    metrics_aggregation_interval: int = field(
        default_factory=lambda: int(os.getenv("ARAGORA_RATE_LIMIT_METRICS_INTERVAL", "60"))
    )
    instance_id: str | None = field(
        default_factory=lambda: os.getenv("ARAGORA_INSTANCE_ID")
        or os.getenv("HOSTNAME")
        or os.getenv("POD_NAME")
    )
    trusted_proxies: str = field(
        default_factory=lambda: os.getenv(
            "ARAGORA_TRUSTED_PROXIES", "127.0.0.1,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"
        )
    )

    def __post_init__(self) -> None:
        """Validate rate limit configuration values."""
        if self.redis_failure_threshold < 1:
            raise ValueError(
                f"redis_failure_threshold must be >= 1, got {self.redis_failure_threshold}"
            )
        if self.metrics_aggregation_interval < 1:
            raise ValueError(
                f"metrics_aggregation_interval must be >= 1, got {self.metrics_aggregation_interval}"
            )


def get_rate_limit_config() -> RateLimitConfig:
    """Factory function for RateLimitConfig."""
    return RateLimitConfig()


# =============================================================================
# XSS Protection Configuration
# =============================================================================


@dataclass(frozen=True)
class XSSProtectionConfig:
    """XSS protection middleware configuration."""

    auto_escape_html: bool = field(
        default_factory=lambda: os.getenv("ARAGORA_AUTO_ESCAPE_HTML", "true").lower() == "true"
    )
    enforce_cookie_security: bool = field(
        default_factory=lambda: os.getenv("ARAGORA_ENFORCE_COOKIE_SECURITY", "true").lower()
        == "true"
    )
    cookie_samesite: str = field(
        default_factory=lambda: os.getenv("ARAGORA_COOKIE_SAMESITE", "Lax")
    )
    cookie_secure: bool = field(
        default_factory=lambda: os.getenv("ARAGORA_COOKIE_SECURE", "true").lower() == "true"
    )
    cookie_httponly: bool = field(
        default_factory=lambda: os.getenv("ARAGORA_COOKIE_HTTPONLY", "true").lower() == "true"
    )
    enable_csp: bool = field(
        default_factory=lambda: os.getenv("ARAGORA_ENABLE_CSP", "true").lower() == "true"
    )
    csp_report_uri: str = field(
        default_factory=lambda: os.getenv("ARAGORA_CSP_REPORT_URI", "/api/csp-report")
    )


def get_xss_protection_config() -> XSSProtectionConfig:
    """Factory function for XSSProtectionConfig."""
    return XSSProtectionConfig()


# =============================================================================
# Server Auth Configuration
# =============================================================================


@dataclass(frozen=True)
class ServerAuthConfig:
    """Server-level authentication configuration."""

    api_token: str | None = field(default_factory=lambda: os.getenv("ARAGORA_API_TOKEN"))
    env: str = field(default_factory=lambda: os.getenv("ARAGORA_ENV", "development"))
    token_ttl: int = field(default_factory=lambda: int(os.getenv("ARAGORA_TOKEN_TTL", "3600")))
    allowed_origins: str | None = field(
        default_factory=lambda: os.getenv("ARAGORA_ALLOWED_ORIGINS")
    )
    cleanup_interval: int = field(
        default_factory=lambda: int(os.getenv("ARAGORA_AUTH_CLEANUP_INTERVAL", "300"))
    )

    def __post_init__(self) -> None:
        """Validate server auth configuration values."""
        if self.token_ttl < 1:
            raise ValueError(f"token_ttl must be >= 1, got {self.token_ttl}")
        if self.cleanup_interval < 1:
            raise ValueError(f"cleanup_interval must be >= 1, got {self.cleanup_interval}")

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.env.lower() == "production"

    @property
    def allowed_origins_list(self) -> list[str]:
        """Get list of allowed CORS origins."""
        if not self.allowed_origins:
            return []
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


def get_server_auth_config() -> ServerAuthConfig:
    """Factory function for ServerAuthConfig."""
    return ServerAuthConfig()


# =============================================================================
# Deprecation Enforcer Configuration
# =============================================================================


@dataclass(frozen=True)
class DeprecationConfig:
    """Deprecation enforcement configuration."""

    block_sunset_endpoints: bool = field(
        default_factory=lambda: os.getenv("ARAGORA_BLOCK_SUNSET_ENDPOINTS", "true").lower()
        == "true"
    )
    log_deprecated_usage: bool = field(
        default_factory=lambda: os.getenv("ARAGORA_LOG_DEPRECATED_USAGE", "true").lower() == "true"
    )


def get_deprecation_config() -> DeprecationConfig:
    """Factory function for DeprecationConfig."""
    return DeprecationConfig()


# =============================================================================
# Token Revocation Configuration
# =============================================================================


@dataclass(frozen=True)
class TokenRevocationConfig:
    """Token revocation store configuration."""

    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379"))


def get_token_revocation_config() -> TokenRevocationConfig:
    """Factory function for TokenRevocationConfig."""
    return TokenRevocationConfig()


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Core environment
    "ServerEnvironmentConfig",
    "get_server_environment_config",
    # SSO
    "SSOConfig",
    "get_sso_config",
    # OAuth
    "OAuthConfig",
    "get_oauth_config",
    # PagerDuty
    "PagerDutyConfig",
    "get_pagerduty_config",
    # Research
    "ResearchConfig",
    "get_research_config",
    # OpenRouter
    "OpenRouterConfig",
    "get_openrouter_config",
    # Sentry
    "SentryConfig",
    "get_sentry_config",
    # Logging
    "LoggingConfig",
    "get_logging_config",
    # Security
    "SecurityConfig",
    "get_security_config",
    # Timeout
    "TimeoutConfig",
    "get_timeout_config",
    # User Auth
    "UserAuthConfig",
    "get_user_auth_config",
    # Audit
    "AuditConfig",
    "get_audit_config",
    # Rate Limit
    "RateLimitConfig",
    "get_rate_limit_config",
    # XSS Protection
    "XSSProtectionConfig",
    "get_xss_protection_config",
    # Server Auth
    "ServerAuthConfig",
    "get_server_auth_config",
    # Deprecation
    "DeprecationConfig",
    "get_deprecation_config",
    # Token Revocation
    "TokenRevocationConfig",
    "get_token_revocation_config",
]
