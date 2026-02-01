"""
Tests for server configuration dataclass validation.

Tests cover:
- TimeoutConfig validation (positive values, max >= default, workers >= 1)
- SentryConfig validation (sample rates between 0 and 1)
- RateLimitConfig validation (thresholds and intervals >= 1)
- ServerAuthConfig validation (TTL and intervals >= 1)
- Default factory functions work correctly
- Environment variable loading
"""

from __future__ import annotations

import os
from unittest import mock

import pytest

from aragora.server.config import (
    AuditConfig,
    DeprecationConfig,
    LoggingConfig,
    OAuthConfig,
    OpenRouterConfig,
    PagerDutyConfig,
    RateLimitConfig,
    ResearchConfig,
    SecurityConfig,
    SentryConfig,
    ServerAuthConfig,
    ServerEnvironmentConfig,
    SSOConfig,
    TimeoutConfig,
    TokenRevocationConfig,
    UserAuthConfig,
    XSSProtectionConfig,
    get_audit_config,
    get_deprecation_config,
    get_logging_config,
    get_oauth_config,
    get_openrouter_config,
    get_pagerduty_config,
    get_rate_limit_config,
    get_research_config,
    get_security_config,
    get_sentry_config,
    get_server_auth_config,
    get_server_environment_config,
    get_sso_config,
    get_timeout_config,
    get_token_revocation_config,
    get_user_auth_config,
    get_xss_protection_config,
)


# =============================================================================
# TimeoutConfig Validation Tests
# =============================================================================


class TestTimeoutConfigValidation:
    """Tests for TimeoutConfig validation."""

    def test_accepts_valid_positive_values(self):
        """Test TimeoutConfig accepts valid positive values."""
        config = TimeoutConfig(
            default_timeout=30.0,
            slow_timeout=120.0,
            max_timeout=600.0,
            timeout_workers=10,
        )
        assert config.default_timeout == 30.0
        assert config.slow_timeout == 120.0
        assert config.max_timeout == 600.0
        assert config.timeout_workers == 10

    def test_accepts_minimum_valid_values(self):
        """Test TimeoutConfig accepts minimum valid values."""
        config = TimeoutConfig(
            default_timeout=0.001,  # Very small but positive
            slow_timeout=0.001,
            max_timeout=0.001,
            timeout_workers=1,
        )
        assert config.default_timeout == 0.001
        assert config.timeout_workers == 1

    def test_rejects_negative_default_timeout(self):
        """Test TimeoutConfig rejects negative default_timeout."""
        with pytest.raises(ValueError) as exc_info:
            TimeoutConfig(
                default_timeout=-1.0,
                slow_timeout=120.0,
                max_timeout=600.0,
                timeout_workers=10,
            )
        assert "default_timeout must be positive" in str(exc_info.value)
        assert "-1.0" in str(exc_info.value)

    def test_rejects_zero_default_timeout(self):
        """Test TimeoutConfig rejects zero default_timeout."""
        with pytest.raises(ValueError) as exc_info:
            TimeoutConfig(
                default_timeout=0.0,
                slow_timeout=120.0,
                max_timeout=600.0,
                timeout_workers=10,
            )
        assert "default_timeout must be positive" in str(exc_info.value)

    def test_rejects_max_timeout_less_than_default(self):
        """Test TimeoutConfig rejects max_timeout < default_timeout."""
        with pytest.raises(ValueError) as exc_info:
            TimeoutConfig(
                default_timeout=100.0,
                slow_timeout=50.0,
                max_timeout=50.0,  # Less than default
                timeout_workers=10,
            )
        assert "max_timeout" in str(exc_info.value)
        assert "default_timeout" in str(exc_info.value)

    def test_accepts_max_timeout_equal_to_default(self):
        """Test TimeoutConfig accepts max_timeout == default_timeout."""
        config = TimeoutConfig(
            default_timeout=30.0,
            slow_timeout=30.0,
            max_timeout=30.0,  # Equal to default is OK
            timeout_workers=1,
        )
        assert config.max_timeout == config.default_timeout

    def test_rejects_zero_timeout_workers(self):
        """Test TimeoutConfig rejects zero timeout_workers."""
        with pytest.raises(ValueError) as exc_info:
            TimeoutConfig(
                default_timeout=30.0,
                slow_timeout=120.0,
                max_timeout=600.0,
                timeout_workers=0,
            )
        assert "timeout_workers must be >= 1" in str(exc_info.value)

    def test_rejects_negative_timeout_workers(self):
        """Test TimeoutConfig rejects negative timeout_workers."""
        with pytest.raises(ValueError) as exc_info:
            TimeoutConfig(
                default_timeout=30.0,
                slow_timeout=120.0,
                max_timeout=600.0,
                timeout_workers=-5,
            )
        assert "timeout_workers must be >= 1" in str(exc_info.value)


# =============================================================================
# SentryConfig Validation Tests
# =============================================================================


class TestSentryConfigValidation:
    """Tests for SentryConfig validation."""

    def test_accepts_valid_sample_rates(self):
        """Test SentryConfig accepts valid sample rates."""
        config = SentryConfig(
            dsn="https://example.sentry.io",
            environment="test",
            traces_sample_rate=0.5,
            profiles_sample_rate=0.1,
        )
        assert config.traces_sample_rate == 0.5
        assert config.profiles_sample_rate == 0.1

    def test_accepts_zero_sample_rate(self):
        """Test SentryConfig accepts 0 sample rate (disabled)."""
        config = SentryConfig(
            traces_sample_rate=0.0,
            profiles_sample_rate=0.0,
        )
        assert config.traces_sample_rate == 0.0
        assert config.profiles_sample_rate == 0.0

    def test_accepts_full_sample_rate(self):
        """Test SentryConfig accepts 1.0 sample rate (100%)."""
        config = SentryConfig(
            traces_sample_rate=1.0,
            profiles_sample_rate=1.0,
        )
        assert config.traces_sample_rate == 1.0
        assert config.profiles_sample_rate == 1.0

    def test_rejects_negative_traces_sample_rate(self):
        """Test SentryConfig rejects negative traces_sample_rate."""
        with pytest.raises(ValueError) as exc_info:
            SentryConfig(
                traces_sample_rate=-0.1,
                profiles_sample_rate=0.1,
            )
        assert "traces_sample_rate must be between 0 and 1" in str(exc_info.value)

    def test_rejects_traces_sample_rate_over_one(self):
        """Test SentryConfig rejects traces_sample_rate > 1."""
        with pytest.raises(ValueError) as exc_info:
            SentryConfig(
                traces_sample_rate=1.5,
                profiles_sample_rate=0.1,
            )
        assert "traces_sample_rate must be between 0 and 1" in str(exc_info.value)

    def test_rejects_negative_profiles_sample_rate(self):
        """Test SentryConfig rejects negative profiles_sample_rate."""
        with pytest.raises(ValueError) as exc_info:
            SentryConfig(
                traces_sample_rate=0.1,
                profiles_sample_rate=-0.5,
            )
        assert "profiles_sample_rate must be between 0 and 1" in str(exc_info.value)

    def test_rejects_profiles_sample_rate_over_one(self):
        """Test SentryConfig rejects profiles_sample_rate > 1."""
        with pytest.raises(ValueError) as exc_info:
            SentryConfig(
                traces_sample_rate=0.1,
                profiles_sample_rate=2.0,
            )
        assert "profiles_sample_rate must be between 0 and 1" in str(exc_info.value)

    def test_is_configured_property(self):
        """Test is_configured property."""
        config_with_dsn = SentryConfig(dsn="https://example.sentry.io")
        assert config_with_dsn.is_configured is True

        config_without_dsn = SentryConfig(dsn=None)
        assert config_without_dsn.is_configured is False


# =============================================================================
# LoggingConfig Validation Tests
# =============================================================================


class TestLoggingConfigValidation:
    """Tests for LoggingConfig validation."""

    def test_accepts_valid_log_level_info(self):
        """Test LoggingConfig accepts INFO log level."""
        config = LoggingConfig(log_level="INFO")
        assert config.log_level == "INFO"

    def test_accepts_valid_log_level_debug(self):
        """Test LoggingConfig accepts DEBUG log level."""
        config = LoggingConfig(log_level="DEBUG")
        assert config.log_level == "DEBUG"

    def test_accepts_valid_log_level_warning(self):
        """Test LoggingConfig accepts WARNING log level."""
        config = LoggingConfig(log_level="WARNING")
        assert config.log_level == "WARNING"

    def test_accepts_valid_log_level_error(self):
        """Test LoggingConfig accepts ERROR log level."""
        config = LoggingConfig(log_level="ERROR")
        assert config.log_level == "ERROR"

    def test_accepts_valid_log_level_critical(self):
        """Test LoggingConfig accepts CRITICAL log level."""
        config = LoggingConfig(log_level="CRITICAL")
        assert config.log_level == "CRITICAL"

    def test_normalizes_lowercase_log_level(self):
        """Test LoggingConfig normalizes lowercase log level to uppercase."""
        config = LoggingConfig(log_level="debug")
        assert config.log_level == "DEBUG"

    def test_normalizes_mixed_case_log_level(self):
        """Test LoggingConfig normalizes mixed case log level to uppercase."""
        config = LoggingConfig(log_level="WaRnInG")
        assert config.log_level == "WARNING"

    def test_invalid_log_level_defaults_to_info(self, caplog):
        """Test LoggingConfig with invalid log level defaults to INFO."""
        with caplog.at_level("WARNING", logger="aragora.server.config"):
            config = LoggingConfig(log_level="INVALID")
        assert config.log_level == "INFO"

    def test_invalid_log_level_logs_warning(self, caplog):
        """Test LoggingConfig logs warning for invalid log level."""
        with caplog.at_level("WARNING", logger="aragora.server.config"):
            LoggingConfig(log_level="NOTVALID")
        assert any("Invalid log_level" in record.message for record in caplog.records)
        assert any("NOTVALID" in record.message for record in caplog.records)

    def test_empty_log_level_defaults_to_info(self, caplog):
        """Test LoggingConfig with empty string defaults to INFO."""
        with caplog.at_level("WARNING", logger="aragora.server.config"):
            config = LoggingConfig(log_level="")
        assert config.log_level == "INFO"

    def test_logging_config_is_frozen(self):
        """Test LoggingConfig is immutable."""
        config = LoggingConfig(log_level="INFO")
        with pytest.raises(AttributeError):
            config.log_level = "DEBUG"


# =============================================================================
# RateLimitConfig Validation Tests
# =============================================================================


class TestRateLimitConfigValidation:
    """Tests for RateLimitConfig validation."""

    def test_accepts_valid_values(self):
        """Test RateLimitConfig accepts valid values."""
        config = RateLimitConfig(
            redis_url="redis://localhost:6379",
            redis_failure_threshold=3,
            metrics_aggregation_interval=60,
        )
        assert config.redis_failure_threshold == 3
        assert config.metrics_aggregation_interval == 60

    def test_accepts_minimum_valid_values(self):
        """Test RateLimitConfig accepts minimum valid values."""
        config = RateLimitConfig(
            redis_failure_threshold=1,
            metrics_aggregation_interval=1,
        )
        assert config.redis_failure_threshold == 1
        assert config.metrics_aggregation_interval == 1

    def test_rejects_zero_redis_failure_threshold(self):
        """Test RateLimitConfig rejects zero redis_failure_threshold."""
        with pytest.raises(ValueError) as exc_info:
            RateLimitConfig(redis_failure_threshold=0)
        assert "redis_failure_threshold must be >= 1" in str(exc_info.value)

    def test_rejects_negative_redis_failure_threshold(self):
        """Test RateLimitConfig rejects negative redis_failure_threshold."""
        with pytest.raises(ValueError) as exc_info:
            RateLimitConfig(redis_failure_threshold=-1)
        assert "redis_failure_threshold must be >= 1" in str(exc_info.value)

    def test_rejects_zero_metrics_aggregation_interval(self):
        """Test RateLimitConfig rejects zero metrics_aggregation_interval."""
        with pytest.raises(ValueError) as exc_info:
            RateLimitConfig(metrics_aggregation_interval=0)
        assert "metrics_aggregation_interval must be >= 1" in str(exc_info.value)

    def test_rejects_negative_metrics_aggregation_interval(self):
        """Test RateLimitConfig rejects negative metrics_aggregation_interval."""
        with pytest.raises(ValueError) as exc_info:
            RateLimitConfig(metrics_aggregation_interval=-10)
        assert "metrics_aggregation_interval must be >= 1" in str(exc_info.value)


# =============================================================================
# ServerAuthConfig Validation Tests
# =============================================================================


class TestServerAuthConfigValidation:
    """Tests for ServerAuthConfig validation."""

    def test_accepts_valid_values(self):
        """Test ServerAuthConfig accepts valid values."""
        config = ServerAuthConfig(
            api_token="test-token",
            token_ttl=3600,
            cleanup_interval=300,
        )
        assert config.token_ttl == 3600
        assert config.cleanup_interval == 300

    def test_accepts_minimum_valid_values(self):
        """Test ServerAuthConfig accepts minimum valid values."""
        config = ServerAuthConfig(
            token_ttl=1,
            cleanup_interval=1,
        )
        assert config.token_ttl == 1
        assert config.cleanup_interval == 1

    def test_rejects_zero_token_ttl(self):
        """Test ServerAuthConfig rejects zero token_ttl."""
        with pytest.raises(ValueError) as exc_info:
            ServerAuthConfig(token_ttl=0)
        assert "token_ttl must be >= 1" in str(exc_info.value)

    def test_rejects_negative_token_ttl(self):
        """Test ServerAuthConfig rejects negative token_ttl."""
        with pytest.raises(ValueError) as exc_info:
            ServerAuthConfig(token_ttl=-100)
        assert "token_ttl must be >= 1" in str(exc_info.value)

    def test_rejects_zero_cleanup_interval(self):
        """Test ServerAuthConfig rejects zero cleanup_interval."""
        with pytest.raises(ValueError) as exc_info:
            ServerAuthConfig(cleanup_interval=0)
        assert "cleanup_interval must be >= 1" in str(exc_info.value)

    def test_rejects_negative_cleanup_interval(self):
        """Test ServerAuthConfig rejects negative cleanup_interval."""
        with pytest.raises(ValueError) as exc_info:
            ServerAuthConfig(cleanup_interval=-50)
        assert "cleanup_interval must be >= 1" in str(exc_info.value)

    def test_is_production_property(self):
        """Test is_production property."""
        prod_config = ServerAuthConfig(env="production")
        assert prod_config.is_production is True

        dev_config = ServerAuthConfig(env="development")
        assert dev_config.is_production is False

    def test_allowed_origins_list_property(self):
        """Test allowed_origins_list property."""
        config = ServerAuthConfig(allowed_origins="http://a.com, http://b.com")
        assert config.allowed_origins_list == ["http://a.com", "http://b.com"]

        config_empty = ServerAuthConfig(allowed_origins=None)
        assert config_empty.allowed_origins_list == []


# =============================================================================
# Environment Variable Loading Tests
# =============================================================================


class TestEnvironmentVariableLoading:
    """Tests for environment variable loading via factory functions."""

    def test_timeout_config_from_env(self):
        """Test TimeoutConfig loads from environment variables."""
        with mock.patch.dict(
            os.environ,
            {
                "ARAGORA_REQUEST_TIMEOUT": "60",
                "ARAGORA_SLOW_REQUEST_TIMEOUT": "240",
                "ARAGORA_MAX_REQUEST_TIMEOUT": "1200",
                "ARAGORA_TIMEOUT_WORKERS": "20",
            },
        ):
            config = get_timeout_config()
            assert config.default_timeout == 60.0
            assert config.slow_timeout == 240.0
            assert config.max_timeout == 1200.0
            assert config.timeout_workers == 20

    def test_sentry_config_from_env(self):
        """Test SentryConfig loads from environment variables."""
        with mock.patch.dict(
            os.environ,
            {
                "SENTRY_DSN": "https://test.sentry.io",
                "SENTRY_ENVIRONMENT": "testing",
                "SENTRY_TRACES_SAMPLE_RATE": "0.5",
                "SENTRY_PROFILES_SAMPLE_RATE": "0.25",
            },
        ):
            config = get_sentry_config()
            assert config.dsn == "https://test.sentry.io"
            assert config.environment == "testing"
            assert config.traces_sample_rate == 0.5
            assert config.profiles_sample_rate == 0.25

    def test_rate_limit_config_from_env(self):
        """Test RateLimitConfig loads from environment variables."""
        with mock.patch.dict(
            os.environ,
            {
                "REDIS_URL": "redis://custom:6380",
                "ARAGORA_REDIS_FAILURE_THRESHOLD": "5",
                "ARAGORA_RATE_LIMIT_METRICS_INTERVAL": "120",
            },
        ):
            config = get_rate_limit_config()
            assert config.redis_url == "redis://custom:6380"
            assert config.redis_failure_threshold == 5
            assert config.metrics_aggregation_interval == 120

    def test_server_auth_config_from_env(self):
        """Test ServerAuthConfig loads from environment variables."""
        with mock.patch.dict(
            os.environ,
            {
                "ARAGORA_API_TOKEN": "secret-token",
                "ARAGORA_ENV": "production",
                "ARAGORA_TOKEN_TTL": "7200",
                "ARAGORA_AUTH_CLEANUP_INTERVAL": "600",
            },
        ):
            config = get_server_auth_config()
            assert config.api_token == "secret-token"
            assert config.env == "production"
            assert config.token_ttl == 7200
            assert config.cleanup_interval == 600


# =============================================================================
# Default Factory Function Tests
# =============================================================================


class TestDefaultFactoryFunctions:
    """Tests for default instantiation of all config dataclasses."""

    def test_server_environment_config_defaults(self):
        """Test ServerEnvironmentConfig can be instantiated with defaults."""
        config = get_server_environment_config()
        assert isinstance(config, ServerEnvironmentConfig)
        assert isinstance(config.env, str)
        assert isinstance(config.is_production, bool)

    def test_sso_config_defaults(self):
        """Test SSOConfig can be instantiated with defaults."""
        config = get_sso_config()
        assert isinstance(config, SSOConfig)
        assert isinstance(config.is_production, bool)
        assert isinstance(config.allowed_hosts_list, list)

    def test_oauth_config_defaults(self):
        """Test OAuthConfig can be instantiated with defaults."""
        config = get_oauth_config()
        assert isinstance(config, OAuthConfig)
        assert isinstance(config.allowed_hosts, frozenset)

    def test_pagerduty_config_defaults(self):
        """Test PagerDutyConfig can be instantiated with defaults."""
        config = get_pagerduty_config()
        assert isinstance(config, PagerDutyConfig)
        assert isinstance(config.is_configured, bool)

    def test_research_config_defaults(self):
        """Test ResearchConfig can be instantiated with defaults."""
        config = get_research_config()
        assert isinstance(config, ResearchConfig)
        assert isinstance(config.has_any_search_api, bool)

    def test_openrouter_config_defaults(self):
        """Test OpenRouterConfig can be instantiated with defaults."""
        config = get_openrouter_config()
        assert isinstance(config, OpenRouterConfig)
        assert isinstance(config.is_available, bool)

    def test_sentry_config_defaults(self):
        """Test SentryConfig can be instantiated with defaults."""
        config = get_sentry_config()
        assert isinstance(config, SentryConfig)
        assert 0 <= config.traces_sample_rate <= 1
        assert 0 <= config.profiles_sample_rate <= 1

    def test_logging_config_defaults(self):
        """Test LoggingConfig can be instantiated with defaults."""
        config = get_logging_config()
        assert isinstance(config, LoggingConfig)
        assert isinstance(config.log_level, str)
        assert isinstance(config.log_format, str)

    def test_security_config_defaults(self):
        """Test SecurityConfig can be instantiated with defaults."""
        config = get_security_config()
        assert isinstance(config, SecurityConfig)
        assert isinstance(config.trusted_proxies_list, list)

    def test_timeout_config_defaults(self):
        """Test TimeoutConfig can be instantiated with defaults."""
        config = get_timeout_config()
        assert isinstance(config, TimeoutConfig)
        assert config.default_timeout > 0
        assert config.max_timeout >= config.default_timeout
        assert config.timeout_workers >= 1

    def test_user_auth_config_defaults(self):
        """Test UserAuthConfig can be instantiated with defaults."""
        config = get_user_auth_config()
        assert isinstance(config, UserAuthConfig)
        assert isinstance(config.is_production, bool)
        assert isinstance(config.is_configured, bool)

    def test_audit_config_defaults(self):
        """Test AuditConfig can be instantiated with defaults."""
        config = get_audit_config()
        assert isinstance(config, AuditConfig)
        assert isinstance(config.log_dir, str)

    def test_rate_limit_config_defaults(self):
        """Test RateLimitConfig can be instantiated with defaults."""
        config = get_rate_limit_config()
        assert isinstance(config, RateLimitConfig)
        assert config.redis_failure_threshold >= 1
        assert config.metrics_aggregation_interval >= 1

    def test_xss_protection_config_defaults(self):
        """Test XSSProtectionConfig can be instantiated with defaults."""
        config = get_xss_protection_config()
        assert isinstance(config, XSSProtectionConfig)
        assert isinstance(config.auto_escape_html, bool)

    def test_server_auth_config_defaults(self):
        """Test ServerAuthConfig can be instantiated with defaults."""
        config = get_server_auth_config()
        assert isinstance(config, ServerAuthConfig)
        assert config.token_ttl >= 1
        assert config.cleanup_interval >= 1

    def test_deprecation_config_defaults(self):
        """Test DeprecationConfig can be instantiated with defaults."""
        config = get_deprecation_config()
        assert isinstance(config, DeprecationConfig)
        assert isinstance(config.block_sunset_endpoints, bool)

    def test_token_revocation_config_defaults(self):
        """Test TokenRevocationConfig can be instantiated with defaults."""
        config = get_token_revocation_config()
        assert isinstance(config, TokenRevocationConfig)
        assert isinstance(config.redis_url, str)


# =============================================================================
# Frozen Dataclass Tests
# =============================================================================


class TestFrozenDataclass:
    """Tests to verify dataclasses are immutable (frozen)."""

    def test_timeout_config_is_frozen(self):
        """Test TimeoutConfig is immutable."""
        config = TimeoutConfig(
            default_timeout=30.0,
            slow_timeout=120.0,
            max_timeout=600.0,
            timeout_workers=10,
        )
        with pytest.raises(AttributeError):
            config.default_timeout = 60.0

    def test_sentry_config_is_frozen(self):
        """Test SentryConfig is immutable."""
        config = SentryConfig(traces_sample_rate=0.5, profiles_sample_rate=0.5)
        with pytest.raises(AttributeError):
            config.traces_sample_rate = 0.9

    def test_rate_limit_config_is_frozen(self):
        """Test RateLimitConfig is immutable."""
        config = RateLimitConfig(redis_failure_threshold=3)
        with pytest.raises(AttributeError):
            config.redis_failure_threshold = 5

    def test_server_auth_config_is_frozen(self):
        """Test ServerAuthConfig is immutable."""
        config = ServerAuthConfig(token_ttl=3600)
        with pytest.raises(AttributeError):
            config.token_ttl = 7200


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_timeout_config_very_large_values(self):
        """Test TimeoutConfig with very large values."""
        config = TimeoutConfig(
            default_timeout=1e10,
            slow_timeout=1e10,
            max_timeout=1e11,
            timeout_workers=10000,
        )
        assert config.default_timeout == 1e10
        assert config.max_timeout == 1e11

    def test_timeout_config_very_small_positive_values(self):
        """Test TimeoutConfig with very small positive values."""
        config = TimeoutConfig(
            default_timeout=1e-10,
            slow_timeout=1e-10,
            max_timeout=1e-10,
            timeout_workers=1,
        )
        assert config.default_timeout == pytest.approx(1e-10)

    def test_sentry_boundary_sample_rates(self):
        """Test SentryConfig at exact boundaries."""
        # Exactly 0
        config_zero = SentryConfig(traces_sample_rate=0.0, profiles_sample_rate=0.0)
        assert config_zero.traces_sample_rate == 0.0

        # Exactly 1
        config_one = SentryConfig(traces_sample_rate=1.0, profiles_sample_rate=1.0)
        assert config_one.traces_sample_rate == 1.0

    def test_rate_limit_exactly_one(self):
        """Test RateLimitConfig with exactly 1 values."""
        config = RateLimitConfig(
            redis_failure_threshold=1,
            metrics_aggregation_interval=1,
        )
        assert config.redis_failure_threshold == 1
        assert config.metrics_aggregation_interval == 1

    def test_server_auth_exactly_one(self):
        """Test ServerAuthConfig with exactly 1 values."""
        config = ServerAuthConfig(
            token_ttl=1,
            cleanup_interval=1,
        )
        assert config.token_ttl == 1
        assert config.cleanup_interval == 1
