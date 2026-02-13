"""Tests for rate limit multi-instance configuration detection and validation.

These tests verify that the rate limiting module correctly detects multi-instance
deployments and warns when Redis is not configured.
"""

from __future__ import annotations

import logging
import sys
import types as _types_mod
from unittest.mock import patch

import pytest

# Pre-stub Slack modules to prevent import chain failures (same as test_rate_limit.py)
_SLACK_ATTRS = [
    "SlackHandler",
    "get_slack_handler",
    "get_slack_integration",
    "get_workspace_store",
    "resolve_workspace",
    "create_tracked_task",
    "_validate_slack_url",
    "SLACK_SIGNING_SECRET",
    "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL",
    "SLACK_ALLOWED_DOMAINS",
    "SignatureVerifierMixin",
    "CommandsMixin",
    "EventsMixin",
    "init_slack_handler",
]
for _mod_name in (
    "aragora.server.handlers.social.slack.handler",
    "aragora.server.handlers.social.slack",
    "aragora.server.handlers.social._slack_impl",
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m


from aragora.server.handlers.utils.rate_limit import (
    _is_multi_instance_mode,
    _is_production_mode,
    _is_redis_configured,
    validate_rate_limit_configuration,
)


# =============================================================================
# Test _is_multi_instance_mode
# =============================================================================


class TestIsMultiInstanceMode:
    """Tests for _is_multi_instance_mode function."""

    def test_multi_instance_detected_from_env_true(self):
        """Should detect multi-instance mode from ARAGORA_MULTI_INSTANCE=true."""
        with patch.dict("os.environ", {"ARAGORA_MULTI_INSTANCE": "true"}, clear=False):
            assert _is_multi_instance_mode() is True

    def test_multi_instance_detected_from_env_1(self):
        """Should detect multi-instance mode from ARAGORA_MULTI_INSTANCE=1."""
        with patch.dict("os.environ", {"ARAGORA_MULTI_INSTANCE": "1"}, clear=False):
            assert _is_multi_instance_mode() is True

    def test_multi_instance_detected_from_env_yes(self):
        """Should detect multi-instance mode from ARAGORA_MULTI_INSTANCE=yes."""
        with patch.dict("os.environ", {"ARAGORA_MULTI_INSTANCE": "yes"}, clear=False):
            assert _is_multi_instance_mode() is True

    def test_multi_instance_not_detected_from_env_false(self):
        """Should not detect multi-instance mode when ARAGORA_MULTI_INSTANCE=false."""
        with patch.dict(
            "os.environ",
            {"ARAGORA_MULTI_INSTANCE": "false", "ARAGORA_REPLICA_COUNT": "1"},
            clear=False,
        ):
            assert _is_multi_instance_mode() is False

    def test_multi_instance_detected_from_replica_count(self):
        """Should detect multi-instance mode from ARAGORA_REPLICA_COUNT > 1."""
        with patch.dict(
            "os.environ",
            {"ARAGORA_MULTI_INSTANCE": "", "ARAGORA_REPLICA_COUNT": "3"},
            clear=False,
        ):
            assert _is_multi_instance_mode() is True

    def test_multi_instance_not_detected_from_replica_count_1(self):
        """Should not detect multi-instance mode when ARAGORA_REPLICA_COUNT=1."""
        with patch.dict(
            "os.environ",
            {"ARAGORA_MULTI_INSTANCE": "", "ARAGORA_REPLICA_COUNT": "1"},
            clear=False,
        ):
            assert _is_multi_instance_mode() is False

    def test_multi_instance_not_detected_when_no_env_vars(self):
        """Should not detect multi-instance mode when env vars are not set."""
        with patch.dict(
            "os.environ",
            {"ARAGORA_MULTI_INSTANCE": "", "ARAGORA_REPLICA_COUNT": ""},
            clear=False,
        ):
            assert _is_multi_instance_mode() is False

    def test_multi_instance_handles_invalid_replica_count(self):
        """Should handle non-integer ARAGORA_REPLICA_COUNT gracefully."""
        with patch.dict(
            "os.environ",
            {"ARAGORA_MULTI_INSTANCE": "", "ARAGORA_REPLICA_COUNT": "invalid"},
            clear=False,
        ):
            assert _is_multi_instance_mode() is False


# =============================================================================
# Test _is_redis_configured
# =============================================================================


class TestIsRedisConfigured:
    """Tests for _is_redis_configured function."""

    def test_redis_configured_via_redis_url(self):
        """Should detect Redis when REDIS_URL is set."""
        with patch.dict(
            "os.environ",
            {"REDIS_URL": "redis://localhost:6379", "ARAGORA_REDIS_URL": ""},
            clear=False,
        ):
            assert _is_redis_configured() is True

    def test_redis_configured_via_aragora_redis_url(self):
        """Should detect Redis when ARAGORA_REDIS_URL is set."""
        with patch.dict(
            "os.environ",
            {"REDIS_URL": "", "ARAGORA_REDIS_URL": "redis://localhost:6379"},
            clear=False,
        ):
            assert _is_redis_configured() is True

    def test_redis_not_configured_when_empty(self):
        """Should not detect Redis when URLs are empty."""
        with patch.dict("os.environ", {"REDIS_URL": "", "ARAGORA_REDIS_URL": ""}, clear=False):
            assert _is_redis_configured() is False

    def test_redis_not_configured_when_whitespace_only(self):
        """Should not detect Redis when URLs are whitespace only."""
        with patch.dict("os.environ", {"REDIS_URL": "   ", "ARAGORA_REDIS_URL": "  "}, clear=False):
            assert _is_redis_configured() is False


# =============================================================================
# Test _is_production_mode
# =============================================================================


class TestIsProductionMode:
    """Tests for _is_production_mode function."""

    def test_production_detected_from_aragora_env(self):
        """Should detect production mode from ARAGORA_ENV=production."""
        with patch.dict("os.environ", {"ARAGORA_ENV": "production"}, clear=False):
            assert _is_production_mode() is True

    def test_production_detected_from_aragora_env_prod(self):
        """Should detect production mode from ARAGORA_ENV=prod."""
        with patch.dict("os.environ", {"ARAGORA_ENV": "prod"}, clear=False):
            assert _is_production_mode() is True

    def test_production_detected_from_node_env(self):
        """Should detect production mode from NODE_ENV=production."""
        with patch.dict(
            "os.environ",
            {"ARAGORA_ENV": "", "NODE_ENV": "production", "ENVIRONMENT": ""},
            clear=False,
        ):
            assert _is_production_mode() is True

    def test_production_detected_from_environment(self):
        """Should detect production mode from ENVIRONMENT=production."""
        with patch.dict(
            "os.environ",
            {"ARAGORA_ENV": "", "NODE_ENV": "", "ENVIRONMENT": "production"},
            clear=False,
        ):
            assert _is_production_mode() is True

    def test_not_production_in_development(self):
        """Should not detect production mode in development."""
        with patch.dict(
            "os.environ",
            {"ARAGORA_ENV": "development", "NODE_ENV": "", "ENVIRONMENT": ""},
            clear=False,
        ):
            assert _is_production_mode() is False


# =============================================================================
# Test validate_rate_limit_configuration
# =============================================================================


class TestValidateRateLimitConfiguration:
    """Tests for validate_rate_limit_configuration function."""

    def test_single_instance_no_warning(self, caplog):
        """Should not log warning in single-instance mode without Redis."""
        with patch.dict(
            "os.environ",
            {
                "ARAGORA_MULTI_INSTANCE": "",
                "ARAGORA_REPLICA_COUNT": "1",
                "REDIS_URL": "",
                "ARAGORA_REDIS_URL": "",
            },
            clear=False,
        ):
            with caplog.at_level(logging.WARNING):
                validate_rate_limit_configuration()

            # Should not have CRITICAL warnings about multi-instance
            critical_records = [r for r in caplog.records if r.levelno >= logging.CRITICAL]
            assert len(critical_records) == 0

    def test_multi_instance_with_redis_no_warning(self, caplog):
        """Should not log warning when Redis is configured in multi-instance mode."""
        with patch.dict(
            "os.environ",
            {
                "ARAGORA_MULTI_INSTANCE": "true",
                "REDIS_URL": "redis://localhost:6379",
                "ARAGORA_RATE_LIMIT_STRICT": "",
            },
            clear=False,
        ):
            with caplog.at_level(logging.INFO):
                validate_rate_limit_configuration()

            # Should log INFO about Redis being configured
            info_records = [
                r
                for r in caplog.records
                if r.levelno == logging.INFO and "Redis configured" in r.message
            ]
            assert len(info_records) == 1

    def test_redis_required_warning_in_multi_instance(self, caplog):
        """Should log CRITICAL warning in multi-instance mode without Redis."""
        with patch.dict(
            "os.environ",
            {
                "ARAGORA_MULTI_INSTANCE": "true",
                "ARAGORA_REPLICA_COUNT": "",
                "REDIS_URL": "",
                "ARAGORA_REDIS_URL": "",
                "ARAGORA_RATE_LIMIT_STRICT": "",
                "ARAGORA_ENV": "",
                "NODE_ENV": "",
                "ENVIRONMENT": "",
            },
            clear=False,
        ):
            with caplog.at_level(logging.WARNING):
                validate_rate_limit_configuration()

            # Should have CRITICAL warning about missing Redis
            critical_records = [r for r in caplog.records if r.levelno >= logging.CRITICAL]
            assert len(critical_records) >= 1

            # Check the warning message content
            warning_text = " ".join(r.message for r in critical_records)
            assert "Multi-instance" in warning_text or "CRITICAL" in warning_text
            assert "Redis" in warning_text

    def test_strict_mode_raises_error(self):
        """Should raise RuntimeError in strict mode without Redis."""
        with patch.dict(
            "os.environ",
            {
                "ARAGORA_MULTI_INSTANCE": "true",
                "REDIS_URL": "",
                "ARAGORA_REDIS_URL": "",
                "ARAGORA_RATE_LIMIT_STRICT": "true",
            },
            clear=False,
        ):
            with pytest.raises(RuntimeError) as exc_info:
                validate_rate_limit_configuration()

            assert "Redis is required" in str(exc_info.value)

    def test_strict_mode_accepts_redis_url(self):
        """Should not raise error in strict mode when Redis is configured."""
        with patch.dict(
            "os.environ",
            {
                "ARAGORA_MULTI_INSTANCE": "true",
                "REDIS_URL": "redis://localhost:6379",
                "ARAGORA_RATE_LIMIT_STRICT": "true",
            },
            clear=False,
        ):
            # Should not raise
            validate_rate_limit_configuration()

    def test_production_mode_adds_extra_warning(self, caplog):
        """Should log additional warning in production mode."""
        with patch.dict(
            "os.environ",
            {
                "ARAGORA_MULTI_INSTANCE": "true",
                "REDIS_URL": "",
                "ARAGORA_REDIS_URL": "",
                "ARAGORA_RATE_LIMIT_STRICT": "",
                "ARAGORA_ENV": "production",
            },
            clear=False,
        ):
            with caplog.at_level(logging.WARNING):
                validate_rate_limit_configuration()

            # Should have production-specific warning
            critical_records = [r for r in caplog.records if r.levelno >= logging.CRITICAL]
            warning_text = " ".join(r.message for r in critical_records)
            assert "PRODUCTION" in warning_text or "security risk" in warning_text

    def test_replica_count_triggers_warning(self, caplog):
        """Should log warning when ARAGORA_REPLICA_COUNT > 1 without Redis."""
        with patch.dict(
            "os.environ",
            {
                "ARAGORA_MULTI_INSTANCE": "",
                "ARAGORA_REPLICA_COUNT": "5",
                "REDIS_URL": "",
                "ARAGORA_REDIS_URL": "",
                "ARAGORA_RATE_LIMIT_STRICT": "",
                "ARAGORA_ENV": "",
            },
            clear=False,
        ):
            with caplog.at_level(logging.WARNING):
                validate_rate_limit_configuration()

            critical_records = [r for r in caplog.records if r.levelno >= logging.CRITICAL]
            assert len(critical_records) >= 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for rate limit configuration."""

    def test_warning_message_content(self, caplog):
        """Should include helpful remediation steps in warning message."""
        with patch.dict(
            "os.environ",
            {
                "ARAGORA_MULTI_INSTANCE": "true",
                "REDIS_URL": "",
                "ARAGORA_REDIS_URL": "",
                "ARAGORA_RATE_LIMIT_STRICT": "",
                "ARAGORA_ENV": "",
            },
            clear=False,
        ):
            with caplog.at_level(logging.WARNING):
                validate_rate_limit_configuration()

            warning_text = " ".join(r.message for r in caplog.records)

            # Should mention the issue
            assert "Multi-instance" in warning_text or "multi-instance" in warning_text.lower()

            # Should mention Redis as the solution
            assert "REDIS_URL" in warning_text or "Redis" in warning_text

    def test_module_level_validation_runs(self):
        """Verify module-level validation is called on import."""
        # This test verifies that the module can be imported without error
        # in single-instance mode (the default test environment)
        import importlib
        import sys

        # Get the actual module object from sys.modules
        rate_limit_mod = sys.modules.get("aragora.server.handlers.utils.rate_limit")
        if rate_limit_mod is None:
            import aragora.server.handlers.utils.rate_limit

            rate_limit_mod = sys.modules["aragora.server.handlers.utils.rate_limit"]

        # Save the _limiters dict and _limiters_lock before reload.
        # importlib.reload() re-executes module-level code, creating a new
        # _limiters dict. This orphans RateLimiter objects captured in decorator
        # closures, making clear_all_limiters() unable to reset them.
        saved_limiters = dict(rate_limit_mod._limiters)
        saved_lock = rate_limit_mod._limiters_lock

        # Re-import to ensure validation runs
        importlib.reload(rate_limit_mod)

        # Restore saved limiters into the new dict so they're not orphaned
        rate_limit_mod._limiters.update(saved_limiters)
        rate_limit_mod._limiters_lock = saved_lock

        # If we get here without error, validation passed
        assert True

    def test_exported_functions_available(self):
        """Verify all multi-instance functions are exported in __all__."""
        import sys

        # Get the actual module object from sys.modules
        rate_limit_mod = sys.modules.get("aragora.server.handlers.utils.rate_limit")
        if rate_limit_mod is None:
            import aragora.server.handlers.utils.rate_limit

            rate_limit_mod = sys.modules["aragora.server.handlers.utils.rate_limit"]

        # Check functions are in __all__
        assert "_is_multi_instance_mode" in rate_limit_mod.__all__
        assert "_is_redis_configured" in rate_limit_mod.__all__
        assert "_is_production_mode" in rate_limit_mod.__all__
        assert "validate_rate_limit_configuration" in rate_limit_mod.__all__

        # Check functions are accessible
        assert callable(rate_limit_mod._is_multi_instance_mode)
        assert callable(rate_limit_mod._is_redis_configured)
        assert callable(rate_limit_mod._is_production_mode)
        assert callable(rate_limit_mod.validate_rate_limit_configuration)
