"""Tests for rate limit multi-instance configuration detection and validation.

These tests verify that the rate limiting module correctly detects multi-instance
deployments (including Kubernetes, Heroku, Fly.io, and Render) and warns when
Redis is not configured.
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
    _reset_multi_instance_cache,
    is_multi_instance,
    validate_rate_limit_configuration,
)

# Environment variables that should be cleared to create a clean
# "single instance" baseline for tests.
_CLEAN_SINGLE_INSTANCE_ENV = {
    "ARAGORA_MULTI_INSTANCE": "",
    "ARAGORA_REPLICA_COUNT": "1",
    "KUBERNETES_SERVICE_HOST": "",
    "HOSTNAME": "",
    "DYNO": "",
    "FLY_ALLOC_ID": "",
    "RENDER_INSTANCE_ID": "",
}

# Strict mode env vars to clear
_CLEAN_STRICT_ENV = {
    "ARAGORA_RATE_LIMIT_STRICT": "",
    "ARAGORA_STRICT_RATE_LIMIT": "",
}

# Redis env vars (unset)
_CLEAN_REDIS_ENV = {
    "REDIS_URL": "",
    "ARAGORA_REDIS_URL": "",
}

# Production env vars (unset)
_CLEAN_PROD_ENV = {
    "ARAGORA_ENV": "",
    "NODE_ENV": "",
    "ENVIRONMENT": "",
}


@pytest.fixture(autouse=True)
def _reset_cache():
    """Reset the multi-instance cache before and after each test."""
    _reset_multi_instance_cache()
    yield
    _reset_multi_instance_cache()


# =============================================================================
# Test _is_multi_instance_mode — explicit flags
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
            {**_CLEAN_SINGLE_INSTANCE_ENV, "ARAGORA_MULTI_INSTANCE": "false"},
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
            {**_CLEAN_SINGLE_INSTANCE_ENV, "ARAGORA_REPLICA_COUNT": "1"},
            clear=False,
        ):
            assert _is_multi_instance_mode() is False

    def test_multi_instance_not_detected_when_no_env_vars(self):
        """Should not detect multi-instance mode when env vars are not set."""
        with patch.dict(
            "os.environ",
            _CLEAN_SINGLE_INSTANCE_ENV,
            clear=False,
        ):
            assert _is_multi_instance_mode() is False

    def test_multi_instance_handles_invalid_replica_count(self):
        """Should handle non-integer ARAGORA_REPLICA_COUNT gracefully."""
        with patch.dict(
            "os.environ",
            {**_CLEAN_SINGLE_INSTANCE_ENV, "ARAGORA_REPLICA_COUNT": "invalid"},
            clear=False,
        ):
            assert _is_multi_instance_mode() is False


# =============================================================================
# Test _is_multi_instance_mode — container/PaaS auto-detection
# =============================================================================


class TestIsMultiInstanceContainerDetection:
    """Tests for container and PaaS auto-detection in _is_multi_instance_mode."""

    def test_kubernetes_service_host_detected(self):
        """Should detect multi-instance mode from KUBERNETES_SERVICE_HOST."""
        with patch.dict(
            "os.environ",
            {**_CLEAN_SINGLE_INSTANCE_ENV, "KUBERNETES_SERVICE_HOST": "10.96.0.1"},
            clear=False,
        ):
            assert _is_multi_instance_mode() is True

    def test_kubernetes_hostname_pattern_detected(self):
        """Should detect K8s pod hostname like 'aragora-web-7f8b9c6d4-x2k9m'."""
        with patch.dict(
            "os.environ",
            {**_CLEAN_SINGLE_INSTANCE_ENV, "HOSTNAME": "aragora-web-7f8b9c6d4-x2k9m"},
            clear=False,
        ):
            assert _is_multi_instance_mode() is True

    def test_kubernetes_hostname_short_replicaset(self):
        """Should detect K8s pod hostname with shorter replicaset hash."""
        with patch.dict(
            "os.environ",
            {**_CLEAN_SINGLE_INSTANCE_ENV, "HOSTNAME": "api-worker-5d7b6-abc12"},
            clear=False,
        ):
            assert _is_multi_instance_mode() is True

    def test_plain_hostname_not_detected(self):
        """Should NOT detect multi-instance from a plain hostname like 'myserver'."""
        with patch.dict(
            "os.environ",
            {**_CLEAN_SINGLE_INSTANCE_ENV, "HOSTNAME": "myserver"},
            clear=False,
        ):
            assert _is_multi_instance_mode() is False

    def test_hostname_with_single_dash_not_detected(self):
        """Should NOT detect from hostname with only one dash segment."""
        with patch.dict(
            "os.environ",
            {**_CLEAN_SINGLE_INSTANCE_ENV, "HOSTNAME": "my-server"},
            clear=False,
        ):
            assert _is_multi_instance_mode() is False

    def test_heroku_dyno_detected(self):
        """Should detect multi-instance mode from DYNO env var (Heroku)."""
        with patch.dict(
            "os.environ",
            {**_CLEAN_SINGLE_INSTANCE_ENV, "DYNO": "web.1"},
            clear=False,
        ):
            assert _is_multi_instance_mode() is True

    def test_fly_alloc_id_detected(self):
        """Should detect multi-instance mode from FLY_ALLOC_ID (Fly.io)."""
        with patch.dict(
            "os.environ",
            {
                **_CLEAN_SINGLE_INSTANCE_ENV,
                "FLY_ALLOC_ID": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            },
            clear=False,
        ):
            assert _is_multi_instance_mode() is True

    def test_render_instance_id_detected(self):
        """Should detect multi-instance mode from RENDER_INSTANCE_ID (Render)."""
        with patch.dict(
            "os.environ",
            {
                **_CLEAN_SINGLE_INSTANCE_ENV,
                "RENDER_INSTANCE_ID": "srv-abc123def456",
            },
            clear=False,
        ):
            assert _is_multi_instance_mode() is True

    def test_empty_container_vars_not_detected(self):
        """Should NOT detect when container vars are present but empty."""
        with patch.dict(
            "os.environ",
            {
                **_CLEAN_SINGLE_INSTANCE_ENV,
                "KUBERNETES_SERVICE_HOST": "",
                "DYNO": "",
                "FLY_ALLOC_ID": "",
                "RENDER_INSTANCE_ID": "",
            },
            clear=False,
        ):
            assert _is_multi_instance_mode() is False


# =============================================================================
# Test is_multi_instance() caching
# =============================================================================


class TestIsMultiInstanceCached:
    """Tests for the cached is_multi_instance() public function."""

    def test_caches_result(self):
        """Should cache the detection result across calls."""
        with patch.dict(
            "os.environ",
            _CLEAN_SINGLE_INSTANCE_ENV,
            clear=False,
        ):
            result1 = is_multi_instance()
            assert result1 is False

        # Now set a multi-instance env var, but result should still be cached
        with patch.dict(
            "os.environ",
            {"ARAGORA_MULTI_INSTANCE": "true"},
            clear=False,
        ):
            result2 = is_multi_instance()
            assert result2 is False  # Still cached as False

    def test_reset_cache_forces_reeval(self):
        """Should re-evaluate after _reset_multi_instance_cache() is called."""
        with patch.dict(
            "os.environ",
            _CLEAN_SINGLE_INSTANCE_ENV,
            clear=False,
        ):
            assert is_multi_instance() is False

        _reset_multi_instance_cache()

        with patch.dict(
            "os.environ",
            {"ARAGORA_MULTI_INSTANCE": "true"},
            clear=False,
        ):
            assert is_multi_instance() is True

    def test_returns_true_when_kubernetes_detected(self):
        """Public function should return True for K8s environments."""
        _reset_multi_instance_cache()
        with patch.dict(
            "os.environ",
            {**_CLEAN_SINGLE_INSTANCE_ENV, "KUBERNETES_SERVICE_HOST": "10.0.0.1"},
            clear=False,
        ):
            assert is_multi_instance() is True


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
                **_CLEAN_SINGLE_INSTANCE_ENV,
                **_CLEAN_REDIS_ENV,
                **_CLEAN_STRICT_ENV,
            },
            clear=False,
        ):
            with caplog.at_level(logging.DEBUG):
                validate_rate_limit_configuration()

            # Should not have any WARNING or higher about multi-instance
            warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
            assert len(warning_records) == 0

    def test_multi_instance_with_redis_no_warning(self, caplog):
        """Should not log warning when Redis is configured in multi-instance mode."""
        with patch.dict(
            "os.environ",
            {
                "ARAGORA_MULTI_INSTANCE": "true",
                "REDIS_URL": "redis://localhost:6379",
                **_CLEAN_STRICT_ENV,
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

    def test_warning_logged_in_multi_instance_without_redis(self, caplog):
        """Should log WARNING in multi-instance mode without Redis."""
        with patch.dict(
            "os.environ",
            {
                "ARAGORA_MULTI_INSTANCE": "true",
                **_CLEAN_REDIS_ENV,
                **_CLEAN_STRICT_ENV,
                **_CLEAN_PROD_ENV,
            },
            clear=False,
        ):
            with caplog.at_level(logging.WARNING):
                validate_rate_limit_configuration()

            # Should have WARNING about per-instance-only rate limiting
            warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
            assert len(warning_records) >= 1

            warning_text = " ".join(r.message for r in warning_records)
            assert "Multi-instance" in warning_text
            assert "Redis" in warning_text
            assert "per-instance" in warning_text

    def test_strict_mode_raises_error_with_old_env_var(self):
        """Should raise RuntimeError when ARAGORA_RATE_LIMIT_STRICT=true."""
        with patch.dict(
            "os.environ",
            {
                "ARAGORA_MULTI_INSTANCE": "true",
                **_CLEAN_REDIS_ENV,
                "ARAGORA_RATE_LIMIT_STRICT": "true",
                "ARAGORA_STRICT_RATE_LIMIT": "",
            },
            clear=False,
        ):
            with pytest.raises(RuntimeError, match="Multi-instance deployment detected"):
                validate_rate_limit_configuration()

    def test_strict_mode_raises_error_with_new_env_var(self):
        """Should raise RuntimeError when ARAGORA_STRICT_RATE_LIMIT=true."""
        with patch.dict(
            "os.environ",
            {
                "ARAGORA_MULTI_INSTANCE": "true",
                **_CLEAN_REDIS_ENV,
                "ARAGORA_RATE_LIMIT_STRICT": "",
                "ARAGORA_STRICT_RATE_LIMIT": "true",
            },
            clear=False,
        ):
            with pytest.raises(RuntimeError, match="Multi-instance deployment detected"):
                validate_rate_limit_configuration()

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
        """Should log additional CRITICAL warning in production mode."""
        with patch.dict(
            "os.environ",
            {
                "ARAGORA_MULTI_INSTANCE": "true",
                **_CLEAN_REDIS_ENV,
                **_CLEAN_STRICT_ENV,
                "ARAGORA_ENV": "production",
            },
            clear=False,
        ):
            with caplog.at_level(logging.WARNING):
                validate_rate_limit_configuration()

            # Should have production-specific CRITICAL warning
            critical_records = [r for r in caplog.records if r.levelno >= logging.CRITICAL]
            warning_text = " ".join(r.message for r in critical_records)
            assert "PRODUCTION" in warning_text or "security risk" in warning_text

    def test_replica_count_triggers_warning(self, caplog):
        """Should log warning when ARAGORA_REPLICA_COUNT > 1 without Redis."""
        with patch.dict(
            "os.environ",
            {
                **_CLEAN_SINGLE_INSTANCE_ENV,
                "ARAGORA_REPLICA_COUNT": "5",
                **_CLEAN_REDIS_ENV,
                **_CLEAN_STRICT_ENV,
                **_CLEAN_PROD_ENV,
            },
            clear=False,
        ):
            with caplog.at_level(logging.WARNING):
                validate_rate_limit_configuration()

            warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
            assert len(warning_records) >= 1

    def test_kubernetes_triggers_warning_without_redis(self, caplog):
        """Should log warning when K8s detected without Redis."""
        with patch.dict(
            "os.environ",
            {
                **_CLEAN_SINGLE_INSTANCE_ENV,
                "KUBERNETES_SERVICE_HOST": "10.96.0.1",
                **_CLEAN_REDIS_ENV,
                **_CLEAN_STRICT_ENV,
                **_CLEAN_PROD_ENV,
            },
            clear=False,
        ):
            with caplog.at_level(logging.WARNING):
                validate_rate_limit_configuration()

            warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
            assert len(warning_records) >= 1
            warning_text = " ".join(r.message for r in warning_records)
            assert "Multi-instance" in warning_text

    def test_heroku_triggers_strict_error(self):
        """Should raise RuntimeError when Heroku detected + strict + no Redis."""
        with patch.dict(
            "os.environ",
            {
                **_CLEAN_SINGLE_INSTANCE_ENV,
                "DYNO": "web.1",
                **_CLEAN_REDIS_ENV,
                "ARAGORA_STRICT_RATE_LIMIT": "true",
                "ARAGORA_RATE_LIMIT_STRICT": "",
            },
            clear=False,
        ):
            with pytest.raises(RuntimeError, match="Multi-instance deployment detected"):
                validate_rate_limit_configuration()

    def test_fly_io_triggers_strict_error(self):
        """Should raise RuntimeError when Fly.io detected + strict + no Redis."""
        with patch.dict(
            "os.environ",
            {
                **_CLEAN_SINGLE_INSTANCE_ENV,
                "FLY_ALLOC_ID": "abc-123",
                **_CLEAN_REDIS_ENV,
                "ARAGORA_STRICT_RATE_LIMIT": "true",
                "ARAGORA_RATE_LIMIT_STRICT": "",
            },
            clear=False,
        ):
            with pytest.raises(RuntimeError, match="Multi-instance deployment detected"):
                validate_rate_limit_configuration()


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
                **_CLEAN_REDIS_ENV,
                **_CLEAN_STRICT_ENV,
                **_CLEAN_PROD_ENV,
            },
            clear=False,
        ):
            with caplog.at_level(logging.WARNING):
                validate_rate_limit_configuration()

            warning_text = " ".join(r.message for r in caplog.records)

            # Should mention the issue
            assert "Multi-instance" in warning_text or "multi-instance" in warning_text.lower()

            # Should mention Redis as the solution
            assert "Redis" in warning_text

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

        # Save state before reload.  importlib.reload() re-executes
        # module-level code, creating new _limiters dict and re-evaluating
        # RATE_LIMITING_DISABLED.  We must restore these on the NEW module
        # so that existing RateLimiter objects (whose __globals__ points to
        # the OLD module dict) and new code all see consistent state.
        saved_limiters = dict(rate_limit_mod._limiters)
        saved_lock = rate_limit_mod._limiters_lock
        saved_disabled = rate_limit_mod.RATE_LIMITING_DISABLED

        # Re-import to ensure validation runs
        importlib.reload(rate_limit_mod)

        # Restore saved state into the reloaded module
        rate_limit_mod._limiters.update(saved_limiters)
        rate_limit_mod._limiters_lock = saved_lock
        rate_limit_mod.RATE_LIMITING_DISABLED = saved_disabled

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
        assert "is_multi_instance" in rate_limit_mod.__all__
        assert "_is_multi_instance_mode" in rate_limit_mod.__all__
        assert "_is_redis_configured" in rate_limit_mod.__all__
        assert "_is_production_mode" in rate_limit_mod.__all__
        assert "_reset_multi_instance_cache" in rate_limit_mod.__all__
        assert "validate_rate_limit_configuration" in rate_limit_mod.__all__

        # Check functions are accessible
        assert callable(rate_limit_mod.is_multi_instance)
        assert callable(rate_limit_mod._is_multi_instance_mode)
        assert callable(rate_limit_mod._is_redis_configured)
        assert callable(rate_limit_mod._is_production_mode)
        assert callable(rate_limit_mod._reset_multi_instance_cache)
        assert callable(rate_limit_mod.validate_rate_limit_configuration)
