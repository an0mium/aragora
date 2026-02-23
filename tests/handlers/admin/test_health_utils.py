"""
Comprehensive tests for health utility functions in
aragora/server/handlers/admin/health_utils.py.

Tests all eight public functions:

  TestCheckFilesystemHealth     - check_filesystem_health()
  TestCheckRedisHealth          - check_redis_health()
  TestCheckAiProvidersHealth    - check_ai_providers_health()
  TestCheckSecurityServices     - check_security_services()
  TestCheckDatabaseHealth       - check_database_health()
  TestGetUptimeInfo             - get_uptime_info()
  TestCheckStripeHealth         - check_stripe_health()
  TestCheckSlackHealth          - check_slack_health()

Coverage: happy paths, error handling, edge cases, input validation, env vars.
Target: 90+ tests, 0 failures.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.admin.health_utils import (
    check_ai_providers_health,
    check_database_health,
    check_filesystem_health,
    check_redis_health,
    check_security_services,
    check_slack_health,
    check_stripe_health,
    get_uptime_info,
)


# ============================================================================
# TestCheckFilesystemHealth
# ============================================================================


class TestCheckFilesystemHealth:
    """Tests for check_filesystem_health() - filesystem write access check."""

    def test_healthy_with_default_temp_dir(self):
        """Default (None) test_dir falls back to tempdir and returns healthy."""
        result = check_filesystem_health()
        assert result["healthy"] is True
        assert "path" in result

    def test_healthy_with_explicit_dir(self, tmp_path):
        """Explicit existing directory returns healthy with that path."""
        result = check_filesystem_health(test_dir=tmp_path)
        assert result["healthy"] is True
        assert result["path"] == str(tmp_path)

    def test_nonexistent_dir_falls_back_to_temp(self, tmp_path):
        """Non-existent directory falls back to tempdir."""
        missing = tmp_path / "does_not_exist"
        result = check_filesystem_health(test_dir=missing)
        assert result["healthy"] is True
        # Falls back to temp dir since missing doesn't exist
        assert "path" in result

    def test_none_dir_falls_back_to_temp(self):
        """Passing None explicitly uses tempdir."""
        result = check_filesystem_health(test_dir=None)
        assert result["healthy"] is True

    def test_cleanup_removes_test_file(self, tmp_path):
        """Health check file is cleaned up after check."""
        check_filesystem_health(test_dir=tmp_path)
        # No leftover .health_check_* files
        health_files = list(tmp_path.glob(".health_check_*"))
        assert len(health_files) == 0

    def test_permission_error_returns_unhealthy(self, tmp_path):
        """PermissionError during write returns unhealthy with 'Permission denied'."""
        with patch.object(Path, "write_text", side_effect=PermissionError("denied")):
            result = check_filesystem_health(test_dir=tmp_path)
            assert result["healthy"] is False
            assert result["error"] == "Permission denied"

    def test_os_error_returns_unhealthy(self, tmp_path):
        """OSError during write returns unhealthy with 'Filesystem error'."""
        with patch.object(Path, "write_text", side_effect=OSError("disk full")):
            result = check_filesystem_health(test_dir=tmp_path)
            assert result["healthy"] is False
            assert result["error"] == "Filesystem error"

    def test_read_verification_failure(self, tmp_path):
        """Read-back mismatch returns unhealthy with 'Read verification failed'."""
        with patch.object(Path, "read_text", return_value="wrong_content"):
            result = check_filesystem_health(test_dir=tmp_path)
            assert result["healthy"] is False
            assert result["error"] == "Read verification failed"

    def test_result_keys(self, tmp_path):
        """Healthy result contains 'healthy' and 'path' keys."""
        result = check_filesystem_health(test_dir=tmp_path)
        assert "healthy" in result
        assert "path" in result

    def test_unhealthy_result_has_error_key(self):
        """Unhealthy result contains 'error' key."""
        with patch.object(Path, "write_text", side_effect=PermissionError("nope")):
            result = check_filesystem_health()
            assert "error" in result

    def test_test_file_name_includes_pid(self, tmp_path):
        """Test file is named with current process ID."""
        pid = os.getpid()
        written_paths = []
        original_write = Path.write_text

        def capture_write(self, content, *args, **kwargs):
            written_paths.append(str(self))
            return original_write(self, content, *args, **kwargs)

        with patch.object(Path, "write_text", capture_write):
            check_filesystem_health(test_dir=tmp_path)

        assert any(str(pid) in p for p in written_paths)


# ============================================================================
# TestCheckRedisHealth
# ============================================================================


class TestCheckRedisHealth:
    """Tests for check_redis_health() - Redis connectivity check."""

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        """Remove Redis env vars to ensure clean state."""
        monkeypatch.delenv("REDIS_URL", raising=False)
        monkeypatch.delenv("CACHE_REDIS_URL", raising=False)

    def test_not_configured_returns_healthy(self):
        """No redis_url and no env var -> healthy but not configured."""
        result = check_redis_health(redis_url=None)
        assert result["healthy"] is True
        assert result["configured"] is False
        assert "note" in result

    def test_empty_string_url_not_configured(self):
        """Empty string url -> not configured."""
        result = check_redis_health(redis_url="")
        assert result["healthy"] is True
        assert result["configured"] is False

    def test_reads_redis_url_from_env(self, monkeypatch):
        """Reads REDIS_URL from environment when redis_url is None."""
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")

        mock_redis = MagicMock()
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis.from_url.return_value = mock_client

        with patch.dict("sys.modules", {"redis": mock_redis}):
            result = check_redis_health(redis_url=None)
            assert result["configured"] is True
            mock_redis.from_url.assert_called_once()

    def test_reads_cache_redis_url_from_env(self, monkeypatch):
        """Falls back to CACHE_REDIS_URL when REDIS_URL is not set."""
        monkeypatch.setenv("CACHE_REDIS_URL", "redis://cache:6379")

        mock_redis = MagicMock()
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis.from_url.return_value = mock_client

        with patch.dict("sys.modules", {"redis": mock_redis}):
            result = check_redis_health(redis_url=None)
            assert result["configured"] is True

    def test_successful_ping_returns_healthy(self):
        """Successful Redis ping -> healthy with latency."""
        mock_redis = MagicMock()
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis.from_url.return_value = mock_client

        with patch.dict("sys.modules", {"redis": mock_redis}):
            result = check_redis_health(redis_url="redis://localhost:6379")
            assert result["healthy"] is True
            assert result["configured"] is True
            assert "latency_ms" in result
            assert isinstance(result["latency_ms"], float)

    def test_ping_returns_false(self):
        """Ping returns False -> unhealthy."""
        mock_redis = MagicMock()
        mock_client = MagicMock()
        mock_client.ping.return_value = False
        mock_redis.from_url.return_value = mock_client

        with patch.dict("sys.modules", {"redis": mock_redis}):
            result = check_redis_health(redis_url="redis://localhost:6379")
            assert result["healthy"] is False
            assert result["error"] == "Ping returned False"

    def test_import_error_returns_warning(self):
        """redis package not installed -> healthy with warning."""
        with patch.dict("sys.modules", {"redis": None}):
            result = check_redis_health(redis_url="redis://localhost:6379")
            assert result["healthy"] is True
            assert result["configured"] is True
            assert "warning" in result

    def test_connection_error_returns_unhealthy(self):
        """ConnectionError -> unhealthy."""
        mock_redis = MagicMock()
        mock_redis.from_url.side_effect = ConnectionError("refused")

        with patch.dict("sys.modules", {"redis": mock_redis}):
            result = check_redis_health(redis_url="redis://localhost:6379")
            assert result["healthy"] is False
            assert result["error"] == "Connection failed"

    def test_timeout_error_returns_unhealthy(self):
        """TimeoutError -> unhealthy."""
        mock_redis = MagicMock()
        mock_redis.from_url.side_effect = TimeoutError("timed out")

        with patch.dict("sys.modules", {"redis": mock_redis}):
            result = check_redis_health(redis_url="redis://localhost:6379")
            assert result["healthy"] is False
            assert result["error"] == "Connection failed"

    def test_os_error_returns_unhealthy(self):
        """OSError -> unhealthy."""
        mock_redis = MagicMock()
        mock_redis.from_url.side_effect = OSError("network unreachable")

        with patch.dict("sys.modules", {"redis": mock_redis}):
            result = check_redis_health(redis_url="redis://localhost:6379")
            assert result["healthy"] is False
            assert result["error"] == "Connection failed"

    def test_value_error_returns_unhealthy(self):
        """ValueError -> unhealthy."""
        mock_redis = MagicMock()
        mock_redis.from_url.side_effect = ValueError("bad url")

        with patch.dict("sys.modules", {"redis": mock_redis}):
            result = check_redis_health(redis_url="redis://localhost:6379")
            assert result["healthy"] is False
            assert result["error"] == "Connection failed"

    def test_runtime_error_returns_unhealthy(self):
        """RuntimeError -> unhealthy."""
        mock_redis = MagicMock()
        mock_redis.from_url.side_effect = RuntimeError("event loop closed")

        with patch.dict("sys.modules", {"redis": mock_redis}):
            result = check_redis_health(redis_url="redis://localhost:6379")
            assert result["healthy"] is False
            assert result["error"] == "Connection failed"

    def test_socket_timeout_is_two_seconds(self):
        """Redis client is created with socket_timeout=2.0."""
        mock_redis = MagicMock()
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis.from_url.return_value = mock_client

        with patch.dict("sys.modules", {"redis": mock_redis}):
            check_redis_health(redis_url="redis://localhost:6379")
            mock_redis.from_url.assert_called_once_with(
                "redis://localhost:6379", socket_timeout=2.0
            )


# ============================================================================
# TestCheckAiProvidersHealth
# ============================================================================


class TestCheckAiProvidersHealth:
    """Tests for check_ai_providers_health() - AI provider availability."""

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        """Remove all AI provider env vars."""
        for var in [
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "GROK_API_KEY",
            "XAI_API_KEY",
            "DEEPSEEK_API_KEY",
            "OPENROUTER_API_KEY",
        ]:
            monkeypatch.delenv(var, raising=False)

    def test_no_providers_configured(self):
        """No API keys set -> all providers False, any_available=False."""
        result = check_ai_providers_health()
        assert result["healthy"] is True
        assert result["any_available"] is False
        assert result["available_count"] == 0
        assert all(v is False for v in result["providers"].values())

    def test_anthropic_key_available(self, monkeypatch):
        """ANTHROPIC_API_KEY set with >10 chars -> anthropic is True."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-12345678901")
        result = check_ai_providers_health()
        assert result["providers"]["anthropic"] is True
        assert result["any_available"] is True
        assert result["available_count"] >= 1

    def test_openai_key_available(self, monkeypatch):
        """OPENAI_API_KEY set -> openai is True."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-12345678901")
        result = check_ai_providers_health()
        assert result["providers"]["openai"] is True

    def test_multiple_providers(self, monkeypatch):
        """Multiple keys set -> multiple providers True."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-12345678901")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-12345678901")
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-key-123456")
        result = check_ai_providers_health()
        assert result["available_count"] == 3
        assert result["any_available"] is True

    def test_short_key_treated_as_unavailable(self, monkeypatch):
        """API key with <=10 chars -> treated as not available."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "short")
        result = check_ai_providers_health()
        assert result["providers"]["anthropic"] is False
        assert result["any_available"] is False

    def test_exactly_10_chars_unavailable(self, monkeypatch):
        """API key with exactly 10 chars -> not available (need >10)."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "1234567890")
        result = check_ai_providers_health()
        assert result["providers"]["anthropic"] is False

    def test_exactly_11_chars_available(self, monkeypatch):
        """API key with 11 chars -> available."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "12345678901")
        result = check_ai_providers_health()
        assert result["providers"]["anthropic"] is True

    def test_all_providers_checked(self):
        """All 7 providers are checked."""
        result = check_ai_providers_health()
        expected_providers = {
            "anthropic",
            "openai",
            "gemini",
            "grok",
            "xai",
            "deepseek",
            "openrouter",
        }
        assert set(result["providers"].keys()) == expected_providers

    def test_always_returns_healthy(self, monkeypatch):
        """Always returns healthy=True regardless of provider availability."""
        result = check_ai_providers_health()
        assert result["healthy"] is True

    def test_empty_string_key_unavailable(self, monkeypatch):
        """Empty string API key -> not available (len 0 <= 10)."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")
        result = check_ai_providers_health()
        assert result["providers"]["anthropic"] is False

    def test_all_seven_providers_available(self, monkeypatch):
        """All seven providers configured -> available_count=7."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-12345678901")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-12345678901")
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-key-1234567")
        monkeypatch.setenv("GROK_API_KEY", "grok-key-12345678")
        monkeypatch.setenv("XAI_API_KEY", "xai-key-123456789")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key-1234")
        monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key-12")
        result = check_ai_providers_health()
        assert result["available_count"] == 7
        assert all(v is True for v in result["providers"].values())


# ============================================================================
# TestCheckSecurityServices
# ============================================================================


class TestCheckSecurityServices:
    """Tests for check_security_services() - security services health."""

    _SECRET = "aragora.config.secrets.get_secret"

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        """Ensure clean env."""
        monkeypatch.delenv("ARAGORA_ENV", raising=False)

    def _patch_encryption(self, return_value=MagicMock()):
        """Patch get_encryption_service to avoid real calls."""
        return patch(
            "aragora.security.encryption.get_encryption_service",
            return_value=return_value,
        )

    def test_healthy_when_all_modules_available(self):
        """All security modules importable -> healthy."""
        with patch(self._SECRET, return_value="some-key"):
            with self._patch_encryption():
                result = check_security_services(is_production=False)
                assert result["healthy"] is True

    def test_encryption_available(self):
        """Encryption service importable -> encryption_available=True."""
        mock_service = MagicMock()
        with patch(self._SECRET, return_value="key"):
            with patch(
                "aragora.security.encryption.get_encryption_service",
                return_value=mock_service,
            ):
                result = check_security_services(is_production=False)
                assert result["encryption_available"] is True

    def test_encryption_configured_when_key_set(self):
        """ARAGORA_ENCRYPTION_KEY set -> encryption_configured=True."""
        with patch(self._SECRET, return_value="my-secret-key"):
            with patch(
                "aragora.security.encryption.get_encryption_service",
                return_value=MagicMock(),
            ):
                result = check_security_services(is_production=False)
                assert result["encryption_configured"] is True

    def test_encryption_not_configured_in_production_unhealthy(self):
        """Production without encryption key -> healthy=False."""
        with patch(self._SECRET, return_value=""):
            with patch(
                "aragora.security.encryption.get_encryption_service",
                return_value=MagicMock(),
            ):
                result = check_security_services(is_production=True)
                assert result["healthy"] is False
                assert "encryption_warning" in result

    def test_encryption_not_configured_in_development_healthy(self):
        """Development without encryption key -> healthy=True."""
        with patch(self._SECRET, return_value=""):
            with patch(
                "aragora.security.encryption.get_encryption_service",
                return_value=MagicMock(),
            ):
                result = check_security_services(is_production=False)
                assert result["healthy"] is True

    def test_encryption_import_error(self):
        """Encryption module not importable -> encryption_available=False."""
        with patch(self._SECRET, return_value="key"):
            with patch.dict("sys.modules", {"aragora.security.encryption": None}):
                result = check_security_services(is_production=False)
                assert result["encryption_available"] is False
                assert "encryption_warning" in result

    def test_encryption_runtime_error(self):
        """Encryption service raises RuntimeError -> error handling."""
        with patch(
            "aragora.security.encryption.get_encryption_service",
            side_effect=RuntimeError("boom"),
        ):
            with patch(self._SECRET, return_value="key"):
                result = check_security_services(is_production=False)
                assert result["encryption_available"] is False
                assert result["encryption_error"] == "Health check failed"

    def test_encryption_value_error(self):
        """Encryption service raises ValueError -> error handling."""
        with patch(
            "aragora.security.encryption.get_encryption_service",
            side_effect=ValueError("bad"),
        ):
            with patch(self._SECRET, return_value="key"):
                result = check_security_services(is_production=False)
                assert result["encryption_available"] is False

    def test_encryption_os_error(self):
        """Encryption service raises OSError -> error handling."""
        with patch(
            "aragora.security.encryption.get_encryption_service",
            side_effect=OSError("file"),
        ):
            with patch(self._SECRET, return_value="key"):
                result = check_security_services(is_production=False)
                assert result["encryption_available"] is False

    def test_encryption_type_error(self):
        """Encryption service raises TypeError -> error handling."""
        with patch(
            "aragora.security.encryption.get_encryption_service",
            side_effect=TypeError("type"),
        ):
            with patch(self._SECRET, return_value="key"):
                result = check_security_services(is_production=False)
                assert result["encryption_available"] is False

    def test_rbac_available(self):
        """RBAC module importable -> rbac_available=True."""
        with patch(self._SECRET, return_value="key"), self._patch_encryption():
            result = check_security_services(is_production=False)
            assert result["rbac_available"] is True

    def test_rbac_import_error(self):
        """RBAC module not importable -> rbac_available=False."""
        with patch(self._SECRET, return_value="key"), self._patch_encryption():
            with patch.dict("sys.modules", {"aragora.rbac": None}):
                result = check_security_services(is_production=False)
                assert result["rbac_available"] is False
                assert "rbac_warning" in result

    def test_audit_logger_configured(self):
        """Audit logger available -> audit_logger_configured check."""
        with patch(self._SECRET, return_value="key"), self._patch_encryption():
            result = check_security_services(is_production=False)
            assert "audit_logger_configured" in result

    def test_audit_logger_import_error(self):
        """Audit logger module missing -> audit_logger_configured=False."""
        with patch(self._SECRET, return_value="key"), self._patch_encryption():
            with patch.dict("sys.modules", {"aragora.server.middleware.audit_logger": None}):
                result = check_security_services(is_production=False)
                assert result["audit_logger_configured"] is False
                assert "audit_warning" in result

    def test_audit_logger_runtime_error(self):
        """Audit logger raises RuntimeError -> audit_error."""
        mock_mod = MagicMock()
        mock_mod.get_audit_logger.side_effect = RuntimeError("broken")
        with patch(self._SECRET, return_value="key"), self._patch_encryption():
            with patch.dict("sys.modules", {"aragora.server.middleware.audit_logger": mock_mod}):
                result = check_security_services(is_production=False)
                assert result["audit_logger_configured"] is False
                assert result["audit_error"] == "Health check failed"

    def test_jwt_auth_available(self):
        """JWT auth module importable -> jwt_auth_available=True."""
        with patch(self._SECRET, return_value="key"), self._patch_encryption():
            result = check_security_services(is_production=False)
            assert "jwt_auth_available" in result

    def test_jwt_auth_import_error(self):
        """JWT auth module missing -> jwt_auth_available=False."""
        with patch(self._SECRET, return_value="key"), self._patch_encryption():
            with patch.dict("sys.modules", {"aragora.billing.jwt_auth": None}):
                result = check_security_services(is_production=False)
                assert result["jwt_auth_available"] is False
                assert "jwt_warning" in result

    def test_is_production_from_env(self, monkeypatch):
        """is_production=None reads from ARAGORA_ENV."""
        monkeypatch.setenv("ARAGORA_ENV", "production")
        with patch(self._SECRET, return_value=""), self._patch_encryption():
            result = check_security_services(is_production=None)
            # Should detect production and flag missing encryption
            assert result["healthy"] is False

    def test_is_production_from_env_development(self, monkeypatch):
        """ARAGORA_ENV=development -> not production."""
        monkeypatch.setenv("ARAGORA_ENV", "development")
        with patch(self._SECRET, return_value=""), self._patch_encryption():
            result = check_security_services(is_production=None)
            assert result["healthy"] is True

    def test_result_always_has_healthy_key(self):
        """Result always includes 'healthy' key."""
        with patch(self._SECRET, return_value="key"), self._patch_encryption():
            result = check_security_services(is_production=False)
            assert "healthy" in result


# ============================================================================
# TestCheckDatabaseHealth
# ============================================================================


class TestCheckDatabaseHealth:
    """Tests for check_database_health() - database connectivity."""

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        """Remove database env vars."""
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("ARAGORA_POSTGRES_DSN", raising=False)

    def test_not_configured_returns_healthy(self):
        """No database_url and no env var -> healthy, not configured."""
        result = check_database_health(database_url=None)
        assert result["healthy"] is True
        assert result["configured"] is False
        assert "note" in result

    def test_empty_string_not_configured(self):
        """Empty string -> not configured."""
        result = check_database_health(database_url="")
        assert result["healthy"] is True
        assert result["configured"] is False

    def test_reads_database_url_from_env(self, monkeypatch):
        """Reads DATABASE_URL from environment."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/db")
        mock_validate = MagicMock(return_value=(True, "Connected"))
        with patch(
            "aragora.server.startup.validate_database_connectivity",
            mock_validate,
        ):
            with patch("asyncio.get_running_loop", side_effect=RuntimeError):
                with patch("asyncio.run", return_value=(True, "Connected")):
                    result = check_database_health(database_url=None)
                    assert result["configured"] is True

    def test_reads_postgres_dsn_from_env(self, monkeypatch):
        """Falls back to ARAGORA_POSTGRES_DSN."""
        monkeypatch.setenv("ARAGORA_POSTGRES_DSN", "postgresql://localhost/db")
        with patch("asyncio.get_running_loop", side_effect=RuntimeError):
            with patch("asyncio.run", return_value=(True, "OK")):
                result = check_database_health(database_url=None)
                assert result["configured"] is True

    def test_healthy_database_no_running_loop(self):
        """Healthy database without running event loop."""
        with patch("asyncio.get_running_loop", side_effect=RuntimeError):
            with patch("asyncio.run", return_value=(True, "Connected")):
                result = check_database_health(database_url="postgresql://localhost/db")
                assert result["healthy"] is True
                assert result["configured"] is True
                assert result["message"] == "Connected"

    def test_unhealthy_database(self):
        """Database reports unhealthy."""
        with patch("asyncio.get_running_loop", side_effect=RuntimeError):
            with patch("asyncio.run", return_value=(False, "Connection refused")):
                result = check_database_health(database_url="postgresql://localhost/db")
                assert result["healthy"] is False
                assert result["configured"] is True

    def test_import_error_skips_check(self):
        """validate_database_connectivity not importable -> check_skipped."""
        with patch.dict("sys.modules", {"aragora.server.startup": None}):
            result = check_database_health(database_url="postgresql://localhost/db")
            assert result["healthy"] is True
            assert result["configured"] is True
            assert result["status"] == "check_skipped"

    def test_connection_error_returns_unhealthy(self):
        """ConnectionError -> unhealthy."""
        with patch("asyncio.get_running_loop", side_effect=RuntimeError):
            with patch("asyncio.run", side_effect=ConnectionError("refused")):
                result = check_database_health(database_url="postgresql://localhost/db")
                assert result["healthy"] is False
                assert result["error"] == "Connection failed"

    def test_timeout_error_returns_unhealthy(self):
        """TimeoutError -> unhealthy."""
        with patch("asyncio.get_running_loop", side_effect=RuntimeError):
            with patch("asyncio.run", side_effect=TimeoutError("timed out")):
                result = check_database_health(database_url="postgresql://localhost/db")
                assert result["healthy"] is False
                assert result["error"] == "Connection failed"

    def test_os_error_returns_unhealthy(self):
        """OSError -> unhealthy."""
        with patch("asyncio.get_running_loop", side_effect=RuntimeError):
            with patch("asyncio.run", side_effect=OSError("network error")):
                result = check_database_health(database_url="postgresql://localhost/db")
                assert result["healthy"] is False
                assert result["error"] == "Connection failed"

    def test_runtime_error_returns_unhealthy(self):
        """RuntimeError from validate -> unhealthy."""
        with patch("asyncio.get_running_loop", side_effect=RuntimeError):
            with patch("asyncio.run", side_effect=RuntimeError("event loop")):
                result = check_database_health(database_url="postgresql://localhost/db")
                assert result["healthy"] is False
                assert result["error"] == "Connection failed"

    def test_value_error_returns_unhealthy(self):
        """ValueError -> unhealthy."""
        with patch("asyncio.get_running_loop", side_effect=RuntimeError):
            with patch("asyncio.run", side_effect=ValueError("bad dsn")):
                result = check_database_health(database_url="postgresql://localhost/db")
                assert result["healthy"] is False
                assert result["error"] == "Connection failed"


# ============================================================================
# TestGetUptimeInfo
# ============================================================================


class TestGetUptimeInfo:
    """Tests for get_uptime_info() - uptime calculation."""

    def test_basic_uptime(self):
        """Returns uptime_seconds and uptime_human."""
        start = time.time() - 60  # 1 minute ago
        result = get_uptime_info(start)
        assert "uptime_seconds" in result
        assert "uptime_human" in result
        assert result["uptime_seconds"] >= 59  # allow slight timing variance

    def test_seconds_only(self):
        """Under 1 hour shows minutes and seconds format."""
        start = time.time() - 90  # 1.5 minutes
        result = get_uptime_info(start)
        assert "m" in result["uptime_human"]
        assert "s" in result["uptime_human"]

    def test_hours_format(self):
        """1-24 hours shows hours/minutes/seconds format."""
        start = time.time() - 3661  # 1h 1m 1s
        result = get_uptime_info(start)
        assert "h" in result["uptime_human"]
        assert "m" in result["uptime_human"]
        assert "s" in result["uptime_human"]

    def test_days_format(self):
        """Over 24 hours shows days/hours/minutes format."""
        start = time.time() - 90000  # ~1 day 1 hour
        result = get_uptime_info(start)
        assert "d" in result["uptime_human"]
        assert "h" in result["uptime_human"]

    def test_uptime_seconds_is_rounded(self):
        """uptime_seconds is rounded to 2 decimal places."""
        start = time.time() - 100
        result = get_uptime_info(start)
        # Check it's a number with at most 2 decimal places
        assert isinstance(result["uptime_seconds"], float)
        decimal_str = str(result["uptime_seconds"])
        if "." in decimal_str:
            decimal_places = len(decimal_str.split(".")[1])
            assert decimal_places <= 2

    def test_zero_uptime(self):
        """Just started -> ~0 seconds uptime."""
        start = time.time()
        result = get_uptime_info(start)
        assert result["uptime_seconds"] < 1
        assert "m" in result["uptime_human"]

    def test_exact_one_day(self):
        """Exactly 1 day shows days format."""
        start = time.time() - 86400
        result = get_uptime_info(start)
        assert "1d" in result["uptime_human"]

    def test_multiple_days(self):
        """Multiple days."""
        start = time.time() - (3 * 86400 + 7200)  # 3 days 2 hours
        result = get_uptime_info(start)
        assert "3d" in result["uptime_human"]
        assert "2h" in result["uptime_human"]

    def test_only_minutes_no_hours(self):
        """Under 1 hour -> no 'h' in format, has 'm' and 's'."""
        start = time.time() - 300  # 5 minutes
        result = get_uptime_info(start)
        assert "h" not in result["uptime_human"]
        assert "d" not in result["uptime_human"]
        assert "m" in result["uptime_human"]


# ============================================================================
# TestCheckStripeHealth
# ============================================================================


class TestCheckStripeHealth:
    """Tests for check_stripe_health() - Stripe API connectivity."""

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        """Remove Stripe env vars."""
        monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)
        monkeypatch.delenv("STRIPE_API_KEY", raising=False)

    def test_not_configured(self):
        """No Stripe key -> healthy, not configured."""
        result = check_stripe_health()
        assert result["healthy"] is True
        assert result["configured"] is False
        assert "note" in result

    def test_reads_stripe_secret_key(self, monkeypatch):
        """Reads STRIPE_SECRET_KEY from environment."""
        monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_123456")

        mock_stripe = MagicMock()
        mock_stripe.Customer.list.return_value = []
        # Need to provide the error module attribute
        mock_error = MagicMock()
        mock_stripe.error = mock_error

        with patch.dict("sys.modules", {"stripe": mock_stripe, "stripe.error": mock_error}):
            result = check_stripe_health()
            assert result["configured"] is True

    def test_reads_stripe_api_key(self, monkeypatch):
        """Falls back to STRIPE_API_KEY."""
        monkeypatch.setenv("STRIPE_API_KEY", "sk_test_123456")

        mock_stripe = MagicMock()
        mock_stripe.Customer.list.return_value = []
        mock_error = MagicMock()
        mock_stripe.error = mock_error

        with patch.dict("sys.modules", {"stripe": mock_stripe, "stripe.error": mock_error}):
            result = check_stripe_health()
            assert result["configured"] is True

    def test_successful_connectivity(self, monkeypatch):
        """Successful API call -> healthy with latency."""
        monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_123456")

        mock_stripe = MagicMock()
        mock_stripe.Customer.list.return_value = []
        mock_error = MagicMock()
        mock_stripe.error = mock_error

        with patch.dict("sys.modules", {"stripe": mock_stripe, "stripe.error": mock_error}):
            result = check_stripe_health()
            assert result["healthy"] is True
            assert result["configured"] is True
            assert "latency_ms" in result

    def test_import_error_returns_warning(self, monkeypatch):
        """stripe package not installed -> healthy with warning."""
        monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_123456")

        with patch.dict("sys.modules", {"stripe": None, "stripe.error": None}):
            result = check_stripe_health()
            assert result["healthy"] is True
            assert result["configured"] is True
            assert "warning" in result

    def test_authentication_error(self, monkeypatch):
        """AuthenticationError -> unhealthy."""
        monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_bad")

        mock_stripe = MagicMock()
        mock_error = MagicMock()
        auth_error = type("AuthenticationError", (Exception,), {})
        api_conn_error = type("APIConnectionError", (Exception,), {})
        mock_error.AuthenticationError = auth_error
        mock_error.APIConnectionError = api_conn_error
        mock_stripe.error = mock_error
        mock_stripe.Customer.list.side_effect = auth_error("bad key")

        with patch.dict("sys.modules", {"stripe": mock_stripe, "stripe.error": mock_error}):
            result = check_stripe_health()
            assert result["healthy"] is False
            assert result["error"] == "Authentication failed"

    def test_api_connection_error(self, monkeypatch):
        """APIConnectionError -> unhealthy."""
        monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_123456")

        mock_stripe = MagicMock()
        mock_error = MagicMock()
        auth_error = type("AuthenticationError", (Exception,), {})
        api_conn_error = type("APIConnectionError", (Exception,), {})
        mock_error.AuthenticationError = auth_error
        mock_error.APIConnectionError = api_conn_error
        mock_stripe.error = mock_error
        mock_stripe.Customer.list.side_effect = api_conn_error("network")

        with patch.dict("sys.modules", {"stripe": mock_stripe, "stripe.error": mock_error}):
            result = check_stripe_health()
            assert result["healthy"] is False
            assert result["error"] == "Connection failed"

    def test_generic_connection_error(self, monkeypatch):
        """Generic ConnectionError -> unhealthy."""
        monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_123456")

        mock_stripe = MagicMock()
        mock_error = MagicMock()
        auth_error = type("AuthenticationError", (Exception,), {})
        api_conn_error = type("APIConnectionError", (Exception,), {})
        mock_error.AuthenticationError = auth_error
        mock_error.APIConnectionError = api_conn_error
        mock_stripe.error = mock_error
        mock_stripe.Customer.list.side_effect = ConnectionError("refused")

        with patch.dict("sys.modules", {"stripe": mock_stripe, "stripe.error": mock_error}):
            result = check_stripe_health()
            assert result["healthy"] is False
            assert result["error"] == "Health check failed"

    def test_timeout_error(self, monkeypatch):
        """TimeoutError -> unhealthy."""
        monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_123456")

        mock_stripe = MagicMock()
        mock_error = MagicMock()
        auth_error = type("AuthenticationError", (Exception,), {})
        api_conn_error = type("APIConnectionError", (Exception,), {})
        mock_error.AuthenticationError = auth_error
        mock_error.APIConnectionError = api_conn_error
        mock_stripe.error = mock_error
        mock_stripe.Customer.list.side_effect = TimeoutError("timed out")

        with patch.dict("sys.modules", {"stripe": mock_stripe, "stripe.error": mock_error}):
            result = check_stripe_health()
            assert result["healthy"] is False
            assert result["error"] == "Health check failed"


# ============================================================================
# TestCheckSlackHealth
# ============================================================================


class TestCheckSlackHealth:
    """Tests for check_slack_health() - Slack API connectivity."""

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        """Remove Slack env vars."""
        monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
        monkeypatch.delenv("SLACK_TOKEN", raising=False)

    def test_not_configured(self):
        """No Slack token -> healthy, not configured."""
        result = check_slack_health()
        assert result["healthy"] is True
        assert result["configured"] is False
        assert "note" in result

    def test_reads_slack_bot_token(self, monkeypatch):
        """Reads SLACK_BOT_TOKEN from environment."""
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test-token")

        mock_httpx = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": True, "team": "TestTeam", "user": "bot"}
        mock_httpx.post.return_value = mock_response
        mock_httpx.TimeoutException = type("TimeoutException", (Exception,), {})

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = check_slack_health()
            assert result["configured"] is True

    def test_reads_slack_token_fallback(self, monkeypatch):
        """Falls back to SLACK_TOKEN."""
        monkeypatch.setenv("SLACK_TOKEN", "xoxb-test-token")

        mock_httpx = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": True, "team": "T", "user": "u"}
        mock_httpx.post.return_value = mock_response
        mock_httpx.TimeoutException = type("TimeoutException", (Exception,), {})

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = check_slack_health()
            assert result["configured"] is True

    def test_successful_auth_test(self, monkeypatch):
        """Successful auth.test -> healthy with team and user."""
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test-token")

        mock_httpx = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "ok": True,
            "team": "MyTeam",
            "user": "mybot",
        }
        mock_httpx.post.return_value = mock_response
        mock_httpx.TimeoutException = type("TimeoutException", (Exception,), {})

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = check_slack_health()
            assert result["healthy"] is True
            assert result["team"] == "MyTeam"
            assert result["user"] == "mybot"
            assert "latency_ms" in result

    def test_auth_test_not_ok(self, monkeypatch):
        """auth.test returns ok=false -> unhealthy."""
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-bad-token")

        mock_httpx = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": False, "error": "invalid_auth"}
        mock_httpx.post.return_value = mock_response
        mock_httpx.TimeoutException = type("TimeoutException", (Exception,), {})

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = check_slack_health()
            assert result["healthy"] is False
            assert result["error"] == "invalid_auth"

    def test_auth_test_not_ok_unknown_error(self, monkeypatch):
        """auth.test ok=false without error key -> 'Unknown error'."""
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-bad-token")

        mock_httpx = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": False}
        mock_httpx.post.return_value = mock_response
        mock_httpx.TimeoutException = type("TimeoutException", (Exception,), {})

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = check_slack_health()
            assert result["healthy"] is False
            assert result["error"] == "Unknown error"

    def test_non_200_status_code(self, monkeypatch):
        """Non-200 HTTP response -> unhealthy with HTTP status."""
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test-token")

        mock_httpx = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_httpx.post.return_value = mock_response
        mock_httpx.TimeoutException = type("TimeoutException", (Exception,), {})

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = check_slack_health()
            assert result["healthy"] is False
            assert result["error"] == "HTTP 500"

    def test_403_status_code(self, monkeypatch):
        """403 response -> unhealthy with HTTP 403."""
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test-token")

        mock_httpx = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_httpx.post.return_value = mock_response
        mock_httpx.TimeoutException = type("TimeoutException", (Exception,), {})

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = check_slack_health()
            assert result["healthy"] is False
            assert result["error"] == "HTTP 403"

    def test_import_error_returns_warning(self, monkeypatch):
        """httpx not installed -> healthy with warning."""
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test-token")

        with patch.dict("sys.modules", {"httpx": None}):
            result = check_slack_health()
            assert result["healthy"] is True
            assert result["configured"] is True
            assert "warning" in result

    def test_timeout_exception(self, monkeypatch):
        """httpx.TimeoutException -> unhealthy with 'Request timeout'."""
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test-token")

        mock_httpx = MagicMock()
        timeout_exc = type("TimeoutException", (Exception,), {})
        mock_httpx.TimeoutException = timeout_exc
        mock_httpx.post.side_effect = timeout_exc("timed out")

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = check_slack_health()
            assert result["healthy"] is False
            assert result["error"] == "Request timeout"

    def test_connection_error(self, monkeypatch):
        """ConnectionError -> unhealthy."""
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test-token")

        mock_httpx = MagicMock()
        mock_httpx.TimeoutException = type("TimeoutException", (Exception,), {})
        mock_httpx.post.side_effect = ConnectionError("refused")

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = check_slack_health()
            assert result["healthy"] is False
            assert result["error"] == "Health check failed"

    def test_os_error(self, monkeypatch):
        """OSError -> unhealthy."""
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test-token")

        mock_httpx = MagicMock()
        mock_httpx.TimeoutException = type("TimeoutException", (Exception,), {})
        mock_httpx.post.side_effect = OSError("network error")

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = check_slack_health()
            assert result["healthy"] is False
            assert result["error"] == "Health check failed"

    def test_value_error(self, monkeypatch):
        """ValueError -> unhealthy."""
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test-token")

        mock_httpx = MagicMock()
        mock_httpx.TimeoutException = type("TimeoutException", (Exception,), {})
        mock_httpx.post.side_effect = ValueError("bad url")

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = check_slack_health()
            assert result["healthy"] is False
            assert result["error"] == "Health check failed"

    def test_sends_bearer_token(self, monkeypatch):
        """Sends Authorization Bearer header with token."""
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test-token")

        mock_httpx = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": True, "team": "T", "user": "u"}
        mock_httpx.post.return_value = mock_response
        mock_httpx.TimeoutException = type("TimeoutException", (Exception,), {})

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            check_slack_health()
            mock_httpx.post.assert_called_once_with(
                "https://slack.com/api/auth.test",
                headers={"Authorization": "Bearer xoxb-test-token"},
                timeout=5.0,
            )

    def test_uses_5_second_timeout(self, monkeypatch):
        """Uses 5 second timeout for HTTP request."""
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test-token")

        mock_httpx = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": True, "team": "T", "user": "u"}
        mock_httpx.post.return_value = mock_response
        mock_httpx.TimeoutException = type("TimeoutException", (Exception,), {})

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            check_slack_health()
            call_kwargs = mock_httpx.post.call_args
            assert call_kwargs[1]["timeout"] == 5.0
