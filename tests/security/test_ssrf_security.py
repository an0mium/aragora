"""
Tests for SSRF security configuration validation.

These tests verify that the SSRF protection module correctly:
- Rejects localhost override in production environments
- Allows localhost override in development/testing environments
- Blocks localhost by default without any environment variables
- Double-checks production mode in _is_hostname_suspicious()
"""

import importlib
import os
import pytest
from unittest.mock import patch

# Import specific functions and classes we need to test
# Note: We need to be careful about module-level validation on import


class TestProductionRejectsLocalhostOverride:
    """Test that ARAGORA_ENV=production + ARAGORA_SSRF_ALLOW_LOCALHOST=true raises error."""

    def test_production_rejects_localhost_override(self, monkeypatch):
        """ARAGORA_ENV=production + ARAGORA_SSRF_ALLOW_LOCALHOST=true should raise SecurityConfigurationError."""
        # Clear any existing env vars first
        monkeypatch.delenv("ARAGORA_ENV", raising=False)
        monkeypatch.delenv("ARAGORA_SSRF_ALLOW_LOCALHOST", raising=False)

        # Set production environment with localhost override
        monkeypatch.setenv("ARAGORA_ENV", "production")
        monkeypatch.setenv("ARAGORA_SSRF_ALLOW_LOCALHOST", "true")

        # Import the module components we need
        from aragora.security.ssrf_protection import (
            SecurityConfigurationError,
            validate_ssrf_security_settings,
        )

        # Should raise SecurityConfigurationError
        with pytest.raises(SecurityConfigurationError) as exc_info:
            validate_ssrf_security_settings()

        # Verify error details
        assert "production" in str(exc_info.value).lower()
        assert exc_info.value.setting == "ARAGORA_SSRF_ALLOW_LOCALHOST"
        assert exc_info.value.environment == "production"

    def test_production_with_uppercase_env_value(self, monkeypatch):
        """Production check should be case-insensitive."""
        monkeypatch.setenv("ARAGORA_ENV", "PRODUCTION")
        monkeypatch.setenv("ARAGORA_SSRF_ALLOW_LOCALHOST", "TRUE")

        from aragora.security.ssrf_protection import (
            SecurityConfigurationError,
            validate_ssrf_security_settings,
        )

        with pytest.raises(SecurityConfigurationError):
            validate_ssrf_security_settings()

    def test_production_without_localhost_override_succeeds(self, monkeypatch):
        """Production environment without localhost override should not raise."""
        monkeypatch.setenv("ARAGORA_ENV", "production")
        monkeypatch.delenv("ARAGORA_SSRF_ALLOW_LOCALHOST", raising=False)

        from aragora.security.ssrf_protection import validate_ssrf_security_settings

        # Should not raise
        validate_ssrf_security_settings()


class TestDevelopmentAllowsLocalhostOverride:
    """Test that ARAGORA_ENV=development allows localhost override."""

    def test_development_allows_localhost_override(self, monkeypatch):
        """ARAGORA_ENV=development should allow localhost override."""
        monkeypatch.setenv("ARAGORA_ENV", "development")
        monkeypatch.setenv("ARAGORA_SSRF_ALLOW_LOCALHOST", "true")

        from aragora.security.ssrf_protection import validate_ssrf_security_settings

        # Should not raise - development allows localhost
        validate_ssrf_security_settings()

    def test_testing_environment_allows_localhost(self, monkeypatch):
        """ARAGORA_ENV=testing should allow localhost override."""
        monkeypatch.setenv("ARAGORA_ENV", "testing")
        monkeypatch.setenv("ARAGORA_SSRF_ALLOW_LOCALHOST", "true")

        from aragora.security.ssrf_protection import validate_ssrf_security_settings

        # Should not raise - testing allows localhost
        validate_ssrf_security_settings()

    def test_local_environment_allows_localhost(self, monkeypatch):
        """ARAGORA_ENV=local should allow localhost override."""
        monkeypatch.setenv("ARAGORA_ENV", "local")
        monkeypatch.setenv("ARAGORA_SSRF_ALLOW_LOCALHOST", "true")

        from aragora.security.ssrf_protection import validate_ssrf_security_settings

        # Should not raise
        validate_ssrf_security_settings()

    def test_empty_env_allows_localhost_override(self, monkeypatch):
        """Empty ARAGORA_ENV (default) should allow localhost override."""
        monkeypatch.delenv("ARAGORA_ENV", raising=False)
        monkeypatch.setenv("ARAGORA_SSRF_ALLOW_LOCALHOST", "true")

        from aragora.security.ssrf_protection import validate_ssrf_security_settings

        # Should not raise - not explicitly production
        validate_ssrf_security_settings()


class TestDefaultBlocksLocalhost:
    """Test that without env vars, localhost is blocked by default."""

    def test_default_blocks_localhost(self, monkeypatch):
        """Without env vars, localhost should be blocked."""
        monkeypatch.delenv("ARAGORA_ENV", raising=False)
        monkeypatch.delenv("ARAGORA_SSRF_ALLOW_LOCALHOST", raising=False)

        from aragora.security.ssrf_protection import _is_hostname_suspicious

        is_suspicious, reason = _is_hostname_suspicious("localhost")
        assert is_suspicious is True
        assert "Localhost" in reason

    def test_default_blocks_127_0_0_1(self, monkeypatch):
        """Without env vars, 127.0.0.1 should be blocked."""
        monkeypatch.delenv("ARAGORA_ENV", raising=False)
        monkeypatch.delenv("ARAGORA_SSRF_ALLOW_LOCALHOST", raising=False)

        from aragora.security.ssrf_protection import _is_hostname_suspicious

        is_suspicious, reason = _is_hostname_suspicious("127.0.0.1")
        assert is_suspicious is True

    def test_default_blocks_ipv6_loopback(self, monkeypatch):
        """Without env vars, ::1 should be blocked."""
        monkeypatch.delenv("ARAGORA_ENV", raising=False)
        monkeypatch.delenv("ARAGORA_SSRF_ALLOW_LOCALHOST", raising=False)

        from aragora.security.ssrf_protection import _is_hostname_suspicious

        is_suspicious, reason = _is_hostname_suspicious("::1")
        assert is_suspicious is True

    def test_validate_url_blocks_localhost_by_default(self, monkeypatch):
        """validate_url should block localhost without env vars."""
        monkeypatch.delenv("ARAGORA_ENV", raising=False)
        monkeypatch.delenv("ARAGORA_SSRF_ALLOW_LOCALHOST", raising=False)

        from aragora.security.ssrf_protection import validate_url

        result = validate_url("http://localhost:8080/api")
        assert result.is_safe is False
        assert "Localhost" in result.error


class TestLocalhostBlockedEvenWithOverrideInProduction:
    """Test that _is_hostname_suspicious() double-checks production mode."""

    def test_localhost_blocked_even_with_override_in_production(self, monkeypatch):
        """Even with ARAGORA_SSRF_ALLOW_LOCALHOST=true, production should block localhost."""
        monkeypatch.setenv("ARAGORA_ENV", "production")
        monkeypatch.setenv("ARAGORA_SSRF_ALLOW_LOCALHOST", "true")

        from aragora.security.ssrf_protection import _is_hostname_suspicious

        is_suspicious, reason = _is_hostname_suspicious("localhost")
        assert is_suspicious is True
        assert "production" in reason.lower()

    def test_127_0_0_1_blocked_in_production_with_override(self, monkeypatch):
        """127.0.0.1 should be blocked in production even with override."""
        monkeypatch.setenv("ARAGORA_ENV", "production")
        monkeypatch.setenv("ARAGORA_SSRF_ALLOW_LOCALHOST", "true")

        from aragora.security.ssrf_protection import _is_hostname_suspicious

        is_suspicious, reason = _is_hostname_suspicious("127.0.0.1")
        assert is_suspicious is True
        assert "production" in reason.lower()

    def test_validate_url_blocks_localhost_in_production_with_override(self, monkeypatch):
        """validate_url should block localhost in production even with override."""
        monkeypatch.setenv("ARAGORA_ENV", "production")
        monkeypatch.setenv("ARAGORA_SSRF_ALLOW_LOCALHOST", "true")

        from aragora.security.ssrf_protection import validate_url

        result = validate_url("http://localhost:3000/api")
        assert result.is_safe is False
        assert "production" in result.error.lower() or "Localhost" in result.error

    def test_all_localhost_variants_blocked_in_production(self, monkeypatch):
        """All localhost hostname variants should be blocked in production."""
        monkeypatch.setenv("ARAGORA_ENV", "production")
        monkeypatch.setenv("ARAGORA_SSRF_ALLOW_LOCALHOST", "true")

        from aragora.security.ssrf_protection import _is_hostname_suspicious

        localhost_variants = [
            "localhost",
            "localhost.localdomain",
            "local",
            "127.0.0.1",
            "::1",
            "[::1]",
            "0.0.0.0",
            "0",
        ]

        for hostname in localhost_variants:
            is_suspicious, reason = _is_hostname_suspicious(hostname)
            assert is_suspicious is True, f"{hostname} should be blocked in production"


class TestSecurityConfigurationErrorAttributes:
    """Test SecurityConfigurationError exception attributes."""

    def test_exception_has_setting_attribute(self):
        """SecurityConfigurationError should have setting attribute."""
        from aragora.security.ssrf_protection import SecurityConfigurationError

        error = SecurityConfigurationError(
            "Test error",
            setting="TEST_SETTING",
            environment="production",
        )
        assert error.setting == "TEST_SETTING"

    def test_exception_has_environment_attribute(self):
        """SecurityConfigurationError should have environment attribute."""
        from aragora.security.ssrf_protection import SecurityConfigurationError

        error = SecurityConfigurationError(
            "Test error",
            setting="TEST_SETTING",
            environment="production",
        )
        assert error.environment == "production"

    def test_exception_message(self):
        """SecurityConfigurationError should have correct message."""
        from aragora.security.ssrf_protection import SecurityConfigurationError

        error = SecurityConfigurationError("Custom error message")
        assert str(error) == "Custom error message"


class TestValidationLogging:
    """Test that appropriate warnings are logged."""

    def test_development_localhost_logs_warning(self, monkeypatch, caplog):
        """Development environment with localhost override should log warning."""
        import logging

        monkeypatch.setenv("ARAGORA_ENV", "development")
        monkeypatch.setenv("ARAGORA_SSRF_ALLOW_LOCALHOST", "true")

        from aragora.security.ssrf_protection import validate_ssrf_security_settings

        with caplog.at_level(logging.WARNING):
            validate_ssrf_security_settings()

        # Check that warning was logged
        assert any(
            "SSRF localhost protection is disabled" in record.message for record in caplog.records
        )

    def test_production_localhost_attempt_logs_warning(self, monkeypatch, caplog):
        """Attempting localhost access in production should log warning."""
        import logging

        monkeypatch.setenv("ARAGORA_ENV", "production")
        monkeypatch.setenv("ARAGORA_SSRF_ALLOW_LOCALHOST", "true")

        from aragora.security.ssrf_protection import _is_hostname_suspicious

        with caplog.at_level(logging.WARNING):
            _is_hostname_suspicious("localhost")

        # Check that warning was logged about production blocking
        assert any("production" in record.message.lower() for record in caplog.records)


class TestIsProductionEnvironment:
    """Test the _is_production_environment helper function."""

    def test_production_returns_true(self, monkeypatch):
        """ARAGORA_ENV=production should return True."""
        monkeypatch.setenv("ARAGORA_ENV", "production")

        from aragora.security.ssrf_protection import _is_production_environment

        assert _is_production_environment() is True

    def test_production_case_insensitive(self, monkeypatch):
        """Production check should be case-insensitive."""
        monkeypatch.setenv("ARAGORA_ENV", "PRODUCTION")

        from aragora.security.ssrf_protection import _is_production_environment

        assert _is_production_environment() is True

    def test_development_returns_false(self, monkeypatch):
        """ARAGORA_ENV=development should return False."""
        monkeypatch.setenv("ARAGORA_ENV", "development")

        from aragora.security.ssrf_protection import _is_production_environment

        assert _is_production_environment() is False

    def test_empty_returns_false(self, monkeypatch):
        """Empty ARAGORA_ENV should return False."""
        monkeypatch.delenv("ARAGORA_ENV", raising=False)

        from aragora.security.ssrf_protection import _is_production_environment

        assert _is_production_environment() is False


class TestModuleLevelValidation:
    """Test that module-level validation works correctly."""

    def test_module_import_with_safe_config(self, monkeypatch):
        """Module should import successfully with safe configuration."""
        monkeypatch.delenv("ARAGORA_ENV", raising=False)
        monkeypatch.delenv("ARAGORA_SSRF_ALLOW_LOCALHOST", raising=False)

        # Force reimport
        import aragora.security.ssrf_protection as ssrf_module

        importlib.reload(ssrf_module)

        # Should import successfully
        assert hasattr(ssrf_module, "validate_url")
        assert hasattr(ssrf_module, "SecurityConfigurationError")

    def test_module_import_fails_with_dangerous_config(self, monkeypatch):
        """Module import should fail with dangerous configuration in production."""
        monkeypatch.setenv("ARAGORA_ENV", "production")
        monkeypatch.setenv("ARAGORA_SSRF_ALLOW_LOCALHOST", "true")

        # Get the exception class before reload (reload may invalidate module refs)
        from aragora.security.ssrf_protection import SecurityConfigurationError
        import aragora.security.ssrf_protection as ssrf_module

        # Should raise SecurityConfigurationError on reload
        with pytest.raises(SecurityConfigurationError):
            importlib.reload(ssrf_module)
