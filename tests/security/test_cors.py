"""
Tests for CORS configuration and enforcement.

Verifies:
- Origin allowlist configuration
- Wildcard origin rejection in production
- Wildcard origin allowed in development (with warning)
- Non-HTTPS origin warnings in production
- Trailing slash stripping
- Preflight request handling
- CORS headers in responses
"""

import logging
import os

import pytest
from unittest.mock import patch


class TestCORSConfig:
    """Tests for CORSConfig class."""

    def test_default_origins_include_localhost(self):
        """Test default origins include development localhost."""
        from aragora.server.cors_config import DEFAULT_ORIGINS

        assert "http://localhost:3000" in DEFAULT_ORIGINS
        assert "http://localhost:8080" in DEFAULT_ORIGINS
        assert "http://127.0.0.1:3000" in DEFAULT_ORIGINS

    def test_default_origins_include_production(self):
        """Test default origins include production domains."""
        from aragora.server.cors_config import DEFAULT_ORIGINS

        assert "https://aragora.ai" in DEFAULT_ORIGINS
        assert "https://www.aragora.ai" in DEFAULT_ORIGINS
        assert "https://api.aragora.ai" in DEFAULT_ORIGINS

    def test_is_origin_allowed_valid(self):
        """Test valid origin is allowed."""
        from aragora.server.cors_config import CORSConfig

        config = CORSConfig()
        assert config.is_origin_allowed("http://localhost:3000")
        assert config.is_origin_allowed("https://aragora.ai")

    def test_is_origin_allowed_invalid(self):
        """Test invalid origin is rejected."""
        from aragora.server.cors_config import CORSConfig

        config = CORSConfig()
        assert not config.is_origin_allowed("https://evil.com")
        assert not config.is_origin_allowed("https://phishing.aragora.ai.evil.com")

    def test_wildcard_origin_rejected_in_production(self):
        """Test that wildcard origin raises ValueError in production."""
        with patch.dict(os.environ, {"ARAGORA_ALLOWED_ORIGINS": "*"}):
            from aragora.server.cors_config import CORSConfig

            with pytest.raises(ValueError, match="Wildcard origin"):
                CORSConfig(_env_mode="production")

    def test_wildcard_origin_allowed_in_development(self):
        """Test that wildcard origin is allowed in development mode with warning."""
        with patch.dict(os.environ, {"ARAGORA_ALLOWED_ORIGINS": "*"}):
            from aragora.server.cors_config import CORSConfig

            # Should NOT raise in development mode
            config = CORSConfig(_env_mode="development")
            # Wildcard means all origins are allowed
            assert config.is_origin_allowed("https://anything.example.com")
            assert config.is_origin_allowed("http://localhost:9999")
            assert config._allow_all is True

    def test_wildcard_origin_allowed_in_dev_mode(self):
        """Test that wildcard origin is allowed when ARAGORA_ENV=dev."""
        with patch.dict(os.environ, {"ARAGORA_ALLOWED_ORIGINS": "*"}):
            from aragora.server.cors_config import CORSConfig

            config = CORSConfig(_env_mode="dev")
            assert config._allow_all is True

    def test_wildcard_origin_allowed_in_local_mode(self):
        """Test that wildcard origin is allowed when ARAGORA_ENV=local."""
        with patch.dict(os.environ, {"ARAGORA_ALLOWED_ORIGINS": "*"}):
            from aragora.server.cors_config import CORSConfig

            config = CORSConfig(_env_mode="local")
            assert config._allow_all is True

    def test_wildcard_origin_allowed_in_test_mode(self):
        """Test that wildcard origin is allowed when ARAGORA_ENV=test."""
        with patch.dict(os.environ, {"ARAGORA_ALLOWED_ORIGINS": "*"}):
            from aragora.server.cors_config import CORSConfig

            config = CORSConfig(_env_mode="test")
            assert config._allow_all is True

    def test_wildcard_origin_rejected_in_staging(self):
        """Test that wildcard origin is rejected in staging (non-dev) mode."""
        with patch.dict(os.environ, {"ARAGORA_ALLOWED_ORIGINS": "*"}):
            from aragora.server.cors_config import CORSConfig

            with pytest.raises(ValueError, match="Wildcard origin"):
                CORSConfig(_env_mode="staging")

    def test_env_origins_override_defaults(self):
        """Test environment variable overrides default origins."""
        with patch.dict(
            os.environ,
            {"ARAGORA_ALLOWED_ORIGINS": "https://custom.com,https://other.com"},
        ):
            from aragora.server.cors_config import CORSConfig

            config = CORSConfig()
            assert config.is_origin_allowed("https://custom.com")
            assert config.is_origin_allowed("https://other.com")
            # Defaults should NOT be included when env is set
            assert not config.is_origin_allowed("http://localhost:3000")

    def test_add_origin_runtime(self):
        """Test adding origin at runtime."""
        from aragora.server.cors_config import CORSConfig

        config = CORSConfig()
        assert not config.is_origin_allowed("https://new-origin.com")

        config.add_origin("https://new-origin.com")
        assert config.is_origin_allowed("https://new-origin.com")

    def test_remove_origin_runtime(self):
        """Test removing origin at runtime."""
        from aragora.server.cors_config import CORSConfig

        config = CORSConfig()
        config.add_origin("https://temp.com")
        assert config.is_origin_allowed("https://temp.com")

        config.remove_origin("https://temp.com")
        assert not config.is_origin_allowed("https://temp.com")

    def test_get_origins_list_returns_list(self):
        """Test get_origins_list returns a list."""
        from aragora.server.cors_config import CORSConfig

        config = CORSConfig()
        origins = config.get_origins_list()
        assert isinstance(origins, list)
        assert len(origins) > 0


class TestCORSOriginValidation:
    """Tests for origin validation edge cases."""

    def test_subdomain_not_auto_allowed(self):
        """Test that subdomains aren't automatically allowed."""
        from aragora.server.cors_config import CORSConfig

        config = CORSConfig()
        # evil.aragora.ai should NOT be allowed even if aragora.ai is
        assert not config.is_origin_allowed("https://evil.aragora.ai")

    def test_http_https_mismatch(self):
        """Test that HTTP/HTTPS mismatch is rejected."""
        from aragora.server.cors_config import CORSConfig

        config = CORSConfig()
        # Production domain with HTTP should be rejected
        assert not config.is_origin_allowed("http://aragora.ai")

    def test_port_mismatch(self):
        """Test that port mismatch is rejected."""
        from aragora.server.cors_config import CORSConfig

        config = CORSConfig()
        # localhost:3001 should be rejected when only 3000 is allowed
        assert not config.is_origin_allowed("http://localhost:3001")

    def test_case_sensitivity(self):
        """Test origin matching is case-sensitive for host."""
        from aragora.server.cors_config import CORSConfig

        config = CORSConfig()
        # Origins are case-sensitive
        assert not config.is_origin_allowed("https://ARAGORA.AI")

    def test_trailing_slash_not_allowed(self):
        """Test that trailing slash variants are rejected."""
        from aragora.server.cors_config import CORSConfig

        config = CORSConfig()
        # Trailing slash should not match
        assert not config.is_origin_allowed("https://aragora.ai/")

    def test_path_in_origin_not_allowed(self):
        """Test that paths in origin are rejected."""
        from aragora.server.cors_config import CORSConfig

        config = CORSConfig()
        assert not config.is_origin_allowed("https://aragora.ai/path")

    def test_trailing_slash_stripped_from_config(self):
        """Test that trailing slashes are stripped from configured origins."""
        with patch.dict(
            os.environ,
            {"ARAGORA_ALLOWED_ORIGINS": "https://example.com/,https://other.com/"},
        ):
            from aragora.server.cors_config import CORSConfig

            config = CORSConfig()
            # Origins should have trailing slashes stripped
            assert config.is_origin_allowed("https://example.com")
            assert config.is_origin_allowed("https://other.com")
            # With trailing slash should NOT match
            assert not config.is_origin_allowed("https://example.com/")

    def test_invalid_origin_no_scheme_rejected(self):
        """Test that origins without a scheme are rejected."""
        with patch.dict(
            os.environ,
            {"ARAGORA_ALLOWED_ORIGINS": "example.com"},
        ):
            from aragora.server.cors_config import CORSConfig

            with pytest.raises(ValueError, match="must include scheme"):
                CORSConfig()

    def test_invalid_origin_no_host_rejected(self):
        """Test that origins without a host are rejected."""
        with patch.dict(
            os.environ,
            {"ARAGORA_ALLOWED_ORIGINS": "https://"},
        ):
            from aragora.server.cors_config import CORSConfig

            with pytest.raises(ValueError, match="must include scheme"):
                CORSConfig()

    def test_invalid_origin_bad_scheme_rejected(self):
        """Test that origins with non-HTTP(S) schemes are rejected."""
        with patch.dict(
            os.environ,
            {"ARAGORA_ALLOWED_ORIGINS": "ftp://files.example.com"},
        ):
            from aragora.server.cors_config import CORSConfig

            with pytest.raises(ValueError, match="scheme must be"):
                CORSConfig()

    def test_valid_https_origins_accepted(self):
        """Test that valid HTTPS origins are accepted without error."""
        with patch.dict(
            os.environ,
            {"ARAGORA_ALLOWED_ORIGINS": ("https://app.example.com,https://api.example.com")},
        ):
            from aragora.server.cors_config import CORSConfig

            config = CORSConfig(_env_mode="production")
            assert config.is_origin_allowed("https://app.example.com")
            assert config.is_origin_allowed("https://api.example.com")


class TestCORSProductionWarnings:
    """Tests for production-mode CORS warnings and restrictions."""

    def test_non_https_origin_warns_in_production(self, caplog):
        """Test that non-HTTPS origins log a warning in production."""
        with patch.dict(
            os.environ,
            {"ARAGORA_ALLOWED_ORIGINS": "http://internal.example.com"},
        ):
            from aragora.server.cors_config import CORSConfig

            with caplog.at_level(logging.WARNING):
                config = CORSConfig(_env_mode="production")

            assert config.is_origin_allowed("http://internal.example.com")
            assert any(
                "Non-HTTPS origin" in record.message and "internal.example.com" in record.message
                for record in caplog.records
            ), f"Expected non-HTTPS warning in logs, got: {[r.message for r in caplog.records]}"

    def test_non_https_origin_no_warning_in_dev(self, caplog):
        """Test that non-HTTPS origins do not warn in development."""
        with patch.dict(
            os.environ,
            {"ARAGORA_ALLOWED_ORIGINS": "http://localhost:3000"},
        ):
            from aragora.server.cors_config import CORSConfig

            with caplog.at_level(logging.WARNING):
                CORSConfig(_env_mode="development")

            non_https_warnings = [r for r in caplog.records if "Non-HTTPS origin" in r.message]
            assert len(non_https_warnings) == 0, (
                f"Did not expect non-HTTPS warning in dev mode, got: "
                f"{[r.message for r in non_https_warnings]}"
            )

    def test_wildcard_warns_in_development(self, caplog):
        """Test that wildcard origin logs a warning in development mode."""
        with patch.dict(os.environ, {"ARAGORA_ALLOWED_ORIGINS": "*"}):
            from aragora.server.cors_config import CORSConfig

            with caplog.at_level(logging.WARNING):
                CORSConfig(_env_mode="development")

            assert any("Wildcard origin" in record.message for record in caplog.records), (
                f"Expected wildcard warning in logs, got: {[r.message for r in caplog.records]}"
            )

    def test_mixed_origins_warns_only_http_in_production(self, caplog):
        """Test that mixed HTTP/HTTPS origins only warn about HTTP ones in production."""
        with patch.dict(
            os.environ,
            {"ARAGORA_ALLOWED_ORIGINS": ("https://acme.example.com,http://plaintext.devbox.io")},
        ):
            from aragora.server.cors_config import CORSConfig

            with caplog.at_level(logging.WARNING):
                CORSConfig(_env_mode="production")

            non_https_warnings = [r for r in caplog.records if "Non-HTTPS origin" in r.message]
            assert len(non_https_warnings) == 1
            assert "plaintext.devbox.io" in non_https_warnings[0].message
            # Should NOT warn about the HTTPS one
            assert "acme.example.com" not in non_https_warnings[0].message


class TestCORSSingletonBehavior:
    """Tests for CORS singleton behavior."""

    def test_singleton_export(self):
        """Test that cors_config is exported as singleton."""
        from aragora.server.cors_config import cors_config

        assert cors_config is not None
        assert hasattr(cors_config, "is_origin_allowed")

    def test_allowed_origins_export(self):
        """Test ALLOWED_ORIGINS is exported for compatibility."""
        from aragora.server.cors_config import ALLOWED_ORIGINS

        assert isinstance(ALLOWED_ORIGINS, list)
        assert len(ALLOWED_ORIGINS) > 0

    def test_ws_allowed_origins_alias(self):
        """Test WS_ALLOWED_ORIGINS is alias for ALLOWED_ORIGINS."""
        from aragora.server.cors_config import ALLOWED_ORIGINS, WS_ALLOWED_ORIGINS

        assert WS_ALLOWED_ORIGINS == ALLOWED_ORIGINS


class TestCORSProductionModeDetection:
    """Tests for _is_production_mode helper."""

    def test_production_mode(self):
        """Test production is detected correctly."""
        from aragora.server.cors_config import _is_production_mode

        assert _is_production_mode("production") is True
        assert _is_production_mode("staging") is True

    def test_dev_modes_not_production(self):
        """Test that dev modes are not production."""
        from aragora.server.cors_config import _is_production_mode

        assert _is_production_mode("development") is False
        assert _is_production_mode("dev") is False
        assert _is_production_mode("local") is False
        assert _is_production_mode("test") is False

    def test_empty_string_not_production(self):
        """Test that empty string (unset env var) is not production."""
        from aragora.server.cors_config import _is_production_mode

        assert _is_production_mode("") is False

    def test_case_insensitive(self):
        """Test that mode detection is case-insensitive."""
        from aragora.server.cors_config import _is_production_mode

        assert _is_production_mode("PRODUCTION") is True
        assert _is_production_mode("Development") is False
        assert _is_production_mode("TEST") is False
