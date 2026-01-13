"""
Tests for CORS configuration and enforcement.

Verifies:
- Origin allowlist configuration
- Wildcard origin rejection
- Preflight request handling
- CORS headers in responses
"""

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

    def test_wildcard_origin_rejected(self):
        """Test that wildcard origin raises ValueError."""
        with patch.dict(os.environ, {"ARAGORA_ALLOWED_ORIGINS": "*"}):
            from importlib import reload
            import aragora.server.cors_config as cors_module

            with pytest.raises(ValueError, match="Wildcard origin"):
                reload(cors_module)

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
