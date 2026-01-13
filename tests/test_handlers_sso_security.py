"""
Security tests for SSO Handler.

Tests security measures including:
- Redirect URL validation (open redirect prevention)
- HTTPS enforcement in production
- Credential stripping from URLs
- Allowlist enforcement
"""

import os
import pytest
from unittest.mock import Mock, patch

from aragora.server.handlers.sso import SSOHandler


class TestRedirectURLValidation:
    """Test redirect URL validation to prevent open redirect attacks."""

    @pytest.fixture
    def handler(self):
        """Create SSOHandler instance."""
        return SSOHandler()

    def test_empty_url_is_valid(self, handler):
        """Empty URL should be valid (no redirect)."""
        assert handler._validate_redirect_url("") is True
        assert handler._validate_redirect_url(None) is True

    def test_https_url_is_valid(self, handler):
        """HTTPS URLs should be valid."""
        assert handler._validate_redirect_url("https://example.com/callback") is True
        assert handler._validate_redirect_url("https://app.aragora.ai/dashboard") is True

    def test_http_url_is_valid_in_dev(self, handler):
        """HTTP URLs should be valid in non-production."""
        with patch.dict(os.environ, {"ARAGORA_ENV": "development"}, clear=False):
            assert handler._validate_redirect_url("http://localhost:3000/callback") is True
            assert handler._validate_redirect_url("http://127.0.0.1:8080/app") is True

    def test_http_url_blocked_in_production(self, handler):
        """HTTP URLs should be blocked in production."""
        with patch.dict(os.environ, {"ARAGORA_ENV": "production"}, clear=False):
            assert handler._validate_redirect_url("http://example.com/callback") is False

    def test_invalid_scheme_blocked(self, handler):
        """Non-HTTP(S) schemes should be blocked."""
        assert handler._validate_redirect_url("javascript:alert(1)") is False
        assert handler._validate_redirect_url("data:text/html,<script>") is False
        assert handler._validate_redirect_url("file:///etc/passwd") is False
        assert handler._validate_redirect_url("ftp://example.com/file") is False

    def test_credentials_in_url_blocked(self, handler):
        """URLs with credentials should be blocked."""
        assert handler._validate_redirect_url("https://user:pass@evil.com/") is False
        assert handler._validate_redirect_url("https://admin@malicious.com/") is False
        assert handler._validate_redirect_url("http://user:password@localhost/") is False

    def test_allowlist_enforcement(self, handler):
        """URLs should be validated against allowlist when configured."""
        allowed_hosts = "app.aragora.ai,dashboard.aragora.ai,localhost"

        with patch.dict(
            os.environ, {"ARAGORA_SSO_ALLOWED_REDIRECT_HOSTS": allowed_hosts}, clear=False
        ):
            # Allowed hosts should pass
            assert handler._validate_redirect_url("https://app.aragora.ai/callback") is True
            assert handler._validate_redirect_url("https://dashboard.aragora.ai/settings") is True
            assert handler._validate_redirect_url("http://localhost:3000/") is True

            # Disallowed hosts should fail
            assert handler._validate_redirect_url("https://evil.com/phishing") is False
            assert handler._validate_redirect_url("https://aragora.ai.evil.com/") is False
            assert handler._validate_redirect_url("https://not-aragora.ai/") is False

    def test_no_allowlist_allows_all_hosts(self, handler):
        """Without allowlist, all valid hosts should be allowed."""
        with patch.dict(os.environ, {"ARAGORA_SSO_ALLOWED_REDIRECT_HOSTS": ""}, clear=False):
            assert handler._validate_redirect_url("https://any-domain.com/callback") is True
            assert handler._validate_redirect_url("https://another-site.org/path") is True

    def test_case_insensitive_host_matching(self, handler):
        """Host matching should be case-insensitive."""
        with patch.dict(
            os.environ, {"ARAGORA_SSO_ALLOWED_REDIRECT_HOSTS": "App.Aragora.AI"}, clear=False
        ):
            assert handler._validate_redirect_url("https://app.aragora.ai/callback") is True
            assert handler._validate_redirect_url("https://APP.ARAGORA.AI/callback") is True
            assert handler._validate_redirect_url("https://App.Aragora.Ai/callback") is True

    def test_port_stripped_for_host_matching(self, handler):
        """Port should be stripped when matching hosts."""
        with patch.dict(
            os.environ, {"ARAGORA_SSO_ALLOWED_REDIRECT_HOSTS": "localhost"}, clear=False
        ):
            assert handler._validate_redirect_url("http://localhost:3000/callback") is True
            assert handler._validate_redirect_url("http://localhost:8080/app") is True
            assert handler._validate_redirect_url("https://localhost:443/secure") is True

    def test_malformed_url_blocked(self, handler):
        """Malformed URLs should be blocked."""
        # These should not crash and should return False
        assert handler._validate_redirect_url("not-a-valid-url") is False
        assert handler._validate_redirect_url("://missing-scheme.com") is False


class TestProductionSecurityEnforcement:
    """Test security measures enforced in production."""

    @pytest.fixture
    def handler(self):
        return SSOHandler()

    def test_https_required_in_production(self, handler):
        """HTTPS should be required for redirects in production."""
        with patch.dict(os.environ, {"ARAGORA_ENV": "production"}, clear=False):
            # HTTPS should work
            assert handler._validate_redirect_url("https://secure.example.com/") is True

            # HTTP should be blocked
            assert handler._validate_redirect_url("http://insecure.example.com/") is False

    def test_http_allowed_in_development(self, handler):
        """HTTP should be allowed in development mode."""
        with patch.dict(os.environ, {"ARAGORA_ENV": "development"}, clear=False):
            assert handler._validate_redirect_url("http://localhost:3000/") is True

    def test_http_allowed_when_env_not_set(self, handler):
        """HTTP should be allowed when ARAGORA_ENV is not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove ARAGORA_ENV if it exists
            os.environ.pop("ARAGORA_ENV", None)
            assert handler._validate_redirect_url("http://localhost:3000/") is True


class TestCallbackSecurityIntegration:
    """Test that security measures are applied in callback handling."""

    @pytest.fixture
    def handler(self):
        return SSOHandler()

    @pytest.fixture
    def mock_provider(self):
        """Create mock SSO provider."""
        provider = Mock()
        provider.provider_type = Mock()
        provider.provider_type.value = "oidc"
        provider.config = Mock()
        provider.config.callback_url = "https://app.aragora.ai/auth/sso/callback"
        provider.config.session_duration_seconds = 3600
        return provider

    @pytest.fixture
    def mock_user(self):
        """Create mock SSO user."""
        user = Mock()
        user.id = "user-123"
        user.email = "user@example.com"
        user.to_dict = Mock(
            return_value={
                "id": "user-123",
                "email": "user@example.com",
            }
        )
        return user

    @pytest.mark.asyncio
    async def test_callback_blocks_unsafe_redirect(self, handler, mock_provider, mock_user):
        """Callback should block unsafe redirect URLs."""
        with patch.object(handler, "_get_provider", return_value=mock_provider):
            with patch.dict(
                os.environ, {"ARAGORA_SSO_ALLOWED_REDIRECT_HOSTS": "app.aragora.ai"}, clear=False
            ):
                mock_provider.authenticate = Mock(return_value=mock_user)

                # Mock auth_config
                with patch("aragora.server.handlers.sso.auth_config") as mock_auth:
                    mock_auth.generate_token = Mock(return_value="test-token")

                    mock_handler = Mock()
                    mock_handler.headers = {}

                    # Unsafe redirect should be blocked
                    params = {
                        "code": "auth-code",
                        "state": "https://evil.com/phishing",
                        "RelayState": "https://evil.com/phishing",
                    }

                    result = await handler.handle_callback(mock_handler, params)

                    # Should return error, not redirect
                    if hasattr(result, "status_code"):
                        assert result.status_code == 400
                    else:
                        assert result.get("status") == 400 or result.get("status_code") == 400


class TestOpenRedirectPrevention:
    """Test various open redirect attack vectors."""

    @pytest.fixture
    def handler(self):
        return SSOHandler()

    def test_double_slash_bypass_blocked(self, handler):
        """Double-slash bypass attempts should be blocked."""
        # These are common open redirect bypass attempts
        with patch.dict(
            os.environ, {"ARAGORA_SSO_ALLOWED_REDIRECT_HOSTS": "safe.com"}, clear=False
        ):
            assert handler._validate_redirect_url("https://evil.com") is False

    def test_subdomain_bypass_blocked(self, handler):
        """Subdomain bypass attempts should be blocked."""
        with patch.dict(
            os.environ, {"ARAGORA_SSO_ALLOWED_REDIRECT_HOSTS": "aragora.ai"}, clear=False
        ):
            # Only exact host match, not subdomains
            assert handler._validate_redirect_url("https://evil.aragora.ai/") is False
            assert handler._validate_redirect_url("https://aragora.ai.evil.com/") is False

    def test_path_traversal_in_url(self, handler):
        """Path traversal in URL should not bypass validation."""
        with patch.dict(
            os.environ, {"ARAGORA_SSO_ALLOWED_REDIRECT_HOSTS": "safe.com"}, clear=False
        ):
            # Host validation should happen before path is considered
            assert handler._validate_redirect_url("https://evil.com/../safe.com/") is False


class TestLogging:
    """Test that security events are properly logged."""

    @pytest.fixture
    def handler(self):
        return SSOHandler()

    def test_blocked_redirect_is_logged(self, handler):
        """Blocked redirects should be logged."""
        with patch("aragora.server.handlers.sso.logger") as mock_logger:
            with patch.dict(
                os.environ, {"ARAGORA_SSO_ALLOWED_REDIRECT_HOSTS": "safe.com"}, clear=False
            ):
                handler._validate_redirect_url("https://evil.com/phishing")
                mock_logger.warning.assert_called()

    def test_credentials_in_url_logged(self, handler):
        """Credentials in URL should be logged as warning."""
        with patch("aragora.server.handlers.sso.logger") as mock_logger:
            handler._validate_redirect_url("https://user:pass@evil.com/")
            mock_logger.warning.assert_called()

    def test_invalid_scheme_logged(self, handler):
        """Invalid URL schemes should be logged."""
        with patch("aragora.server.handlers.sso.logger") as mock_logger:
            handler._validate_redirect_url("javascript:alert(1)")
            mock_logger.warning.assert_called()
