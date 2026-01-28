"""
Tests for Apple OAuth Provider - Sign in with Apple implementation.

Tests cover:
- Authorization URL generation
- Token exchange
- ID token signature verification (JWKS-based)
- User info extraction from ID token
- First-auth user data handling (name only on first sign-in)
- JWT client secret generation
- Token refresh
- Token revocation
- Configuration validation
- JWKS caching behavior

Security test categories:
- Signature verification: Invalid/expired tokens rejected
- Claim validation: iss, aud, exp checked
- Nonce validation: Replay attack prevention
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.oauth_providers.apple import (
    AppleOAuthProvider,
    APPLE_AUTH_URL,
    APPLE_TOKEN_URL,
    APPLE_KEYS_URL,
    APPLE_REVOCATION_URL,
    JWKS_CACHE_TTL,
)
from aragora.server.handlers.oauth_providers.base import (
    OAuthProviderConfig,
    OAuthTokens,
)


# ===========================================================================
# Test Fixtures
# ===========================================================================

# Mock ES256 private key (for testing only - not a real key)
MOCK_PRIVATE_KEY = """-----BEGIN EC PRIVATE KEY-----
MHQCAQEEICNsLqwfRoS+T9BO+aJO2qJpD4tP4H8NqVKc6PvE3K9soAcGBSuBBAAK
oUQDQgAE5K3L6p0Dl6TrH0w0Kw5C0Tw0k0fU1HJ5TQWP5D3eYVQJK7H9B+8NKXJL
mnb5YRpDm/vFq3OZBgoR3gJPKLJCiw==
-----END EC PRIVATE KEY-----"""


@pytest.fixture
def apple_config():
    """Create a minimal Apple OAuth configuration."""
    return OAuthProviderConfig(
        client_id="com.example.app.web",
        client_secret="",  # Apple uses JWT
        redirect_uri="https://example.com/callback",
        scopes=["name", "email"],
        authorization_endpoint=APPLE_AUTH_URL,
        token_endpoint=APPLE_TOKEN_URL,
        revocation_endpoint=APPLE_REVOCATION_URL,
        team_id="TEAMID123",
        key_id="KEYID456",
        private_key=MOCK_PRIVATE_KEY,
    )


@pytest.fixture
def apple_provider(apple_config):
    """Create an Apple OAuth provider with test configuration."""
    provider = AppleOAuthProvider(config=apple_config)
    yield provider
    provider.close()


@pytest.fixture
def unconfigured_provider():
    """Create an Apple OAuth provider without configuration."""
    config = OAuthProviderConfig(
        client_id="",
        client_secret="",
        redirect_uri="",
        scopes=[],
    )
    provider = AppleOAuthProvider(config=config)
    yield provider
    provider.close()


@pytest.fixture(autouse=True)
def clear_jwks_cache():
    """Clear JWKS cache before and after each test."""
    AppleOAuthProvider._jwks_cache = None
    AppleOAuthProvider._jwks_cache_expiry = 0
    yield
    AppleOAuthProvider._jwks_cache = None
    AppleOAuthProvider._jwks_cache_expiry = 0


# ===========================================================================
# Test Data
# ===========================================================================

# Sample JWKS response from Apple
MOCK_JWKS = {
    "keys": [
        {
            "kty": "RSA",
            "kid": "test-key-id-1",
            "use": "sig",
            "alg": "RS256",
            "n": "xzGF4ysF8y9z6p3tFbPvVvYmY",
            "e": "AQAB",
        }
    ]
}


def create_mock_id_token(
    claims: Dict[str, Any],
    kid: str = "test-key-id-1",
    alg: str = "RS256",
) -> str:
    """Create a mock ID token for testing (unverifiable)."""
    header = {"alg": alg, "kid": kid, "typ": "JWT"}
    header_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).rstrip(b"=")
    payload_b64 = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=")
    sig_b64 = base64.urlsafe_b64encode(b"fake_signature").rstrip(b"=")
    return f"{header_b64.decode()}.{payload_b64.decode()}.{sig_b64.decode()}"


# ===========================================================================
# Configuration Tests
# ===========================================================================


class TestAppleOAuthConfiguration:
    """Tests for Apple OAuth provider configuration."""

    def test_is_configured_with_all_required_fields(self, apple_provider):
        """Test that provider reports configured when all fields present."""
        assert apple_provider.is_configured is True

    def test_is_configured_missing_client_id(self, apple_config):
        """Test that provider reports not configured without client_id."""
        apple_config.client_id = ""
        provider = AppleOAuthProvider(config=apple_config)
        assert provider.is_configured is False
        provider.close()

    def test_is_configured_missing_team_id(self, apple_config):
        """Test that provider reports not configured without team_id."""
        apple_config.team_id = ""
        provider = AppleOAuthProvider(config=apple_config)
        assert provider.is_configured is False
        provider.close()

    def test_is_configured_missing_key_id(self, apple_config):
        """Test that provider reports not configured without key_id."""
        apple_config.key_id = ""
        provider = AppleOAuthProvider(config=apple_config)
        assert provider.is_configured is False
        provider.close()

    def test_is_configured_missing_private_key(self, apple_config):
        """Test that provider reports not configured without private_key."""
        apple_config.private_key = ""
        provider = AppleOAuthProvider(config=apple_config)
        assert provider.is_configured is False
        provider.close()

    def test_provider_name(self, apple_provider):
        """Test that provider name is 'apple'."""
        assert apple_provider.PROVIDER_NAME == "apple"


# ===========================================================================
# Authorization URL Tests
# ===========================================================================


class TestAppleAuthorizationUrl:
    """Tests for authorization URL generation."""

    def test_authorization_url_contains_required_params(self, apple_provider):
        """Test that authorization URL includes all required parameters."""
        url = apple_provider.get_authorization_url(state="test-state-123")

        assert APPLE_AUTH_URL in url
        assert "client_id=com.example.app.web" in url
        assert "redirect_uri=" in url
        assert "response_type=code" in url
        assert "scope=name+email" in url or "scope=name%20email" in url
        assert "state=test-state-123" in url
        assert "response_mode=form_post" in url

    def test_authorization_url_custom_redirect_uri(self, apple_provider):
        """Test custom redirect URI override."""
        url = apple_provider.get_authorization_url(
            state="test",
            redirect_uri="https://custom.example.com/cb",
        )
        assert "redirect_uri=https%3A%2F%2Fcustom.example.com%2Fcb" in url

    def test_authorization_url_custom_scopes(self, apple_provider):
        """Test custom scopes override."""
        url = apple_provider.get_authorization_url(
            state="test",
            scopes=["email"],
        )
        assert "scope=email" in url
        assert "name" not in url.split("scope=")[1].split("&")[0]

    def test_authorization_url_with_nonce(self, apple_provider):
        """Test nonce parameter for replay protection."""
        url = apple_provider.get_authorization_url(
            state="test",
            nonce="unique-nonce-789",
        )
        assert "nonce=unique-nonce-789" in url

    def test_authorization_url_custom_response_mode(self, apple_provider):
        """Test custom response_mode parameter."""
        url = apple_provider.get_authorization_url(
            state="test",
            response_mode="query",
        )
        assert "response_mode=query" in url


# ===========================================================================
# JWT Client Secret Tests
# ===========================================================================


class TestAppleClientSecretGeneration:
    """Tests for JWT client secret generation."""

    def test_client_secret_is_valid_jwt(self, apple_provider):
        """Test that generated client secret is a valid JWT."""
        with patch("jwt.encode") as mock_encode:
            mock_encode.return_value = "mock.jwt.token"
            secret = apple_provider._generate_client_secret()

            assert secret == "mock.jwt.token"
            mock_encode.assert_called_once()

            call_args = mock_encode.call_args
            payload = call_args[0][0]
            assert payload["iss"] == "TEAMID123"
            assert payload["sub"] == "com.example.app.web"
            assert payload["aud"] == "https://appleid.apple.com"
            assert "iat" in payload
            assert "exp" in payload

            headers = call_args[1]["headers"]
            assert headers["alg"] == "ES256"
            assert headers["kid"] == "KEYID456"

    def test_client_secret_expiry_is_180_days(self, apple_provider):
        """Test that client secret expires in 180 days (Apple maximum)."""
        with patch("jwt.encode") as mock_encode:
            mock_encode.return_value = "mock.jwt.token"
            apple_provider._generate_client_secret()

            payload = mock_encode.call_args[0][0]
            expected_exp = payload["iat"] + (86400 * 180)
            assert payload["exp"] == expected_exp


# ===========================================================================
# ID Token Decode Tests (Without Verification)
# ===========================================================================


class TestAppleIdTokenDecode:
    """Tests for ID token decoding without verification."""

    def test_decode_id_token_without_verification(self, apple_provider):
        """Test decoding ID token without signature verification."""
        claims = {
            "sub": "001234.abcd5678.1234",
            "email": "user@privaterelay.appleid.com",
            "email_verified": True,
            "iss": "https://appleid.apple.com",
            "aud": "com.example.app.web",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
        }
        token = create_mock_id_token(claims)

        decoded = apple_provider._decode_id_token(token, verify=False)

        assert decoded["sub"] == claims["sub"]
        assert decoded["email"] == claims["email"]

    def test_decode_id_token_invalid_format_rejected(self, apple_provider):
        """Test that malformed tokens are rejected."""
        with pytest.raises(ValueError, match="Invalid ID token format"):
            apple_provider._decode_id_token("not.a.valid.token.format", verify=False)

    def test_decode_id_token_two_parts_rejected(self, apple_provider):
        """Test that tokens with wrong number of parts are rejected."""
        with pytest.raises(ValueError, match="Invalid ID token format"):
            apple_provider._decode_id_token("only.two", verify=False)


# ===========================================================================
# ID Token Verification Tests
# ===========================================================================


class TestAppleIdTokenVerification:
    """Tests for ID token signature verification."""

    def test_verify_id_token_success(self, apple_provider):
        """Test successful ID token verification."""
        claims = {
            "sub": "001234.abcd5678.1234",
            "email": "user@example.com",
            "iss": "https://appleid.apple.com",
            "aud": "com.example.app.web",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
        }

        with patch.object(apple_provider, "_get_signing_key") as mock_get_key:
            mock_key = MagicMock()
            mock_get_key.return_value = (mock_key, "RS256")

            with patch("jwt.decode") as mock_decode:
                mock_decode.return_value = claims

                result = apple_provider._verify_id_token("mock.token.here")

                assert result == claims
                mock_decode.assert_called_once()

    def test_verify_id_token_expired_rejected(self, apple_provider):
        """Test that expired tokens are rejected."""
        import jwt

        with patch.object(apple_provider, "_get_signing_key") as mock_get_key:
            mock_key = MagicMock()
            mock_get_key.return_value = (mock_key, "RS256")

            with patch("jwt.decode") as mock_decode:
                mock_decode.side_effect = jwt.ExpiredSignatureError("Token expired")

                with pytest.raises(ValueError, match="ID token has expired"):
                    apple_provider._verify_id_token("expired.token.here")

    def test_verify_id_token_invalid_audience_rejected(self, apple_provider):
        """Test that tokens with wrong audience are rejected."""
        import jwt

        with patch.object(apple_provider, "_get_signing_key") as mock_get_key:
            mock_key = MagicMock()
            mock_get_key.return_value = (mock_key, "RS256")

            with patch("jwt.decode") as mock_decode:
                mock_decode.side_effect = jwt.InvalidAudienceError("Wrong audience")

                with pytest.raises(ValueError, match="audience does not match"):
                    apple_provider._verify_id_token("wrong.aud.token")

    def test_verify_id_token_invalid_issuer_rejected(self, apple_provider):
        """Test that tokens from wrong issuer are rejected."""
        import jwt

        with patch.object(apple_provider, "_get_signing_key") as mock_get_key:
            mock_key = MagicMock()
            mock_get_key.return_value = (mock_key, "RS256")

            with patch("jwt.decode") as mock_decode:
                mock_decode.side_effect = jwt.InvalidIssuerError("Wrong issuer")

                with pytest.raises(ValueError, match="issuer is not Apple"):
                    apple_provider._verify_id_token("wrong.iss.token")

    def test_verify_id_token_invalid_signature_rejected(self, apple_provider):
        """Test that tokens with invalid signatures are rejected."""
        import jwt

        with patch.object(apple_provider, "_get_signing_key") as mock_get_key:
            mock_key = MagicMock()
            mock_get_key.return_value = (mock_key, "RS256")

            with patch("jwt.decode") as mock_decode:
                mock_decode.side_effect = jwt.InvalidSignatureError("Bad signature")

                with pytest.raises(ValueError, match="signature verification failed"):
                    apple_provider._verify_id_token("bad.sig.token")

    def test_verify_id_token_nonce_mismatch_rejected(self, apple_provider):
        """Test that tokens with wrong nonce are rejected."""
        claims = {
            "sub": "001234.abcd5678.1234",
            "nonce": "expected-nonce",
            "iss": "https://appleid.apple.com",
            "aud": "com.example.app.web",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
        }

        with patch.object(apple_provider, "_get_signing_key") as mock_get_key:
            mock_key = MagicMock()
            mock_get_key.return_value = (mock_key, "RS256")

            with patch("jwt.decode") as mock_decode:
                mock_decode.return_value = claims

                with pytest.raises(ValueError, match="nonce does not match"):
                    apple_provider._verify_id_token(
                        "mock.token.here",
                        nonce="different-nonce",
                    )


# ===========================================================================
# User Info Extraction Tests
# ===========================================================================


class TestAppleUserInfoExtraction:
    """Tests for user info extraction from ID token."""

    def test_get_user_info_from_id_token_basic(self, apple_provider):
        """Test extracting basic user info from ID token."""
        claims = {
            "sub": "001234.abcd5678.1234",
            "email": "user@privaterelay.appleid.com",
            "email_verified": True,
            "iss": "https://appleid.apple.com",
            "aud": "com.example.app.web",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
        }

        with patch.object(apple_provider, "_decode_id_token") as mock_decode:
            mock_decode.return_value = claims

            user_info = apple_provider.get_user_info_from_id_token("mock.token")

            assert user_info.provider == "apple"
            assert user_info.provider_user_id == "001234.abcd5678.1234"
            assert user_info.email == "user@privaterelay.appleid.com"
            assert user_info.email_verified is True
            assert user_info.name is None

    def test_get_user_info_first_auth_with_name(self, apple_provider):
        """Test extracting user info on first authorization (includes name)."""
        claims = {
            "sub": "001234.abcd5678.1234",
            "email": "user@example.com",
            "email_verified": True,
            "iss": "https://appleid.apple.com",
            "aud": "com.example.app.web",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
        }
        user_data = {
            "name": {
                "firstName": "John",
                "lastName": "Doe",
            },
            "email": "user@example.com",
        }

        with patch.object(apple_provider, "_decode_id_token") as mock_decode:
            mock_decode.return_value = claims

            user_info = apple_provider.get_user_info_from_id_token(
                "mock.token",
                user_data=user_data,
            )

            assert user_info.name == "John Doe"
            assert user_info.given_name == "John"
            assert user_info.family_name == "Doe"

    def test_get_user_info_first_name_only(self, apple_provider):
        """Test extracting user info when only first name provided."""
        claims = {
            "sub": "001234.abcd5678.1234",
            "iss": "https://appleid.apple.com",
            "aud": "com.example.app.web",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
        }
        user_data = {"name": {"firstName": "John"}}

        with patch.object(apple_provider, "_decode_id_token") as mock_decode:
            mock_decode.return_value = claims

            user_info = apple_provider.get_user_info_from_id_token(
                "mock.token",
                user_data=user_data,
            )

            assert user_info.name == "John"
            assert user_info.given_name == "John"
            assert user_info.family_name is None

    def test_get_user_info_missing_sub_raises(self, apple_provider):
        """Test that missing 'sub' claim raises ValueError."""
        claims = {
            "email": "user@example.com",
            "iss": "https://appleid.apple.com",
        }

        with patch.object(apple_provider, "_decode_id_token") as mock_decode:
            mock_decode.return_value = claims

            with pytest.raises(ValueError, match="No subject in Apple ID token"):
                apple_provider.get_user_info_from_id_token("mock.token")

    def test_get_user_info_from_callback(self, apple_provider):
        """Test extracting user info from callback response."""
        claims = {
            "sub": "001234.abcd5678.1234",
            "email": "user@example.com",
            "email_verified": True,
            "iss": "https://appleid.apple.com",
            "aud": "com.example.app.web",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
        }
        tokens = OAuthTokens(
            access_token="access_token_123",
            id_token="mock.id.token",
        )
        user_json = '{"name": {"firstName": "Jane", "lastName": "Smith"}}'

        with patch.object(apple_provider, "_decode_id_token") as mock_decode:
            mock_decode.return_value = claims

            user_info = apple_provider.get_user_info_from_callback(tokens, user_json)

            assert user_info.provider_user_id == "001234.abcd5678.1234"
            assert user_info.name == "Jane Smith"

    def test_get_user_info_from_callback_no_id_token_raises(self, apple_provider):
        """Test that callback without ID token raises ValueError."""
        tokens = OAuthTokens(access_token="access_token_123", id_token=None)

        with pytest.raises(ValueError, match="No ID token"):
            apple_provider.get_user_info_from_callback(tokens)

    def test_get_user_info_standard_raises_not_implemented(self, apple_provider):
        """Test that standard get_user_info raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="userinfo endpoint"):
            apple_provider.get_user_info("some_access_token")


# ===========================================================================
# JWKS Caching Tests
# ===========================================================================


class TestAppleJwksCaching:
    """Tests for JWKS caching behavior."""

    def test_jwks_fetched_and_cached(self, apple_provider):
        """Test that JWKS is fetched and cached."""
        mock_response = MagicMock()
        mock_response.json.return_value = MOCK_JWKS
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response

        with patch.object(apple_provider, "_get_http_client", return_value=mock_client):
            jwks = apple_provider._fetch_apple_jwks()

            assert jwks == MOCK_JWKS
            assert AppleOAuthProvider._jwks_cache == MOCK_JWKS
            assert AppleOAuthProvider._jwks_cache_expiry > time.time()

    def test_jwks_cache_used_when_valid(self, apple_provider):
        """Test that cached JWKS is used when not expired."""
        AppleOAuthProvider._jwks_cache = MOCK_JWKS
        AppleOAuthProvider._jwks_cache_expiry = time.time() + 3600

        mock_client = MagicMock()

        with patch.object(apple_provider, "_get_http_client", return_value=mock_client):
            jwks = apple_provider._fetch_apple_jwks()

            assert jwks == MOCK_JWKS
            mock_client.get.assert_not_called()

    def test_jwks_refetched_when_expired(self, apple_provider):
        """Test that JWKS is refetched when cache expires."""
        AppleOAuthProvider._jwks_cache = {"keys": [{"kid": "old"}]}
        AppleOAuthProvider._jwks_cache_expiry = time.time() - 1

        mock_response = MagicMock()
        mock_response.json.return_value = MOCK_JWKS
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response

        with patch.object(apple_provider, "_get_http_client", return_value=mock_client):
            jwks = apple_provider._fetch_apple_jwks()

            assert jwks == MOCK_JWKS
            mock_client.get.assert_called_once_with(APPLE_KEYS_URL)

    def test_jwks_fallback_to_expired_cache_on_error(self, apple_provider):
        """Test that expired cache is used as fallback on fetch error."""
        old_jwks = {"keys": [{"kid": "old"}]}
        AppleOAuthProvider._jwks_cache = old_jwks
        AppleOAuthProvider._jwks_cache_expiry = time.time() - 1

        mock_client = MagicMock()
        mock_client.get.side_effect = Exception("Network error")

        with patch.object(apple_provider, "_get_http_client", return_value=mock_client):
            jwks = apple_provider._fetch_apple_jwks()

            assert jwks == old_jwks


# ===========================================================================
# Token Exchange Tests
# ===========================================================================


class TestAppleTokenExchange:
    """Tests for token exchange."""

    def test_exchange_code_success(self, apple_provider):
        """Test successful code exchange."""
        with patch.object(apple_provider, "_generate_client_secret") as mock_secret:
            mock_secret.return_value = "mock.client.secret"

            with patch.object(apple_provider, "_request_tokens") as mock_request:
                mock_tokens = OAuthTokens(
                    access_token="access_123",
                    id_token="id.token.here",
                    refresh_token="refresh_456",
                )
                mock_request.return_value = mock_tokens

                tokens = apple_provider.exchange_code("auth_code_789")

                assert tokens.access_token == "access_123"
                assert tokens.id_token == "id.token.here"
                assert tokens.refresh_token == "refresh_456"

                mock_request.assert_called_once()
                call_data = mock_request.call_args[0][1]
                assert call_data["code"] == "auth_code_789"
                assert call_data["client_secret"] == "mock.client.secret"


# ===========================================================================
# Token Refresh Tests
# ===========================================================================


class TestAppleTokenRefresh:
    """Tests for token refresh."""

    def test_refresh_access_token_success(self, apple_provider):
        """Test successful token refresh."""
        with patch.object(apple_provider, "_generate_client_secret") as mock_secret:
            mock_secret.return_value = "mock.client.secret"

            with patch.object(apple_provider, "_request_tokens") as mock_request:
                mock_tokens = OAuthTokens(
                    access_token="new_access_123",
                    id_token="new.id.token",
                )
                mock_request.return_value = mock_tokens

                tokens = apple_provider.refresh_access_token("refresh_token_old")

                assert tokens.access_token == "new_access_123"

                call_data = mock_request.call_args[0][1]
                assert call_data["refresh_token"] == "refresh_token_old"
                assert call_data["grant_type"] == "refresh_token"


# ===========================================================================
# Token Revocation Tests
# ===========================================================================


class TestAppleTokenRevocation:
    """Tests for token revocation."""

    def test_revoke_token_success(self, apple_provider):
        """Test successful token revocation."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response

        with patch.object(apple_provider, "_generate_client_secret") as mock_secret:
            mock_secret.return_value = "mock.client.secret"

            with patch.object(apple_provider, "_get_http_client", return_value=mock_client):
                result = apple_provider.revoke_token("token_to_revoke")

                assert result is True
                mock_client.post.assert_called_once()
                call_data = mock_client.post.call_args[1]["data"]
                assert call_data["token"] == "token_to_revoke"

    def test_revoke_token_failure(self, apple_provider):
        """Test token revocation failure."""
        mock_client = MagicMock()
        mock_client.post.side_effect = Exception("Network error")

        with patch.object(apple_provider, "_generate_client_secret") as mock_secret:
            mock_secret.return_value = "mock.client.secret"

            with patch.object(apple_provider, "_get_http_client", return_value=mock_client):
                result = apple_provider.revoke_token("token_to_revoke")

                assert result is False

    def test_revoke_token_with_type_hint(self, apple_provider):
        """Test token revocation with token_type hint."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response

        with patch.object(apple_provider, "_generate_client_secret") as mock_secret:
            mock_secret.return_value = "mock.client.secret"

            with patch.object(apple_provider, "_get_http_client", return_value=mock_client):
                apple_provider.revoke_token("refresh_token", token_type="refresh_token")

                call_data = mock_client.post.call_args[1]["data"]
                assert call_data["token_type_hint"] == "refresh_token"


# ===========================================================================
# Integration-style Tests
# ===========================================================================


class TestAppleOAuthIntegration:
    """Integration-style tests for the full OAuth flow."""

    def test_full_flow_first_authorization(self, apple_provider):
        """Test complete first authorization flow with user name."""
        auth_url = apple_provider.get_authorization_url(
            state="csrf-state",
            nonce="replay-nonce",
        )
        assert "state=csrf-state" in auth_url
        assert "nonce=replay-nonce" in auth_url

        with patch.object(apple_provider, "_generate_client_secret") as mock_secret:
            mock_secret.return_value = "mock.secret"
            with patch.object(apple_provider, "_request_tokens") as mock_request:
                mock_request.return_value = OAuthTokens(
                    access_token="access",
                    id_token="id.token.here",
                )
                tokens = apple_provider.exchange_code("auth_code")
                assert tokens.id_token == "id.token.here"

        claims = {
            "sub": "user.id.123",
            "email": "user@example.com",
            "email_verified": True,
            "iss": "https://appleid.apple.com",
            "aud": "com.example.app.web",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
        }
        user_json = '{"name": {"firstName": "Alice", "lastName": "Wonder"}}'

        with patch.object(apple_provider, "_decode_id_token") as mock_decode:
            mock_decode.return_value = claims

            user_info = apple_provider.get_user_info_from_callback(
                OAuthTokens(access_token="access", id_token="id.token"),
                user_json=user_json,
            )

            assert user_info.provider_user_id == "user.id.123"
            assert user_info.email == "user@example.com"
            assert user_info.name == "Alice Wonder"
            assert user_info.email_verified is True

    def test_subsequent_authorization_no_name(self, apple_provider):
        """Test subsequent authorization (no name provided by Apple)."""
        claims = {
            "sub": "user.id.123",
            "email": "user@privaterelay.appleid.com",
            "email_verified": "true",
            "iss": "https://appleid.apple.com",
            "aud": "com.example.app.web",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
        }

        with patch.object(apple_provider, "_decode_id_token") as mock_decode:
            mock_decode.return_value = claims

            user_info = apple_provider.get_user_info_from_callback(
                OAuthTokens(access_token="access", id_token="id.token"),
                user_json=None,
            )

            assert user_info.provider_user_id == "user.id.123"
            assert user_info.name is None
            assert user_info.email_verified is True
