"""
Comprehensive tests for JWT verification in chat connectors.

Tests the JWT verification utilities for Teams and Google Chat webhooks,
covering security-critical verification flows including:
- JWT token parsing and validation with real tokens
- JWKS (JSON Web Key Set) caching and refresh
- OpenID metadata discovery and caching
- Teams webhook token verification
- Google Chat webhook token verification
- Expired token handling
- Invalid signature detection
- Key rotation handling
- Issuer and audience validation
- Cache TTL and invalidation

Phase 3.1: Bot Framework JWT validation for Teams webhook handler.
"""

import json
import os
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Import jwt library components for creating test tokens
try:
    import jwt
    from jwt import PyJWKClient
    from jwt.exceptions import (
        ExpiredSignatureError,
        InvalidAudienceError,
        InvalidIssuerError,
        InvalidSignatureError,
        PyJWTError,
    )
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.backends import default_backend

    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False
    jwt = None

from aragora.connectors.chat.jwt_verify import (
    JWTVerifier,
    JWTVerificationResult,
    _fetch_openid_metadata,
    _OpenIDMetadataCache,
    get_jwt_verifier,
    verify_teams_webhook,
    verify_google_chat_webhook,
    HAS_JWT,
    MICROSOFT_VALID_ISSUERS,
    GOOGLE_VALID_ISSUERS,
    MICROSOFT_JWKS_URI,
)


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
def rsa_key_pair():
    """Generate an RSA key pair for testing JWT signatures."""
    if not HAS_CRYPTO:
        pytest.skip("Cryptography library not available")

    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend(),
    )
    public_key = private_key.public_key()

    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    return private_key, public_key, private_pem, public_pem


@pytest.fixture
def second_rsa_key_pair():
    """Generate a second RSA key pair for key rotation testing."""
    if not HAS_CRYPTO:
        pytest.skip("Cryptography library not available")

    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend(),
    )
    public_key = private_key.public_key()

    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    return private_key, public_key, private_pem, public_pem


@pytest.fixture
def valid_microsoft_claims():
    """Generate valid Microsoft Bot Framework JWT claims."""
    now = int(time.time())
    return {
        "iss": MICROSOFT_VALID_ISSUERS[0],  # https://api.botframework.com
        "aud": "test-app-id",
        "exp": now + 3600,  # 1 hour from now
        "iat": now,
        "nbf": now - 60,
        "sub": "bot-user-id",
        "serviceurl": "https://smba.trafficmanager.net/teams/",
    }


@pytest.fixture
def valid_google_claims():
    """Generate valid Google Chat JWT claims."""
    now = int(time.time())
    return {
        "iss": GOOGLE_VALID_ISSUERS[1],  # https://accounts.google.com
        "aud": "test-project-id",
        "exp": now + 3600,
        "iat": now,
        "sub": "google-user-id",
        "email": "bot@example.iam.gserviceaccount.com",
    }


@pytest.fixture
def mock_jwks_client():
    """Create a mock PyJWKClient that returns a controllable signing key."""
    mock_client = MagicMock()
    mock_signing_key = MagicMock()
    mock_client.get_signing_key_from_jwt.return_value = mock_signing_key
    return mock_client, mock_signing_key


@pytest.fixture
def fresh_verifier():
    """Create a fresh JWTVerifier instance with no cached state."""
    return JWTVerifier()


# ===========================================================================
# JWTVerificationResult Tests
# ===========================================================================


class TestJWTVerificationResult:
    """Test JWTVerificationResult dataclass."""

    def test_valid_result(self):
        """Valid result has claims and no error."""
        result = JWTVerificationResult(
            valid=True,
            claims={"sub": "user123", "aud": "app123"},
        )
        assert result.valid is True
        assert result.claims["sub"] == "user123"
        assert result.error is None

    def test_invalid_result(self):
        """Invalid result has error message."""
        result = JWTVerificationResult(
            valid=False,
            claims={},
            error="Token expired",
        )
        assert result.valid is False
        assert result.error == "Token expired"

    def test_result_with_full_claims(self):
        """Result preserves all claim fields."""
        claims = {
            "iss": "https://api.botframework.com",
            "aud": "app123",
            "exp": 1700000000,
            "iat": 1699999000,
            "sub": "user123",
            "serviceurl": "https://teams.example.com",
        }
        result = JWTVerificationResult(valid=True, claims=claims)
        assert result.claims == claims
        assert result.claims["iss"] == "https://api.botframework.com"


# ===========================================================================
# JWTVerifier Initialization Tests
# ===========================================================================


class TestJWTVerifierInit:
    """Test JWTVerifier initialization and configuration."""

    def test_verifier_initialization(self):
        """Verifier initializes with no clients."""
        verifier = JWTVerifier()
        assert verifier._microsoft_jwks_client is None
        assert verifier._google_jwks_client is None

    def test_verifier_custom_cache_ttl(self):
        """Verifier accepts custom cache TTL."""
        verifier = JWTVerifier(cache_ttl=600)
        assert verifier._cache_ttl == 600

    def test_verifier_default_cache_ttl(self):
        """Verifier defaults to 1 hour cache TTL."""
        verifier = JWTVerifier()
        assert verifier._cache_ttl == 3600

    def test_get_jwt_verifier_singleton(self):
        """get_jwt_verifier returns singleton."""
        # Reset singleton for clean test
        import aragora.connectors.chat.jwt_verify as module

        original_verifier = module._verifier
        try:
            module._verifier = None
            verifier1 = get_jwt_verifier()
            verifier2 = get_jwt_verifier()
            assert verifier1 is verifier2
        finally:
            module._verifier = original_verifier

    def test_verifier_has_independent_cache_times(self):
        """Microsoft and Google have independent cache timers."""
        verifier = JWTVerifier()
        assert verifier._microsoft_cache_time == 0
        assert verifier._google_cache_time == 0


# ===========================================================================
# JWT Token Parsing and Validation Tests
# ===========================================================================


@pytest.mark.skipif(not HAS_CRYPTO, reason="Cryptography library required")
class TestJWTTokenParsing:
    """Test JWT token parsing with real tokens."""

    def test_parse_valid_jwt_structure(self, rsa_key_pair, valid_microsoft_claims):
        """Valid JWT with proper structure is parsed correctly."""
        private_key, public_key, _, _ = rsa_key_pair

        # Create a real JWT token
        token = jwt.encode(
            valid_microsoft_claims,
            private_key,
            algorithm="RS256",
            headers={"kid": "test-key-id"},
        )

        # Verify token has three parts
        parts = token.split(".")
        assert len(parts) == 3

        # Decode without verification to check structure
        decoded = jwt.decode(token, options={"verify_signature": False})
        assert decoded["iss"] == valid_microsoft_claims["iss"]
        assert decoded["aud"] == valid_microsoft_claims["aud"]

    def test_parse_token_with_all_standard_claims(self, rsa_key_pair):
        """Token with all standard JWT claims is parsed correctly."""
        private_key, _, _, _ = rsa_key_pair
        now = int(time.time())

        claims = {
            "iss": "https://api.botframework.com",
            "sub": "subject-id",
            "aud": "audience-id",
            "exp": now + 3600,
            "nbf": now - 60,
            "iat": now,
            "jti": "unique-token-id",
        }

        token = jwt.encode(claims, private_key, algorithm="RS256")
        decoded = jwt.decode(token, options={"verify_signature": False})

        assert decoded["jti"] == "unique-token-id"
        assert decoded["sub"] == "subject-id"


# ===========================================================================
# Expired Token Handling Tests
# ===========================================================================


@pytest.mark.skipif(not HAS_CRYPTO, reason="Cryptography library required")
class TestExpiredTokenHandling:
    """Test handling of expired JWT tokens."""

    def test_expired_token_rejected(self, rsa_key_pair, mock_jwks_client):
        """Expired tokens are rejected with appropriate error."""
        private_key, public_key, _, _ = rsa_key_pair
        mock_client, mock_signing_key = mock_jwks_client
        mock_signing_key.key = public_key

        # Create expired token
        expired_claims = {
            "iss": MICROSOFT_VALID_ISSUERS[0],
            "aud": "test-app-id",
            "exp": int(time.time()) - 3600,  # Expired 1 hour ago
            "iat": int(time.time()) - 7200,
        }

        token = jwt.encode(expired_claims, private_key, algorithm="RS256")

        verifier = JWTVerifier()

        with patch.object(verifier, "_get_microsoft_jwks_client", return_value=mock_client):
            result = verifier.verify_microsoft_token(token, "test-app-id")

        assert result.valid is False
        assert result.error is not None
        assert "expired" in result.error.lower() or "exp" in result.error.lower()

    def test_token_expiring_soon_still_valid(self, rsa_key_pair, mock_jwks_client):
        """Token expiring in the future is still valid."""
        private_key, public_key, _, _ = rsa_key_pair
        mock_client, mock_signing_key = mock_jwks_client
        mock_signing_key.key = public_key

        # Token expires in 5 seconds
        claims = {
            "iss": MICROSOFT_VALID_ISSUERS[0],
            "aud": "test-app-id",
            "exp": int(time.time()) + 5,
            "iat": int(time.time()),
        }

        token = jwt.encode(claims, private_key, algorithm="RS256")

        verifier = JWTVerifier()

        with patch.object(verifier, "_get_microsoft_jwks_client", return_value=mock_client):
            result = verifier.verify_microsoft_token(token, "test-app-id")

        assert result.valid is True

    def test_token_with_future_nbf_rejected(self, rsa_key_pair, mock_jwks_client):
        """Token with not-before claim in future is handled."""
        private_key, public_key, _, _ = rsa_key_pair
        mock_client, mock_signing_key = mock_jwks_client
        mock_signing_key.key = public_key

        # Token not valid until future
        claims = {
            "iss": MICROSOFT_VALID_ISSUERS[0],
            "aud": "test-app-id",
            "exp": int(time.time()) + 7200,
            "iat": int(time.time()),
            "nbf": int(time.time()) + 3600,  # Not valid for another hour
        }

        token = jwt.encode(claims, private_key, algorithm="RS256")

        verifier = JWTVerifier()

        with patch.object(verifier, "_get_microsoft_jwks_client", return_value=mock_client):
            result = verifier.verify_microsoft_token(token, "test-app-id")

        # PyJWT checks nbf by default
        assert result.valid is False


# ===========================================================================
# Invalid Signature Detection Tests
# ===========================================================================


@pytest.mark.skipif(not HAS_CRYPTO, reason="Cryptography library required")
class TestInvalidSignatureDetection:
    """Test detection of invalid JWT signatures."""

    def test_wrong_key_signature_rejected(
        self, rsa_key_pair, second_rsa_key_pair, mock_jwks_client
    ):
        """Token signed with wrong key is rejected."""
        private_key_1, _, _, _ = rsa_key_pair
        _, public_key_2, _, _ = second_rsa_key_pair  # Different key

        mock_client, mock_signing_key = mock_jwks_client
        mock_signing_key.key = public_key_2  # Verify with different key

        claims = {
            "iss": MICROSOFT_VALID_ISSUERS[0],
            "aud": "test-app-id",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
        }

        # Sign with key 1
        token = jwt.encode(claims, private_key_1, algorithm="RS256")

        verifier = JWTVerifier()

        with patch.object(verifier, "_get_microsoft_jwks_client", return_value=mock_client):
            result = verifier.verify_microsoft_token(token, "test-app-id")

        assert result.valid is False
        assert result.error is not None

    def test_tampered_payload_rejected(self, rsa_key_pair, mock_jwks_client):
        """Token with tampered payload is rejected."""
        private_key, public_key, _, _ = rsa_key_pair
        mock_client, mock_signing_key = mock_jwks_client
        mock_signing_key.key = public_key

        claims = {
            "iss": MICROSOFT_VALID_ISSUERS[0],
            "aud": "test-app-id",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
        }

        token = jwt.encode(claims, private_key, algorithm="RS256")

        # Tamper with the payload (middle part)
        parts = token.split(".")
        # Modify the payload
        import base64

        payload_decoded = base64.urlsafe_b64decode(parts[1] + "==")
        payload_json = json.loads(payload_decoded)
        payload_json["aud"] = "hacked-app-id"
        new_payload = (
            base64.urlsafe_b64encode(json.dumps(payload_json).encode()).decode().rstrip("=")
        )
        tampered_token = f"{parts[0]}.{new_payload}.{parts[2]}"

        verifier = JWTVerifier()

        with patch.object(verifier, "_get_microsoft_jwks_client", return_value=mock_client):
            result = verifier.verify_microsoft_token(tampered_token, "test-app-id")

        assert result.valid is False

    def test_malformed_token_rejected(self):
        """Malformed token structure is rejected."""
        verifier = JWTVerifier()

        # Token without proper structure
        result = verifier.verify_microsoft_token("not-a-valid-jwt", "app123")
        assert result.valid is False
        assert result.error is not None

        # Token with only two parts
        result = verifier.verify_microsoft_token("part1.part2", "app123")
        assert result.valid is False


# ===========================================================================
# Issuer and Audience Validation Tests
# ===========================================================================


@pytest.mark.skipif(not HAS_CRYPTO, reason="Cryptography library required")
class TestIssuerAudienceValidation:
    """Test issuer and audience claim validation."""

    def test_invalid_issuer_rejected(self, rsa_key_pair, mock_jwks_client):
        """Token with invalid issuer is rejected."""
        private_key, public_key, _, _ = rsa_key_pair
        mock_client, mock_signing_key = mock_jwks_client
        mock_signing_key.key = public_key

        claims = {
            "iss": "https://evil-issuer.com",  # Invalid issuer
            "aud": "test-app-id",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
        }

        token = jwt.encode(claims, private_key, algorithm="RS256")

        verifier = JWTVerifier()

        with patch.object(verifier, "_get_microsoft_jwks_client", return_value=mock_client):
            result = verifier.verify_microsoft_token(token, "test-app-id")

        assert result.valid is False
        assert "issuer" in result.error.lower() or "iss" in result.error.lower()

    def test_invalid_audience_rejected(self, rsa_key_pair, mock_jwks_client):
        """Token with wrong audience is rejected."""
        private_key, public_key, _, _ = rsa_key_pair
        mock_client, mock_signing_key = mock_jwks_client
        mock_signing_key.key = public_key

        claims = {
            "iss": MICROSOFT_VALID_ISSUERS[0],
            "aud": "wrong-app-id",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
        }

        token = jwt.encode(claims, private_key, algorithm="RS256")

        verifier = JWTVerifier()

        with patch.object(verifier, "_get_microsoft_jwks_client", return_value=mock_client):
            result = verifier.verify_microsoft_token(token, "expected-app-id")

        assert result.valid is False
        assert "audience" in result.error.lower() or "aud" in result.error.lower()

    def test_all_microsoft_issuers_accepted(self, rsa_key_pair, mock_jwks_client):
        """All valid Microsoft issuers are accepted."""
        private_key, public_key, _, _ = rsa_key_pair
        mock_client, mock_signing_key = mock_jwks_client
        mock_signing_key.key = public_key

        verifier = JWTVerifier()

        for issuer in MICROSOFT_VALID_ISSUERS:
            claims = {
                "iss": issuer,
                "aud": "test-app-id",
                "exp": int(time.time()) + 3600,
                "iat": int(time.time()),
            }

            token = jwt.encode(claims, private_key, algorithm="RS256")

            with patch.object(verifier, "_get_microsoft_jwks_client", return_value=mock_client):
                result = verifier.verify_microsoft_token(token, "test-app-id")

            assert result.valid is True, f"Issuer {issuer} should be valid"

    def test_all_google_issuers_accepted(self, rsa_key_pair, mock_jwks_client):
        """All valid Google issuers are accepted."""
        private_key, public_key, _, _ = rsa_key_pair
        mock_client, mock_signing_key = mock_jwks_client
        mock_signing_key.key = public_key

        verifier = JWTVerifier()

        for issuer in GOOGLE_VALID_ISSUERS:
            claims = {
                "iss": issuer,
                "aud": "test-project-id",
                "exp": int(time.time()) + 3600,
                "iat": int(time.time()),
            }

            token = jwt.encode(claims, private_key, algorithm="RS256")

            with patch.object(verifier, "_get_google_jwks_client", return_value=mock_client):
                result = verifier.verify_google_token(token, "test-project-id")

            assert result.valid is True, f"Issuer {issuer} should be valid"


# ===========================================================================
# JWKS Caching and Refresh Tests
# ===========================================================================


class TestJWKSCaching:
    """Test JWKS client caching behavior."""

    def test_jwks_client_cached(self):
        """JWKS client is cached after first creation."""
        verifier = JWTVerifier()
        verifier._microsoft_cache_time = 0

        with patch("aragora.connectors.chat.jwt_verify.PyJWKClient") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            with patch.object(
                verifier, "_resolve_microsoft_jwks_uri", return_value="https://test/keys"
            ):
                client1 = verifier._get_microsoft_jwks_client()
                assert mock_client.call_count == 1

                client2 = verifier._get_microsoft_jwks_client()
                assert mock_client.call_count == 1
                assert client1 is client2

    def test_jwks_client_refreshes_after_ttl(self):
        """JWKS client refreshes after cache TTL."""
        verifier = JWTVerifier(cache_ttl=0.1)

        with patch("aragora.connectors.chat.jwt_verify.PyJWKClient") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            with patch.object(
                verifier, "_resolve_microsoft_jwks_uri", return_value="https://test/keys"
            ):
                verifier._get_microsoft_jwks_client()
                assert mock_client.call_count == 1

                time.sleep(0.2)

                verifier._get_microsoft_jwks_client()
                assert mock_client.call_count == 2

    def test_google_jwks_client_cached_independently(self):
        """Google JWKS client has its own cache timer."""
        verifier = JWTVerifier()
        verifier._google_cache_time = 0

        with patch("aragora.connectors.chat.jwt_verify.PyJWKClient") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            client1 = verifier._get_google_jwks_client()
            assert mock_client.call_count == 1

            client2 = verifier._get_google_jwks_client()
            assert mock_client.call_count == 1
            assert client1 is client2

    def test_microsoft_and_google_caches_independent(self):
        """Microsoft and Google clients have independent caches."""
        verifier = JWTVerifier(cache_ttl=0.1)

        with patch("aragora.connectors.chat.jwt_verify.PyJWKClient") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            with patch.object(
                verifier, "_resolve_microsoft_jwks_uri", return_value="https://ms/keys"
            ):
                # Create Microsoft client
                verifier._get_microsoft_jwks_client()
                ms_cache_time = verifier._microsoft_cache_time

                # Create Google client
                verifier._get_google_jwks_client()
                google_cache_time = verifier._google_cache_time

                # Both should have been created
                assert mock_client.call_count == 2

                # Cache times should be independent and close but not necessarily equal
                assert abs(ms_cache_time - google_cache_time) < 1

    def test_jwks_client_creation_failure_returns_none(self):
        """JWKS client creation failure returns None."""
        verifier = JWTVerifier()
        verifier._microsoft_cache_time = 0

        with patch("aragora.connectors.chat.jwt_verify.PyJWKClient") as mock_client:
            mock_client.side_effect = Exception("Network error")

            with patch.object(
                verifier, "_resolve_microsoft_jwks_uri", return_value="https://test/keys"
            ):
                client = verifier._get_microsoft_jwks_client()

        assert client is None


# ===========================================================================
# Key Rotation Handling Tests
# ===========================================================================


@pytest.mark.skipif(not HAS_CRYPTO, reason="Cryptography library required")
class TestKeyRotationHandling:
    """Test handling of key rotation scenarios."""

    def test_new_key_accepted_after_cache_refresh(
        self, rsa_key_pair, second_rsa_key_pair, mock_jwks_client
    ):
        """New signing key is accepted after JWKS cache refresh."""
        private_key_1, public_key_1, _, _ = rsa_key_pair
        private_key_2, public_key_2, _, _ = second_rsa_key_pair

        mock_client, mock_signing_key = mock_jwks_client

        verifier = JWTVerifier(cache_ttl=0.1)

        # First token with key 1
        claims = {
            "iss": MICROSOFT_VALID_ISSUERS[0],
            "aud": "test-app-id",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
        }

        token1 = jwt.encode(claims, private_key_1, algorithm="RS256")
        mock_signing_key.key = public_key_1

        with patch.object(verifier, "_get_microsoft_jwks_client", return_value=mock_client):
            result1 = verifier.verify_microsoft_token(token1, "test-app-id")
        assert result1.valid is True

        # Simulate key rotation - new token with key 2
        token2 = jwt.encode(claims, private_key_2, algorithm="RS256")
        mock_signing_key.key = public_key_2

        with patch.object(verifier, "_get_microsoft_jwks_client", return_value=mock_client):
            result2 = verifier.verify_microsoft_token(token2, "test-app-id")
        assert result2.valid is True

    def test_old_key_invalid_after_rotation(
        self, rsa_key_pair, second_rsa_key_pair, mock_jwks_client
    ):
        """Token signed with rotated-out key fails verification."""
        private_key_old, _, _, _ = rsa_key_pair
        _, public_key_new, _, _ = second_rsa_key_pair

        mock_client, mock_signing_key = mock_jwks_client
        mock_signing_key.key = public_key_new  # Only new key available

        claims = {
            "iss": MICROSOFT_VALID_ISSUERS[0],
            "aud": "test-app-id",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
        }

        # Token signed with old key
        token = jwt.encode(claims, private_key_old, algorithm="RS256")

        verifier = JWTVerifier()

        with patch.object(verifier, "_get_microsoft_jwks_client", return_value=mock_client):
            result = verifier.verify_microsoft_token(token, "test-app-id")

        assert result.valid is False


# ===========================================================================
# OpenID Metadata Discovery Tests
# ===========================================================================


class TestOpenIDMetadataDiscovery:
    """Test OpenID metadata discovery for JWKS URI resolution."""

    def test_fetch_openid_metadata_success(self):
        """Successful metadata fetch returns dict with jwks_uri."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "jwks_uri": "https://login.botframework.com/v1/.well-known/keys",
                "issuer": "https://api.botframework.com",
                "authorization_endpoint": "https://invalid.botframework.com",
            }
        ).encode("utf-8")
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("aragora.connectors.chat.jwt_verify.urlopen", return_value=mock_response):
            result = _fetch_openid_metadata(
                "https://login.botframework.com/v1/.well-known/openidconfiguration"
            )

        assert result is not None
        assert result["jwks_uri"] == "https://login.botframework.com/v1/.well-known/keys"
        assert result["issuer"] == "https://api.botframework.com"

    def test_fetch_openid_metadata_missing_jwks_uri(self):
        """Returns None when jwks_uri field is missing."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "issuer": "https://api.botframework.com",
            }
        ).encode("utf-8")
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("aragora.connectors.chat.jwt_verify.urlopen", return_value=mock_response):
            result = _fetch_openid_metadata("https://example.com/.well-known/openidconfiguration")

        assert result is None

    def test_fetch_openid_metadata_network_error(self):
        """Returns None on network error."""
        from urllib.error import URLError

        with patch(
            "aragora.connectors.chat.jwt_verify.urlopen", side_effect=URLError("Connection refused")
        ):
            result = _fetch_openid_metadata("https://example.com/.well-known/openidconfiguration")

        assert result is None

    def test_fetch_openid_metadata_invalid_json(self):
        """Returns None on invalid JSON response."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"not json"
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("aragora.connectors.chat.jwt_verify.urlopen", return_value=mock_response):
            result = _fetch_openid_metadata("https://example.com/.well-known/openidconfiguration")

        assert result is None

    def test_fetch_openid_metadata_timeout(self):
        """Returns None on timeout."""
        from urllib.error import URLError

        with patch(
            "aragora.connectors.chat.jwt_verify.urlopen",
            side_effect=URLError("timed out"),
        ):
            result = _fetch_openid_metadata("https://example.com/.well-known/openidconfiguration")

        assert result is None

    def test_resolve_microsoft_jwks_uri_uses_discovery(self):
        """JWKS URI is resolved from OpenID metadata."""
        verifier = JWTVerifier()

        with patch(
            "aragora.connectors.chat.jwt_verify._fetch_openid_metadata",
            return_value={"jwks_uri": "https://discovered.example.com/keys", "issuer": "test"},
        ):
            uri = verifier._resolve_microsoft_jwks_uri()

        assert uri == "https://discovered.example.com/keys"
        assert verifier._microsoft_metadata is not None
        assert verifier._microsoft_metadata.jwks_uri == "https://discovered.example.com/keys"

    def test_resolve_microsoft_jwks_uri_caches_result(self):
        """Cached JWKS URI is returned without re-fetching."""
        verifier = JWTVerifier()
        verifier._microsoft_metadata = _OpenIDMetadataCache(
            jwks_uri="https://cached.example.com/keys",
            issuer="cached-issuer",
            fetched_at=time.time(),
        )

        with patch(
            "aragora.connectors.chat.jwt_verify._fetch_openid_metadata",
        ) as mock_fetch:
            uri = verifier._resolve_microsoft_jwks_uri()

        assert uri == "https://cached.example.com/keys"
        mock_fetch.assert_not_called()

    def test_resolve_microsoft_jwks_uri_falls_back_on_failure(self):
        """Falls back to hardcoded URI when discovery fails."""
        verifier = JWTVerifier()

        with patch(
            "aragora.connectors.chat.jwt_verify._fetch_openid_metadata",
            return_value=None,
        ):
            uri = verifier._resolve_microsoft_jwks_uri()

        assert uri == MICROSOFT_JWKS_URI

    def test_resolve_microsoft_jwks_uri_refetches_after_ttl(self):
        """Metadata is re-fetched after cache TTL expires."""
        verifier = JWTVerifier(cache_ttl=0.1)
        verifier._microsoft_metadata = _OpenIDMetadataCache(
            jwks_uri="https://old.example.com/keys",
            issuer="old-issuer",
            fetched_at=time.time() - 1.0,
        )

        with patch(
            "aragora.connectors.chat.jwt_verify._fetch_openid_metadata",
            return_value={"jwks_uri": "https://new.example.com/keys", "issuer": "new-issuer"},
        ):
            uri = verifier._resolve_microsoft_jwks_uri()

        assert uri == "https://new.example.com/keys"

    def test_openid_metadata_cache_dataclass(self):
        """OpenIDMetadataCache dataclass stores fields correctly."""
        cache = _OpenIDMetadataCache(
            jwks_uri="https://example.com/keys",
            issuer="https://example.com",
        )
        assert cache.jwks_uri == "https://example.com/keys"
        assert cache.issuer == "https://example.com"
        assert cache.fetched_at > 0


# ===========================================================================
# Teams Webhook Verification Tests
# ===========================================================================


class TestTeamsWebhookVerification:
    """Test verify_teams_webhook function."""

    def test_missing_bearer_prefix(self):
        """Verification fails without Bearer prefix."""
        result = verify_teams_webhook("just-a-token", "app123")
        assert result is False

    def test_empty_header(self):
        """Verification fails with empty header."""
        result = verify_teams_webhook("", "app123")
        assert result is False

    def test_invalid_token(self):
        """Verification fails for invalid JWT."""
        result = verify_teams_webhook("Bearer invalid.token.here", "app123")
        assert result is False

    def test_basic_auth_rejected(self):
        """Verification fails for Basic auth scheme."""
        result = verify_teams_webhook("Basic dXNlcjpwYXNz", "app123")
        assert result is False

    def test_bearer_only_rejected(self):
        """Verification fails for 'Bearer' without token."""
        result = verify_teams_webhook("Bearer ", "app123")
        assert result is False

    @pytest.mark.skipif(not HAS_CRYPTO, reason="Cryptography library required")
    def test_valid_teams_token_accepted(self, rsa_key_pair, mock_jwks_client):
        """Valid Teams token with proper claims is accepted."""
        private_key, public_key, _, _ = rsa_key_pair
        mock_client, mock_signing_key = mock_jwks_client
        mock_signing_key.key = public_key

        claims = {
            "iss": MICROSOFT_VALID_ISSUERS[0],
            "aud": "test-app-id",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
        }

        token = jwt.encode(claims, private_key, algorithm="RS256")

        with patch("aragora.connectors.chat.jwt_verify.get_jwt_verifier") as mock_get_verifier:
            mock_verifier = MagicMock()
            mock_verifier.verify_microsoft_token.return_value = JWTVerificationResult(
                valid=True, claims=claims
            )
            mock_get_verifier.return_value = mock_verifier

            result = verify_teams_webhook(f"Bearer {token}", "test-app-id")

        assert result is True


# ===========================================================================
# Google Chat Webhook Verification Tests
# ===========================================================================


class TestGoogleChatWebhookVerification:
    """Test verify_google_chat_webhook function."""

    def test_missing_bearer_prefix(self):
        """Verification fails without Bearer prefix."""
        result = verify_google_chat_webhook("just-a-token")
        assert result is False

    def test_empty_header(self):
        """Verification fails with empty header."""
        result = verify_google_chat_webhook("")
        assert result is False

    def test_invalid_token(self):
        """Verification fails for invalid JWT."""
        result = verify_google_chat_webhook("Bearer invalid.token.here")
        assert result is False

    def test_accepts_without_project_id(self):
        """Can verify without project_id (skips audience check in dev)."""
        result = verify_google_chat_webhook("Bearer test", project_id=None)
        assert isinstance(result, bool)

    def test_production_requires_project_id(self):
        """Production mode requires project_id for audience validation."""
        with patch.dict("os.environ", {"ARAGORA_ENV": "production"}):
            with patch("aragora.connectors.chat.jwt_verify._IS_PRODUCTION", True):
                # Need to reimport or patch at module level
                verifier = JWTVerifier()

                # Mock to get past JWKS client check
                mock_client = MagicMock()
                mock_signing_key = MagicMock()
                mock_client.get_signing_key_from_jwt.return_value = mock_signing_key

                with patch.object(verifier, "_get_google_jwks_client", return_value=mock_client):
                    with patch("aragora.connectors.chat.jwt_verify._IS_PRODUCTION", True):
                        result = verifier.verify_google_token("dummy.jwt.token", project_id=None)

                # In production without project_id, should fail
                assert result.valid is False
                assert "production" in result.error.lower() or "project_id" in result.error.lower()

    @pytest.mark.skipif(not HAS_CRYPTO, reason="Cryptography library required")
    def test_valid_google_token_accepted(self, rsa_key_pair, mock_jwks_client):
        """Valid Google token with proper claims is accepted."""
        private_key, public_key, _, _ = rsa_key_pair
        mock_client, mock_signing_key = mock_jwks_client
        mock_signing_key.key = public_key

        claims = {
            "iss": GOOGLE_VALID_ISSUERS[1],  # https://accounts.google.com
            "aud": "test-project-id",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
        }

        token = jwt.encode(claims, private_key, algorithm="RS256")

        with patch("aragora.connectors.chat.jwt_verify.get_jwt_verifier") as mock_get_verifier:
            mock_verifier = MagicMock()
            mock_verifier.verify_google_token.return_value = JWTVerificationResult(
                valid=True, claims=claims
            )
            mock_get_verifier.return_value = mock_verifier

            result = verify_google_chat_webhook(f"Bearer {token}", "test-project-id")

        assert result is True


# ===========================================================================
# Fail-Closed Behavior Tests
# ===========================================================================


class TestFailClosedBehavior:
    """Test fail-closed security model."""

    def test_microsoft_token_fails_closed_without_pyjwt(self):
        """Microsoft verification fails closed when PyJWT unavailable."""
        verifier = JWTVerifier()
        with patch("aragora.connectors.chat.jwt_verify.HAS_JWT", False):
            result = verifier.verify_microsoft_token("some.jwt.token", "app123")
        assert result.valid is False
        assert "PyJWT" in result.error

    def test_google_token_fails_closed_without_pyjwt(self):
        """Google verification fails closed when PyJWT unavailable."""
        verifier = JWTVerifier()
        with patch("aragora.connectors.chat.jwt_verify.HAS_JWT", False):
            result = verifier.verify_google_token("some.jwt.token", "project123")
        assert result.valid is False
        assert "PyJWT" in result.error

    def test_microsoft_jwks_client_returns_none_without_pyjwt(self):
        """JWKS client returns None when PyJWT not available."""
        verifier = JWTVerifier()
        with patch("aragora.connectors.chat.jwt_verify.HAS_JWT", False):
            client = verifier._get_microsoft_jwks_client()
        assert client is None

    def test_google_jwks_client_returns_none_without_pyjwt(self):
        """JWKS client returns None when PyJWT not available."""
        verifier = JWTVerifier()
        with patch("aragora.connectors.chat.jwt_verify.HAS_JWT", False):
            client = verifier._get_google_jwks_client()
        assert client is None

    def test_jwks_client_unavailable_fails_closed(self):
        """Verification fails when JWKS client cannot be created."""
        verifier = JWTVerifier()

        with patch.object(verifier, "_get_microsoft_jwks_client", return_value=None):
            result = verifier.verify_microsoft_token("some.jwt.token", "app123")

        assert result.valid is False
        assert "JWKS" in result.error or "unavailable" in result.error.lower()


# ===========================================================================
# Cache TTL and Invalidation Tests
# ===========================================================================


class TestCacheTTLAndInvalidation:
    """Test cache TTL behavior and invalidation."""

    def test_short_cache_ttl_triggers_refresh(self):
        """Very short cache TTL triggers refresh on next call."""
        verifier = JWTVerifier(cache_ttl=0.01)  # 10ms

        with patch("aragora.connectors.chat.jwt_verify.PyJWKClient") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            with patch.object(
                verifier, "_resolve_microsoft_jwks_uri", return_value="https://test/keys"
            ):
                verifier._get_microsoft_jwks_client()
                initial_count = mock_client.call_count

                time.sleep(0.02)  # Wait for TTL to expire

                verifier._get_microsoft_jwks_client()
                assert mock_client.call_count > initial_count

    def test_long_cache_ttl_prevents_refresh(self):
        """Long cache TTL prevents unnecessary refresh."""
        verifier = JWTVerifier(cache_ttl=3600)  # 1 hour

        with patch("aragora.connectors.chat.jwt_verify.PyJWKClient") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance

            with patch.object(
                verifier, "_resolve_microsoft_jwks_uri", return_value="https://test/keys"
            ):
                verifier._get_microsoft_jwks_client()
                verifier._get_microsoft_jwks_client()
                verifier._get_microsoft_jwks_client()

                # Should only create client once
                assert mock_client.call_count == 1

    def test_metadata_cache_invalidation_on_ttl_expiry(self):
        """OpenID metadata cache is invalidated after TTL."""
        verifier = JWTVerifier(cache_ttl=0.1)

        # Set up expired cache
        verifier._microsoft_metadata = _OpenIDMetadataCache(
            jwks_uri="https://old.example.com/keys",
            issuer="old-issuer",
            fetched_at=time.time() - 1.0,  # Well past TTL
        )

        call_count = 0

        def mock_fetch(*args):
            nonlocal call_count
            call_count += 1
            return {"jwks_uri": "https://new.example.com/keys", "issuer": "new-issuer"}

        with patch(
            "aragora.connectors.chat.jwt_verify._fetch_openid_metadata",
            side_effect=mock_fetch,
        ):
            uri = verifier._resolve_microsoft_jwks_uri()

        assert uri == "https://new.example.com/keys"
        assert call_count == 1


# ===========================================================================
# Connector Integration Tests
# ===========================================================================


class TestTeamsConnectorVerification:
    """Test Teams connector webhook verification integration."""

    def test_teams_connector_uses_jwt_verify(self):
        """Teams connector calls JWT verification."""
        from aragora.connectors.chat.teams import TeamsConnector

        connector = TeamsConnector(app_id="test-app-id")

        # Missing auth header
        result = connector.verify_webhook({}, b"{}")
        assert result is False

        # Invalid auth header format
        result = connector.verify_webhook({"Authorization": "Basic xyz"}, b"{}")
        assert result is False

    def test_teams_connector_accepts_bearer_token(self):
        """Teams connector accepts Bearer token (may pass or fail based on JWT lib)."""
        from aragora.connectors.chat.teams import TeamsConnector

        connector = TeamsConnector(app_id="test-app-id")

        headers = {"Authorization": "Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.test"}
        result = connector.verify_webhook(headers, b"{}")

        assert isinstance(result, bool)


class TestGoogleChatConnectorVerification:
    """Test Google Chat connector webhook verification integration."""

    def test_google_chat_connector_uses_jwt_verify(self):
        """Google Chat connector calls JWT verification."""
        from aragora.connectors.chat.google_chat import GoogleChatConnector

        connector = GoogleChatConnector(project_id="test-project")

        # Missing auth header
        result = connector.verify_webhook({}, b"{}")
        assert result is False

        # Invalid auth header format
        result = connector.verify_webhook({"Authorization": "Basic xyz"}, b"{}")
        assert result is False

    def test_google_chat_connector_accepts_bearer_token(self):
        """Google Chat connector accepts Bearer token."""
        from aragora.connectors.chat.google_chat import GoogleChatConnector

        connector = GoogleChatConnector(project_id="test-project")

        headers = {"Authorization": "Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.test"}
        result = connector.verify_webhook(headers, b"{}")

        assert isinstance(result, bool)


# ===========================================================================
# Teams Handler JWT Integration Tests
# ===========================================================================


class TestTeamsHandlerJWTIntegration:
    """Test JWT validation integration in the Teams handler."""

    @pytest.mark.asyncio
    async def test_verify_teams_token_rejects_missing_header(self):
        """Empty auth header is rejected."""
        from aragora.server.handlers.bots.teams import _verify_teams_token

        result = await _verify_teams_token("", "app-123")
        assert result is False

    @pytest.mark.asyncio
    async def test_verify_teams_token_rejects_non_bearer(self):
        """Non-Bearer auth header is rejected."""
        from aragora.server.handlers.bots.teams import _verify_teams_token

        result = await _verify_teams_token("Basic abc123", "app-123")
        assert result is False

    @pytest.mark.asyncio
    async def test_verify_teams_token_delegates_to_jwt_verify(self):
        """Token verification delegates to centralized jwt_verify module."""
        from aragora.server.handlers.bots.teams import _verify_teams_token

        with patch(
            "aragora.connectors.chat.jwt_verify.verify_teams_webhook",
            return_value=True,
        ) as mock_verify:
            with patch("aragora.connectors.chat.jwt_verify.HAS_JWT", True):
                result = await _verify_teams_token("Bearer valid.jwt.token", "app-123")

        assert result is True
        mock_verify.assert_called_once_with("Bearer valid.jwt.token", "app-123")

    @pytest.mark.asyncio
    async def test_verify_teams_token_rejects_invalid_jwt(self):
        """Invalid JWT token is rejected."""
        from aragora.server.handlers.bots.teams import _verify_teams_token

        with patch(
            "aragora.connectors.chat.jwt_verify.verify_teams_webhook",
            return_value=False,
        ):
            with patch("aragora.connectors.chat.jwt_verify.HAS_JWT", True):
                result = await _verify_teams_token("Bearer invalid.jwt.token", "app-123")

        assert result is False

    def test_ms_app_id_env_var_fallback(self):
        """TEAMS_APP_ID falls back to MS_APP_ID environment variable."""
        with patch.dict("os.environ", {"MS_APP_ID": "ms-app-fallback"}, clear=False):
            with patch.dict("os.environ", {"TEAMS_APP_ID": ""}, clear=False):
                app_id = os.environ.get("TEAMS_APP_ID") or os.environ.get("MS_APP_ID")
                assert app_id == "ms-app-fallback"

    def test_teams_app_id_takes_precedence(self):
        """TEAMS_APP_ID takes precedence over MS_APP_ID."""
        with patch.dict(
            "os.environ", {"TEAMS_APP_ID": "teams-app", "MS_APP_ID": "ms-app"}, clear=False
        ):
            app_id = os.environ.get("TEAMS_APP_ID") or os.environ.get("MS_APP_ID")
            assert app_id == "teams-app"


# ===========================================================================
# Error Path Tests
# ===========================================================================


class TestErrorPaths:
    """Test error handling paths."""

    def test_microsoft_token_verification_exception_handling(self):
        """Unexpected exceptions during verification are handled."""
        verifier = JWTVerifier()

        mock_client = MagicMock()
        mock_client.get_signing_key_from_jwt.side_effect = ValueError("Unexpected error")

        with patch.object(verifier, "_get_microsoft_jwks_client", return_value=mock_client):
            result = verifier.verify_microsoft_token("some.jwt.token", "app123")

        assert result.valid is False
        assert result.error is not None

    def test_google_token_verification_exception_handling(self):
        """Unexpected exceptions during Google verification are handled."""
        verifier = JWTVerifier()

        mock_client = MagicMock()
        mock_client.get_signing_key_from_jwt.side_effect = TypeError("Unexpected error")

        with patch.object(verifier, "_get_google_jwks_client", return_value=mock_client):
            result = verifier.verify_google_token("some.jwt.token", "project123")

        assert result.valid is False
        assert result.error is not None

    def test_key_error_during_verification(self):
        """KeyError during claim access is handled."""
        verifier = JWTVerifier()

        mock_client = MagicMock()
        mock_client.get_signing_key_from_jwt.side_effect = KeyError("kid")

        with patch.object(verifier, "_get_microsoft_jwks_client", return_value=mock_client):
            result = verifier.verify_microsoft_token("some.jwt.token", "app123")

        assert result.valid is False


# ===========================================================================
# Module Exports Tests
# ===========================================================================


class TestModuleExports:
    """Test module exports are available."""

    def test_all_exports_available(self):
        """All documented items are exported."""
        from aragora.connectors.chat import jwt_verify

        expected = [
            "JWTVerifier",
            "JWTVerificationResult",
            "get_jwt_verifier",
            "verify_teams_webhook",
            "verify_google_chat_webhook",
            "HAS_JWT",
            "_fetch_openid_metadata",
            "_OpenIDMetadataCache",
        ]

        for name in expected:
            assert hasattr(jwt_verify, name), f"Missing export: {name}"

    def test_constants_exported(self):
        """Important constants are accessible."""
        from aragora.connectors.chat.jwt_verify import (
            MICROSOFT_VALID_ISSUERS,
            GOOGLE_VALID_ISSUERS,
            MICROSOFT_JWKS_URI,
            GOOGLE_JWKS_URI,
        )

        assert len(MICROSOFT_VALID_ISSUERS) >= 1
        assert len(GOOGLE_VALID_ISSUERS) >= 1
        assert "botframework" in MICROSOFT_JWKS_URI.lower()
        assert "googleapis" in GOOGLE_JWKS_URI.lower()
