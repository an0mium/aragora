"""
Tests for JWT Authentication module.

Tests cover:
- TokenBlacklist class (singleton, revocation, cleanup)
- JWT generation (access and refresh tokens)
- JWT decoding and validation
- Secret validation
- JWTPayload dataclass
- Security: algorithm validation, expiration
"""

from __future__ import annotations

import pytest
import time
import threading
import os
from unittest.mock import patch, MagicMock

from aragora.billing.jwt_auth import (
    TokenBlacklist,
    get_token_blacklist,
    JWTPayload,
    create_access_token,
    create_refresh_token,
    decode_jwt,
    validate_access_token,
    _validate_secret_strength,
    _base64url_encode,
    _base64url_decode,
    MIN_SECRET_LENGTH,
    ALLOWED_ALGORITHMS,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def reset_blacklist():
    """Reset blacklist before each test."""
    blacklist = get_token_blacklist()
    blacklist.clear()
    yield
    blacklist.clear()


@pytest.fixture
def test_secret():
    """Provide a test secret for JWT operations."""
    return "test-secret-key-that-is-long-enough-for-tests-12345"


@pytest.fixture
def mock_env_secret(test_secret):
    """Mock environment to use test secret."""
    with patch.dict(os.environ, {
        "ARAGORA_JWT_SECRET": test_secret,
        "ARAGORA_ENVIRONMENT": "development"
    }):
        yield test_secret


# ============================================================================
# TokenBlacklist Tests
# ============================================================================

class TestTokenBlacklist:
    """Tests for TokenBlacklist class."""

    def test_singleton_pattern(self):
        """Test TokenBlacklist is a singleton."""
        blacklist1 = TokenBlacklist()
        blacklist2 = TokenBlacklist()
        assert blacklist1 is blacklist2

    def test_get_token_blacklist_returns_instance(self):
        """Test get_token_blacklist returns blacklist instance."""
        blacklist = get_token_blacklist()
        assert isinstance(blacklist, TokenBlacklist)

    def test_revoke_adds_token(self):
        """Test revoke() adds token JTI to internal blacklist."""
        blacklist = get_token_blacklist()
        future_expiry = time.time() + 3600

        blacklist.revoke("token-jti-123", future_expiry)

        # Check internal storage directly since is_revoked hashes its input
        assert "token-jti-123" in blacklist._blacklist

    def test_is_revoked_returns_true_for_revoked(self, mock_env_secret):
        """Test is_revoked returns True for revoked tokens."""
        blacklist = get_token_blacklist()

        # Create a real token and revoke it using revoke_token
        token = create_access_token(user_id="test-user", email="test@test.com")
        result = blacklist.revoke_token(token)

        assert result is True
        assert blacklist.is_revoked(token) is True

    def test_is_revoked_returns_false_for_valid(self):
        """Test is_revoked returns False for non-revoked tokens."""
        blacklist = get_token_blacklist()

        assert blacklist.is_revoked("valid-token") is False

    def test_cleanup_expired_removes_old_tokens(self):
        """Test cleanup_expired removes expired tokens."""
        blacklist = get_token_blacklist()
        # Add token that already expired
        past_expiry = time.time() - 100
        blacklist.revoke("expired-token", past_expiry)

        count = blacklist.cleanup_expired()

        assert count == 1
        assert blacklist.is_revoked("expired-token") is False

    def test_cleanup_keeps_unexpired_tokens(self):
        """Test cleanup keeps tokens that haven't expired."""
        blacklist = get_token_blacklist()
        future_expiry = time.time() + 3600
        blacklist.revoke("future-token-jti", future_expiry)

        blacklist.cleanup_expired()

        # Check internal storage directly since is_revoked hashes its input
        assert "future-token-jti" in blacklist._blacklist

    def test_size_returns_count(self):
        """Test size() returns number of revoked tokens."""
        blacklist = get_token_blacklist()
        future_expiry = time.time() + 3600

        blacklist.revoke("token-1", future_expiry)
        blacklist.revoke("token-2", future_expiry)

        assert blacklist.size() == 2

    def test_clear_removes_all(self):
        """Test clear() removes all tokens."""
        blacklist = get_token_blacklist()
        future_expiry = time.time() + 3600
        blacklist.revoke("token-1", future_expiry)
        blacklist.revoke("token-2", future_expiry)

        blacklist.clear()

        assert blacklist.size() == 0

    def test_thread_safety_concurrent_revoke(self):
        """Test blacklist is thread-safe for concurrent revocation."""
        blacklist = get_token_blacklist()
        future_expiry = time.time() + 3600
        errors = []

        def revoke_token(token_id):
            try:
                blacklist.revoke(f"token-{token_id}", future_expiry)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=revoke_token, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert blacklist.size() == 10


# ============================================================================
# JWTPayload Tests
# ============================================================================

class TestJWTPayload:
    """Tests for JWTPayload dataclass."""

    def test_create_payload(self):
        """Test creating a JWT payload."""
        now = int(time.time())
        payload = JWTPayload(
            sub="user-123",
            email="test@example.com",
            org_id="org-456",
            role="admin",
            iat=now,
            exp=now + 3600,
            type="access"
        )

        assert payload.sub == "user-123"
        assert payload.email == "test@example.com"
        assert payload.role == "admin"

    def test_to_dict(self):
        """Test payload to_dict method."""
        now = int(time.time())
        payload = JWTPayload(
            sub="user-123",
            email="test@example.com",
            org_id="org-456",
            role="member",
            iat=now,
            exp=now + 3600,
        )

        data = payload.to_dict()

        assert data["sub"] == "user-123"
        assert data["email"] == "test@example.com"
        assert data["org_id"] == "org-456"
        assert data["type"] == "access"

    def test_from_dict(self):
        """Test payload from_dict class method."""
        data = {
            "sub": "user-789",
            "email": "user@test.com",
            "org_id": None,
            "role": "admin",
            "iat": 1000,
            "exp": 5000,
            "type": "refresh"
        }

        payload = JWTPayload.from_dict(data)

        assert payload.sub == "user-789"
        assert payload.email == "user@test.com"
        assert payload.type == "refresh"

    def test_from_dict_with_defaults(self):
        """Test from_dict uses defaults for missing fields."""
        data = {"sub": "user-123"}

        payload = JWTPayload.from_dict(data)

        assert payload.email == ""
        assert payload.role == "member"
        assert payload.type == "access"

    def test_is_expired_false_for_future(self):
        """Test is_expired returns False for future expiry."""
        now = int(time.time())
        payload = JWTPayload(
            sub="user", email="", org_id=None, role="",
            iat=now, exp=now + 3600
        )

        assert payload.is_expired is False

    def test_is_expired_true_for_past(self):
        """Test is_expired returns True for past expiry."""
        now = int(time.time())
        payload = JWTPayload(
            sub="user", email="", org_id=None, role="",
            iat=now - 7200, exp=now - 3600
        )

        assert payload.is_expired is True

    def test_user_id_alias(self):
        """Test user_id is alias for sub."""
        payload = JWTPayload(
            sub="user-abc", email="", org_id=None, role="",
            iat=0, exp=0
        )

        assert payload.user_id == "user-abc"
        assert payload.user_id == payload.sub


# ============================================================================
# Secret Validation Tests
# ============================================================================

class TestSecretValidation:
    """Tests for secret validation functions."""

    def test_validate_secret_strength_accepts_long(self):
        """Test validation accepts secrets >= MIN_SECRET_LENGTH."""
        long_secret = "a" * MIN_SECRET_LENGTH
        assert _validate_secret_strength(long_secret) is True

    def test_validate_secret_strength_rejects_short(self):
        """Test validation rejects secrets < MIN_SECRET_LENGTH."""
        short_secret = "a" * (MIN_SECRET_LENGTH - 1)
        assert _validate_secret_strength(short_secret) is False

    def test_validate_secret_strength_exact_length(self):
        """Test validation accepts exact minimum length."""
        exact_secret = "x" * MIN_SECRET_LENGTH
        assert _validate_secret_strength(exact_secret) is True


# ============================================================================
# Base64 URL Encoding Tests
# ============================================================================

class TestBase64UrlEncoding:
    """Tests for base64 URL encoding/decoding."""

    def test_encode_decode_roundtrip(self):
        """Test encoding then decoding returns original."""
        original = b"Hello, World! This is a test."
        encoded = _base64url_encode(original)
        decoded = _base64url_decode(encoded)
        assert decoded == original

    def test_encode_no_padding(self):
        """Test encoding removes padding characters."""
        data = b"test"
        encoded = _base64url_encode(data)
        assert "=" not in encoded

    def test_decode_handles_missing_padding(self):
        """Test decoding handles missing padding."""
        # "test" encodes to "dGVzdA==" in standard base64
        # URL-safe without padding is "dGVzdA"
        decoded = _base64url_decode("dGVzdA")
        assert decoded == b"test"


# ============================================================================
# JWT Generation Tests
# ============================================================================

class TestJWTGeneration:
    """Tests for JWT token generation."""

    def test_create_access_token_format(self, mock_env_secret):
        """Test access token has correct JWT format (3 parts)."""
        token = create_access_token(
            user_id="user-123",
            email="test@example.com"
        )

        parts = token.split(".")
        assert len(parts) == 3

    def test_create_access_token_decodable(self, mock_env_secret):
        """Test created access token can be decoded."""
        token = create_access_token(
            user_id="user-123",
            email="test@example.com",
            role="admin"
        )

        payload = decode_jwt(token)

        assert payload is not None
        assert payload.sub == "user-123"
        assert payload.email == "test@example.com"
        assert payload.role == "admin"
        assert payload.type == "access"

    def test_create_access_token_with_org(self, mock_env_secret):
        """Test access token includes org_id."""
        token = create_access_token(
            user_id="user-123",
            email="test@example.com",
            org_id="org-456"
        )

        payload = decode_jwt(token)

        assert payload.org_id == "org-456"

    def test_create_access_token_default_expiry(self, mock_env_secret):
        """Test access token has default expiry."""
        now = int(time.time())
        token = create_access_token(
            user_id="user-123",
            email="test@example.com"
        )

        payload = decode_jwt(token)

        # Default is 24 hours
        assert payload.exp > now
        assert payload.exp <= now + (24 * 3600) + 10  # Allow some slack

    def test_create_refresh_token_format(self, mock_env_secret):
        """Test refresh token has correct format."""
        token = create_refresh_token(user_id="user-123")

        parts = token.split(".")
        assert len(parts) == 3

    def test_create_refresh_token_type(self, mock_env_secret):
        """Test refresh token has type='refresh'."""
        token = create_refresh_token(user_id="user-123")

        payload = decode_jwt(token)

        assert payload is not None
        assert payload.type == "refresh"

    def test_create_refresh_token_longer_expiry(self, mock_env_secret):
        """Test refresh token has longer expiry than access token."""
        now = int(time.time())
        access_token = create_access_token(user_id="user-123", email="test@example.com")
        refresh_token = create_refresh_token(user_id="user-123")

        access_payload = decode_jwt(access_token)
        refresh_payload = decode_jwt(refresh_token)

        assert refresh_payload.exp > access_payload.exp


# ============================================================================
# JWT Decoding Tests
# ============================================================================

class TestJWTDecoding:
    """Tests for JWT token decoding."""

    def test_decode_valid_token(self, mock_env_secret):
        """Test decoding a valid token."""
        token = create_access_token(
            user_id="user-123",
            email="test@example.com"
        )

        payload = decode_jwt(token)

        assert payload is not None
        assert payload.sub == "user-123"

    def test_decode_expired_token_returns_none(self, mock_env_secret):
        """Test decoding expired token returns None."""
        # Create token with past expiry
        with patch("aragora.billing.auth.tokens.time") as mock_time:
            # First call is for token creation, make it in the past
            mock_time.time.return_value = 1000

            token = create_access_token(
                user_id="user-123",
                email="test@example.com",
                expiry_hours=1
            )

        # Decode at current time (token is expired)
        payload = decode_jwt(token)

        assert payload is None

    def test_decode_invalid_format_returns_none(self):
        """Test decoding invalid format returns None."""
        invalid_tokens = [
            "not-a-jwt",
            "only.two.parts.invalid",
            "",
            "a.b",
            "a.b.c.d",
        ]

        for invalid in invalid_tokens:
            assert decode_jwt(invalid) is None

    def test_decode_invalid_signature_returns_none(self, mock_env_secret):
        """Test decoding token with wrong signature returns None."""
        token = create_access_token(
            user_id="user-123",
            email="test@example.com"
        )

        # Tamper with signature
        parts = token.split(".")
        parts[2] = "invalid_signature_here"
        tampered = ".".join(parts)

        payload = decode_jwt(tampered)

        assert payload is None

    def test_decode_rejects_none_algorithm(self, mock_env_secret):
        """Test decoding rejects 'none' algorithm attack."""
        # Create a token with 'none' algorithm (attack vector)
        import json
        header = {"alg": "none", "typ": "JWT"}
        payload = {"sub": "attacker", "email": "evil@hack.com", "exp": int(time.time()) + 3600}

        header_b64 = _base64url_encode(json.dumps(header).encode())
        payload_b64 = _base64url_encode(json.dumps(payload).encode())

        # 'none' algorithm means no signature
        malicious_token = f"{header_b64}.{payload_b64}."

        result = decode_jwt(malicious_token)

        assert result is None

    def test_decode_only_allows_hs256(self, mock_env_secret):
        """Test only HS256 algorithm is allowed."""
        assert "HS256" in ALLOWED_ALGORITHMS
        assert len(ALLOWED_ALGORITHMS) == 1


# ============================================================================
# Token Validation Tests
# ============================================================================

class TestTokenValidation:
    """Tests for token validation."""

    def test_validate_access_token_success(self, mock_env_secret):
        """Test validating a valid access token."""
        token = create_access_token(
            user_id="user-123",
            email="test@example.com"
        )

        payload = validate_access_token(token, use_persistent_blacklist=False)

        assert payload is not None
        assert payload.sub == "user-123"

    def test_validate_access_token_wrong_type(self, mock_env_secret):
        """Test validating refresh token as access token fails."""
        refresh_token = create_refresh_token(user_id="user-123")

        payload = validate_access_token(refresh_token, use_persistent_blacklist=False)

        # Should fail because type is 'refresh', not 'access'
        assert payload is None


# ============================================================================
# Integration Tests
# ============================================================================

class TestJWTIntegration:
    """Integration tests for JWT authentication."""

    def test_full_token_lifecycle(self, mock_env_secret):
        """Test complete token lifecycle: create, decode, validate."""
        # Create token
        token = create_access_token(
            user_id="integration-user",
            email="integration@test.com",
            org_id="org-123",
            role="admin"
        )

        # Decode and verify claims
        payload = decode_jwt(token)
        assert payload is not None
        assert payload.sub == "integration-user"
        assert payload.org_id == "org-123"
        assert payload.role == "admin"
        assert payload.type == "access"
        assert not payload.is_expired

        # Validate as access token
        validated = validate_access_token(token, use_persistent_blacklist=False)
        assert validated is not None

    def test_blacklist_integration(self, mock_env_secret):
        """Test token revocation via blacklist."""
        blacklist = get_token_blacklist()

        # Create token
        token = create_access_token(
            user_id="revoke-user",
            email="revoke@test.com"
        )

        # Verify token works before revocation
        payload = decode_jwt(token)
        assert payload is not None
        assert blacklist.is_revoked(token) is False

        # Revoke the token using revoke_token (high-level API)
        result = blacklist.revoke_token(token)
        assert result is True

        # Token should now be in blacklist
        assert blacklist.is_revoked(token) is True

    def test_token_roundtrip(self, mock_env_secret):
        """Test token creation and full validation roundtrip."""
        user_id = "roundtrip-user"
        email = "roundtrip@test.com"

        # Create
        access_token = create_access_token(user_id=user_id, email=email)
        refresh_token = create_refresh_token(user_id=user_id)

        # Verify access token
        access_payload = decode_jwt(access_token)
        assert access_payload.sub == user_id
        assert access_payload.email == email
        assert access_payload.type == "access"

        # Verify refresh token
        refresh_payload = decode_jwt(refresh_token)
        assert refresh_payload.sub == user_id
        assert refresh_payload.type == "refresh"
        assert refresh_payload.exp > access_payload.exp


# ============================================================================
# Token Version Tests (for logout-all functionality)
# ============================================================================

class TestTokenVersion:
    """Tests for token version (logout-all) functionality."""

    def test_token_includes_version(self, mock_env_secret):
        """Test token includes tv (token version) claim."""
        token = create_access_token(
            user_id="user-123",
            email="test@example.com",
            token_version=5,
        )
        payload = decode_jwt(token)
        assert payload is not None
        assert payload.tv == 5

    def test_token_default_version_is_1(self, mock_env_secret):
        """Test token defaults to version 1."""
        token = create_access_token(
            user_id="user-123",
            email="test@example.com",
        )
        payload = decode_jwt(token)
        assert payload is not None
        assert payload.tv == 1

    def test_refresh_token_includes_version(self, mock_env_secret):
        """Test refresh token includes tv claim."""
        token = create_refresh_token(user_id="user-123", token_version=3)
        payload = decode_jwt(token)
        assert payload is not None
        assert payload.tv == 3

    def test_payload_to_dict_includes_tv(self):
        """Test payload to_dict includes tv field."""
        now = int(time.time())
        payload = JWTPayload(
            sub="user-123",
            email="test@example.com",
            org_id=None,
            role="member",
            iat=now,
            exp=now + 3600,
            tv=7,
        )
        data = payload.to_dict()
        assert data["tv"] == 7

    def test_payload_from_dict_reads_tv(self):
        """Test payload from_dict reads tv field."""
        data = {
            "sub": "user-123",
            "email": "test@example.com",
            "org_id": None,
            "role": "member",
            "iat": 1000,
            "exp": 5000,
            "tv": 10,
        }
        payload = JWTPayload.from_dict(data)
        assert payload.tv == 10

    def test_payload_from_dict_defaults_tv_to_1(self):
        """Test payload from_dict defaults tv to 1."""
        data = {"sub": "user-123"}
        payload = JWTPayload.from_dict(data)
        assert payload.tv == 1

    def test_validate_access_token_rejects_old_version(self, mock_env_secret):
        """Test validate_access_token rejects token with old version."""
        # Create token with version 1
        token = create_access_token(
            user_id="user-123",
            email="test@example.com",
            token_version=1,
        )

        # Create mock user store where user has version 2
        mock_user = MagicMock()
        mock_user.token_version = 2
        mock_store = MagicMock()
        mock_store.get_user_by_id.return_value = mock_user

        # Token should be rejected
        payload = validate_access_token(token, user_store=mock_store)
        assert payload is None
        mock_store.get_user_by_id.assert_called_once_with("user-123")

    def test_validate_access_token_accepts_current_version(self, mock_env_secret):
        """Test validate_access_token accepts token with current version."""
        # Create token with version 2
        token = create_access_token(
            user_id="user-123",
            email="test@example.com",
            token_version=2,
        )

        # Create mock user store where user has version 2
        mock_user = MagicMock()
        mock_user.token_version = 2
        mock_store = MagicMock()
        mock_store.get_user_by_id.return_value = mock_user

        # Token should be accepted
        payload = validate_access_token(token, user_store=mock_store)
        assert payload is not None
        assert payload.tv == 2

    def test_validate_access_token_accepts_newer_version(self, mock_env_secret):
        """Test validate_access_token accepts token with newer version (edge case)."""
        # Token with version 3
        token = create_access_token(
            user_id="user-123",
            email="test@example.com",
            token_version=3,
        )

        # User has version 2 (less than token)
        mock_user = MagicMock()
        mock_user.token_version = 2
        mock_store = MagicMock()
        mock_store.get_user_by_id.return_value = mock_user

        # Token should be accepted (version 3 >= 2)
        payload = validate_access_token(token, user_store=mock_store)
        assert payload is not None

    def test_validate_access_token_works_without_user_store(self, mock_env_secret):
        """Test validate_access_token works when user_store is not provided."""
        token = create_access_token(
            user_id="user-123",
            email="test@example.com",
            token_version=1,
        )

        # Should validate successfully without version check
        payload = validate_access_token(token, user_store=None)
        assert payload is not None

    def test_validate_access_token_handles_store_error(self, mock_env_secret):
        """Test validate_access_token handles user store errors gracefully."""
        token = create_access_token(
            user_id="user-123",
            email="test@example.com",
            token_version=1,
        )

        # Create mock store that raises an error
        mock_store = MagicMock()
        mock_store.get_user_by_id.side_effect = Exception("Database error")

        # Should still validate (don't block on store errors)
        payload = validate_access_token(token, user_store=mock_store)
        assert payload is not None

    def test_validate_access_token_handles_user_not_found(self, mock_env_secret):
        """Test validate_access_token handles non-existent user."""
        token = create_access_token(
            user_id="nonexistent-user",
            email="test@example.com",
            token_version=1,
        )

        # User not found
        mock_store = MagicMock()
        mock_store.get_user_by_id.return_value = None

        # Should still validate (user might be deleted but token valid)
        payload = validate_access_token(token, user_store=mock_store)
        assert payload is not None
