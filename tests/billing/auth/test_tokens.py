"""
Comprehensive tests for JWT token creation, encoding, and validation.

Tests cover:
- Base64 URL-safe encoding/decoding
- JWTPayload dataclass
- Token encoding and decoding
- Security: algorithm confusion attacks, tampering, expiration
- Access token creation with expiry bounds
- Refresh token creation with expiry bounds
- Token validation with blacklist checking
- Token version validation (logout-all support)
- Secret rotation support
- MFA pending tokens
- TokenPair creation
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from aragora.billing.auth.tokens import (
    JWTPayload,
    TokenPair,
    _base64url_decode,
    _base64url_encode,
    create_access_token,
    create_mfa_pending_token,
    create_refresh_token,
    create_token_pair,
    decode_jwt,
    validate_access_token,
    validate_mfa_pending_token,
    validate_refresh_token,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# Base64 URL Encoding/Decoding Tests
# =============================================================================


class TestBase64UrlEncoding:
    """Test base64 URL-safe encoding utilities."""

    def test_encode_simple_data(self):
        """Test encoding simple data."""
        data = b"hello world"
        encoded = _base64url_encode(data)
        assert isinstance(encoded, str)
        # Should not contain padding
        assert "=" not in encoded
        # Should use URL-safe characters
        assert "+" not in encoded
        assert "/" not in encoded

    def test_decode_simple_data(self):
        """Test decoding simple data."""
        data = b"hello world"
        encoded = _base64url_encode(data)
        decoded = _base64url_decode(encoded)
        assert decoded == data

    def test_roundtrip_various_lengths(self):
        """Test roundtrip with various lengths requiring padding."""
        for length in [1, 2, 3, 4, 5, 10, 100, 255]:
            data = bytes(range(length % 256)) * (length // 256 + 1)
            data = data[:length]
            encoded = _base64url_encode(data)
            decoded = _base64url_decode(encoded)
            assert decoded == data, f"Failed for length {length}"

    def test_encode_json_payload(self):
        """Test encoding JSON payload (typical JWT use)."""
        payload = {"sub": "user123", "exp": 1234567890}
        data = json.dumps(payload).encode("utf-8")
        encoded = _base64url_encode(data)
        decoded = _base64url_decode(encoded)
        assert json.loads(decoded.decode("utf-8")) == payload

    def test_decode_handles_padding(self):
        """Test decoder handles inputs with and without padding."""
        data = b"test"
        # Standard base64 (with padding)
        standard = base64.urlsafe_b64encode(data).decode("utf-8")
        # Without padding
        no_padding = standard.rstrip("=")
        # Both should decode correctly
        assert _base64url_decode(standard) == data
        assert _base64url_decode(no_padding) == data


# =============================================================================
# JWTPayload Dataclass Tests
# =============================================================================


class TestJWTPayload:
    """Test JWTPayload dataclass."""

    def test_create_payload(self):
        """Test creating a payload."""
        now = int(time.time())
        payload = JWTPayload(
            sub="user123",
            email="test@example.com",
            org_id="org456",
            role="admin",
            iat=now,
            exp=now + 3600,
            type="access",
            tv=1,
        )
        assert payload.sub == "user123"
        assert payload.email == "test@example.com"
        assert payload.org_id == "org456"
        assert payload.role == "admin"
        assert payload.type == "access"
        assert payload.tv == 1

    def test_to_dict(self):
        """Test converting payload to dictionary."""
        now = int(time.time())
        payload = JWTPayload(
            sub="user123",
            email="test@example.com",
            org_id="org456",
            role="admin",
            iat=now,
            exp=now + 3600,
        )
        d = payload.to_dict()
        assert d["sub"] == "user123"
        assert d["email"] == "test@example.com"
        assert d["org_id"] == "org456"
        assert d["role"] == "admin"
        assert d["iat"] == now
        assert d["exp"] == now + 3600
        assert d["type"] == "access"
        assert d["tv"] == 1

    def test_from_dict(self):
        """Test creating payload from dictionary."""
        data = {
            "sub": "user123",
            "email": "test@example.com",
            "org_id": "org456",
            "role": "admin",
            "iat": 1234567890,
            "exp": 1234571490,
            "type": "refresh",
            "tv": 2,
        }
        payload = JWTPayload.from_dict(data)
        assert payload.sub == "user123"
        assert payload.email == "test@example.com"
        assert payload.org_id == "org456"
        assert payload.role == "admin"
        assert payload.type == "refresh"
        assert payload.tv == 2

    def test_from_dict_with_defaults(self):
        """Test from_dict uses defaults for missing fields."""
        data = {"iat": 1234567890, "exp": 1234571490}
        payload = JWTPayload.from_dict(data)
        assert payload.sub == ""
        assert payload.email == ""
        assert payload.org_id is None
        assert payload.role == "member"
        assert payload.type == "access"
        assert payload.tv == 1

    def test_is_expired_false(self):
        """Test is_expired returns False for future expiration."""
        now = int(time.time())
        payload = JWTPayload(
            sub="user123",
            email="test@example.com",
            org_id=None,
            role="member",
            iat=now,
            exp=now + 3600,  # 1 hour in future
        )
        assert payload.is_expired is False

    def test_is_expired_true(self):
        """Test is_expired returns True for past expiration."""
        now = int(time.time())
        payload = JWTPayload(
            sub="user123",
            email="test@example.com",
            org_id=None,
            role="member",
            iat=now - 7200,
            exp=now - 3600,  # 1 hour in past
        )
        assert payload.is_expired is True

    def test_user_id_alias(self):
        """Test user_id is alias for sub."""
        payload = JWTPayload(
            sub="user123",
            email="test@example.com",
            org_id=None,
            role="member",
            iat=0,
            exp=0,
        )
        assert payload.user_id == "user123"
        assert payload.user_id == payload.sub


# =============================================================================
# Token Encoding/Decoding Tests
# =============================================================================


class TestTokenEncoding:
    """Test JWT encoding and decoding."""

    def test_create_and_decode_access_token(self):
        """Test creating and decoding an access token."""
        token = create_access_token(
            user_id="user123",
            email="test@example.com",
            org_id="org456",
            role="admin",
        )
        # Should have 3 parts
        assert token.count(".") == 2
        # Should be decodable
        payload = decode_jwt(token)
        assert payload is not None
        assert payload.sub == "user123"
        assert payload.email == "test@example.com"
        assert payload.org_id == "org456"
        assert payload.role == "admin"
        assert payload.type == "access"

    def test_create_and_decode_refresh_token(self):
        """Test creating and decoding a refresh token."""
        token = create_refresh_token(user_id="user123")
        payload = decode_jwt(token)
        assert payload is not None
        assert payload.sub == "user123"
        assert payload.type == "refresh"
        # Refresh tokens have minimal claims
        assert payload.email == ""
        assert payload.org_id is None

    def test_token_format(self):
        """Test token has correct JWT format."""
        token = create_access_token(
            user_id="user123",
            email="test@example.com",
        )
        parts = token.split(".")
        assert len(parts) == 3

        # Decode header
        header = json.loads(_base64url_decode(parts[0]).decode("utf-8"))
        assert header["alg"] == "HS256"
        assert header["typ"] == "JWT"

        # Decode payload
        payload = json.loads(_base64url_decode(parts[1]).decode("utf-8"))
        assert payload["sub"] == "user123"
        assert payload["email"] == "test@example.com"


# =============================================================================
# Security Tests
# =============================================================================


class TestSecurityValidation:
    """Test security validation in token decoding."""

    def test_reject_none_algorithm_attack(self):
        """Test rejection of 'none' algorithm attack."""
        # Craft a token with alg=none
        header = {"alg": "none", "typ": "JWT"}
        payload = {
            "sub": "attacker",
            "email": "attacker@evil.com",
            "org_id": None,
            "role": "admin",
            "iat": int(time.time()),
            "exp": int(time.time()) + 3600,
            "type": "access",
            "tv": 1,
        }
        header_b64 = _base64url_encode(json.dumps(header).encode("utf-8"))
        payload_b64 = _base64url_encode(json.dumps(payload).encode("utf-8"))
        # Token with empty signature
        malicious_token = f"{header_b64}.{payload_b64}."

        result = decode_jwt(malicious_token)
        assert result is None, "Should reject none algorithm"

    def test_reject_algorithm_confusion(self):
        """Test rejection of algorithm confusion (e.g., RS256 when expecting HS256)."""
        header = {"alg": "RS256", "typ": "JWT"}
        payload = {
            "sub": "attacker",
            "email": "attacker@evil.com",
            "org_id": None,
            "role": "admin",
            "iat": int(time.time()),
            "exp": int(time.time()) + 3600,
            "type": "access",
            "tv": 1,
        }
        header_b64 = _base64url_encode(json.dumps(header).encode("utf-8"))
        payload_b64 = _base64url_encode(json.dumps(payload).encode("utf-8"))
        fake_sig = _base64url_encode(b"fake_signature")
        malicious_token = f"{header_b64}.{payload_b64}.{fake_sig}"

        result = decode_jwt(malicious_token)
        assert result is None, "Should reject RS256 algorithm"

    def test_reject_tampered_payload(self):
        """Test rejection of token with tampered payload."""
        # Create a valid token
        token = create_access_token(
            user_id="user123",
            email="original@example.com",
            role="member",
        )
        parts = token.split(".")

        # Tamper with payload to change role
        payload = json.loads(_base64url_decode(parts[1]).decode("utf-8"))
        payload["role"] = "admin"  # Elevate privileges
        tampered_payload = _base64url_encode(json.dumps(payload).encode("utf-8"))

        # Reconstruct token with original signature
        tampered_token = f"{parts[0]}.{tampered_payload}.{parts[2]}"

        result = decode_jwt(tampered_token)
        assert result is None, "Should reject tampered payload"

    def test_reject_tampered_header(self):
        """Test rejection of token with tampered header."""
        token = create_access_token(user_id="user123", email="test@example.com")
        parts = token.split(".")

        # Tamper with header
        header = json.loads(_base64url_decode(parts[0]).decode("utf-8"))
        header["extra"] = "field"
        tampered_header = _base64url_encode(json.dumps(header).encode("utf-8"))

        tampered_token = f"{tampered_header}.{parts[1]}.{parts[2]}"

        result = decode_jwt(tampered_token)
        assert result is None, "Should reject tampered header"

    def test_reject_invalid_token_format(self):
        """Test rejection of invalid token format."""
        assert decode_jwt("") is None
        assert decode_jwt("invalid") is None
        assert decode_jwt("only.two") is None
        assert decode_jwt("too.many.parts.here") is None
        assert decode_jwt("...") is None

    def test_reject_invalid_base64(self):
        """Test rejection of invalid base64 encoding."""
        # Invalid base64 in header
        assert decode_jwt("!!!.valid.valid") is None
        # Invalid base64 in payload (with valid header)
        token = create_access_token(user_id="test", email="test@example.com")
        parts = token.split(".")
        assert decode_jwt(f"{parts[0]}.!!!.{parts[2]}") is None

    def test_reject_expired_token(self):
        """Test rejection of expired token."""
        # Create a token that's already expired by manipulating time
        with patch("aragora.billing.auth.tokens.time.time") as mock_time:
            # Create token at time T
            mock_time.return_value = 1000000
            token = create_access_token(
                user_id="user123",
                email="test@example.com",
                expiry_hours=1,
            )

        # Try to decode at time T + 2 hours (after expiry)
        with patch("aragora.billing.auth.tokens.time.time") as mock_time:
            mock_time.return_value = 1000000 + 7200  # 2 hours later
            result = decode_jwt(token)
            assert result is None, "Should reject expired token"

    def test_accept_valid_token(self):
        """Test acceptance of properly signed valid token."""
        token = create_access_token(
            user_id="user123",
            email="test@example.com",
            org_id="org456",
            role="admin",
        )
        result = decode_jwt(token)
        assert result is not None
        assert result.sub == "user123"
        assert result.role == "admin"


# =============================================================================
# Access Token Tests
# =============================================================================


class TestAccessTokenCreation:
    """Test access token creation."""

    def test_default_expiry(self):
        """Test default expiry is applied."""
        token = create_access_token(user_id="user123", email="test@example.com")
        payload = decode_jwt(token)
        assert payload is not None
        # Default should be from config (24 hours typically)
        now = int(time.time())
        # Expiry should be in the future
        assert payload.exp > now

    def test_custom_expiry(self):
        """Test custom expiry hours."""
        token = create_access_token(
            user_id="user123",
            email="test@example.com",
            expiry_hours=12,
        )
        payload = decode_jwt(token)
        assert payload is not None
        now = int(time.time())
        # Should expire in roughly 12 hours
        expected_exp = now + (12 * 3600)
        assert abs(payload.exp - expected_exp) < 5  # Within 5 seconds

    def test_expiry_capped_at_max(self):
        """Test expiry is capped at maximum allowed."""
        # Request 1000 hours (way over max of 168)
        token = create_access_token(
            user_id="user123",
            email="test@example.com",
            expiry_hours=1000,
        )
        payload = decode_jwt(token)
        assert payload is not None
        now = int(time.time())
        # Should be capped at 168 hours (7 days)
        max_exp = now + (168 * 3600)
        assert payload.exp <= max_exp + 5  # Within 5 seconds

    def test_expiry_minimum_enforced(self):
        """Test minimum expiry of 1 hour is enforced."""
        token = create_access_token(
            user_id="user123",
            email="test@example.com",
            expiry_hours=0,  # Too low
        )
        payload = decode_jwt(token)
        assert payload is not None
        now = int(time.time())
        # Should be at least 1 hour
        min_exp = now + 3600
        assert payload.exp >= min_exp - 5  # Within 5 seconds

    def test_token_version(self):
        """Test token version is included."""
        token = create_access_token(
            user_id="user123",
            email="test@example.com",
            token_version=5,
        )
        payload = decode_jwt(token)
        assert payload is not None
        assert payload.tv == 5

    def test_org_id_optional(self):
        """Test org_id can be None."""
        token = create_access_token(
            user_id="user123",
            email="test@example.com",
            org_id=None,
        )
        payload = decode_jwt(token)
        assert payload is not None
        assert payload.org_id is None


# =============================================================================
# Refresh Token Tests
# =============================================================================


class TestRefreshTokenCreation:
    """Test refresh token creation."""

    def test_default_expiry(self):
        """Test default refresh token expiry."""
        token = create_refresh_token(user_id="user123")
        payload = decode_jwt(token)
        assert payload is not None
        assert payload.type == "refresh"
        # Should have longer expiry than access tokens
        now = int(time.time())
        assert payload.exp > now + (7 * 86400)  # At least 7 days

    def test_custom_expiry_days(self):
        """Test custom expiry in days."""
        token = create_refresh_token(user_id="user123", expiry_days=7)
        payload = decode_jwt(token)
        assert payload is not None
        now = int(time.time())
        expected_exp = now + (7 * 86400)
        assert abs(payload.exp - expected_exp) < 5

    def test_expiry_capped_at_max(self):
        """Test refresh token expiry capped at 90 days."""
        token = create_refresh_token(user_id="user123", expiry_days=365)
        payload = decode_jwt(token)
        assert payload is not None
        now = int(time.time())
        max_exp = now + (90 * 86400)
        assert payload.exp <= max_exp + 5

    def test_expiry_minimum_enforced(self):
        """Test minimum expiry of 1 day is enforced."""
        token = create_refresh_token(user_id="user123", expiry_days=0)
        payload = decode_jwt(token)
        assert payload is not None
        now = int(time.time())
        min_exp = now + 86400
        assert payload.exp >= min_exp - 5

    def test_refresh_token_minimal_claims(self):
        """Test refresh tokens have minimal claims."""
        token = create_refresh_token(user_id="user123")
        payload = decode_jwt(token)
        assert payload is not None
        assert payload.sub == "user123"
        assert payload.email == ""
        assert payload.org_id is None
        assert payload.role == ""

    def test_token_version(self):
        """Test token version in refresh token."""
        token = create_refresh_token(user_id="user123", token_version=3)
        payload = decode_jwt(token)
        assert payload is not None
        assert payload.tv == 3


# =============================================================================
# Token Validation Tests
# =============================================================================


class TestAccessTokenValidation:
    """Test access token validation."""

    def test_validate_valid_access_token(self):
        """Test validation of valid access token."""
        token = create_access_token(user_id="user123", email="test@example.com")
        payload = validate_access_token(token, use_persistent_blacklist=False)
        assert payload is not None
        assert payload.sub == "user123"
        assert payload.type == "access"

    def test_reject_refresh_token_as_access(self):
        """Test refresh token rejected when validating as access."""
        token = create_refresh_token(user_id="user123")
        payload = validate_access_token(token, use_persistent_blacklist=False)
        assert payload is None

    def test_reject_revoked_token(self):
        """Test revoked token is rejected."""
        from aragora.billing.auth.blacklist import get_token_blacklist

        token = create_access_token(user_id="user123", email="test@example.com")

        # Revoke the token
        blacklist = get_token_blacklist()
        blacklist.revoke_token(token)

        payload = validate_access_token(token, use_persistent_blacklist=False)
        assert payload is None

        # Clean up
        blacklist.clear()

    def test_token_version_validation(self):
        """Test token version validation with user store."""
        # Create token with version 1
        token = create_access_token(
            user_id="user123",
            email="test@example.com",
            token_version=1,
        )

        # Mock user store with higher version (user logged out everywhere)
        mock_store = MagicMock()
        mock_user = MagicMock()
        mock_user.token_version = 2  # User's current version is higher
        mock_store.get_user_by_id.return_value = mock_user

        payload = validate_access_token(
            token,
            use_persistent_blacklist=False,
            user_store=mock_store,
        )
        assert payload is None, "Should reject token with old version"

    def test_token_version_valid(self):
        """Test valid token version passes validation."""
        token = create_access_token(
            user_id="user123",
            email="test@example.com",
            token_version=2,
        )

        mock_store = MagicMock()
        mock_user = MagicMock()
        mock_user.token_version = 2  # Same version
        mock_store.get_user_by_id.return_value = mock_user

        payload = validate_access_token(
            token,
            use_persistent_blacklist=False,
            user_store=mock_store,
        )
        assert payload is not None


class TestRefreshTokenValidation:
    """Test refresh token validation."""

    def test_validate_valid_refresh_token(self):
        """Test validation of valid refresh token."""
        token = create_refresh_token(user_id="user123")
        payload = validate_refresh_token(token, use_persistent_blacklist=False)
        assert payload is not None
        assert payload.sub == "user123"
        assert payload.type == "refresh"

    def test_reject_access_token_as_refresh(self):
        """Test access token rejected when validating as refresh."""
        token = create_access_token(user_id="user123", email="test@example.com")
        payload = validate_refresh_token(token, use_persistent_blacklist=False)
        assert payload is None

    def test_reject_revoked_refresh_token(self):
        """Test revoked refresh token is rejected."""
        from aragora.billing.auth.blacklist import get_token_blacklist

        token = create_refresh_token(user_id="user123")

        blacklist = get_token_blacklist()
        blacklist.revoke_token(token)

        payload = validate_refresh_token(token, use_persistent_blacklist=False)
        assert payload is None

        blacklist.clear()

    def test_token_version_validation(self):
        """Test refresh token version validation."""
        token = create_refresh_token(user_id="user123", token_version=1)

        mock_store = MagicMock()
        mock_user = MagicMock()
        mock_user.token_version = 2
        mock_store.get_user_by_id.return_value = mock_user

        payload = validate_refresh_token(
            token,
            use_persistent_blacklist=False,
            user_store=mock_store,
        )
        assert payload is None


# =============================================================================
# Secret Rotation Tests
# =============================================================================


class TestSecretRotation:
    """Test JWT secret rotation support."""

    def test_decode_with_previous_secret(self):
        """Test tokens signed with previous secret are still valid during rotation."""
        from aragora.billing.auth import config

        # Save original secrets
        original_secret = config.JWT_SECRET

        # Create token with "old" secret
        old_secret = "old_secret_that_is_long_enough_for_tests"
        config.JWT_SECRET = old_secret

        token = create_access_token(user_id="user123", email="test@example.com")

        # Now rotate to new secret, keep old as previous
        new_secret = "new_secret_that_is_long_enough_for_tests"
        config.JWT_SECRET = new_secret
        config.JWT_SECRET_PREVIOUS = old_secret
        # Set rotation timestamp to now (within grace period)
        config.JWT_SECRET_ROTATED_AT = str(int(time.time()))

        # Token should still be valid (using previous secret)
        payload = decode_jwt(token)
        assert payload is not None
        assert payload.sub == "user123"

        # Restore original
        config.JWT_SECRET = original_secret
        config.JWT_SECRET_PREVIOUS = ""
        config.JWT_SECRET_ROTATED_AT = ""


# =============================================================================
# MFA Pending Token Tests
# =============================================================================


class TestMFAPendingTokens:
    """Test MFA pending token creation and validation."""

    def test_create_mfa_pending_token(self):
        """Test creating MFA pending token."""
        token = create_mfa_pending_token(user_id="user123", email="test@example.com")
        payload = decode_jwt(token)
        assert payload is not None
        assert payload.sub == "user123"
        assert payload.email == "test@example.com"
        assert payload.type == "mfa_pending"

    def test_mfa_pending_token_short_expiry(self):
        """Test MFA pending token has short expiry (5 minutes)."""
        token = create_mfa_pending_token(user_id="user123", email="test@example.com")
        payload = decode_jwt(token)
        assert payload is not None
        now = int(time.time())
        # Should expire in ~5 minutes
        expected_exp = now + (5 * 60)
        assert abs(payload.exp - expected_exp) < 5

    def test_validate_mfa_pending_token(self):
        """Test validating MFA pending token."""
        token = create_mfa_pending_token(user_id="user123", email="test@example.com")
        payload = validate_mfa_pending_token(token)
        assert payload is not None
        assert payload.type == "mfa_pending"

    def test_reject_access_token_as_mfa_pending(self):
        """Test access token rejected as MFA pending."""
        token = create_access_token(user_id="user123", email="test@example.com")
        payload = validate_mfa_pending_token(token)
        assert payload is None

    def test_reject_refresh_token_as_mfa_pending(self):
        """Test refresh token rejected as MFA pending."""
        token = create_refresh_token(user_id="user123")
        payload = validate_mfa_pending_token(token)
        assert payload is None

    def test_reject_used_mfa_pending_token(self):
        """Test MFA pending token can only be used once (replay protection)."""
        from aragora.billing.auth.blacklist import get_token_blacklist

        token = create_mfa_pending_token(user_id="user123", email="test@example.com")

        # First validation should succeed
        payload = validate_mfa_pending_token(token)
        assert payload is not None

        # Revoke after use (simulating real flow)
        blacklist = get_token_blacklist()
        blacklist.revoke_token(token)

        # Second validation should fail
        payload = validate_mfa_pending_token(token)
        assert payload is None

        blacklist.clear()


# =============================================================================
# TokenPair Tests
# =============================================================================


class TestTokenPair:
    """Test TokenPair creation."""

    def test_create_token_pair(self):
        """Test creating access/refresh token pair."""
        pair = create_token_pair(
            user_id="user123",
            email="test@example.com",
            org_id="org456",
            role="admin",
        )
        assert isinstance(pair, TokenPair)
        assert pair.access_token is not None
        assert pair.refresh_token is not None
        assert pair.token_type == "Bearer"
        assert pair.expires_in > 0

    def test_token_pair_to_dict(self):
        """Test TokenPair to_dict for API response."""
        pair = create_token_pair(
            user_id="user123",
            email="test@example.com",
        )
        d = pair.to_dict()
        assert "access_token" in d
        assert "refresh_token" in d
        assert d["token_type"] == "Bearer"
        assert "expires_in" in d

    def test_token_pair_tokens_valid(self):
        """Test both tokens in pair are valid."""
        pair = create_token_pair(
            user_id="user123",
            email="test@example.com",
            org_id="org456",
            role="admin",
            token_version=2,
        )

        access_payload = validate_access_token(
            pair.access_token, use_persistent_blacklist=False
        )
        refresh_payload = validate_refresh_token(
            pair.refresh_token, use_persistent_blacklist=False
        )

        assert access_payload is not None
        assert access_payload.sub == "user123"
        assert access_payload.type == "access"
        assert access_payload.tv == 2

        assert refresh_payload is not None
        assert refresh_payload.sub == "user123"
        assert refresh_payload.type == "refresh"
        assert refresh_payload.tv == 2


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_user_id(self):
        """Test handling of empty user ID."""
        token = create_access_token(user_id="", email="test@example.com")
        payload = decode_jwt(token)
        assert payload is not None
        assert payload.sub == ""

    def test_unicode_in_claims(self):
        """Test Unicode characters in claims."""
        token = create_access_token(
            user_id="user_日本語",
            email="test@例え.jp",
        )
        payload = decode_jwt(token)
        assert payload is not None
        assert payload.sub == "user_日本語"
        assert payload.email == "test@例え.jp"

    def test_special_characters_in_claims(self):
        """Test special characters in claims."""
        token = create_access_token(
            user_id="user+special=chars&more",
            email="test+alias@example.com",
        )
        payload = decode_jwt(token)
        assert payload is not None
        assert payload.sub == "user+special=chars&more"

    def test_malformed_json_in_payload(self):
        """Test handling of malformed JSON in payload."""
        from aragora.billing.auth.config import get_secret

        header = {"alg": "HS256", "typ": "JWT"}
        header_b64 = _base64url_encode(json.dumps(header).encode("utf-8"))
        # Invalid JSON in payload
        payload_b64 = _base64url_encode(b"not valid json {{{")

        # Sign it properly
        message = f"{header_b64}.{payload_b64}"
        signature = hmac.new(
            get_secret(),
            message.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        sig_b64 = _base64url_encode(signature)

        token = f"{header_b64}.{payload_b64}.{sig_b64}"
        result = decode_jwt(token)
        assert result is None
