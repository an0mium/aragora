"""
Tests for aragora.server.handlers.auth.signup_handlers - Self-service signup flow.

Tests cover:
- User registration (validation, conflicts, invite flow)
- Email verification (token validation, expiry, JWT generation)
- Resend verification (enumeration prevention)
- Organization setup (name, slug, plan validation)
- Team invitations (send, check, accept)
- Onboarding (completion, status tracking)

Security test categories:
- Token security: cryptographic randomness, single-use, expiry
- Input validation: email, password, injection prevention
- Enumeration prevention: consistent responses regardless of email existence
- Rate limiting: all endpoints should be rate limited
"""

from __future__ import annotations

import json
import secrets
import time
from typing import Any, Dict
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from aragora.server.handlers.auth.signup_handlers import (
    handle_signup,
    handle_verify_email,
    handle_resend_verification,
    handle_setup_organization,
    handle_invite,
    handle_check_invite,
    handle_accept_invite,
    handle_onboarding_complete,
    handle_onboarding_status,
    get_signup_handlers,
    _pending_signups,
    _pending_signups_lock,
    _pending_invites,
    _pending_invites_lock,
    _onboarding_status,
    _onboarding_lock,
    VERIFICATION_TTL,
    INVITE_TTL,
    EMAIL_REGEX,
    MIN_PASSWORD_LENGTH,
    _generate_verification_token,
    _hash_password,
    _validate_password,
    _cleanup_expired_tokens,
)
from aragora.server.handlers.utils.responses import HandlerResult


# ===========================================================================
# Helper Functions
# ===========================================================================


def parse_result(result: HandlerResult) -> tuple[int, Dict[str, Any]]:
    """Parse HandlerResult into (status_code, body_dict)."""
    body = json.loads(result.body.decode("utf-8"))
    return result.status_code, body


def get_data(result: HandlerResult) -> Dict[str, Any]:
    """Get the 'data' from a success response."""
    _, body = parse_result(result)
    return body.get("data", body)


def get_error(result: HandlerResult) -> str:
    """Get the error message from an error response."""
    _, body = parse_result(result)
    error = body.get("error", "")
    if isinstance(error, dict):
        return error.get("message", "")
    return error


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def clear_signup_state():
    """Clear all in-memory stores before and after each test."""
    with _pending_signups_lock:
        _pending_signups.clear()
    with _pending_invites_lock:
        _pending_invites.clear()
    with _onboarding_lock:
        _onboarding_status.clear()
    yield
    with _pending_signups_lock:
        _pending_signups.clear()
    with _pending_invites_lock:
        _pending_invites.clear()
    with _onboarding_lock:
        _onboarding_status.clear()


@pytest.fixture
def valid_signup_data() -> Dict[str, Any]:
    """Valid signup request data."""
    return {
        "email": "newuser@example.com",
        "password": "SecurePass123!",
        "name": "New User",
        "company_name": "Test Company",
    }


@pytest.fixture
def pending_signup() -> tuple[str, Dict[str, Any]]:
    """Create a pending signup for verification tests."""
    token = _generate_verification_token()
    signup_data = {
        "email": "pending@example.com",
        "password_hash": _hash_password("SecurePass123!"),
        "name": "Pending User",
        "company_name": "Test Company",
        "invite_token": None,
        "invite_data": None,
        "created_at": time.time(),
        "verified": False,
    }
    with _pending_signups_lock:
        _pending_signups[token] = signup_data
    return token, signup_data


@pytest.fixture
def expired_signup() -> tuple[str, Dict[str, Any]]:
    """Create an expired signup."""
    token = _generate_verification_token()
    signup_data = {
        "email": "expired@example.com",
        "password_hash": _hash_password("SecurePass123!"),
        "name": "Expired User",
        "company_name": "",
        "created_at": time.time() - VERIFICATION_TTL - 100,  # Expired
        "verified": False,
    }
    with _pending_signups_lock:
        _pending_signups[token] = signup_data
    return token, signup_data


@pytest.fixture
def valid_invite() -> tuple[str, Dict[str, Any]]:
    """Create a valid team invitation."""
    token = _generate_verification_token()
    invite_data = {
        "email": "invited@example.com",
        "organization_id": "org_abc123",
        "role": "member",
        "invited_by": "user_xyz",
        "created_at": time.time(),
    }
    with _pending_invites_lock:
        _pending_invites[token] = invite_data
    return token, invite_data


@pytest.fixture
def expired_invite() -> tuple[str, Dict[str, Any]]:
    """Create an expired invitation."""
    token = _generate_verification_token()
    invite_data = {
        "email": "expired_invite@example.com",
        "organization_id": "org_abc123",
        "role": "member",
        "invited_by": "user_xyz",
        "created_at": time.time() - INVITE_TTL - 100,  # Expired
    }
    with _pending_invites_lock:
        _pending_invites[token] = invite_data
    return token, invite_data


# ===========================================================================
# Test Helper Functions
# ===========================================================================


class TestHelperFunctions:
    """Tests for internal helper functions."""

    def test_generate_verification_token_uniqueness(self):
        """Verification tokens should be unique."""
        tokens = [_generate_verification_token() for _ in range(100)]
        assert len(set(tokens)) == 100, "Tokens should be unique"

    def test_generate_verification_token_length(self):
        """Verification tokens should be reasonably long for security."""
        token = _generate_verification_token()
        assert len(token) >= 32, "Token should be at least 32 characters"

    def test_hash_password_consistency(self):
        """Same password should produce same hash with same salt."""
        password = "TestPassword123"
        hash1 = _hash_password(password)
        hash2 = _hash_password(password)
        assert hash1 == hash2

    def test_hash_password_different_for_different_passwords(self):
        """Different passwords should produce different hashes."""
        hash1 = _hash_password("Password1")
        hash2 = _hash_password("Password2")
        assert hash1 != hash2

    def test_validate_password_too_short(self):
        """Password shorter than minimum should fail."""
        errors = _validate_password("Abc1")
        assert any("8 characters" in e for e in errors)

    def test_validate_password_no_lowercase(self):
        """Password without lowercase should fail."""
        errors = _validate_password("ABCDEFGH1")
        assert any("lowercase" in e for e in errors)

    def test_validate_password_no_uppercase(self):
        """Password without uppercase should fail."""
        errors = _validate_password("abcdefgh1")
        assert any("uppercase" in e for e in errors)

    def test_validate_password_no_number(self):
        """Password without number should fail."""
        errors = _validate_password("Abcdefghi")
        assert any("number" in e for e in errors)

    def test_validate_password_valid(self):
        """Valid password should return no errors."""
        errors = _validate_password("SecurePass1")
        assert errors == []

    def test_cleanup_expired_tokens(self, expired_signup, expired_invite):
        """Cleanup should remove expired tokens."""
        token1, _ = expired_signup
        token2, _ = expired_invite

        # Add a valid token
        valid_token = _generate_verification_token()
        with _pending_signups_lock:
            _pending_signups[valid_token] = {
                "email": "valid@example.com",
                "created_at": time.time(),
            }

        _cleanup_expired_tokens()

        with _pending_signups_lock:
            assert token1 not in _pending_signups
            assert valid_token in _pending_signups

        with _pending_invites_lock:
            assert token2 not in _pending_invites

    def test_email_regex_valid_emails(self):
        """Valid emails should match the regex."""
        valid_emails = [
            "test@example.com",
            "user.name@domain.org",
            "user+tag@company.co.uk",
            "a@b.io",
        ]
        for email in valid_emails:
            assert EMAIL_REGEX.match(email), f"{email} should be valid"

    def test_email_regex_invalid_emails(self):
        """Invalid emails should not match the regex."""
        invalid_emails = [
            "notanemail",
            "@nodomain.com",
            "no@tld",
            "spaces in@email.com",
            "",
        ]
        for email in invalid_emails:
            assert not EMAIL_REGEX.match(email), f"{email} should be invalid"


# ===========================================================================
# Test handle_signup
# ===========================================================================


class TestHandleSignup:
    """Tests for handle_signup endpoint."""

    @pytest.mark.asyncio
    async def test_signup_success(self, valid_signup_data):
        """Valid signup should return success with verification token."""
        result = await handle_signup(valid_signup_data)
        status, _ = parse_result(result)
        body = get_data(result)

        assert status == 200
        assert "verification_token" in body
        assert body["email"] == "newuser@example.com"
        assert "expires_in" in body
        assert body["expires_in"] == VERIFICATION_TTL

    @pytest.mark.asyncio
    async def test_signup_email_normalized(self, valid_signup_data):
        """Email should be lowercased and trimmed."""
        valid_signup_data["email"] = "  USER@EXAMPLE.COM  "
        result = await handle_signup(valid_signup_data)
        status, _ = parse_result(result)
        body = get_data(result)

        assert status == 200
        assert body["email"] == "user@example.com"

    @pytest.mark.asyncio
    async def test_signup_invalid_email_format(self, valid_signup_data):
        """Invalid email format should return 400."""
        valid_signup_data["email"] = "not-an-email"
        result = await handle_signup(valid_signup_data)
        status, _ = parse_result(result)
        error = get_error(result)

        assert status == 400
        assert "Invalid email" in error

    @pytest.mark.asyncio
    async def test_signup_empty_email(self, valid_signup_data):
        """Empty email should return 400."""
        valid_signup_data["email"] = ""
        result = await handle_signup(valid_signup_data)
        status, _ = parse_result(result)
        error = get_error(result)

        assert status == 400
        assert "Invalid email" in error

    @pytest.mark.asyncio
    async def test_signup_weak_password_too_short(self, valid_signup_data):
        """Password too short should return 400."""
        valid_signup_data["password"] = "Short1"
        result = await handle_signup(valid_signup_data)
        status, _ = parse_result(result)
        error = get_error(result)

        assert status == 400
        assert "Password" in error

    @pytest.mark.asyncio
    async def test_signup_weak_password_no_uppercase(self, valid_signup_data):
        """Password without uppercase should return 400."""
        valid_signup_data["password"] = "lowercase123"
        result = await handle_signup(valid_signup_data)
        status, _ = parse_result(result)

        assert status == 400

    @pytest.mark.asyncio
    async def test_signup_weak_password_no_lowercase(self, valid_signup_data):
        """Password without lowercase should return 400."""
        valid_signup_data["password"] = "UPPERCASE123"
        result = await handle_signup(valid_signup_data)
        status, _ = parse_result(result)

        assert status == 400

    @pytest.mark.asyncio
    async def test_signup_weak_password_no_number(self, valid_signup_data):
        """Password without number should return 400."""
        valid_signup_data["password"] = "NoNumbersHere"
        result = await handle_signup(valid_signup_data)
        status, _ = parse_result(result)

        assert status == 400

    @pytest.mark.asyncio
    async def test_signup_name_too_short(self, valid_signup_data):
        """Name too short should return 400."""
        valid_signup_data["name"] = "A"
        result = await handle_signup(valid_signup_data)
        status, _ = parse_result(result)
        error = get_error(result)

        assert status == 400
        assert "Name" in error

    @pytest.mark.asyncio
    async def test_signup_empty_name(self, valid_signup_data):
        """Empty name should return 400."""
        valid_signup_data["name"] = ""
        result = await handle_signup(valid_signup_data)
        status, _ = parse_result(result)

        assert status == 400

    @pytest.mark.asyncio
    async def test_signup_email_already_pending(self, valid_signup_data, pending_signup):
        """Email already pending verification should return 409."""
        valid_signup_data["email"] = "pending@example.com"
        result = await handle_signup(valid_signup_data)
        status, _ = parse_result(result)
        error = get_error(result)

        assert status == 409
        assert "pending" in error.lower()

    @pytest.mark.asyncio
    async def test_signup_with_valid_invite(self, valid_signup_data, valid_invite):
        """Signup with valid invite token should succeed."""
        token, invite_data = valid_invite
        valid_signup_data["email"] = invite_data["email"]
        valid_signup_data["invite_token"] = token

        result = await handle_signup(valid_signup_data)
        status, _ = parse_result(result)

        assert status == 200

    @pytest.mark.asyncio
    async def test_signup_with_invite_email_mismatch(self, valid_signup_data, valid_invite):
        """Signup with invite but different email should return 400."""
        token, _ = valid_invite
        valid_signup_data["invite_token"] = token
        # Email doesn't match invite

        result = await handle_signup(valid_signup_data)
        status, _ = parse_result(result)
        error = get_error(result)

        assert status == 400
        assert "match" in error.lower()

    @pytest.mark.asyncio
    async def test_signup_with_expired_invite(self, valid_signup_data, expired_invite):
        """Signup with expired invite should return 400."""
        token, invite_data = expired_invite
        valid_signup_data["email"] = invite_data["email"]
        valid_signup_data["invite_token"] = token

        result = await handle_signup(valid_signup_data)
        status, _ = parse_result(result)
        error = get_error(result)

        assert status == 400
        assert "expired" in error.lower()

    @pytest.mark.asyncio
    async def test_signup_stores_pending_record(self, valid_signup_data):
        """Signup should store pending signup record."""
        result = await handle_signup(valid_signup_data)
        body = get_data(result)
        token = body["verification_token"]

        with _pending_signups_lock:
            assert token in _pending_signups
            record = _pending_signups[token]
            assert record["email"] == "newuser@example.com"
            assert record["name"] == "New User"
            assert record["verified"] is False
            assert "password_hash" in record

    @pytest.mark.asyncio
    async def test_signup_password_not_stored_plaintext(self, valid_signup_data):
        """Password should not be stored in plaintext."""
        result = await handle_signup(valid_signup_data)
        body = get_data(result)
        token = body["verification_token"]

        with _pending_signups_lock:
            record = _pending_signups[token]
            assert "password" not in record
            assert record["password_hash"] != valid_signup_data["password"]

    @pytest.mark.asyncio
    async def test_signup_optional_company_name(self):
        """Signup without company name should succeed."""
        data = {
            "email": "nocompany@example.com",
            "password": "SecurePass123!",
            "name": "No Company",
        }
        result = await handle_signup(data)
        status, _ = parse_result(result)

        assert status == 200

    @pytest.mark.asyncio
    async def test_signup_name_trimmed(self, valid_signup_data):
        """Name should be trimmed of whitespace."""
        valid_signup_data["name"] = "  Spaced Name  "
        result = await handle_signup(valid_signup_data)
        body = get_data(result)
        token = body["verification_token"]

        with _pending_signups_lock:
            record = _pending_signups[token]
            assert record["name"] == "Spaced Name"


# ===========================================================================
# Test handle_verify_email
# ===========================================================================


class TestHandleVerifyEmail:
    """Tests for handle_verify_email endpoint."""

    @pytest.mark.asyncio
    async def test_verify_success(self, pending_signup):
        """Valid verification should return success with JWT."""
        token, _ = pending_signup

        with patch("aragora.billing.jwt_auth.create_access_token") as mock_jwt:
            mock_jwt.return_value = "jwt_token_123"
            result = await handle_verify_email({"token": token})

        status, _ = parse_result(result)
        body = get_data(result)
        assert status == 200
        assert "user_id" in body
        assert body["email"] == "pending@example.com"
        assert body["access_token"] == "jwt_token_123"
        assert body["token_type"] == "bearer"

    @pytest.mark.asyncio
    async def test_verify_removes_pending_signup(self, pending_signup):
        """Verification should remove pending signup."""
        token, _ = pending_signup

        with patch("aragora.billing.jwt_auth.create_access_token") as mock_jwt:
            mock_jwt.return_value = "jwt_token_123"
            await handle_verify_email({"token": token})

        with _pending_signups_lock:
            assert token not in _pending_signups

    @pytest.mark.asyncio
    async def test_verify_empty_token(self):
        """Empty token should return 400."""
        result = await handle_verify_email({"token": ""})
        status, _ = parse_result(result)
        error = get_error(result)

        assert status == 400
        assert "required" in error.lower()

    @pytest.mark.asyncio
    async def test_verify_missing_token(self):
        """Missing token should return 400."""
        result = await handle_verify_email({})
        status, _ = parse_result(result)

        assert status == 400

    @pytest.mark.asyncio
    async def test_verify_invalid_token(self):
        """Invalid token should return 400."""
        result = await handle_verify_email({"token": "invalid_token_123"})
        status, _ = parse_result(result)
        error = get_error(result)

        assert status == 400
        assert "invalid" in error.lower()

    @pytest.mark.asyncio
    async def test_verify_expired_token(self, expired_signup):
        """Expired token should return 400 and remove pending signup."""
        token, _ = expired_signup
        result = await handle_verify_email({"token": token})
        status, _ = parse_result(result)
        error = get_error(result)

        assert status == 400
        assert "expired" in error.lower()

        # Should remove expired record
        with _pending_signups_lock:
            assert token not in _pending_signups

    @pytest.mark.asyncio
    async def test_verify_token_single_use(self, pending_signup):
        """Token should only work once."""
        token, _ = pending_signup

        with patch("aragora.billing.jwt_auth.create_access_token") as mock_jwt:
            mock_jwt.return_value = "jwt_token_123"
            result1 = await handle_verify_email({"token": token})
            result2 = await handle_verify_email({"token": token})

        status1, _ = parse_result(result1)
        status2, _ = parse_result(result2)
        assert status1 == 200
        assert status2 == 400

    @pytest.mark.asyncio
    async def test_verify_needs_org_setup_without_invite(self, pending_signup):
        """User without invite needs org setup."""
        token, _ = pending_signup

        with patch("aragora.billing.jwt_auth.create_access_token") as mock_jwt:
            mock_jwt.return_value = "jwt_token_123"
            result = await handle_verify_email({"token": token})

        body = get_data(result)
        assert body["needs_org_setup"] is True
        assert body["organization_id"] is None

    @pytest.mark.asyncio
    async def test_verify_with_invite_no_org_setup(self, valid_invite):
        """User with invite doesn't need org setup."""
        invite_token, invite_data = valid_invite

        # Create pending signup with invite
        signup_token = _generate_verification_token()
        signup_data = {
            "email": invite_data["email"],
            "password_hash": _hash_password("SecurePass123!"),
            "name": "Invited User",
            "company_name": "",
            "invite_token": invite_token,
            "invite_data": invite_data,
            "created_at": time.time(),
            "verified": False,
        }
        with _pending_signups_lock:
            _pending_signups[signup_token] = signup_data

        with patch("aragora.billing.jwt_auth.create_access_token") as mock_jwt:
            mock_jwt.return_value = "jwt_token_123"
            result = await handle_verify_email({"token": signup_token})

        body = get_data(result)
        assert body["needs_org_setup"] is False
        assert body["organization_id"] == "org_abc123"

    @pytest.mark.asyncio
    async def test_verify_removes_used_invite(self, valid_invite):
        """Verification should remove the used invite."""
        invite_token, invite_data = valid_invite

        signup_token = _generate_verification_token()
        signup_data = {
            "email": invite_data["email"],
            "password_hash": _hash_password("SecurePass123!"),
            "name": "Invited User",
            "company_name": "",
            "invite_token": invite_token,
            "invite_data": invite_data,
            "created_at": time.time(),
            "verified": False,
        }
        with _pending_signups_lock:
            _pending_signups[signup_token] = signup_data

        with patch("aragora.billing.jwt_auth.create_access_token") as mock_jwt:
            mock_jwt.return_value = "jwt_token_123"
            await handle_verify_email({"token": signup_token})

        with _pending_invites_lock:
            assert invite_token not in _pending_invites

    @pytest.mark.asyncio
    async def test_verify_generates_user_id(self, pending_signup):
        """Verification should generate a user ID."""
        token, _ = pending_signup

        with patch("aragora.billing.jwt_auth.create_access_token") as mock_jwt:
            mock_jwt.return_value = "jwt_token_123"
            result = await handle_verify_email({"token": token})

        body = get_data(result)
        user_id = body["user_id"]
        assert user_id.startswith("user_")


# ===========================================================================
# Test handle_resend_verification
# ===========================================================================


class TestHandleResendVerification:
    """Tests for handle_resend_verification endpoint."""

    @pytest.mark.asyncio
    async def test_resend_with_pending_signup(self, pending_signup):
        """Resend for pending signup should succeed."""
        _, signup_data = pending_signup
        result = await handle_resend_verification({"email": signup_data["email"]})
        status, _ = parse_result(result)
        body = get_data(result)

        assert status == 200
        assert (
            "resent" in body.get("message", "").lower() or "sent" in body.get("message", "").lower()
        )

    @pytest.mark.asyncio
    async def test_resend_nonexistent_email(self):
        """Resend for nonexistent email should still return success (enumeration prevention)."""
        result = await handle_resend_verification({"email": "nonexistent@example.com"})
        status, _ = parse_result(result)

        assert status == 200
        # Should not reveal whether email exists

    @pytest.mark.asyncio
    async def test_resend_enumeration_prevention(self, pending_signup):
        """Response should be identical for existing and non-existing emails."""
        _, signup_data = pending_signup

        result_existing = await handle_resend_verification({"email": signup_data["email"]})
        result_nonexistent = await handle_resend_verification({"email": "nobody@example.com"})

        status1, _ = parse_result(result_existing)
        status2, _ = parse_result(result_nonexistent)
        # Both should return 200
        assert status1 == 200
        assert status2 == 200

    @pytest.mark.asyncio
    async def test_resend_empty_email(self):
        """Empty email should return 400."""
        result = await handle_resend_verification({"email": ""})
        status, _ = parse_result(result)

        assert status == 400

    @pytest.mark.asyncio
    async def test_resend_missing_email(self):
        """Missing email should return 400."""
        result = await handle_resend_verification({})
        status, _ = parse_result(result)

        assert status == 400

    @pytest.mark.asyncio
    async def test_resend_already_verified(self, pending_signup):
        """Resend for already verified email should return success (enumeration prevention)."""
        token, signup_data = pending_signup

        # Mark as verified
        with _pending_signups_lock:
            _pending_signups[token]["verified"] = True

        result = await handle_resend_verification({"email": signup_data["email"]})
        status, _ = parse_result(result)

        # Should still return success to prevent enumeration
        assert status == 200


# ===========================================================================
# Test handle_setup_organization
# ===========================================================================


class TestHandleSetupOrganization:
    """Tests for handle_setup_organization endpoint."""

    @pytest.mark.asyncio
    async def test_setup_org_success(self):
        """Valid org setup should return success."""
        result = await handle_setup_organization(
            {
                "name": "Test Organization",
                "slug": "test-org",
                "plan": "team",
                "billing_email": "billing@testorg.com",
            },
            user_id="user_123",
        )

        status, _ = parse_result(result)
        body = get_data(result)
        assert status == 200
        assert "organization" in body
        assert body["organization"]["name"] == "Test Organization"
        assert body["organization"]["slug"] == "test-org"
        assert body["organization"]["plan"] == "team"
        assert body["organization"]["owner_id"] == "user_123"

    @pytest.mark.asyncio
    async def test_setup_org_generates_id(self):
        """Org setup should generate organization ID."""
        result = await handle_setup_organization(
            {
                "name": "New Org",
            },
            user_id="user_123",
        )

        body = get_data(result)
        org_id = body["organization"]["id"]
        assert org_id.startswith("org_")

    @pytest.mark.asyncio
    async def test_setup_org_auto_generates_slug(self):
        """Org setup should auto-generate slug if not provided."""
        result = await handle_setup_organization(
            {
                "name": "My Amazing Company",
            },
            user_id="user_123",
        )

        body = get_data(result)
        slug = body["organization"]["slug"]
        assert slug == "my-amazing-company"

    @pytest.mark.asyncio
    async def test_setup_org_slug_special_chars(self):
        """Slug generation should handle special characters."""
        result = await handle_setup_organization(
            {
                "name": "Test & Company, Inc.",
            },
            user_id="user_123",
        )

        body = get_data(result)
        slug = body["organization"]["slug"]
        assert "&" not in slug
        assert "," not in slug
        assert "." not in slug

    @pytest.mark.asyncio
    async def test_setup_org_name_required(self):
        """Missing name should return 400."""
        result = await handle_setup_organization({}, user_id="user_123")
        status, _ = parse_result(result)

        assert status == 400

    @pytest.mark.asyncio
    async def test_setup_org_name_too_short(self):
        """Name too short should return 400."""
        result = await handle_setup_organization(
            {
                "name": "A",
            },
            user_id="user_123",
        )
        status, _ = parse_result(result)

        assert status == 400

    @pytest.mark.asyncio
    async def test_setup_org_invalid_slug_too_short(self):
        """Slug too short should return 400."""
        result = await handle_setup_organization(
            {
                "name": "Test Org",
                "slug": "ab",  # Too short
            },
            user_id="user_123",
        )
        status, _ = parse_result(result)
        error = get_error(result)

        assert status == 400
        assert "slug" in error.lower()

    @pytest.mark.asyncio
    async def test_setup_org_invalid_slug_special_chars(self):
        """Slug with invalid characters should return 400."""
        result = await handle_setup_organization(
            {
                "name": "Test Org",
                "slug": "test_org!",  # Invalid chars
            },
            user_id="user_123",
        )
        status, _ = parse_result(result)

        assert status == 400

    @pytest.mark.asyncio
    async def test_setup_org_default_plan_free(self):
        """Default plan should be free."""
        result = await handle_setup_organization(
            {
                "name": "Free Org",
            },
            user_id="user_123",
        )

        body = get_data(result)
        assert body["organization"]["plan"] == "free"

    @pytest.mark.asyncio
    async def test_setup_org_billing_email_optional(self):
        """Billing email should be optional."""
        result = await handle_setup_organization(
            {
                "name": "No Billing Email",
            },
            user_id="user_123",
        )
        status, _ = parse_result(result)
        body = get_data(result)

        assert status == 200
        assert body["organization"]["billing_email"] is None


# ===========================================================================
# Test handle_invite
# ===========================================================================


@patch("aragora.server.handlers.auth.signup_handlers._check_permission", return_value=None)
class TestHandleInvite:
    """Tests for handle_invite endpoint."""

    @pytest.mark.asyncio
    async def test_invite_success(self, mock_check):
        """Valid invite should return success."""
        result = await handle_invite(
            {
                "email": "newhire@example.com",
                "organization_id": "org_123",
                "role": "member",
            },
            user_id="admin_user",
        )

        status, _ = parse_result(result)
        body = get_data(result)
        assert status == 200
        assert "invite_token" in body
        assert body["email"] == "newhire@example.com"
        assert "invite_url" in body
        assert body["expires_in"] == INVITE_TTL

    @pytest.mark.asyncio
    async def test_invite_stores_record(self, mock_check):
        """Invite should store invitation record."""
        result = await handle_invite(
            {
                "email": "newhire@example.com",
                "organization_id": "org_123",
                "role": "admin",
            },
            user_id="admin_user",
        )

        body = get_data(result)
        token = body["invite_token"]
        with _pending_invites_lock:
            assert token in _pending_invites
            invite = _pending_invites[token]
            assert invite["email"] == "newhire@example.com"
            assert invite["organization_id"] == "org_123"
            assert invite["role"] == "admin"
            assert invite["invited_by"] == "admin_user"

    @pytest.mark.asyncio
    async def test_invite_invalid_email(self, mock_check):
        """Invalid email should return 400."""
        result = await handle_invite(
            {
                "email": "not-an-email",
                "organization_id": "org_123",
            },
            user_id="admin_user",
        )

        status, _ = parse_result(result)
        error = get_error(result)
        assert status == 400
        assert "email" in error.lower()

    @pytest.mark.asyncio
    async def test_invite_empty_email(self, mock_check):
        """Empty email should return 400."""
        result = await handle_invite(
            {
                "email": "",
                "organization_id": "org_123",
            },
            user_id="admin_user",
        )

        status, _ = parse_result(result)
        assert status == 400

    @pytest.mark.asyncio
    async def test_invite_missing_org_id(self, mock_check):
        """Missing organization ID should return 400."""
        result = await handle_invite(
            {
                "email": "user@example.com",
            },
            user_id="admin_user",
        )

        status, _ = parse_result(result)
        error = get_error(result)
        assert status == 400
        assert "organization" in error.lower()

    @pytest.mark.asyncio
    async def test_invite_default_role_member(self, mock_check):
        """Default role should be member."""
        result = await handle_invite(
            {
                "email": "user@example.com",
                "organization_id": "org_123",
            },
            user_id="admin_user",
        )

        body = get_data(result)
        token = body["invite_token"]
        with _pending_invites_lock:
            assert _pending_invites[token]["role"] == "member"

    @pytest.mark.asyncio
    async def test_invite_invalid_role(self, mock_check):
        """Invalid role should return 400."""
        result = await handle_invite(
            {
                "email": "user@example.com",
                "organization_id": "org_123",
                "role": "superuser",  # Invalid
            },
            user_id="admin_user",
        )

        status, _ = parse_result(result)
        error = get_error(result)
        assert status == 400
        assert "role" in error.lower()

    @pytest.mark.asyncio
    async def test_invite_valid_roles(self, mock_check):
        """All valid roles should be accepted."""
        for role in ["admin", "member", "viewer"]:
            result = await handle_invite(
                {
                    "email": f"{role}@example.com",
                    "organization_id": f"org_{role}",
                    "role": role,
                },
                user_id="admin_user",
            )

            status, _ = parse_result(result)
            assert status == 200, f"Role {role} should be valid"

    @pytest.mark.asyncio
    async def test_invite_duplicate_pending(self, mock_check, valid_invite):
        """Duplicate pending invite should return 409."""
        _, invite_data = valid_invite

        result = await handle_invite(
            {
                "email": invite_data["email"],
                "organization_id": invite_data["organization_id"],
            },
            user_id="admin_user",
        )

        status, _ = parse_result(result)
        error = get_error(result)
        assert status == 409
        assert "pending" in error.lower()

    @pytest.mark.asyncio
    async def test_invite_email_normalized(self, mock_check):
        """Email should be normalized."""
        result = await handle_invite(
            {
                "email": "  USER@EXAMPLE.COM  ",
                "organization_id": "org_123",
            },
            user_id="admin_user",
        )

        body = get_data(result)
        token = body["invite_token"]
        with _pending_invites_lock:
            assert _pending_invites[token]["email"] == "user@example.com"

    @pytest.mark.asyncio
    async def test_invite_same_email_different_org(self, valid_invite):
        """Same email for different org should succeed."""
        _, invite_data = valid_invite

        result = await handle_invite(
            {
                "email": invite_data["email"],
                "organization_id": "different_org_456",  # Different org
            },
            user_id="admin_user",
        )

        status, _ = parse_result(result)
        assert status == 200


# ===========================================================================
# Test handle_check_invite
# ===========================================================================


class TestHandleCheckInvite:
    """Tests for handle_check_invite endpoint."""

    @pytest.mark.asyncio
    async def test_check_invite_valid(self, valid_invite):
        """Valid invite should return details."""
        token, invite_data = valid_invite
        result = await handle_check_invite({"token": token})

        status, _ = parse_result(result)
        body = get_data(result)
        assert status == 200
        assert body["valid"] is True
        assert body["email"] == invite_data["email"]
        assert body["organization_id"] == invite_data["organization_id"]
        assert body["role"] == invite_data["role"]
        assert "expires_at" in body

    @pytest.mark.asyncio
    async def test_check_invite_missing_token(self):
        """Missing token should return 400."""
        result = await handle_check_invite({})
        status, _ = parse_result(result)

        assert status == 400

    @pytest.mark.asyncio
    async def test_check_invite_empty_token(self):
        """Empty token should return 400."""
        result = await handle_check_invite({"token": ""})
        status, _ = parse_result(result)

        assert status == 400

    @pytest.mark.asyncio
    async def test_check_invite_invalid_token(self):
        """Invalid token should return 404."""
        result = await handle_check_invite({"token": "invalid_token_xyz"})
        status, _ = parse_result(result)

        assert status == 404

    @pytest.mark.asyncio
    async def test_check_invite_expired(self, expired_invite):
        """Expired invite should return 400."""
        token, _ = expired_invite
        result = await handle_check_invite({"token": token})

        status, _ = parse_result(result)
        error = get_error(result)
        assert status == 400
        assert "expired" in error.lower()

    @pytest.mark.asyncio
    async def test_check_invite_does_not_consume_token(self, valid_invite):
        """Checking should not consume the invite."""
        token, _ = valid_invite

        result1 = await handle_check_invite({"token": token})
        result2 = await handle_check_invite({"token": token})

        status1, _ = parse_result(result1)
        status2, _ = parse_result(result2)
        assert status1 == 200
        assert status2 == 200


# ===========================================================================
# Test handle_accept_invite
# ===========================================================================


class TestHandleAcceptInvite:
    """Tests for handle_accept_invite endpoint."""

    @pytest.mark.asyncio
    async def test_accept_invite_success(self, valid_invite):
        """Valid accept should return success."""
        token, invite_data = valid_invite
        result = await handle_accept_invite({"token": token}, user_id="existing_user")

        status, _ = parse_result(result)
        body = get_data(result)
        assert status == 200
        assert body["organization_id"] == invite_data["organization_id"]
        assert body["role"] == invite_data["role"]

    @pytest.mark.asyncio
    async def test_accept_invite_removes_token(self, valid_invite):
        """Accepting should remove the invite."""
        token, _ = valid_invite
        await handle_accept_invite({"token": token}, user_id="existing_user")

        with _pending_invites_lock:
            assert token not in _pending_invites

    @pytest.mark.asyncio
    async def test_accept_invite_missing_token(self):
        """Missing token should return 400."""
        result = await handle_accept_invite({}, user_id="existing_user")
        status, _ = parse_result(result)

        assert status == 400

    @pytest.mark.asyncio
    async def test_accept_invite_empty_token(self):
        """Empty token should return 400."""
        result = await handle_accept_invite({"token": ""}, user_id="existing_user")
        status, _ = parse_result(result)

        assert status == 400

    @pytest.mark.asyncio
    async def test_accept_invite_invalid_token(self):
        """Invalid token should return 404."""
        result = await handle_accept_invite({"token": "invalid"}, user_id="existing_user")
        status, _ = parse_result(result)

        assert status == 404

    @pytest.mark.asyncio
    async def test_accept_invite_expired(self, expired_invite):
        """Expired invite should return 400 and be removed."""
        token, _ = expired_invite
        result = await handle_accept_invite({"token": token}, user_id="existing_user")

        status, _ = parse_result(result)
        error = get_error(result)
        assert status == 400
        assert "expired" in error.lower()

        with _pending_invites_lock:
            assert token not in _pending_invites

    @pytest.mark.asyncio
    async def test_accept_invite_single_use(self, valid_invite):
        """Invite should only be usable once."""
        token, _ = valid_invite

        result1 = await handle_accept_invite({"token": token}, user_id="user1")
        result2 = await handle_accept_invite({"token": token}, user_id="user2")

        status1, _ = parse_result(result1)
        status2, _ = parse_result(result2)
        assert status1 == 200
        assert status2 == 404


# ===========================================================================
# Test handle_onboarding_complete
# ===========================================================================


class TestHandleOnboardingComplete:
    """Tests for handle_onboarding_complete endpoint."""

    @pytest.mark.asyncio
    async def test_complete_success(self):
        """Complete onboarding should return success."""
        result = await handle_onboarding_complete(
            {"first_debate_id": "debate_123", "template_used": "quick_start"},
            user_id="user_123",
            organization_id="org_123",
        )

        status_code, _ = parse_result(result)
        body = get_data(result)
        assert status_code == 200
        assert body["completed"] is True
        assert body["organization_id"] == "org_123"
        assert "completed_at" in body

    @pytest.mark.asyncio
    async def test_complete_stores_status(self):
        """Completion should store onboarding status."""
        await handle_onboarding_complete(
            {"first_debate_id": "debate_123"},
            user_id="user_123",
            organization_id="org_123",
        )

        with _onboarding_lock:
            assert "org_123" in _onboarding_status
            status = _onboarding_status["org_123"]
            assert status["completed"] is True
            assert status["completed_by"] == "user_123"
            assert status["first_debate_id"] == "debate_123"

    @pytest.mark.asyncio
    async def test_complete_optional_fields(self):
        """Optional fields should be handled."""
        result = await handle_onboarding_complete(
            {},
            user_id="user_123",
            organization_id="org_123",
        )

        status_code, _ = parse_result(result)
        assert status_code == 200

    @pytest.mark.asyncio
    async def test_complete_overwrites_previous(self):
        """Completing again should overwrite."""
        await handle_onboarding_complete(
            {"first_debate_id": "debate_1"},
            user_id="user_123",
            organization_id="org_123",
        )
        await handle_onboarding_complete(
            {"first_debate_id": "debate_2"},
            user_id="user_456",
            organization_id="org_123",
        )

        with _onboarding_lock:
            status = _onboarding_status["org_123"]
            assert status["first_debate_id"] == "debate_2"
            assert status["completed_by"] == "user_456"


# ===========================================================================
# Test handle_onboarding_status
# ===========================================================================


class TestHandleOnboardingStatus:
    """Tests for handle_onboarding_status endpoint."""

    @pytest.mark.asyncio
    async def test_status_not_started(self):
        """Status for new org should show not completed."""
        result = await handle_onboarding_status(organization_id="new_org")

        status_code, _ = parse_result(result)
        body = get_data(result)
        assert status_code == 200
        assert body["completed"] is False
        assert body["organization_id"] == "new_org"
        assert "steps" in body
        assert body["steps"]["signup"] is True
        assert body["steps"]["organization_created"] is True
        assert body["steps"]["first_debate"] is False

    @pytest.mark.asyncio
    async def test_status_completed(self):
        """Status for completed org should show details."""
        # Complete onboarding first
        await handle_onboarding_complete(
            {"first_debate_id": "debate_123", "template_used": "quick_start"},
            user_id="user_123",
            organization_id="org_123",
        )

        result = await handle_onboarding_status(organization_id="org_123")

        status_code, _ = parse_result(result)
        body = get_data(result)
        assert status_code == 200
        assert body["completed"] is True
        assert body["first_debate_id"] == "debate_123"
        assert body["template_used"] == "quick_start"
        assert body["steps"]["first_debate"] is True

    @pytest.mark.asyncio
    async def test_status_includes_all_steps(self):
        """Status should include all onboarding steps."""
        result = await handle_onboarding_status(organization_id="org_123")

        body = get_data(result)
        expected_steps = ["signup", "organization_created", "first_debate", "first_receipt"]
        for step in expected_steps:
            assert step in body["steps"], f"Missing step: {step}"


# ===========================================================================
# Test get_signup_handlers
# ===========================================================================


class TestGetSignupHandlers:
    """Tests for handler registration."""

    def test_all_handlers_registered(self):
        """All handlers should be registered."""
        handlers = get_signup_handlers()

        expected = [
            "signup",
            "verify_email",
            "resend_verification",
            "setup_organization",
            "invite",
            "check_invite",
            "accept_invite",
            "onboarding_complete",
            "onboarding_status",
        ]

        for name in expected:
            assert name in handlers, f"Handler {name} should be registered"

    def test_handlers_are_callable(self):
        """All handlers should be callable."""
        handlers = get_signup_handlers()

        for name, handler in handlers.items():
            assert callable(handler), f"Handler {name} should be callable"
