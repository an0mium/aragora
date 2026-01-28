"""
Tests for aragora.server.handlers.auth.signup_handlers - Self-service signup endpoints.

Tests cover:
- User registration (email/password validation, duplicate handling)
- Email verification (token validation, expiry)
- Resend verification
- Organization setup (name/slug validation)
- Team invitations (invite, check, accept)
- Onboarding completion tracking
"""

from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import patch

import pytest


# ===========================================================================
# Response Helpers
# ===========================================================================


def get_response_data(body: dict) -> dict:
    """Extract data from response body, handling wrapped format.

    The API uses format: {"success": true, "data": {...}}
    This helper extracts the data portion consistently.
    """
    if "data" in body:
        return body["data"]
    return body


def get_response_error(body: dict) -> str:
    """Extract error message from response body."""
    return body.get("error", "")


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def clear_signup_state():
    """Clear all pending signups and invites before each test."""
    from aragora.server.handlers.auth import signup_handlers

    signup_handlers._pending_signups.clear()
    signup_handlers._pending_invites.clear()
    signup_handlers._onboarding_status.clear()
    yield
    signup_handlers._pending_signups.clear()
    signup_handlers._pending_invites.clear()
    signup_handlers._onboarding_status.clear()


@pytest.fixture
def valid_signup_data():
    """Valid user signup data."""
    return {
        "email": "newuser@example.com",
        "password": "SecureP@ss123",
        "name": "Test User",
        "company_name": "Test Company",
    }


@pytest.fixture
def weak_password_data():
    """Signup data with weak password."""
    return {
        "email": "newuser@example.com",
        "password": "weak",  # Too short, no uppercase, no number
        "name": "Test User",
    }


# ===========================================================================
# Test User Registration
# ===========================================================================


class TestSignup:
    """Tests for handle_signup - user registration."""

    async def test_signup_success(self, valid_signup_data):
        """Test successful user signup."""
        from aragora.server.handlers.auth.signup_handlers import handle_signup

        result = await handle_signup(valid_signup_data)

        assert result.status_code == 200
        body = json.loads(result.body)
        data = get_response_data(body)
        assert "verification_token" in data
        assert data["email"] == "newuser@example.com"
        assert data["expires_in"] > 0

    async def test_signup_invalid_email(self, valid_signup_data):
        """Test signup fails with invalid email."""
        from aragora.server.handlers.auth.signup_handlers import handle_signup

        valid_signup_data["email"] = "not-an-email"
        result = await handle_signup(valid_signup_data)

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "email" in get_response_error(body).lower()

    async def test_signup_empty_email(self, valid_signup_data):
        """Test signup fails with empty email."""
        from aragora.server.handlers.auth.signup_handlers import handle_signup

        valid_signup_data["email"] = ""
        result = await handle_signup(valid_signup_data)

        assert result.status_code == 400

    async def test_signup_weak_password(self, weak_password_data):
        """Test signup fails with weak password."""
        from aragora.server.handlers.auth.signup_handlers import handle_signup

        result = await handle_signup(weak_password_data)

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "password" in get_response_error(body).lower()

    async def test_signup_missing_name(self, valid_signup_data):
        """Test signup fails without name."""
        from aragora.server.handlers.auth.signup_handlers import handle_signup

        valid_signup_data["name"] = ""
        result = await handle_signup(valid_signup_data)

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "name" in get_response_error(body).lower()

    async def test_signup_short_name(self, valid_signup_data):
        """Test signup fails with very short name."""
        from aragora.server.handlers.auth.signup_handlers import handle_signup

        valid_signup_data["name"] = "X"
        result = await handle_signup(valid_signup_data)

        assert result.status_code == 400

    async def test_signup_duplicate_pending(self, valid_signup_data):
        """Test signup fails if email already pending verification."""
        from aragora.server.handlers.auth.signup_handlers import handle_signup

        # First signup
        await handle_signup(valid_signup_data)

        # Second signup with same email
        result = await handle_signup(valid_signup_data)

        assert result.status_code == 409
        body = json.loads(result.body)
        assert "pending" in get_response_error(body).lower()

    async def test_signup_normalizes_email(self, valid_signup_data):
        """Test signup normalizes email to lowercase."""
        from aragora.server.handlers.auth.signup_handlers import handle_signup

        valid_signup_data["email"] = "NewUser@Example.COM"
        result = await handle_signup(valid_signup_data)

        body = json.loads(result.body)
        data = get_response_data(body)
        assert data["email"] == "newuser@example.com"

    async def test_signup_with_invite_token(self, valid_signup_data):
        """Test signup with valid invitation token."""
        from aragora.server.handlers.auth.signup_handlers import (
            _pending_invites,
            handle_signup,
        )

        # Create invite
        invite_token = "test_invite_token_123"
        _pending_invites[invite_token] = {
            "email": "newuser@example.com",
            "organization_id": "org_123",
            "role": "member",
            "created_at": time.time(),
        }

        valid_signup_data["invite_token"] = invite_token
        result = await handle_signup(valid_signup_data)

        assert result.status_code == 200

    async def test_signup_with_wrong_invite_email(self, valid_signup_data):
        """Test signup fails when invite email doesn't match."""
        from aragora.server.handlers.auth.signup_handlers import (
            _pending_invites,
            handle_signup,
        )

        # Create invite for different email
        invite_token = "test_invite_token_123"
        _pending_invites[invite_token] = {
            "email": "different@example.com",
            "organization_id": "org_123",
            "role": "member",
            "created_at": time.time(),
        }

        valid_signup_data["invite_token"] = invite_token
        result = await handle_signup(valid_signup_data)

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "match" in get_response_error(body).lower()


# ===========================================================================
# Test Email Verification
# ===========================================================================


class TestVerifyEmail:
    """Tests for handle_verify_email - email verification."""

    async def test_verify_email_success(self, valid_signup_data):
        """Test successful email verification."""
        from aragora.server.handlers.auth.signup_handlers import (
            handle_signup,
            handle_verify_email,
        )

        # First signup
        signup_result = await handle_signup(valid_signup_data)
        signup_body = json.loads(signup_result.body)
        token = get_response_data(signup_body)["verification_token"]

        # Mock JWT creation
        with patch("aragora.billing.jwt_auth.create_access_token", return_value="jwt_token_123"):
            result = await handle_verify_email({"token": token})

        assert result.status_code == 200
        body = json.loads(result.body)
        data = get_response_data(body)
        assert "access_token" in data
        assert data["email"] == "newuser@example.com"
        assert "user_id" in data

    async def test_verify_email_invalid_token(self):
        """Test verification fails with invalid token."""
        from aragora.server.handlers.auth.signup_handlers import handle_verify_email

        result = await handle_verify_email({"token": "invalid_token"})

        assert result.status_code == 400
        body = json.loads(result.body)
        error = get_response_error(body)
        assert "invalid" in error.lower() or "expired" in error.lower()

    async def test_verify_email_missing_token(self):
        """Test verification fails without token."""
        from aragora.server.handlers.auth.signup_handlers import handle_verify_email

        result = await handle_verify_email({})

        assert result.status_code == 400

    async def test_verify_email_expired_token(self, valid_signup_data):
        """Test verification fails with expired token."""
        from aragora.server.handlers.auth.signup_handlers import (
            VERIFICATION_TTL,
            _pending_signups,
            handle_signup,
            handle_verify_email,
        )

        # Signup
        signup_result = await handle_signup(valid_signup_data)
        signup_body = json.loads(signup_result.body)
        token = get_response_data(signup_body)["verification_token"]

        # Make token expired
        _pending_signups[token]["created_at"] = time.time() - VERIFICATION_TTL - 100

        result = await handle_verify_email({"token": token})

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "expired" in get_response_error(body).lower()

    async def test_verify_email_removes_token(self, valid_signup_data):
        """Test that verification consumes the token."""
        from aragora.server.handlers.auth.signup_handlers import (
            _pending_signups,
            handle_signup,
            handle_verify_email,
        )

        signup_result = await handle_signup(valid_signup_data)
        signup_body = json.loads(signup_result.body)
        token = get_response_data(signup_body)["verification_token"]

        with patch("aragora.billing.jwt_auth.create_access_token", return_value="jwt"):
            await handle_verify_email({"token": token})

        assert token not in _pending_signups


# ===========================================================================
# Test Resend Verification
# ===========================================================================


class TestResendVerification:
    """Tests for handle_resend_verification."""

    async def test_resend_verification_success(self, valid_signup_data):
        """Test successful resend verification."""
        from aragora.server.handlers.auth.signup_handlers import (
            handle_resend_verification,
            handle_signup,
        )

        # First signup
        await handle_signup(valid_signup_data)

        # Resend
        result = await handle_resend_verification({"email": "newuser@example.com"})

        assert result.status_code == 200

    async def test_resend_verification_unknown_email(self):
        """Test resend with unknown email doesn't reveal existence."""
        from aragora.server.handlers.auth.signup_handlers import handle_resend_verification

        result = await handle_resend_verification({"email": "unknown@example.com"})

        # Should return 200 to prevent email enumeration
        assert result.status_code == 200

    async def test_resend_verification_missing_email(self):
        """Test resend fails without email."""
        from aragora.server.handlers.auth.signup_handlers import handle_resend_verification

        result = await handle_resend_verification({})

        assert result.status_code == 400


# ===========================================================================
# Test Organization Setup
# ===========================================================================


class TestSetupOrganization:
    """Tests for handle_setup_organization."""

    @patch("aragora.server.handlers.auth.signup_handlers._check_permission", return_value=None)
    async def test_setup_organization_success(self, mock_check):
        """Test successful organization creation."""
        from aragora.server.handlers.auth.signup_handlers import handle_setup_organization

        result = await handle_setup_organization(
            {
                "name": "Acme Corp",
                "plan": "team",
                "billing_email": "billing@acme.com",
            }
        )

        assert result.status_code == 200
        body = json.loads(result.body)
        data = get_response_data(body)
        assert "organization" in data
        org = data["organization"]
        assert org["name"] == "Acme Corp"
        assert org["plan"] == "team"
        assert "id" in org
        assert org["slug"]  # Auto-generated from name

    @patch("aragora.server.handlers.auth.signup_handlers._check_permission", return_value=None)
    async def test_setup_organization_with_slug(self, mock_check):
        """Test organization creation with custom slug."""
        from aragora.server.handlers.auth.signup_handlers import handle_setup_organization

        result = await handle_setup_organization(
            {
                "name": "Acme Corp",
                "slug": "acme-corp-2024",
            }
        )

        body = json.loads(result.body)
        data = get_response_data(body)
        assert data["organization"]["slug"] == "acme-corp-2024"

    @patch("aragora.server.handlers.auth.signup_handlers._check_permission", return_value=None)
    async def test_setup_organization_invalid_slug(self, mock_check):
        """Test organization creation fails with invalid slug."""
        from aragora.server.handlers.auth.signup_handlers import handle_setup_organization

        result = await handle_setup_organization(
            {
                "name": "Acme Corp",
                "slug": "AC",  # Too short
            }
        )

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "slug" in get_response_error(body).lower()

    @patch("aragora.server.handlers.auth.signup_handlers._check_permission", return_value=None)
    async def test_setup_organization_missing_name(self, mock_check):
        """Test organization creation fails without name."""
        from aragora.server.handlers.auth.signup_handlers import handle_setup_organization

        result = await handle_setup_organization({})

        assert result.status_code == 400

    @patch("aragora.server.handlers.auth.signup_handlers._check_permission", return_value=None)
    async def test_setup_organization_short_name(self, mock_check):
        """Test organization creation fails with very short name."""
        from aragora.server.handlers.auth.signup_handlers import handle_setup_organization

        result = await handle_setup_organization({"name": "X"})

        assert result.status_code == 400


# ===========================================================================
# Test Team Invitations
# ===========================================================================


@patch("aragora.server.handlers.auth.signup_handlers._check_permission", return_value=None)
class TestInvite:
    """Tests for handle_invite - team member invitation."""

    async def test_invite_success(self, mock_check):
        """Test successful team invitation."""
        from aragora.server.handlers.auth.signup_handlers import handle_invite

        result = await handle_invite(
            {
                "email": "teammate@example.com",
                "organization_id": "org_123",
                "role": "member",
            }
        )

        assert result.status_code == 200
        body = json.loads(result.body)
        data = get_response_data(body)
        assert "invite_token" in data
        assert data["email"] == "teammate@example.com"
        assert "expires_in" in data

    async def test_invite_invalid_email(self, mock_check):
        """Test invite fails with invalid email."""
        from aragora.server.handlers.auth.signup_handlers import handle_invite

        result = await handle_invite(
            {
                "email": "not-an-email",
                "organization_id": "org_123",
            }
        )

        assert result.status_code == 400

    async def test_invite_missing_org_id(self, mock_check):
        """Test invite fails without organization ID."""
        from aragora.server.handlers.auth.signup_handlers import handle_invite

        result = await handle_invite({"email": "teammate@example.com"})

        assert result.status_code == 400

    async def test_invite_invalid_role(self, mock_check):
        """Test invite fails with invalid role."""
        from aragora.server.handlers.auth.signup_handlers import handle_invite

        result = await handle_invite(
            {
                "email": "teammate@example.com",
                "organization_id": "org_123",
                "role": "superadmin",  # Invalid
            }
        )

        assert result.status_code == 400

    async def test_invite_duplicate_pending(self, mock_check):
        """Test invite fails if already pending for same email/org."""
        from aragora.server.handlers.auth.signup_handlers import handle_invite

        data = {
            "email": "teammate@example.com",
            "organization_id": "org_123",
            "role": "member",
        }

        # First invite
        await handle_invite(data)

        # Duplicate invite
        result = await handle_invite(data)

        assert result.status_code == 409

    async def test_invite_default_role(self, mock_check):
        """Test invite defaults to member role."""
        from aragora.server.handlers.auth.signup_handlers import (
            _pending_invites,
            handle_invite,
        )

        result = await handle_invite(
            {
                "email": "teammate@example.com",
                "organization_id": "org_123",
            }
        )

        body = json.loads(result.body)
        token = get_response_data(body)["invite_token"]
        assert _pending_invites[token]["role"] == "member"


@patch("aragora.server.handlers.auth.signup_handlers._check_permission", return_value=None)
class TestCheckInvite:
    """Tests for handle_check_invite - invitation validation."""

    async def test_check_invite_valid(self, mock_check):
        """Test checking a valid invitation."""
        from aragora.server.handlers.auth.signup_handlers import (
            handle_check_invite,
            handle_invite,
        )

        # Create invite
        invite_result = await handle_invite(
            {
                "email": "teammate@example.com",
                "organization_id": "org_123",
                "role": "admin",
            }
        )
        invite_body = json.loads(invite_result.body)
        token = get_response_data(invite_body)["invite_token"]

        # Check invite
        result = await handle_check_invite({"token": token})

        assert result.status_code == 200
        body = json.loads(result.body)
        data = get_response_data(body)
        assert data["valid"] is True
        assert data["email"] == "teammate@example.com"
        assert data["organization_id"] == "org_123"
        assert data["role"] == "admin"

    async def test_check_invite_invalid_token(self, mock_check):
        """Test checking an invalid invitation."""
        from aragora.server.handlers.auth.signup_handlers import handle_check_invite

        result = await handle_check_invite({"token": "invalid_token"})

        assert result.status_code == 404

    async def test_check_invite_expired(self, mock_check):
        """Test checking an expired invitation."""
        from aragora.server.handlers.auth.signup_handlers import (
            INVITE_TTL,
            _pending_invites,
            handle_check_invite,
            handle_invite,
        )

        # Create invite
        invite_result = await handle_invite(
            {
                "email": "teammate@example.com",
                "organization_id": "org_123",
            }
        )
        invite_body = json.loads(invite_result.body)
        token = get_response_data(invite_body)["invite_token"]

        # Make it expired
        _pending_invites[token]["created_at"] = time.time() - INVITE_TTL - 100

        result = await handle_check_invite({"token": token})

        assert result.status_code == 400
        body = json.loads(result.body)
        assert "expired" in get_response_error(body).lower()

    async def test_check_invite_missing_token(self, mock_check):
        """Test check invite fails without token."""
        from aragora.server.handlers.auth.signup_handlers import handle_check_invite

        result = await handle_check_invite({})

        assert result.status_code == 400


@patch("aragora.server.handlers.auth.signup_handlers._check_permission", return_value=None)
class TestAcceptInvite:
    """Tests for handle_accept_invite - accepting team invitation."""

    async def test_accept_invite_success(self, mock_check):
        """Test successfully accepting an invitation."""
        from aragora.server.handlers.auth.signup_handlers import (
            _pending_invites,
            handle_accept_invite,
            handle_invite,
        )

        # Create invite
        invite_result = await handle_invite(
            {
                "email": "teammate@example.com",
                "organization_id": "org_123",
                "role": "admin",
            }
        )
        invite_body = json.loads(invite_result.body)
        token = get_response_data(invite_body)["invite_token"]

        # Accept invite
        result = await handle_accept_invite({"token": token})

        assert result.status_code == 200
        body = json.loads(result.body)
        data = get_response_data(body)
        assert data["organization_id"] == "org_123"
        assert data["role"] == "admin"

        # Invite should be removed
        assert token not in _pending_invites

    async def test_accept_invite_invalid_token(self, mock_check):
        """Test accepting an invalid invitation."""
        from aragora.server.handlers.auth.signup_handlers import handle_accept_invite

        result = await handle_accept_invite({"token": "invalid_token"})

        assert result.status_code == 404

    async def test_accept_invite_expired(self, mock_check):
        """Test accepting an expired invitation."""
        from aragora.server.handlers.auth.signup_handlers import (
            INVITE_TTL,
            _pending_invites,
            handle_accept_invite,
            handle_invite,
        )

        # Create invite
        invite_result = await handle_invite(
            {
                "email": "teammate@example.com",
                "organization_id": "org_123",
            }
        )
        invite_body = json.loads(invite_result.body)
        token = get_response_data(invite_body)["invite_token"]

        # Make it expired
        _pending_invites[token]["created_at"] = time.time() - INVITE_TTL - 100

        result = await handle_accept_invite({"token": token})

        assert result.status_code == 400

    async def test_accept_invite_missing_token(self, mock_check):
        """Test accept invite fails without token."""
        from aragora.server.handlers.auth.signup_handlers import handle_accept_invite

        result = await handle_accept_invite({})

        assert result.status_code == 400


# ===========================================================================
# Test Onboarding
# ===========================================================================


class TestOnboarding:
    """Tests for onboarding completion tracking."""

    @patch("aragora.server.handlers.auth.signup_handlers._check_permission", return_value=None)
    async def test_onboarding_complete_success(self, mock_check):
        """Test marking onboarding as complete."""
        from aragora.server.handlers.auth.signup_handlers import handle_onboarding_complete

        result = await handle_onboarding_complete(
            {
                "first_debate_id": "debate_123",
                "template_used": "research",
            },
            user_id="user_123",
            organization_id="org_123",
        )

        assert result.status_code == 200
        body = json.loads(result.body)
        data = get_response_data(body)
        assert data["completed"] is True
        assert data["organization_id"] == "org_123"

    @patch("aragora.server.handlers.auth.signup_handlers._check_permission", return_value=None)
    async def test_onboarding_status_not_completed(self, mock_check):
        """Test onboarding status when not completed."""
        from aragora.server.handlers.auth.signup_handlers import handle_onboarding_status

        result = await handle_onboarding_status(organization_id="org_new")

        assert result.status_code == 200
        body = json.loads(result.body)
        data = get_response_data(body)
        assert data["completed"] is False
        assert "steps" in data

    @patch("aragora.server.handlers.auth.signup_handlers._check_permission", return_value=None)
    async def test_onboarding_status_completed(self, mock_check):
        """Test onboarding status after completion."""
        from aragora.server.handlers.auth.signup_handlers import (
            handle_onboarding_complete,
            handle_onboarding_status,
        )

        # Complete onboarding
        await handle_onboarding_complete(
            {"first_debate_id": "debate_123"},
            organization_id="org_123",
        )

        # Check status
        result = await handle_onboarding_status(organization_id="org_123")

        body = json.loads(result.body)
        data = get_response_data(body)
        assert data["completed"] is True
        assert data["first_debate_id"] == "debate_123"


# ===========================================================================
# Test Password Validation
# ===========================================================================


class TestPasswordValidation:
    """Tests for password validation helper."""

    def test_valid_password(self):
        """Test valid password passes validation."""
        from aragora.server.handlers.auth.signup_handlers import _validate_password

        errors = _validate_password("SecureP@ss123")
        assert len(errors) == 0

    def test_short_password(self):
        """Test short password fails validation."""
        from aragora.server.handlers.auth.signup_handlers import _validate_password

        errors = _validate_password("Short1")
        assert len(errors) > 0
        assert any("8" in e for e in errors)

    def test_no_lowercase(self):
        """Test password without lowercase fails."""
        from aragora.server.handlers.auth.signup_handlers import _validate_password

        errors = _validate_password("ALLUPPERCASE123")
        assert any("lowercase" in e.lower() for e in errors)

    def test_no_uppercase(self):
        """Test password without uppercase fails."""
        from aragora.server.handlers.auth.signup_handlers import _validate_password

        errors = _validate_password("alllowercase123")
        assert any("uppercase" in e.lower() for e in errors)

    def test_no_number(self):
        """Test password without number fails."""
        from aragora.server.handlers.auth.signup_handlers import _validate_password

        errors = _validate_password("NoNumberHere")
        assert any("number" in e.lower() for e in errors)


# ===========================================================================
# Test Token Cleanup
# ===========================================================================


class TestTokenCleanup:
    """Tests for _cleanup_expired_tokens helper."""

    def test_cleanup_removes_expired_signups(self):
        """Test that expired signup tokens are removed."""
        from aragora.server.handlers.auth.signup_handlers import (
            VERIFICATION_TTL,
            _cleanup_expired_tokens,
            _pending_signups,
        )

        # Add expired and valid tokens
        _pending_signups["expired"] = {"created_at": time.time() - VERIFICATION_TTL - 100}
        _pending_signups["valid"] = {"created_at": time.time()}

        _cleanup_expired_tokens()

        assert "expired" not in _pending_signups
        assert "valid" in _pending_signups

    def test_cleanup_removes_expired_invites(self):
        """Test that expired invite tokens are removed."""
        from aragora.server.handlers.auth.signup_handlers import (
            INVITE_TTL,
            _cleanup_expired_tokens,
            _pending_invites,
        )

        # Add expired and valid tokens
        _pending_invites["expired"] = {"created_at": time.time() - INVITE_TTL - 100}
        _pending_invites["valid"] = {"created_at": time.time()}

        _cleanup_expired_tokens()

        assert "expired" not in _pending_invites
        assert "valid" in _pending_invites


# ===========================================================================
# Test Handler Registration
# ===========================================================================


class TestHandlerRegistration:
    """Tests for get_signup_handlers - handler function exports."""

    def test_get_signup_handlers_returns_dict(self):
        """Test that get_signup_handlers returns handler mapping."""
        from aragora.server.handlers.auth.signup_handlers import get_signup_handlers

        handlers = get_signup_handlers()

        assert isinstance(handlers, dict)
        assert "signup" in handlers
        assert "verify_email" in handlers
        assert "resend_verification" in handlers
        assert "setup_organization" in handlers
        assert "invite" in handlers
        assert "check_invite" in handlers
        assert "accept_invite" in handlers
        assert "onboarding_complete" in handlers
        assert "onboarding_status" in handlers

    def test_all_handlers_are_callable(self):
        """Test that all exported handlers are callable."""
        from aragora.server.handlers.auth.signup_handlers import get_signup_handlers

        handlers = get_signup_handlers()

        for name, handler in handlers.items():
            assert callable(handler), f"Handler '{name}' is not callable"
