"""
HTTP API Handlers for Self-Service Signup.

Provides REST APIs for self-service organization/user signup:
- User registration
- Email verification
- Organization creation
- Team invitations

Endpoints:
- POST /api/v1/auth/signup - Register new user
- POST /api/v1/auth/verify-email - Verify email address
- POST /api/v1/auth/resend-verification - Resend verification email
- POST /api/v1/auth/setup-organization - Create organization after signup
- POST /api/v1/auth/invite - Invite team member
- POST /api/v1/auth/accept-invite - Accept team invitation
- GET /api/v1/auth/check-invite - Check invitation validity
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import secrets
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from aragora.server.handlers.base import (
    HandlerResult,
    error_response,
    success_response,
)

logger = logging.getLogger(__name__)

# In-memory storage (replace with DB in production)
_pending_signups: Dict[str, Dict[str, Any]] = {}
_pending_signups_lock = threading.Lock()

_pending_invites: Dict[str, Dict[str, Any]] = {}
_pending_invites_lock = threading.Lock()

# Verification token TTL (24 hours)
VERIFICATION_TTL = 86400

# Invite TTL (7 days)
INVITE_TTL = 604800

# Email regex
EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")

# Password requirements
MIN_PASSWORD_LENGTH = 8


def _generate_verification_token() -> str:
    """Generate a secure verification token."""
    return secrets.token_urlsafe(32)


def _hash_password(password: str) -> str:
    """Hash password (in production, use bcrypt/argon2)."""
    # Simple hash for demo - use proper password hashing in production
    salt = os.environ.get("PASSWORD_SALT", "aragora-salt")
    return hashlib.sha256(f"{password}{salt}".encode()).hexdigest()


def _validate_password(password: str) -> List[str]:
    """Validate password strength."""
    errors = []

    if len(password) < MIN_PASSWORD_LENGTH:
        errors.append(f"Password must be at least {MIN_PASSWORD_LENGTH} characters")

    if not re.search(r"[a-z]", password):
        errors.append("Password must contain a lowercase letter")

    if not re.search(r"[A-Z]", password):
        errors.append("Password must contain an uppercase letter")

    if not re.search(r"\d", password):
        errors.append("Password must contain a number")

    return errors


def _cleanup_expired_tokens():
    """Remove expired verification tokens and invites."""
    now = time.time()

    with _pending_signups_lock:
        expired = [
            token
            for token, data in _pending_signups.items()
            if now - data.get("created_at", 0) > VERIFICATION_TTL
        ]
        for token in expired:
            del _pending_signups[token]

    with _pending_invites_lock:
        expired = [
            token
            for token, data in _pending_invites.items()
            if now - data.get("created_at", 0) > INVITE_TTL
        ]
        for token in expired:
            del _pending_invites[token]


# =============================================================================
# User Registration
# =============================================================================


async def handle_signup(
    data: Dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Register a new user.

    POST /api/v1/auth/signup
    Body: {
        email: str,
        password: str,
        name: str,
        company_name: str (optional),
        invite_token: str (optional) - If joining via invitation
    }
    """
    try:
        email = data.get("email", "").lower().strip()
        password = data.get("password", "")
        name = data.get("name", "").strip()
        company_name = data.get("company_name", "").strip()
        invite_token = data.get("invite_token")

        # Validate email
        if not email or not EMAIL_REGEX.match(email):
            return error_response("Invalid email address", status=400)

        # Validate password
        password_errors = _validate_password(password)
        if password_errors:
            return error_response(
                "Password requirements not met",
                status=400,
            )

        # Validate name
        if not name or len(name) < 2:
            return error_response("Name must be at least 2 characters", status=400)

        # Check if email already registered (would check DB in production)
        # For now, check pending signups
        with _pending_signups_lock:
            for signup_data in _pending_signups.values():
                if signup_data.get("email") == email:
                    return error_response(
                        "Email already pending verification",
                        status=409,
                    )

        # Check for invitation
        invite_data = None
        if invite_token:
            with _pending_invites_lock:
                invite_data = _pending_invites.get(invite_token)
                if invite_data:
                    if invite_data.get("email") != email:
                        return error_response(
                            "Email does not match invitation",
                            status=400,
                        )
                    if time.time() - invite_data.get("created_at", 0) > INVITE_TTL:
                        return error_response(
                            "Invitation has expired",
                            status=400,
                        )

        # Generate verification token
        verification_token = _generate_verification_token()

        # Store pending signup
        signup_record = {
            "email": email,
            "password_hash": _hash_password(password),
            "name": name,
            "company_name": company_name,
            "invite_token": invite_token,
            "invite_data": invite_data,
            "created_at": time.time(),
            "verified": False,
        }

        with _pending_signups_lock:
            _pending_signups[verification_token] = signup_record

        # Cleanup old tokens periodically
        if len(_pending_signups) % 10 == 0:
            _cleanup_expired_tokens()

        # In production: send verification email
        logger.info(f"Signup initiated for {email}, verification token: {verification_token}")

        return success_response(
            {
                "message": "Verification email sent",
                "email": email,
                "verification_token": verification_token,  # Remove in production
                "expires_in": VERIFICATION_TTL,
            }
        )

    except Exception as e:
        logger.exception("Signup failed")
        return error_response(f"Signup failed: {str(e)}", status=500)


async def handle_verify_email(
    data: Dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Verify email address.

    POST /api/v1/auth/verify-email
    Body: {
        token: str
    }
    """
    try:
        token = data.get("token", "")

        if not token:
            return error_response("Verification token is required", status=400)

        with _pending_signups_lock:
            signup_data = _pending_signups.get(token)

            if not signup_data:
                return error_response("Invalid or expired token", status=400)

            if time.time() - signup_data.get("created_at", 0) > VERIFICATION_TTL:
                del _pending_signups[token]
                return error_response("Verification token has expired", status=400)

            # Mark as verified
            signup_data["verified"] = True
            signup_data["verified_at"] = time.time()

        # In production: create user in database
        email = signup_data["email"]
        name = signup_data["name"]

        # Generate user ID
        user_id = f"user_{secrets.token_hex(8)}"

        # Create JWT token
        from aragora.billing.jwt_auth import create_access_token

        access_token = create_access_token(user_id=user_id, email=email)

        # Remove from pending signups
        with _pending_signups_lock:
            del _pending_signups[token]

        # If this was an invite, remove the invite
        if signup_data.get("invite_token"):
            with _pending_invites_lock:
                _pending_invites.pop(signup_data["invite_token"], None)

        return success_response(
            {
                "message": "Email verified successfully",
                "user_id": user_id,
                "email": email,
                "name": name,
                "access_token": access_token,
                "token_type": "bearer",
                "needs_org_setup": not signup_data.get("invite_data"),
                "organization_id": signup_data.get("invite_data", {}).get("organization_id"),
            }
        )

    except Exception as e:
        logger.exception("Email verification failed")
        return error_response(f"Verification failed: {str(e)}", status=500)


async def handle_resend_verification(
    data: Dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Resend verification email.

    POST /api/v1/auth/resend-verification
    Body: {
        email: str
    }
    """
    try:
        email = data.get("email", "").lower().strip()

        if not email:
            return error_response("Email is required", status=400)

        # Find pending signup by email
        found_token = None
        with _pending_signups_lock:
            for token, signup_data in _pending_signups.items():
                if signup_data.get("email") == email and not signup_data.get("verified"):
                    found_token = token
                    break

        if not found_token:
            # Don't reveal if email exists
            return success_response(
                {
                    "message": "If email is pending verification, a new email will be sent",
                }
            )

        # In production: resend verification email
        logger.info(f"Resending verification for {email}")

        return success_response(
            {
                "message": "Verification email resent",
                "email": email,
            }
        )

    except Exception as e:
        logger.exception("Resend verification failed")
        return error_response(f"Resend failed: {str(e)}", status=500)


# =============================================================================
# Organization Setup
# =============================================================================


async def handle_setup_organization(
    data: Dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Create organization after signup.

    POST /api/v1/auth/setup-organization
    Body: {
        name: str,
        slug: str (optional),
        plan: str (optional - free, team, enterprise),
        billing_email: str (optional)
    }
    """
    try:
        name = data.get("name", "").strip()
        slug = data.get("slug", "").lower().strip()
        plan = data.get("plan", "free")
        billing_email = data.get("billing_email", "").lower().strip()

        if not name or len(name) < 2:
            return error_response("Organization name is required", status=400)

        # Generate slug if not provided
        if not slug:
            slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")

        # Validate slug
        if not re.match(r"^[a-z0-9][a-z0-9-]{2,30}[a-z0-9]$", slug):
            return error_response(
                "Slug must be 4-32 characters, alphanumeric with hyphens",
                status=400,
            )

        # Generate organization ID
        org_id = f"org_{secrets.token_hex(8)}"

        # In production: create organization in database
        organization = {
            "id": org_id,
            "name": name,
            "slug": slug,
            "plan": plan,
            "billing_email": billing_email or None,
            "owner_id": user_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "member_count": 1,
        }

        logger.info(f"Organization created: {org_id} ({name}) by {user_id}")

        return success_response(
            {
                "organization": organization,
                "message": "Organization created successfully",
            }
        )

    except Exception as e:
        logger.exception("Organization setup failed")
        return error_response(f"Setup failed: {str(e)}", status=500)


# =============================================================================
# Team Invitations
# =============================================================================


async def handle_invite(
    data: Dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Invite team member to organization.

    POST /api/v1/auth/invite
    Body: {
        email: str,
        organization_id: str,
        role: str (optional - admin, member, viewer)
    }
    """
    try:
        email = data.get("email", "").lower().strip()
        organization_id = data.get("organization_id", "")
        role = data.get("role", "member")

        if not email or not EMAIL_REGEX.match(email):
            return error_response("Invalid email address", status=400)

        if not organization_id:
            return error_response("Organization ID is required", status=400)

        valid_roles = {"admin", "member", "viewer"}
        if role not in valid_roles:
            return error_response(
                f"Invalid role. Must be one of: {', '.join(valid_roles)}",
                status=400,
            )

        # Check for existing pending invite
        with _pending_invites_lock:
            for token, invite in _pending_invites.items():
                if (
                    invite.get("email") == email
                    and invite.get("organization_id") == organization_id
                ):
                    return error_response(
                        "Invitation already pending for this email",
                        status=409,
                    )

        # Generate invite token
        invite_token = _generate_verification_token()

        # Store invitation
        invite_record = {
            "email": email,
            "organization_id": organization_id,
            "role": role,
            "invited_by": user_id,
            "created_at": time.time(),
        }

        with _pending_invites_lock:
            _pending_invites[invite_token] = invite_record

        # In production: send invitation email
        invite_url = f"/invite/{invite_token}"
        logger.info(f"Invitation sent to {email} for org {organization_id}")

        return success_response(
            {
                "message": "Invitation sent",
                "email": email,
                "invite_token": invite_token,  # Remove in production
                "invite_url": invite_url,
                "expires_in": INVITE_TTL,
            }
        )

    except Exception as e:
        logger.exception("Invite failed")
        return error_response(f"Invite failed: {str(e)}", status=500)


async def handle_check_invite(
    data: Dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Check invitation validity.

    GET /api/v1/auth/check-invite
    Query params:
        token: str
    """
    try:
        token = data.get("token", "")

        if not token:
            return error_response("Token is required", status=400)

        with _pending_invites_lock:
            invite = _pending_invites.get(token)

        if not invite:
            return error_response("Invalid invitation", status=404)

        if time.time() - invite.get("created_at", 0) > INVITE_TTL:
            return error_response("Invitation has expired", status=400)

        return success_response(
            {
                "valid": True,
                "email": invite.get("email"),
                "organization_id": invite.get("organization_id"),
                "role": invite.get("role"),
                "expires_at": invite.get("created_at", 0) + INVITE_TTL,
            }
        )

    except Exception as e:
        logger.exception("Check invite failed")
        return error_response(f"Check failed: {str(e)}", status=500)


async def handle_accept_invite(
    data: Dict[str, Any],
    user_id: str = "default",
) -> HandlerResult:
    """
    Accept team invitation (for existing users).

    POST /api/v1/auth/accept-invite
    Body: {
        token: str
    }
    """
    try:
        token = data.get("token", "")

        if not token:
            return error_response("Token is required", status=400)

        with _pending_invites_lock:
            invite = _pending_invites.get(token)

            if not invite:
                return error_response("Invalid invitation", status=404)

            if time.time() - invite.get("created_at", 0) > INVITE_TTL:
                del _pending_invites[token]
                return error_response("Invitation has expired", status=400)

            # Remove invitation
            del _pending_invites[token]

        # In production: add user to organization
        organization_id = invite.get("organization_id")
        role = invite.get("role")

        logger.info(f"User {user_id} joined org {organization_id} as {role}")

        return success_response(
            {
                "message": "Successfully joined organization",
                "organization_id": organization_id,
                "role": role,
            }
        )

    except Exception as e:
        logger.exception("Accept invite failed")
        return error_response(f"Accept failed: {str(e)}", status=500)


# =============================================================================
# Handler Registration
# =============================================================================


def get_signup_handlers() -> Dict[str, Any]:
    """Get all signup handlers for registration."""
    return {
        "signup": handle_signup,
        "verify_email": handle_verify_email,
        "resend_verification": handle_resend_verification,
        "setup_organization": handle_setup_organization,
        "invite": handle_invite,
        "check_invite": handle_check_invite,
        "accept_invite": handle_accept_invite,
    }


__all__ = [
    "handle_signup",
    "handle_verify_email",
    "handle_resend_verification",
    "handle_setup_organization",
    "handle_invite",
    "handle_check_invite",
    "handle_accept_invite",
    "get_signup_handlers",
]
