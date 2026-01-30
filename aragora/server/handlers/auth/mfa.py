"""
MFA (Multi-Factor Authentication) Handlers.

Handles MFA-related endpoints:
- POST /api/auth/mfa/setup - Generate MFA secret and provisioning URI
- POST /api/auth/mfa/enable - Enable MFA after verifying setup code
- POST /api/auth/mfa/disable - Disable MFA
- DELETE /api/auth/mfa - Disable MFA (alias)
- POST /api/auth/mfa/verify - Verify MFA code during login
- POST /api/auth/mfa/backup-codes - Regenerate backup codes
"""

from __future__ import annotations

import hashlib
import json as json_module
import logging
import secrets as py_secrets
from typing import TYPE_CHECKING

from aragora.billing.jwt_auth import extract_user_from_request

from ..base import HandlerResult, error_response, json_response, handle_errors, log_request
from ..utils.rate_limit import rate_limit

if TYPE_CHECKING:
    from .handler import AuthHandler

# Unified audit logging
try:
    from aragora.audit.unified import audit_security

    AUDIT_AVAILABLE = True
except ImportError:
    AUDIT_AVAILABLE = False
    audit_security = None

logger = logging.getLogger(__name__)


@rate_limit(requests_per_minute=5, limiter_name="mfa_setup")
@handle_errors("MFA setup")
@log_request("MFA setup")
def handle_mfa_setup(handler_instance: "AuthHandler", handler) -> HandlerResult:
    """Generate MFA secret and provisioning URI for setup."""
    # RBAC check: authentication.create permission required
    if error := handler_instance._check_permission(handler, "authentication.create"):
        return error

    try:
        import pyotp
    except ImportError:
        return error_response("MFA not available (pyotp not installed)", 503)

    # Get current user (already verified by _check_permission)
    user_store = handler_instance._get_user_store()
    auth_ctx = extract_user_from_request(handler, user_store)

    user = user_store.get_user_by_id(auth_ctx.user_id)
    if not user:
        return error_response("User not found", 404)

    if user.mfa_enabled:
        return error_response("MFA is already enabled", 400)

    # Generate new secret
    secret = pyotp.random_base32()

    # Store secret temporarily (not enabled yet)
    user_store.update_user(user.id, mfa_secret=secret)

    # Generate provisioning URI for authenticator apps
    totp = pyotp.TOTP(secret)
    provisioning_uri = totp.provisioning_uri(name=user.email, issuer_name="Aragora")

    return json_response(
        {
            "secret": secret,
            "provisioning_uri": provisioning_uri,
            "message": "Scan QR code or enter secret in your authenticator app, then call /api/auth/mfa/enable with verification code",
        }
    )


@rate_limit(requests_per_minute=5, limiter_name="mfa_enable")
@handle_errors("MFA enable")
@log_request("MFA enable")
def handle_mfa_enable(handler_instance: "AuthHandler", handler) -> HandlerResult:
    """Enable MFA after verifying setup code."""
    # RBAC check: authentication.update permission required
    if error := handler_instance._check_permission(handler, "authentication.update"):
        return error

    try:
        import pyotp
    except ImportError:
        return error_response("MFA not available", 503)

    body = handler_instance.read_json_body(handler)
    if body is None:
        return error_response("Invalid JSON body", 400)

    code = body.get("code", "").strip()
    if not code:
        return error_response("Verification code is required", 400)

    # Get current user (already verified by _check_permission)
    user_store = handler_instance._get_user_store()
    auth_ctx = extract_user_from_request(handler, user_store)

    user = user_store.get_user_by_id(auth_ctx.user_id)
    if not user:
        return error_response("User not found", 404)

    if user.mfa_enabled:
        return error_response("MFA is already enabled", 400)

    if not user.mfa_secret:
        return error_response("MFA not set up. Call /api/auth/mfa/setup first", 400)

    # Verify the code
    totp = pyotp.TOTP(user.mfa_secret)
    if not totp.verify(code, valid_window=1):
        return error_response("Invalid verification code", 400)

    # Generate backup codes
    backup_codes = [py_secrets.token_hex(4) for _ in range(10)]
    backup_hashes = [hashlib.sha256(c.encode()).hexdigest() for c in backup_codes]

    user_store.update_user(
        user.id,
        mfa_enabled=True,
        mfa_backup_codes=json_module.dumps(backup_hashes),
    )

    # Invalidate all existing sessions by incrementing token version
    user_store.increment_token_version(user.id)

    logger.info(f"MFA enabled for user: {user.email}")

    # Audit log: MFA enabled
    if AUDIT_AVAILABLE and audit_security:
        audit_security(
            event_type="encryption",
            actor_id=user.id,
            reason="mfa_enabled",
        )

    return json_response(
        {
            "message": "MFA enabled successfully",
            "backup_codes": backup_codes,
            "warning": "Save these backup codes securely. They cannot be shown again.",
            "sessions_invalidated": True,
        }
    )


@rate_limit(requests_per_minute=5, limiter_name="mfa_disable")
@handle_errors("MFA disable")
@log_request("MFA disable")
def handle_mfa_disable(handler_instance: "AuthHandler", handler) -> HandlerResult:
    """Disable MFA for the user."""
    # RBAC check: authentication.update permission required
    if error := handler_instance._check_permission(handler, "authentication.update"):
        return error

    try:
        import pyotp
    except ImportError:
        return error_response("MFA not available", 503)

    body = handler_instance.read_json_body(handler)
    if body is None:
        return error_response("Invalid JSON body", 400)

    # Require password or MFA code to disable
    code = body.get("code", "").strip()
    password = body.get("password", "").strip()

    if not code and not password:
        return error_response("MFA code or password required to disable MFA", 400)

    # Get current user (already verified by _check_permission)
    user_store = handler_instance._get_user_store()
    auth_ctx = extract_user_from_request(handler, user_store)

    user = user_store.get_user_by_id(auth_ctx.user_id)
    if not user:
        return error_response("User not found", 404)

    if not user.mfa_enabled:
        return error_response("MFA is not enabled", 400)

    # Verify with code or password
    if code:
        totp = pyotp.TOTP(user.mfa_secret)
        if not totp.verify(code, valid_window=1):
            return error_response("Invalid MFA code", 400)
    elif password:
        if not user.verify_password(password):
            return error_response("Invalid password", 400)

    # Disable MFA
    user_store.update_user(
        user.id,
        mfa_enabled=False,
        mfa_secret=None,
        mfa_backup_codes=None,
    )

    logger.info(f"MFA disabled for user: {user.email}")

    # Audit log: MFA disabled
    if AUDIT_AVAILABLE and audit_security:
        audit_security(
            event_type="encryption",
            actor_id=user.id,
            reason="mfa_disabled",
        )

    return json_response({"message": "MFA disabled successfully"})


@rate_limit(requests_per_minute=10, limiter_name="mfa_verify")
@handle_errors("MFA verify")
@log_request("MFA verify")
def handle_mfa_verify(handler_instance: "AuthHandler", handler) -> HandlerResult:
    """Verify MFA code during login."""
    from aragora.billing.jwt_auth import create_token_pair, validate_mfa_pending_token

    try:
        import pyotp
    except ImportError:
        return error_response("MFA not available", 503)

    body = handler_instance.read_json_body(handler)
    if body is None:
        return error_response("Invalid JSON body", 400)

    code = body.get("code", "").strip()
    pending_token = body.get("pending_token", "").strip()

    if not code:
        return error_response("MFA code is required", 400)

    if not pending_token:
        return error_response("Pending token is required", 400)

    # Validate the pending token to identify the user
    pending_payload = validate_mfa_pending_token(pending_token)
    if not pending_payload:
        return error_response("Invalid or expired pending token", 401)

    user_store = handler_instance._get_user_store()
    if not user_store:
        return error_response("Authentication service unavailable", 503)

    user = user_store.get_user_by_id(pending_payload.sub)
    if not user:
        return error_response("User not found", 404)

    if not user.mfa_enabled or not user.mfa_secret:
        return error_response("MFA not enabled for this user", 400)

    # Try TOTP code first
    totp = pyotp.TOTP(user.mfa_secret)
    if totp.verify(code, valid_window=1):
        # Blacklist pending token to prevent replay
        from aragora.billing.jwt_auth import get_token_blacklist

        blacklist = get_token_blacklist()
        blacklist.revoke_token(pending_token)

        # Valid TOTP code - create full tokens
        tokens = create_token_pair(
            user_id=user.id,
            email=user.email,
            org_id=user.org_id,
            role=user.role,
        )
        token_dict = tokens.to_dict()
        logger.info(f"MFA verified for user: {user.email}")
        return json_response(
            {
                "message": "MFA verification successful",
                "user": user.to_dict(),
                "tokens": token_dict,
            }
        )

    # Try backup code
    if user.mfa_backup_codes:
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        backup_hashes = json_module.loads(user.mfa_backup_codes)

        if code_hash in backup_hashes:
            # Valid backup code - remove it
            backup_hashes.remove(code_hash)
            user_store.update_user(
                user.id,
                mfa_backup_codes=json_module.dumps(backup_hashes),
            )

            # Blacklist pending token to prevent replay
            from aragora.billing.jwt_auth import get_token_blacklist

            blacklist = get_token_blacklist()
            blacklist.revoke_token(pending_token)

            tokens = create_token_pair(
                user_id=user.id,
                email=user.email,
                org_id=user.org_id,
                role=user.role,
            )
            token_dict = tokens.to_dict()
            remaining = len(backup_hashes)

            logger.info(f"Backup code used for user: {user.email}, {remaining} remaining")

            return json_response(
                {
                    "message": "MFA verification successful (backup code used)",
                    "user": user.to_dict(),
                    "tokens": token_dict,
                    "backup_codes_remaining": remaining,
                    "warning": (
                        f"Backup code used. {remaining} remaining." if remaining < 5 else None
                    ),
                }
            )

    return error_response("Invalid MFA code", 400)


@rate_limit(requests_per_minute=3, limiter_name="mfa_backup")
@handle_errors("MFA backup codes")
@log_request("MFA backup codes")
def handle_mfa_backup_codes(handler_instance: "AuthHandler", handler) -> HandlerResult:
    """Regenerate MFA backup codes."""
    # RBAC check: authentication.read permission required
    if error := handler_instance._check_permission(handler, "authentication.read"):
        return error

    try:
        import pyotp
    except ImportError:
        return error_response("MFA not available", 503)

    body = handler_instance.read_json_body(handler)
    if body is None:
        return error_response("Invalid JSON body", 400)

    # Require current MFA code to regenerate backup codes
    code = body.get("code", "").strip()
    if not code:
        return error_response("Current MFA code is required", 400)

    # Get current user (already verified by _check_permission)
    user_store = handler_instance._get_user_store()
    auth_ctx = extract_user_from_request(handler, user_store)

    user = user_store.get_user_by_id(auth_ctx.user_id)
    if not user:
        return error_response("User not found", 404)

    if not user.mfa_enabled or not user.mfa_secret:
        return error_response("MFA not enabled", 400)

    # Verify current code
    totp = pyotp.TOTP(user.mfa_secret)
    if not totp.verify(code, valid_window=1):
        return error_response("Invalid MFA code", 400)

    # Generate new backup codes
    backup_codes = [py_secrets.token_hex(4) for _ in range(10)]
    backup_hashes = [hashlib.sha256(c.encode()).hexdigest() for c in backup_codes]

    user_store.update_user(
        user.id,
        mfa_backup_codes=json_module.dumps(backup_hashes),
    )

    logger.info(f"Backup codes regenerated for user: {user.email}")

    return json_response(
        {
            "backup_codes": backup_codes,
            "warning": "Save these backup codes securely. They cannot be shown again.",
        }
    )


__all__ = [
    "handle_mfa_setup",
    "handle_mfa_enable",
    "handle_mfa_disable",
    "handle_mfa_verify",
    "handle_mfa_backup_codes",
]
