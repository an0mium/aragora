"""
Password Management Handlers.

Handles password-related endpoints:
- POST /api/auth/password - Change password
- POST /api/auth/password/change - Change password (alias)
- POST /api/auth/password/forgot - Request password reset
- POST /api/auth/password/reset - Reset password with token
- POST /api/auth/forgot-password - Request password reset (legacy)
- POST /api/auth/reset-password - Reset password with token (legacy)
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING

from aragora.billing.jwt_auth import extract_user_from_request

from ..base import HandlerResult, error_response, json_response, handle_errors, log_request
from ..utils.rate_limit import get_client_ip, rate_limit
from .validation import validate_email, validate_password

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


@rate_limit(requests_per_minute=3, limiter_name="auth_change_password")
@handle_errors("change password")
def handle_change_password(handler_instance: "AuthHandler", handler) -> HandlerResult:
    """Change user password."""
    # RBAC check: authentication.read permission required
    if error := handler_instance._check_permission(handler, "authentication.read"):
        return error

    # Get current user (already verified by _check_permission)
    user_store = handler_instance._get_user_store()
    auth_ctx = extract_user_from_request(handler, user_store)

    # Parse request body
    body = handler_instance.read_json_body(handler)
    if body is None:
        return error_response("Invalid JSON body", 400)

    current_password = body.get("current_password", "")
    new_password = body.get("new_password", "")

    if not current_password or not new_password:
        return error_response("Current and new password required", 400)

    # Validate new password
    valid, err = validate_password(new_password)
    if not valid:
        return error_response(err, 400)

    # Get user store
    if not user_store:
        return error_response("Authentication service unavailable", 503)

    # Get user
    user = user_store.get_user_by_id(auth_ctx.user_id)
    if not user:
        return error_response("User not found", 404)

    # Verify current password
    if not user.verify_password(current_password):
        return error_response("Current password is incorrect", 401)

    # Set new password
    from aragora.billing.models import hash_password

    password_hash, password_salt = hash_password(new_password)
    user_store.update_user(
        user.id,
        password_hash=password_hash,
        password_salt=password_salt,
    )

    # Invalidate all existing sessions by incrementing token version
    user_store.increment_token_version(user.id)

    logger.info(f"Password changed for user: {user.email}")

    return json_response(
        {
            "message": "Password changed successfully",
            "sessions_invalidated": True,
        }
    )


@rate_limit(requests_per_minute=3, limiter_name="auth_forgot_password")
@handle_errors("forgot password")
@log_request("forgot password")
def handle_forgot_password(handler_instance: "AuthHandler", handler) -> HandlerResult:
    """
    Handle forgot password request.

    Generates a password reset token and sends an email with reset link.
    Always returns success to prevent email enumeration attacks.
    """
    from aragora.storage.password_reset_store import get_password_reset_store

    # Parse request body
    body = handler_instance.read_json_body(handler)
    if body is None:
        return error_response("Invalid JSON body", 400)

    email = body.get("email", "").strip().lower()
    if not email:
        return error_response("Email is required", 400)

    # Validate email format
    valid, err = validate_email(email)
    if not valid:
        return error_response(err, 400)

    # Get client IP for logging
    client_ip = get_client_ip(handler)

    # Get user store to check if email exists
    user_store = handler_instance._get_user_store()
    if not user_store:
        return error_response("Service unavailable", 503)

    # Check if user exists (but don't reveal this to prevent enumeration)
    user = user_store.get_user_by_email(email)

    # Generate reset token only if user exists
    if user and user.is_active:
        store = get_password_reset_store()
        token, rate_error = store.create_token(email)

        if rate_error:
            # Rate limit hit - still return generic success for security
            logger.warning(f"Password reset rate limited: email={email}, ip={client_ip}")
        elif token:
            # Build reset link (frontend URL with token)
            base_url = os.environ.get("ARAGORA_FRONTEND_URL", "https://aragora.ai")
            reset_link = f"{base_url}/reset-password?token={token}"

            # Send email asynchronously
            send_password_reset_email(user, reset_link)

            logger.info(f"Password reset requested: email={email}, ip={client_ip}")

            # Audit log: password reset requested
            if AUDIT_AVAILABLE and audit_security:
                audit_security(
                    event_type="anomaly",
                    actor_id=email,
                    ip_address=client_ip,
                    reason="password_reset_requested",
                )
    else:
        # User doesn't exist - log but return same response
        logger.debug(f"Password reset for non-existent email: {email}, ip={client_ip}")

    # Always return success to prevent email enumeration
    return json_response(
        {
            "message": "If an account exists with that email, a password reset link has been sent.",
            "email": email,
        }
    )


def send_password_reset_email(user, reset_link: str) -> None:
    """Send password reset email to user (fire-and-forget)."""

    async def send_email():
        try:
            from aragora.integrations.email import (
                EmailConfig,
                EmailIntegration,
                EmailRecipient,
            )

            # Check if email is configured
            smtp_host = os.environ.get("SMTP_HOST", "")
            sendgrid_key = os.environ.get("SENDGRID_API_KEY", "")
            ses_key = os.environ.get("AWS_ACCESS_KEY_ID", "")

            if not smtp_host and not sendgrid_key and not ses_key:
                logger.warning(
                    "No email provider configured - password reset email not sent. "
                    "Configure SMTP_HOST, SENDGRID_API_KEY, or AWS_ACCESS_KEY_ID."
                )
                return

            config = EmailConfig(
                smtp_host=smtp_host,
                sendgrid_api_key=sendgrid_key,
                from_email=os.environ.get("ARAGORA_FROM_EMAIL", "noreply@aragora.ai"),
                from_name="Aragora",
            )

            async with EmailIntegration(config) as email_client:
                recipient = EmailRecipient(email=user.email, name=user.name)

                subject = "Reset Your Password - Aragora"
                html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #00ff00, #00cc00); color: #000; padding: 20px; text-align: center; border-radius: 8px 8px 0 0; }}
        .content {{ background: #f9f9f9; padding: 30px; border: 1px solid #e0e0e0; }}
        .footer {{ background: #333; color: #999; padding: 15px; text-align: center; font-size: 12px; border-radius: 0 0 8px 8px; }}
        .button {{ display: inline-block; background: #00cc00; color: #000; padding: 12px 24px; text-decoration: none; border-radius: 4px; font-weight: bold; margin: 20px 0; }}
        .warning {{ background: #fff3cd; border: 1px solid #ffc107; padding: 10px; border-radius: 4px; margin-top: 20px; font-size: 13px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Password Reset Request</h1>
        </div>
        <div class="content">
            <p>Hi {user.name or "there"},</p>
            <p>We received a request to reset your password for your Aragora account.</p>
            <p>Click the button below to reset your password:</p>
            <div style="text-align: center;">
                <a href="{reset_link}" class="button">Reset Password</a>
            </div>
            <p style="font-size: 13px; color: #666;">
                Or copy and paste this link into your browser:<br>
                <code style="word-break: break-all;">{reset_link}</code>
            </p>
            <div class="warning">
                <strong>Security Notice:</strong> This link will expire in 1 hour.
                If you didn't request this password reset, you can safely ignore this email.
                Your password will not be changed unless you click the link above.
            </div>
        </div>
        <div class="footer">
            <p>Aragora AI - Multi-Agent Debate Platform</p>
            <p>This is an automated message. Please do not reply.</p>
        </div>
    </div>
</body>
</html>
"""

                text_body = f"""Password Reset Request
======================

Hi {user.name or "there"},

We received a request to reset your password for your Aragora account.

Click the link below to reset your password:
{reset_link}

This link will expire in 1 hour.

If you didn't request this password reset, you can safely ignore this email.
Your password will not be changed unless you click the link above.

---
Aragora AI - Multi-Agent Debate Platform
This is an automated message. Please do not reply.
"""

                success = await email_client._send_email(recipient, subject, html_body, text_body)
                if success:
                    logger.info(f"Password reset email sent to: {user.email}")
                else:
                    logger.error(f"Failed to send password reset email to: {user.email}")

        except ImportError as e:
            logger.warning(f"Email integration not available: {e}")
        except Exception as e:
            logger.error(f"Error sending password reset email: {e}")

    # Run email sending in background (don't block the response)
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(send_email())
        else:
            loop.run_until_complete(send_email())
    except RuntimeError:
        # No event loop - create one for this task
        asyncio.run(send_email())


@rate_limit(requests_per_minute=5, limiter_name="auth_reset_password")
@handle_errors("reset password")
@log_request("reset password")
def handle_reset_password(handler_instance: "AuthHandler", handler) -> HandlerResult:
    """
    Handle password reset with token.

    Validates the reset token and updates the user's password.
    """
    from aragora.billing.models import hash_password
    from aragora.storage.password_reset_store import get_password_reset_store

    # Parse request body
    body = handler_instance.read_json_body(handler)
    if body is None:
        return error_response("Invalid JSON body", 400)

    token = body.get("token", "").strip()
    new_password = body.get("password", "") or body.get("new_password", "")

    if not token:
        return error_response("Reset token is required", 400)

    if not new_password:
        return error_response("New password is required", 400)

    # Validate password requirements
    valid, err = validate_password(new_password)
    if not valid:
        return error_response(err, 400)

    # Get client IP for logging
    client_ip = get_client_ip(handler)

    # Validate the reset token
    store = get_password_reset_store()
    email, token_error = store.validate_token(token)

    if token_error:
        logger.warning(f"Invalid password reset attempt: ip={client_ip}, error={token_error}")
        return error_response(token_error, 400)

    # Get user store
    user_store = handler_instance._get_user_store()
    if not user_store:
        return error_response("Service unavailable", 503)

    # Get user by email
    user = user_store.get_user_by_email(email)
    if not user:
        # This shouldn't happen if token is valid, but handle it
        store.consume_token(token)
        return error_response("User not found", 404)

    if not user.is_active:
        store.consume_token(token)
        return error_response("Account is disabled", 401)

    # Hash the new password
    password_hash, password_salt = hash_password(new_password)

    # Update password
    user_store.update_user(
        user.id,
        password_hash=password_hash,
        password_salt=password_salt,
    )

    # Invalidate all existing sessions
    user_store.increment_token_version(user.id)

    # Consume the reset token and invalidate any other tokens for this email
    store.consume_token(token)
    store.invalidate_tokens_for_email(email)

    logger.info(f"Password reset completed: email={email}, ip={client_ip}")

    # Audit log: password reset completed
    if AUDIT_AVAILABLE and audit_security:
        audit_security(
            event_type="encryption",
            actor_id=user.id,
            ip_address=client_ip,
            reason="password_reset_completed",
        )

    return json_response(
        {
            "message": "Password has been reset successfully. Please log in with your new password.",
            "sessions_invalidated": True,
        }
    )


__all__ = [
    "handle_change_password",
    "handle_forgot_password",
    "handle_reset_password",
    "send_password_reset_email",
]
