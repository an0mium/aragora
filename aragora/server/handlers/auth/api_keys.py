"""
API Key Management Handlers.

Handles API key-related endpoints:
- POST /api/auth/api-key - Generate API key
- DELETE /api/auth/api-key - Revoke API key
- GET /api/auth/api-keys - List API keys
- DELETE /api/auth/api-keys/:prefix - Revoke API key by prefix
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aragora.billing.jwt_auth import extract_user_from_request

from ..base import HandlerResult, error_response, json_response, handle_errors
from ..utils.rate_limit import rate_limit

if TYPE_CHECKING:
    from .handler import AuthHandler

# Unified audit logging
try:
    from aragora.audit.unified import audit_admin

    AUDIT_AVAILABLE = True
except ImportError:
    AUDIT_AVAILABLE = False
    audit_admin = None

logger = logging.getLogger(__name__)


@rate_limit(requests_per_minute=3, limiter_name="auth_api_key_gen")
@handle_errors("generate API key")
def handle_generate_api_key(handler_instance: "AuthHandler", handler) -> HandlerResult:
    """Generate a new API key for the user."""
    # RBAC check: api_key.create permission required
    if error := handler_instance._check_permission(handler, "api_key.create"):
        return error

    # Get current user (already verified by _check_permission)
    user_store = handler_instance._get_user_store()
    auth_ctx = extract_user_from_request(handler, user_store)

    # Get user store
    if not user_store:
        return error_response("Authentication service unavailable", 503)

    # Get user
    user = user_store.get_user_by_id(auth_ctx.user_id)
    if not user:
        return error_response("User not found", 404)

    # Check if user's tier allows API access
    if user.org_id:
        org = user_store.get_organization_by_id(user.org_id)
        if org and not org.limits.api_access:
            return error_response("API access requires Professional tier or higher", 403)

    # Generate new API key using secure hash-based storage
    # The plaintext key is only returned once; we store the hash
    api_key = user.generate_api_key(expires_days=365)

    # Persist the hashed key fields (api_key_hash, api_key_prefix, expiry)
    user_store.update_user(
        user.id,
        api_key_hash=user.api_key_hash,
        api_key_prefix=user.api_key_prefix,
        api_key_created_at=user.api_key_created_at,
        api_key_expires_at=user.api_key_expires_at,
    )

    logger.info(f"API key generated for user: {user.email} (prefix: {user.api_key_prefix})")

    # Audit log: API key generated
    if AUDIT_AVAILABLE and audit_admin:
        audit_admin(
            admin_id=user.id,
            action="api_key_generated",
            target_type="api_key",
            target_id=user.api_key_prefix,
        )

    # Return the key (only shown once - plaintext is never stored)
    return json_response(
        {
            "api_key": api_key,
            "prefix": user.api_key_prefix,
            "expires_at": (
                user.api_key_expires_at.isoformat() if user.api_key_expires_at else None
            ),
            "message": "Save this key - it will not be shown again",
        }
    )


@rate_limit(requests_per_minute=5, limiter_name="auth_revoke_api_key")
@handle_errors("revoke API key")
def handle_revoke_api_key(handler_instance: "AuthHandler", handler) -> HandlerResult:
    """Revoke the user's API key."""
    # RBAC check: api_key.revoke permission required
    if error := handler_instance._check_permission(handler, "api_key.revoke"):
        return error

    # Get current user (already verified by _check_permission)
    user_store = handler_instance._get_user_store()
    auth_ctx = extract_user_from_request(handler, user_store)

    # Get user store
    if not user_store:
        return error_response("Authentication service unavailable", 503)

    # Get user
    user = user_store.get_user_by_id(auth_ctx.user_id)
    if not user:
        return error_response("User not found", 404)

    # Revoke API key - clear all hashed fields
    user_store.update_user(
        user.id,
        api_key_hash=None,
        api_key_prefix=None,
        api_key_created_at=None,
        api_key_expires_at=None,
    )

    logger.info(f"API key revoked for user: {user.email}")

    # Audit log: API key revoked
    if AUDIT_AVAILABLE and audit_admin:
        audit_admin(
            admin_id=user.id,
            action="api_key_revoked",
            target_type="api_key",
            target_id=user.id,
        )

    return json_response({"message": "API key revoked"})


@rate_limit(requests_per_minute=10, limiter_name="auth_list_api_keys")
@handle_errors("list API keys")
def handle_list_api_keys(handler_instance: "AuthHandler", handler) -> HandlerResult:
    """List API keys for the current user."""
    # RBAC check: reuse api_key.create permission for self-service listing
    if error := handler_instance._check_permission(handler, "api_key.create"):
        return error

    user_store = handler_instance._get_user_store()
    auth_ctx = extract_user_from_request(handler, user_store)

    if not user_store:
        return error_response("Authentication service unavailable", 503)

    user = user_store.get_user_by_id(auth_ctx.user_id)
    if not user:
        return error_response("User not found", 404)

    keys = []
    if user.api_key_prefix:
        keys.append(
            {
                "prefix": user.api_key_prefix,
                "created_at": (
                    user.api_key_created_at.isoformat() if user.api_key_created_at else None
                ),
                "expires_at": (
                    user.api_key_expires_at.isoformat() if user.api_key_expires_at else None
                ),
            }
        )

    return json_response({"keys": keys, "count": len(keys)})


@rate_limit(requests_per_minute=5, limiter_name="auth_revoke_api_key_prefix")
@handle_errors("revoke API key (prefix)")
def handle_revoke_api_key_prefix(
    handler_instance: "AuthHandler", handler, prefix: str
) -> HandlerResult:
    """Revoke the user's API key by prefix."""
    if error := handler_instance._check_permission(handler, "api_key.revoke"):
        return error

    user_store = handler_instance._get_user_store()
    auth_ctx = extract_user_from_request(handler, user_store)

    if not user_store:
        return error_response("Authentication service unavailable", 503)

    user = user_store.get_user_by_id(auth_ctx.user_id)
    if not user:
        return error_response("User not found", 404)

    if not user.api_key_prefix or user.api_key_prefix != prefix:
        return error_response("API key not found", 404)

    user_store.update_user(
        user.id,
        api_key_hash=None,
        api_key_prefix=None,
        api_key_created_at=None,
        api_key_expires_at=None,
    )

    logger.info(f"API key revoked for user: {user.email} (prefix: {prefix})")

    if AUDIT_AVAILABLE and audit_admin:
        audit_admin(
            admin_id=user.id,
            action="api_key_revoked",
            target_type="api_key",
            target_id=prefix,
        )

    return json_response({"message": "API key revoked"})


__all__ = [
    "handle_generate_api_key",
    "handle_revoke_api_key",
    "handle_list_api_keys",
    "handle_revoke_api_key_prefix",
]
