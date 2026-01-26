"""
MFA Enforcement Middleware.

Provides decorators and utilities for enforcing Multi-Factor Authentication
on sensitive routes, particularly admin endpoints.

SOC 2 Control: CC5-01 - Enforce MFA for administrative access

Usage:
    from aragora.server.middleware.mfa import require_mfa, require_admin_mfa

    @require_admin_mfa
    def admin_endpoint(self, handler, user: User):
        # Only accessible if user is admin AND has MFA enabled
        return {"admin": True}
"""

import logging
from functools import wraps
from typing import Any, Callable, Optional

from aragora.server.middleware.user_auth import User, get_current_user

logger = logging.getLogger(__name__)


def require_mfa(func: Callable) -> Callable:
    """
    Decorator that requires MFA to be enabled for the current user.

    This is for routes where ALL users must have MFA enabled, regardless
    of their role.

    Usage:
        @require_mfa
        def sensitive_endpoint(self, handler, user: User):
            ...
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        from aragora.server.handlers.base import error_response

        # Extract handler from args/kwargs
        handler = kwargs.get("handler")
        if handler is None:
            for arg in args:
                if hasattr(arg, "headers"):
                    handler = arg
                    break

        if handler is None:
            return error_response("No request handler", 500)

        # Authenticate user
        user = get_current_user(handler)
        if not user:
            return error_response("Authentication required", 401)

        # Check MFA status via user store
        user_store = _get_user_store_from_handler(handler)
        if user_store:
            full_user = user_store.get_user_by_id(user.id)
            if full_user:
                if not getattr(full_user, "mfa_enabled", False):
                    logger.warning(f"MFA required but not enabled for user: {user.id}")
                    return error_response(
                        "MFA required. Please enable MFA at /api/auth/mfa/setup",
                        403,
                        code="MFA_REQUIRED",
                    )
        else:
            # If no user store, check metadata (fallback)
            mfa_enabled = user.metadata.get("mfa_enabled", False)
            if not mfa_enabled:
                logger.warning(f"MFA required but status unknown for user: {user.id}")
                return error_response(
                    "MFA required. Please enable MFA at /api/auth/mfa/setup",
                    403,
                    code="MFA_REQUIRED",
                )

        kwargs["user"] = user
        return func(*args, **kwargs)

    return wrapper


def require_admin_mfa(func: Callable) -> Callable:
    """
    Decorator that requires MFA for admin/owner users.

    Non-admin users are allowed without MFA (they get standard require_user behavior).
    Admin and owner roles MUST have MFA enabled.

    This implements SOC 2 Control CC5-01: Enforce MFA for administrative access.

    Usage:
        @require_admin_mfa
        def admin_endpoint(self, handler, user: User):
            # Admins must have MFA, regular users don't need it
            ...
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        from aragora.server.handlers.base import error_response

        # Extract handler from args/kwargs
        handler = kwargs.get("handler")
        if handler is None:
            for arg in args:
                if hasattr(arg, "headers"):
                    handler = arg
                    break

        if handler is None:
            return error_response("No request handler", 500)

        # Authenticate user
        user = get_current_user(handler)
        if not user:
            return error_response("Authentication required", 401)

        # Check if user is admin/owner
        admin_roles = {"admin", "owner", "superadmin"}
        if user.role in admin_roles:
            # Admin users MUST have MFA enabled
            user_store = _get_user_store_from_handler(handler)
            mfa_enabled = False

            if user_store:
                full_user = user_store.get_user_by_id(user.id)
                if full_user:
                    mfa_enabled = getattr(full_user, "mfa_enabled", False)
            else:
                # Fallback to metadata
                mfa_enabled = user.metadata.get("mfa_enabled", False)

            if not mfa_enabled:
                logger.warning(
                    f"Admin MFA enforcement: user {user.id} ({user.role}) "
                    f"attempted admin access without MFA"
                )
                return error_response(
                    "Administrative access requires MFA. Please enable MFA at /api/auth/mfa/setup",
                    403,
                    code="ADMIN_MFA_REQUIRED",
                )

        kwargs["user"] = user
        return func(*args, **kwargs)

    return wrapper


def require_admin_with_mfa(func: Callable) -> Callable:
    """
    Decorator that requires BOTH admin role AND MFA enabled.

    Use this for the most sensitive admin operations.

    Usage:
        @require_admin_with_mfa
        def delete_all_data(self, handler, user: User):
            # Only admins with MFA can access
            ...
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        from aragora.server.handlers.base import error_response

        # Extract handler from args/kwargs
        handler = kwargs.get("handler")
        if handler is None:
            for arg in args:
                if hasattr(arg, "headers"):
                    handler = arg
                    break

        if handler is None:
            return error_response("No request handler", 500)

        # Authenticate user
        user = get_current_user(handler)
        if not user:
            return error_response("Authentication required", 401)

        # Require admin role
        if not user.is_admin:
            return error_response("Admin access required", 403)

        # Require MFA
        user_store = _get_user_store_from_handler(handler)
        mfa_enabled = False

        if user_store:
            full_user = user_store.get_user_by_id(user.id)
            if full_user:
                mfa_enabled = getattr(full_user, "mfa_enabled", False)
        else:
            mfa_enabled = user.metadata.get("mfa_enabled", False)

        if not mfa_enabled:
            logger.warning(
                f"Admin+MFA enforcement: admin {user.id} attempted sensitive operation without MFA"
            )
            return error_response(
                "This operation requires MFA. Please enable MFA at /api/auth/mfa/setup",
                403,
                code="MFA_REQUIRED",
            )

        kwargs["user"] = user
        return func(*args, **kwargs)

    return wrapper


def check_mfa_status(user_id: str, user_store: Any) -> dict:
    """
    Check MFA status for a user.

    Returns:
        dict with mfa_enabled, mfa_secret_set, backup_codes_remaining
    """
    if not user_store:
        return {
            "mfa_enabled": False,
            "mfa_secret_set": False,
            "backup_codes_remaining": 0,
            "error": "User store not available",
        }

    user = user_store.get_user_by_id(user_id)
    if not user:
        return {
            "mfa_enabled": False,
            "mfa_secret_set": False,
            "backup_codes_remaining": 0,
            "error": "User not found",
        }

    backup_count = 0
    if getattr(user, "mfa_backup_codes", None):
        import json

        try:
            backup_hashes = json.loads(user.mfa_backup_codes)
            backup_count = len(backup_hashes)
        except (json.JSONDecodeError, TypeError):
            pass

    return {
        "mfa_enabled": getattr(user, "mfa_enabled", False),
        "mfa_secret_set": bool(getattr(user, "mfa_secret", None)),
        "backup_codes_remaining": backup_count,
    }


def enforce_admin_mfa_policy(
    user: User,
    user_store: Any,
    grace_period_days: Optional[int] = None,
) -> Optional[dict]:
    """
    Check if admin user complies with MFA policy.

    SOC 2 Control: CC5-01 - Enforce MFA for administrative access.

    Args:
        user: The authenticated user
        user_store: User storage backend
        grace_period_days: Days to allow before enforcing (uses settings if None)

    Returns:
        None if compliant, or dict with enforcement details if not
    """
    # Load settings
    from aragora.config.settings import get_settings

    settings = get_settings()

    # Check if MFA enforcement is enabled
    if not settings.security.admin_mfa_required:
        return None  # MFA enforcement disabled

    # Use settings-based grace period if not explicitly provided
    if grace_period_days is None:
        grace_period_days = settings.security.admin_mfa_grace_period_days

    admin_roles = {"admin", "owner", "superadmin"}
    if user.role not in admin_roles:
        return None  # Non-admins are always compliant

    status = check_mfa_status(user.id, user_store)

    if status.get("mfa_enabled"):
        # Check backup codes
        if status.get("backup_codes_remaining", 0) < 3:
            return {
                "compliant": True,
                "warning": "Low backup codes",
                "backup_codes_remaining": status["backup_codes_remaining"],
                "action": "Consider regenerating backup codes",
            }
        return None  # Fully compliant

    # MFA not enabled - check grace period
    from datetime import datetime, timedelta, timezone

    if user_store:
        full_user = user_store.get_user_by_id(user.id)
        if full_user:
            # Prefer mfa_grace_period_started_at (set when user becomes admin)
            # Fall back to created_at for backward compatibility
            grace_start = getattr(full_user, "mfa_grace_period_started_at", None)
            if grace_start is None:
                grace_start = getattr(full_user, "created_at", None)

            if grace_start:
                if isinstance(grace_start, str):
                    try:
                        grace_start = datetime.fromisoformat(grace_start.replace("Z", "+00:00"))
                    except ValueError:
                        grace_start = None

                if grace_start:
                    grace_end = grace_start + timedelta(days=grace_period_days)
                    now = datetime.now(timezone.utc)
                    # Ensure timezone-aware comparison
                    if grace_start.tzinfo is None:
                        now = datetime.utcnow()

                    if now < grace_end:
                        days_remaining = (grace_end - now).days
                        return {
                            "compliant": False,
                            "enforced": False,
                            "grace_period_remaining_days": days_remaining,
                            "grace_period_started_at": grace_start.isoformat(),
                            "action": "Please enable MFA before grace period ends",
                        }

    # No grace period or expired
    return {
        "compliant": False,
        "enforced": True,
        "action": "MFA is required for admin access",
    }


def _get_user_store_from_handler(handler: Any) -> Any:
    """
    Extract user store from handler context.

    The user store is typically injected via the handler's context.
    """
    # Try common patterns for accessing user store
    if hasattr(handler, "ctx"):
        ctx = handler.ctx
        if isinstance(ctx, dict):
            return ctx.get("user_store")
        if hasattr(ctx, "user_store"):
            return ctx.user_store

    if hasattr(handler, "server"):
        server = handler.server
        if hasattr(server, "ctx"):
            ctx = server.ctx
            if isinstance(ctx, dict):
                return ctx.get("user_store")

    if hasattr(handler, "app"):
        app = handler.app
        if hasattr(app, "ctx"):
            return app.ctx.get("user_store")
        if hasattr(app, "user_store"):
            return app.user_store

    return None


def require_mfa_fresh(max_age_minutes: int = 15) -> Callable:
    """
    Decorator that requires recent MFA verification for sensitive operations.

    Use this for step-up authentication on sensitive operations like:
    - User role changes
    - API key generation
    - Payment method changes
    - Security settings modifications

    Args:
        max_age_minutes: Maximum age of MFA verification (default: 15)

    Usage:
        @require_mfa_fresh(max_age_minutes=10)
        def sensitive_operation(self, handler, user: User):
            # Only accessible if MFA was verified in the last 10 minutes
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            from aragora.server.handlers.base import error_response

            # Extract handler from args/kwargs
            handler = kwargs.get("handler")
            if handler is None:
                for arg in args:
                    if hasattr(arg, "headers"):
                        handler = arg
                        break

            if handler is None:
                return error_response("No request handler", 500)

            # Authenticate user
            user = get_current_user(handler)
            if not user:
                return error_response("Authentication required", 401)

            # Check MFA is enabled
            user_store = _get_user_store_from_handler(handler)
            mfa_enabled = False

            if user_store:
                full_user = user_store.get_user_by_id(user.id)
                if full_user:
                    mfa_enabled = getattr(full_user, "mfa_enabled", False)
            else:
                mfa_enabled = user.metadata.get("mfa_enabled", False)

            if not mfa_enabled:
                return error_response(
                    "MFA must be enabled for this operation. "
                    "Please enable MFA at /api/auth/mfa/setup",
                    403,
                    code="MFA_REQUIRED",
                )

            # Check MFA freshness
            session_manager = _get_session_manager_from_handler(handler)
            token_jti = getattr(user, "token_jti", None) or user.metadata.get("jti")

            if session_manager and token_jti:
                max_age_seconds = max_age_minutes * 60
                if not session_manager.is_session_mfa_fresh(user.id, token_jti, max_age_seconds):
                    logger.warning(
                        f"MFA step-up required for user {user.id}: "
                        f"MFA not fresh (max age: {max_age_minutes} min)"
                    )
                    return error_response(
                        "This operation requires recent MFA verification. "
                        "Please re-verify MFA at /api/auth/mfa/verify",
                        403,
                        code="MFA_STEP_UP_REQUIRED",
                    )
            else:
                # No session tracking available - require MFA verification
                logger.warning(
                    f"MFA step-up required but session tracking unavailable for user {user.id}"
                )
                return error_response(
                    "This operation requires MFA verification. "
                    "Please verify MFA at /api/auth/mfa/verify",
                    403,
                    code="MFA_STEP_UP_REQUIRED",
                )

            kwargs["user"] = user
            return func(*args, **kwargs)

        return wrapper

    return decorator


def _get_session_manager_from_handler(handler: Any) -> Any:
    """Extract session manager from handler context."""
    # Try common patterns for accessing session manager
    if hasattr(handler, "ctx"):
        ctx = handler.ctx
        if isinstance(ctx, dict):
            return ctx.get("session_manager")
        if hasattr(ctx, "session_manager"):
            return ctx.session_manager

    if hasattr(handler, "server"):
        server = handler.server
        if hasattr(server, "ctx"):
            ctx = server.ctx
            if isinstance(ctx, dict):
                return ctx.get("session_manager")

    if hasattr(handler, "app"):
        app = handler.app
        if hasattr(app, "ctx"):
            ctx = app.ctx
            if isinstance(ctx, dict):
                return ctx.get("session_manager")
        if hasattr(app, "session_manager"):
            return app.session_manager

    # Fallback: try to get the global session manager
    try:
        from aragora.billing.auth.sessions import get_session_manager

        return get_session_manager()
    except Exception:
        return None


__all__ = [
    "require_mfa",
    "require_admin_mfa",
    "require_admin_with_mfa",
    "require_mfa_fresh",
    "check_mfa_status",
    "enforce_admin_mfa_policy",
]
