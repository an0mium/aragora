"""
Impersonation Session Enforcement Middleware.

Enforces impersonation session validation for all requests containing
the X-Impersonation-Session-ID header. This middleware ensures:

1. Session exists and is active
2. Session hasn't expired
3. Impersonator is authorized (session admin matches requester)
4. All impersonation actions are logged to audit trail
5. Impersonation context is added to request for downstream handlers

Security Features:
- Blocks requests with invalid/expired impersonation sessions (403)
- Logs all impersonation access attempts
- Adds impersonation context to request for audit trails
- Supports session refresh to extend active sessions

Usage:
    from aragora.server.middleware import (
        impersonation_middleware,
        require_valid_impersonation,
        get_impersonation_context,
    )

    # As middleware decorator (only enforces if header present)
    @impersonation_middleware
    def handle_request(handler):
        ctx = get_impersonation_context(handler)
        if ctx.is_impersonated:
            # Log that this action was via impersonation
            ...

    # As explicit requirement (fails if not impersonating)
    @require_valid_impersonation
    def admin_impersonation_action(handler):
        # Only reachable with valid impersonation session
        ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import TYPE_CHECKING, Any
from collections.abc import Callable

from aragora.auth.impersonation import (
    ImpersonationManager,
    ImpersonationSession,
    get_impersonation_manager,
)
from aragora.server.middleware.audit_logger import (
    AuditCategory,
    AuditSeverity,
    audit_event,
)
from aragora.server.middleware.user_auth import extract_client_ip, get_current_user

if TYPE_CHECKING:
    from aragora.server.handlers.base import HandlerResult

logger = logging.getLogger(__name__)


# Header name for impersonation session
IMPERSONATION_SESSION_HEADER = "X-Impersonation-Session-ID"

# Context key for storing impersonation info on request
IMPERSONATION_CONTEXT_KEY = "impersonation_context"


@dataclass
class ImpersonationContext:
    """
    Context information about the current impersonation state.

    Added to request context when impersonation header is present and validated.
    """

    is_impersonated: bool = False
    session_id: str | None = None
    admin_user_id: str | None = None
    admin_email: str | None = None
    target_user_id: str | None = None
    target_email: str | None = None
    reason: str | None = None
    started_at: datetime | None = None
    expires_at: datetime | None = None
    actions_performed: int = 0
    # Additional metadata for audit trails
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_session(cls, session: ImpersonationSession) -> ImpersonationContext:
        """Create context from an impersonation session."""
        return cls(
            is_impersonated=True,
            session_id=session.session_id,
            admin_user_id=session.admin_user_id,
            admin_email=session.admin_email,
            target_user_id=session.target_user_id,
            target_email=session.target_email,
            reason=session.reason,
            started_at=session.started_at,
            expires_at=session.expires_at,
            actions_performed=session.actions_performed,
        )

    def to_audit_dict(self) -> dict[str, Any]:
        """Convert to dictionary for audit logging."""
        return {
            "is_impersonated": self.is_impersonated,
            "session_id": self.session_id,
            "admin_user_id": self.admin_user_id,
            "admin_email": self.admin_email,
            "target_user_id": self.target_user_id,
            "target_email": self.target_email,
            "reason": self.reason,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "actions_performed": self.actions_performed,
        }


def _extract_handler(*args, **kwargs) -> Any:
    """Extract handler from function arguments."""
    handler = kwargs.get("handler")
    if handler is None:
        for arg in args:
            if hasattr(arg, "headers"):
                handler = arg
                break
    return handler


def _get_user_agent(handler: Any) -> str:
    """Extract User-Agent from handler."""
    if handler is None:
        return "unknown"
    headers = getattr(handler, "headers", {})
    if isinstance(headers, dict):
        return headers.get("User-Agent", headers.get("user-agent", "unknown"))
    # Handle header objects with get method
    if hasattr(headers, "get"):
        return headers.get("User-Agent") or headers.get("user-agent") or "unknown"
    return "unknown"


def _get_session_id_from_header(handler: Any) -> str | None:
    """Extract impersonation session ID from request header."""
    if handler is None:
        return None
    headers = getattr(handler, "headers", {})
    if isinstance(headers, dict):
        return headers.get(
            IMPERSONATION_SESSION_HEADER,
            headers.get(IMPERSONATION_SESSION_HEADER.lower()),
        )
    # Handle header objects with get method
    if hasattr(headers, "get"):
        return headers.get(IMPERSONATION_SESSION_HEADER) or headers.get(
            IMPERSONATION_SESSION_HEADER.lower()
        )
    return None


def _error_response(message: str, status: int = 403) -> HandlerResult:
    """Create an error response."""
    from aragora.server.handlers.base import error_response

    return error_response(message, status)


def _set_impersonation_context(handler: Any, ctx: ImpersonationContext) -> None:
    """Set impersonation context on handler for downstream use."""
    if handler is None:
        return

    # Try to set on handler.ctx dict
    handler_ctx = getattr(handler, "ctx", None)
    if handler_ctx is not None:
        if isinstance(handler_ctx, dict):
            handler_ctx[IMPERSONATION_CONTEXT_KEY] = ctx
        elif hasattr(handler_ctx, "__setattr__"):
            setattr(handler_ctx, IMPERSONATION_CONTEXT_KEY, ctx)

    # Also set directly on handler as fallback
    try:
        setattr(handler, IMPERSONATION_CONTEXT_KEY, ctx)
    except (AttributeError, TypeError):
        pass


def get_impersonation_context(handler: Any) -> ImpersonationContext:
    """
    Get impersonation context from handler.

    Returns empty context (is_impersonated=False) if not found.

    Args:
        handler: The HTTP handler object

    Returns:
        ImpersonationContext with impersonation details or empty context
    """
    if handler is None:
        return ImpersonationContext()

    # Try handler attribute first
    ctx = getattr(handler, IMPERSONATION_CONTEXT_KEY, None)
    if ctx is not None:
        return ctx

    # Try handler.ctx dict
    handler_ctx = getattr(handler, "ctx", None)
    if handler_ctx is not None:
        if isinstance(handler_ctx, dict):
            ctx = handler_ctx.get(IMPERSONATION_CONTEXT_KEY)
        elif hasattr(handler_ctx, IMPERSONATION_CONTEXT_KEY):
            ctx = getattr(handler_ctx, IMPERSONATION_CONTEXT_KEY)
        if ctx is not None:
            return ctx

    return ImpersonationContext()


def validate_impersonation_session(
    session_id: str,
    requester_user_id: str | None,
    ip_address: str,
    user_agent: str,
    manager: ImpersonationManager | None = None,
) -> tuple[ImpersonationSession | None, str | None]:
    """
    Validate an impersonation session.

    Args:
        session_id: The session ID to validate
        requester_user_id: User ID of the requester (must match session admin)
        ip_address: IP address of the request
        user_agent: User agent of the request
        manager: Optional impersonation manager (uses global if not provided)

    Returns:
        Tuple of (session or None, error_message or None)
    """
    if manager is None:
        manager = get_impersonation_manager()

    # Validate session exists and is not expired
    session = manager.validate_session(session_id)
    if session is None:
        audit_event(
            action="impersonation.session_invalid",
            actor=requester_user_id or "unknown",
            resource="impersonation/session",
            resource_id=session_id[:16] if session_id else None,
            outcome="denied",
            severity=AuditSeverity.WARNING,
            category=AuditCategory.AUTHORIZATION,
            details={
                "reason": "session_not_found_or_expired",
                "ip_address": ip_address,
                "user_agent": user_agent,
            },
        )
        return None, "Impersonation session not found or expired"

    # Verify the requester is the session admin (impersonator authorization)
    if requester_user_id and session.admin_user_id != requester_user_id:
        audit_event(
            action="impersonation.unauthorized_access",
            actor=requester_user_id,
            resource="impersonation/session",
            resource_id=session_id[:16],
            outcome="denied",
            severity=AuditSeverity.WARNING,
            category=AuditCategory.AUTHORIZATION,
            details={
                "reason": "requester_not_session_admin",
                "session_admin": session.admin_user_id,
                "ip_address": ip_address,
                "user_agent": user_agent,
            },
        )
        return None, "Not authorized for this impersonation session"

    return session, None


def log_impersonation_access(
    session: ImpersonationSession,
    action_type: str,
    ip_address: str,
    user_agent: str,
    additional_details: dict[str, Any] | None = None,
    manager: ImpersonationManager | None = None,
) -> None:
    """
    Log an impersonation access to the audit trail.

    Args:
        session: The impersonation session
        action_type: Type of action being performed
        ip_address: IP address of the request
        user_agent: User agent of the request
        additional_details: Additional details to include in audit
        manager: Optional impersonation manager
    """
    if manager is None:
        manager = get_impersonation_manager()

    details = {
        "action_type": action_type,
        "target_user_id": session.target_user_id,
        "target_email": session.target_email,
        **(additional_details or {}),
    }

    # Log via impersonation manager (increments action count)
    manager.log_impersonation_action(
        session_id=session.session_id,
        action_type=action_type,
        action_details=details,
        ip_address=ip_address,
        user_agent=user_agent,
    )

    # Also log via audit system
    audit_event(
        action=f"impersonation.{action_type}",
        actor=session.admin_user_id,
        resource="impersonation/action",
        resource_id=session.session_id[:16],
        outcome="success",
        severity=AuditSeverity.INFO,
        category=AuditCategory.AUTHORIZATION,
        details=details,
    )


def refresh_impersonation_session(
    session_id: str,
    requester_user_id: str,
    ip_address: str,
    user_agent: str,
    extension: timedelta | None = None,
    manager: ImpersonationManager | None = None,
) -> tuple[ImpersonationSession | None, str]:
    """
    Refresh an impersonation session to extend its expiration.

    Args:
        session_id: The session ID to refresh
        requester_user_id: User ID of the requester
        ip_address: IP address of the request
        user_agent: User agent of the request
        extension: How much to extend (defaults to DEFAULT_SESSION_DURATION)
        manager: Optional impersonation manager

    Returns:
        Tuple of (refreshed session or None, message)
    """
    if manager is None:
        manager = get_impersonation_manager()

    # First validate the session
    session, error = validate_impersonation_session(
        session_id=session_id,
        requester_user_id=requester_user_id,
        ip_address=ip_address,
        user_agent=user_agent,
        manager=manager,
    )

    if session is None:
        return None, error or "Session validation failed"

    # Calculate new expiration
    extension = extension or manager.DEFAULT_SESSION_DURATION
    new_expires = datetime.now(timezone.utc) + extension

    # Cap at MAX_SESSION_DURATION from original start
    max_expires = session.started_at + manager.MAX_SESSION_DURATION
    if new_expires > max_expires:
        new_expires = max_expires

    # Update session expiration
    session.expires_at = new_expires

    # Log the refresh
    audit_event(
        action="impersonation.session_refreshed",
        actor=requester_user_id,
        resource="impersonation/session",
        resource_id=session_id[:16],
        outcome="success",
        severity=AuditSeverity.INFO,
        category=AuditCategory.AUTHORIZATION,
        details={
            "new_expires_at": new_expires.isoformat(),
            "target_user_id": session.target_user_id,
            "ip_address": ip_address,
        },
    )

    return session, f"Session refreshed until {new_expires.isoformat()}"


def impersonation_middleware(func: Callable) -> Callable:
    """
    Middleware decorator that validates impersonation sessions.

    Only enforces validation if the X-Impersonation-Session-ID header is present.
    If the header is present but invalid, returns 403.
    If valid, adds ImpersonationContext to the request and logs the access.

    Usage:
        @impersonation_middleware
        def handle_request(handler):
            ctx = get_impersonation_context(handler)
            if ctx.is_impersonated:
                # Request is via impersonation
                ...
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        handler = _extract_handler(*args, **kwargs)

        # Check for impersonation header
        session_id = _get_session_id_from_header(handler)
        if not session_id:
            # No impersonation header - proceed normally with empty context
            _set_impersonation_context(handler, ImpersonationContext())
            return func(*args, **kwargs)

        # Get current user and request info
        current_user = get_current_user(handler)
        requester_user_id = current_user.id if current_user else None
        ip_address = extract_client_ip(handler) or "unknown"
        user_agent = _get_user_agent(handler)

        # Validate the impersonation session
        session, error = validate_impersonation_session(
            session_id=session_id,
            requester_user_id=requester_user_id,
            ip_address=ip_address,
            user_agent=user_agent,
        )

        if session is None:
            logger.warning(
                "Impersonation session validation failed: %s (session_id=%s...)", error, session_id[:16] if session_id else 'none'
            )
            return _error_response(error or "Invalid impersonation session", 403)

        # Create and set impersonation context
        ctx = ImpersonationContext.from_session(session)
        _set_impersonation_context(handler, ctx)

        # Log the impersonation access
        log_impersonation_access(
            session=session,
            action_type="request",
            ip_address=ip_address,
            user_agent=user_agent,
            additional_details={
                "endpoint": getattr(handler, "path", "unknown"),
                "method": getattr(handler, "method", "unknown"),
            },
        )

        return func(*args, **kwargs)

    return wrapper


def require_valid_impersonation(func: Callable) -> Callable:
    """
    Decorator that requires a valid impersonation session.

    Unlike impersonation_middleware, this decorator fails if no valid
    impersonation session is present. Use for endpoints that should
    only be accessible during impersonation.

    Usage:
        @require_valid_impersonation
        def end_impersonation_session(handler):
            # Only reachable with valid impersonation
            ctx = get_impersonation_context(handler)
            ...
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        handler = _extract_handler(*args, **kwargs)

        if handler is None:
            logger.warning("require_valid_impersonation: No handler provided")
            return _error_response("Internal server error", 500)

        # Check for impersonation header
        session_id = _get_session_id_from_header(handler)
        if not session_id:
            return _error_response(
                "Impersonation session required. Provide X-Impersonation-Session-ID header.",
                403,
            )

        # Get current user and request info
        current_user = get_current_user(handler)
        requester_user_id = current_user.id if current_user else None
        ip_address = extract_client_ip(handler) or "unknown"
        user_agent = _get_user_agent(handler)

        # Validate the impersonation session
        session, error = validate_impersonation_session(
            session_id=session_id,
            requester_user_id=requester_user_id,
            ip_address=ip_address,
            user_agent=user_agent,
        )

        if session is None:
            return _error_response(error or "Invalid impersonation session", 403)

        # Create and set impersonation context
        ctx = ImpersonationContext.from_session(session)
        _set_impersonation_context(handler, ctx)

        # Log the impersonation access
        log_impersonation_access(
            session=session,
            action_type="privileged_request",
            ip_address=ip_address,
            user_agent=user_agent,
            additional_details={
                "endpoint": getattr(handler, "path", "unknown"),
                "method": getattr(handler, "method", "unknown"),
            },
        )

        return func(*args, **kwargs)

    return wrapper


__all__ = [
    # Constants
    "IMPERSONATION_SESSION_HEADER",
    "IMPERSONATION_CONTEXT_KEY",
    # Context
    "ImpersonationContext",
    # Functions
    "get_impersonation_context",
    "validate_impersonation_session",
    "log_impersonation_access",
    "refresh_impersonation_session",
    # Decorators
    "impersonation_middleware",
    "require_valid_impersonation",
]
