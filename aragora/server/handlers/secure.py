"""
Secure Handler Base Class.

Provides a security-enhanced base handler that automatically:
- Extracts and verifies authentication from JWT tokens
- Enforces RBAC permissions on endpoints
- Logs security events to the audit trail
- Emits security metrics
- Handles encryption/decryption of sensitive fields

Usage:
    from aragora.server.handlers.secure import SecureHandler, secure_endpoint

    class MyHandler(SecureHandler):

        @secure_endpoint(permission="myresource.read")
        async def handle_get(self, request, auth_context):
            # auth_context is automatically injected and verified
            return json_response({"data": "..."})

        @secure_endpoint(permission="myresource.write", audit=True)
        async def handle_post(self, request, auth_context):
            # This action will be logged to the audit trail
            return json_response({"created": True})
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, Optional, TypeVar, ParamSpec

from aragora.rbac.models import AuthorizationContext
from aragora.rbac.decorators import PermissionDeniedError, RoleRequiredError

from .base import BaseHandler, HandlerResult, error_response, ServerContext
from .utils.auth import get_auth_context, UnauthorizedError, ForbiddenError

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


class SecureHandler(BaseHandler):
    """
    Security-enhanced base handler.

    Extends BaseHandler with automatic security features including:
    - JWT-based authentication extraction
    - RBAC permission enforcement
    - Security audit logging
    - Metrics emission

    Subclasses should use the @secure_endpoint decorator on methods
    that require security enforcement.
    """

    # Default permission requirements for HTTP methods
    # Subclasses can override these
    DEFAULT_METHOD_PERMISSIONS: dict[str, str | None] = {
        "GET": None,  # Read operations may not require permission
        "POST": None,  # Subclass should define
        "PUT": None,
        "PATCH": None,
        "DELETE": None,
    }

    # Resource type for audit logging (subclasses should override)
    RESOURCE_TYPE: str = "unknown"

    def __init__(self, server_context: ServerContext):
        """Initialize with server context."""
        super().__init__(server_context)
        self._auth_context: Optional[AuthorizationContext] = None

    async def get_auth_context(
        self,
        request: Any,
        require_auth: bool = True,
    ) -> AuthorizationContext:
        """
        Get authentication context for the current request.

        Args:
            request: HTTP request object
            require_auth: If True, raises error when not authenticated

        Returns:
            AuthorizationContext with user information

        Raises:
            UnauthorizedError: If require_auth=True and not authenticated
        """
        return await get_auth_context(request, require_auth=require_auth)

    def check_permission(
        self,
        auth_context: AuthorizationContext,
        permission: str,
        resource_id: Optional[str] = None,
    ) -> bool:
        """
        Check if user has a specific permission.

        Args:
            auth_context: User's authorization context
            permission: Permission key to check
            resource_id: Optional resource ID for resource-specific checks

        Returns:
            True if permission is granted

        Raises:
            ForbiddenError: If permission is denied
        """
        from aragora.rbac.checker import get_permission_checker
        from aragora.observability.metrics.security import record_rbac_decision

        checker = get_permission_checker()
        decision = checker.check_permission(auth_context, permission, resource_id)

        record_rbac_decision(permission, decision.allowed)

        if not decision.allowed:
            raise ForbiddenError(
                f"Permission denied: {permission}",
                permission=permission,
            )

        return True

    async def audit_action(
        self,
        auth_context: AuthorizationContext,
        action: str,
        resource_id: str,
        resource_type: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
        request: Optional[Any] = None,
    ) -> None:
        """
        Log an action to the security audit trail.

        Args:
            auth_context: User's authorization context
            action: Action performed (e.g., "create", "update", "delete")
            resource_id: ID of the affected resource
            resource_type: Type of resource (defaults to RESOURCE_TYPE)
            details: Additional details to log
            request: Optional request for IP/user-agent extraction
        """
        from aragora.observability.immutable_log import get_audit_log

        ip_address = None
        user_agent = None

        if request and hasattr(request, "headers"):
            ip_address = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
            if not ip_address and hasattr(request, "remote"):
                ip_address = request.remote
            user_agent = request.headers.get("User-Agent")

        await get_audit_log().append(
            event_type=f"{resource_type or self.RESOURCE_TYPE}.{action}",
            actor=auth_context.user_id,
            actor_type="user",
            resource_type=resource_type or self.RESOURCE_TYPE,
            resource_id=resource_id,
            action=action,
            workspace_id=auth_context.workspace_id,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent,
        )

    def encrypt_response_fields(
        self,
        data: dict[str, Any],
        fields: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Encrypt sensitive fields in response data.

        This is typically NOT needed as we don't send encrypted data to clients.
        But useful for data being stored or passed to other services.

        Args:
            data: Data dictionary
            fields: Specific fields to encrypt (default: auto-detect)

        Returns:
            Data with specified fields encrypted
        """
        from aragora.storage.encrypted_fields import encrypt_sensitive

        return encrypt_sensitive(data)

    def decrypt_request_fields(
        self,
        data: dict[str, Any],
        fields: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Decrypt sensitive fields from stored/received data.

        Args:
            data: Data dictionary with encrypted fields
            fields: Specific fields to decrypt (default: auto-detect)

        Returns:
            Data with fields decrypted
        """
        from aragora.storage.encrypted_fields import decrypt_sensitive

        return decrypt_sensitive(data)

    def handle_security_error(
        self,
        error: Exception,
        request: Optional[Any] = None,
    ) -> HandlerResult:
        """
        Handle security-related errors and return appropriate response.

        Args:
            error: The security exception
            request: Optional request for context

        Returns:
            Appropriate error response
        """
        from aragora.observability.metrics.security import (
            record_auth_failure,
            record_blocked_request,
        )

        if isinstance(error, UnauthorizedError):
            record_auth_failure("jwt", "invalid_token")
            return error_response("Authentication required", 401)

        if isinstance(error, ForbiddenError):
            record_blocked_request("permission_denied", "user")
            return error_response(
                f"Access denied: {error.permission or 'insufficient permissions'}",
                403,
            )

        if isinstance(error, PermissionDeniedError):
            record_blocked_request("rbac_denied", "user")
            return error_response(
                f"Permission denied: {error.permission_key or 'unknown'}",
                403,
            )

        if isinstance(error, RoleRequiredError):
            record_blocked_request("role_required", "user")
            return error_response(
                f"Required role not found: {error.required_roles}",
                403,
            )

        # Unknown security error
        logger.error(f"Unexpected security error: {error}")
        return error_response("Security error", 500)


def secure_endpoint(
    permission: Optional[str] = None,
    require_auth: bool = True,
    audit: bool = False,
    audit_action: Optional[str] = None,
    resource_id_param: Optional[str] = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for secure endpoint methods.

    Automatically handles:
    - Authentication extraction and verification
    - Permission checking
    - Audit logging
    - Error handling

    Args:
        permission: Required permission (e.g., "debates.create")
        require_auth: Whether authentication is required
        audit: Whether to log the action to audit trail
        audit_action: Custom action name for audit (default: method name)
        resource_id_param: Parameter name containing resource ID

    Usage:
        class MyHandler(SecureHandler):
            @secure_endpoint(permission="items.read")
            async def handle_get(self, request, auth_context, **kwargs):
                return json_response({"items": []})

            @secure_endpoint(permission="items.create", audit=True)
            async def handle_post(self, request, auth_context, **kwargs):
                # Creates are audited
                return json_response({"created": True})
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def wrapper(
            self: SecureHandler,
            request: Any,
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> T:
            from aragora.observability.metrics.security import (
                record_auth_attempt,
                track_rbac_evaluation,
            )

            start_time = time.perf_counter()

            try:
                # 1. Extract authentication
                auth_context = await self.get_auth_context(
                    request,
                    require_auth=require_auth,
                )

                record_auth_attempt(
                    "jwt",
                    success=auth_context.user_id != "anonymous",
                )

                # 2. Check permission if specified
                if permission:
                    resource_id = None
                    if resource_id_param:
                        resource_id = kwargs.get(resource_id_param)

                    with track_rbac_evaluation():
                        self.check_permission(auth_context, permission, resource_id)

                # 3. Call the actual handler
                result = await func(self, request, auth_context, *args, **kwargs)

                # 4. Audit if requested
                if audit:
                    action_name = audit_action or func.__name__.replace("handle_", "")
                    resource_id = kwargs.get(resource_id_param, "unknown")
                    await self.audit_action(
                        auth_context,
                        action=action_name,
                        resource_id=str(resource_id),
                        request=request,
                        details={
                            "duration_ms": (time.perf_counter() - start_time) * 1000,
                        },
                    )

                return result

            except (UnauthorizedError, ForbiddenError, PermissionDeniedError, RoleRequiredError) as e:
                return self.handle_security_error(e, request)

            except Exception as e:
                logger.exception(f"Error in secure endpoint {func.__name__}: {e}")
                raise

        return wrapper  # type: ignore

    return decorator


def audit_sensitive_access(
    resource_type: str,
    action: str = "access",
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to audit access to sensitive data.

    Use this for endpoints that access sensitive information
    like API keys, tokens, or PII.

    Args:
        resource_type: Type of sensitive resource being accessed
        action: Action being performed

    Usage:
        @audit_sensitive_access("api_key", "read")
        async def get_api_key(self, request, auth_context):
            ...
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def wrapper(
            self: SecureHandler,
            request: Any,
            auth_context: AuthorizationContext,
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> T:
            from aragora.observability.metrics.security import record_secret_access
            from aragora.observability.security_audit import audit_secret_access

            # Record metric
            record_secret_access(resource_type, action)

            # Log to audit trail
            await audit_secret_access(
                actor=auth_context.user_id,
                secret_type=resource_type,
                store=self.RESOURCE_TYPE,
                operation=action,
                workspace_id=auth_context.workspace_id,
            )

            return await func(self, request, auth_context, *args, **kwargs)

        return wrapper  # type: ignore

    return decorator


# Export commonly used items
__all__ = [
    "SecureHandler",
    "secure_endpoint",
    "audit_sensitive_access",
    "UnauthorizedError",
    "ForbiddenError",
]
