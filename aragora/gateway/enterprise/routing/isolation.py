"""
Data Isolation for Multi-Tenant Router.

Provides tenant data isolation, context building, and cross-tenant access control.

Features:
- Cross-tenant access prevention
- Tenant context propagation
- Isolation level enforcement
- Context hash generation for verification

Usage:
    from aragora.gateway.enterprise.routing.isolation import (
        CrossTenantAccessError,
        TenantContextBuilder,
        validate_tenant_access,
    )

    # Build tenant context for external requests
    builder = TenantContextBuilder()
    context = builder.build_context(tenant_id, config, request)

    # Validate tenant access
    validate_tenant_access(requesting_tenant, target_tenant)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from aragora.tenancy.context import (
    TenantContext,
    get_current_tenant_id,
)
from aragora.tenancy.isolation import IsolationLevel


# =============================================================================
# Exceptions
# =============================================================================


class CrossTenantAccessError(Exception):
    """Raised when cross-tenant data access is attempted."""

    def __init__(
        self,
        requesting_tenant: str,
        target_tenant: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        message = f"Cross-tenant access denied: {requesting_tenant} -> {target_tenant}"
        super().__init__(message)
        self.message = message
        self.requesting_tenant = requesting_tenant
        self.target_tenant = target_tenant
        self.code = "CROSS_TENANT_ACCESS"
        self.details = {
            "requesting_tenant": requesting_tenant,
            "target_tenant": target_tenant,
            **(details or {}),
        }


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class IsolationConfig:
    """
    Configuration for tenant isolation.

    Attributes:
        level: Isolation level for the tenant.
        allow_cross_tenant_read: Whether cross-tenant reads are allowed.
        allow_cross_tenant_write: Whether cross-tenant writes are allowed.
        allowed_peer_tenants: Set of tenant IDs allowed for cross-tenant access.
        audit_all_access: Whether to audit all data access.
    """

    level: IsolationLevel = IsolationLevel.STRICT
    allow_cross_tenant_read: bool = False
    allow_cross_tenant_write: bool = False
    allowed_peer_tenants: set[str] = field(default_factory=set)
    audit_all_access: bool = True


@dataclass
class TenantAccessContext:
    """
    Context for tenant data access.

    Attributes:
        tenant_id: The tenant ID.
        isolation_level: The isolation level.
        tenant_hash: Hash for tenant verification.
        correlation_id: Request correlation ID.
        headers: HTTP headers for the request.
        timestamp: When the context was created.
    """

    tenant_id: str
    isolation_level: str
    tenant_hash: str
    correlation_id: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


# =============================================================================
# Tenant Context Builder
# =============================================================================


class TenantContextBuilder:
    """
    Builds tenant context for propagation to external services.
    """

    def build_context(
        self,
        tenant_id: str,
        isolation_level: IsolationLevel,
        context_headers: dict[str, str] | None = None,
        request: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        """
        Build tenant context for propagation.

        Args:
            tenant_id: Tenant identifier.
            isolation_level: Isolation level for the tenant.
            context_headers: Additional headers to include.
            request: Original request data.

        Returns:
            Dictionary of tenant context values.
        """
        context = {
            "X-Tenant-ID": tenant_id,
            "X-Aragora-Tenant": tenant_id,
            "X-Isolation-Level": isolation_level.value,
        }

        # Add any configured context headers
        if context_headers:
            context.update(context_headers)

        # Add request correlation ID if present
        if request and "correlation_id" in request:
            context["X-Correlation-ID"] = str(request["correlation_id"])

        # Add tenant hash for verification
        tenant_hash = self.compute_tenant_hash(tenant_id)
        context["X-Tenant-Hash"] = tenant_hash

        return context

    def build_headers(
        self,
        tenant_id: str,
        context_headers: dict[str, str] | None = None,
        endpoint_headers: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """
        Build request headers for external request.

        Args:
            tenant_id: Tenant identifier.
            context_headers: Tenant context headers.
            endpoint_headers: Endpoint-specific headers.

        Returns:
            Dictionary of request headers.
        """
        headers: dict[str, str] = {}

        # Add endpoint-specific headers
        if endpoint_headers:
            headers.update(endpoint_headers)

        # Add tenant context headers
        headers["X-Tenant-ID"] = tenant_id
        headers["X-Aragora-Tenant"] = tenant_id

        # Add any configured context headers
        if context_headers:
            headers.update(context_headers)

        # Add routing metadata
        headers["X-Aragora-Router"] = "TenantRouter"
        headers["X-Aragora-Timestamp"] = datetime.now(timezone.utc).isoformat()

        return headers

    @staticmethod
    def compute_tenant_hash(tenant_id: str) -> str:
        """
        Compute a verification hash for a tenant ID.

        Args:
            tenant_id: Tenant identifier.

        Returns:
            First 16 characters of SHA-256 hash.
        """
        return hashlib.sha256(tenant_id.encode()).hexdigest()[:16]


# =============================================================================
# Access Validation
# =============================================================================


def validate_tenant_access(
    requesting_tenant: str | None,
    target_tenant: str,
    config: IsolationConfig | None = None,
) -> None:
    """
    Validate that a tenant access is allowed.

    Args:
        requesting_tenant: The tenant making the request (from context).
        target_tenant: The tenant being accessed.
        config: Optional isolation configuration.

    Raises:
        CrossTenantAccessError: If cross-tenant access is not allowed.
    """
    # If no context tenant, allow (will be set by the router)
    if requesting_tenant is None:
        return

    # Same tenant is always allowed
    if requesting_tenant == target_tenant:
        return

    # Check if cross-tenant is allowed by config
    if config is not None:
        if config.allow_cross_tenant_read:
            return
        if target_tenant in config.allowed_peer_tenants:
            return

    # Cross-tenant access denied
    raise CrossTenantAccessError(requesting_tenant, target_tenant)


def get_tenant_from_context() -> str | None:
    """
    Get the current tenant ID from context.

    Returns:
        Current tenant ID or None if not set.
    """
    return get_current_tenant_id()


# =============================================================================
# Tenant Routing Context
# =============================================================================


class TenantRoutingContextManager:
    """
    Context manager for tenant-scoped routing.

    Ensures tenant context is set and request completion is tracked.

    Usage:
        async with TenantRoutingContextManager(router, "acme-corp") as ctx:
            decision = await ctx.route(request)
            response = await make_external_request(decision)
            await ctx.complete(success=True, latency_ms=response.elapsed_ms)
    """

    def __init__(
        self,
        router: Any,  # TenantRouter - avoid circular import
        tenant_id: str,
    ) -> None:
        """
        Initialize routing context.

        Args:
            router: TenantRouter instance.
            tenant_id: Tenant identifier.
        """
        self._router = router
        self._tenant_id = tenant_id
        self._tenant_context: TenantContext | None = None
        self._decision: Any | None = None  # RoutingDecision
        self._completed = False

    async def __aenter__(self) -> TenantRoutingContextManager:
        """Enter tenant routing context."""
        self._tenant_context = TenantContext(tenant_id=self._tenant_id)
        await self._tenant_context.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit tenant routing context."""
        # Auto-complete with failure if not explicitly completed and exception occurred
        if not self._completed and self._decision and exc_type is not None:
            await self.complete(
                success=False,
                latency_ms=0,
                error=str(exc_val) if exc_val else "Unknown error",
            )

        if self._tenant_context:
            await self._tenant_context.__aexit__(exc_type, exc_val, exc_tb)

    async def route(
        self,
        request: dict[str, Any] | None = None,
        operation: str | None = None,
        bytes_size: int = 0,
    ) -> Any:  # RoutingDecision
        """
        Route a request within this tenant context.

        Args:
            request: Request data for context.
            operation: Operation type for permission checking.
            bytes_size: Size of request in bytes.

        Returns:
            RoutingDecision with target endpoint.
        """
        self._decision = await self._router.route(
            tenant_id=self._tenant_id,
            request=request,
            operation=operation,
            bytes_size=bytes_size,
        )
        return self._decision

    async def complete(
        self,
        success: bool,
        latency_ms: float,
        error: str | None = None,
    ) -> None:
        """
        Mark the request as complete.

        Args:
            success: Whether the request succeeded.
            latency_ms: Request latency in milliseconds.
            error: Error message if request failed.
        """
        if self._decision and not self._completed:
            await self._router.complete_request(
                tenant_id=self._tenant_id,
                endpoint_url=self._decision.target_endpoint,
                success=success,
                latency_ms=latency_ms,
                error=error,
            )
            self._completed = True


__all__ = [
    "CrossTenantAccessError",
    "IsolationConfig",
    "TenantAccessContext",
    "TenantContextBuilder",
    "TenantRoutingContextManager",
    "validate_tenant_access",
    "get_tenant_from_context",
]
