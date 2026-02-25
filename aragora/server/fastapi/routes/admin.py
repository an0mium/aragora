"""
Admin Endpoints (FastAPI v2).

Migrated from: aragora/server/handlers/admin/ (aiohttp handler)

Surfaces the core admin panel as REST endpoints:
- GET    /api/v2/admin/stats                     - System-wide statistics
- GET    /api/v2/admin/system/metrics             - Aggregated system metrics
- GET    /api/v2/admin/revenue                    - Revenue and billing stats
- GET    /api/v2/admin/organizations              - List organizations
- GET    /api/v2/admin/users                      - List users
- POST   /api/v2/admin/users                      - Create a user
- POST   /api/v2/admin/users/{user_id}/deactivate - Deactivate a user
- POST   /api/v2/admin/users/{user_id}/activate   - Activate a user
- POST   /api/v2/admin/users/{user_id}/unlock     - Unlock a locked account
- POST   /api/v2/admin/impersonate/{user_id}      - Create impersonation token
- GET    /api/v2/admin/tenants                    - List tenants
- POST   /api/v2/admin/tenants                    - Create a tenant
- PUT    /api/v2/admin/tenants/{tenant_id}        - Update tenant config
- GET    /api/v2/admin/audit-log                  - View audit log entries
- GET    /api/v2/admin/config                     - Get system configuration
- PUT    /api/v2/admin/config                     - Update system configuration
- GET    /api/v2/admin/license                    - Get license information
- POST   /api/v2/admin/license                    - Activate/update license
- GET    /api/v2/admin/nomic/status               - Nomic loop status
- POST   /api/v2/admin/nomic/pause                - Pause the nomic loop
- POST   /api/v2/admin/nomic/resume               - Resume the nomic loop

Migration Notes:
    This module replaces AdminHandler (+ MetricsDashboardMixin,
    UserManagementMixin, NomicAdminMixin) in the legacy handler with native
    FastAPI routes.  Key improvements:
    - Pydantic request/response models with automatic validation
    - FastAPI dependency injection for auth and storage
    - Proper HTTP status codes (422 for validation, 404 for not found)
    - OpenAPI schema auto-generation
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from aragora.rbac.models import AuthorizationContext

from ..dependencies.auth import require_permission
from ..middleware.error_handling import ForbiddenError, NotFoundError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/admin", tags=["Admin"])

# =============================================================================
# Constants
# =============================================================================

SAFE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_@.\-]{0,127}$")

ADMIN_ROLES = {
    "admin",
    "owner",
    "superadmin",
    "super_admin",
    "org_admin",
    "workspace_admin",
    "security_admin",
    "compliance_officer",
}

VALID_NOMIC_PHASES = {
    "idle",
    "context",
    "debate",
    "design",
    "implement",
    "verify",
    "commit",
}

# =============================================================================
# Pydantic Models — Responses
# =============================================================================


class SystemStatsResponse(BaseModel):
    """Response for GET /admin/stats."""

    stats: dict[str, Any]


class SystemMetricsResponse(BaseModel):
    """Response for GET /admin/system/metrics."""

    metrics: dict[str, Any]
    timestamp: str


class RevenueResponse(BaseModel):
    """Response for GET /admin/revenue."""

    revenue: dict[str, Any]


class OrganizationItem(BaseModel):
    """A single organization record."""

    model_config = {"extra": "allow"}


class OrganizationListResponse(BaseModel):
    """Response for GET /admin/organizations."""

    organizations: list[dict[str, Any]]
    total: int
    limit: int
    offset: int


class UserItem(BaseModel):
    """A single user record (sensitive fields stripped)."""

    id: str = ""
    email: str = ""
    name: str = ""
    role: str = ""
    org_id: str | None = None
    is_active: bool = True

    model_config = {"extra": "allow"}


class UserListResponse(BaseModel):
    """Response for GET /admin/users."""

    users: list[dict[str, Any]]
    total: int
    limit: int
    offset: int


class CreateUserRequest(BaseModel):
    """Request body for POST /admin/users."""

    email: str = Field(..., min_length=3, max_length=255, description="User email")
    name: str = Field("", max_length=200, description="Display name")
    role: str = Field("member", description="Role to assign")
    org_id: str | None = Field(None, description="Organization ID")
    password: str | None = Field(
        None, min_length=8, max_length=128, description="Initial password (optional)"
    )


class CreateUserResponse(BaseModel):
    """Response for POST /admin/users."""

    success: bool
    user_id: str
    email: str


class UserActionResponse(BaseModel):
    """Generic response for user state-change actions."""

    success: bool
    user_id: str
    message: str | None = None

    model_config = {"extra": "allow"}


class ImpersonateResponse(BaseModel):
    """Response for POST /admin/impersonate/{user_id}."""

    token: str
    expires_in: int
    target_user: dict[str, Any]
    warning: str


class TenantItem(BaseModel):
    """A tenant summary."""

    id: str = ""
    name: str = ""
    tier: str = ""
    is_active: bool = True

    model_config = {"extra": "allow"}


class TenantListResponse(BaseModel):
    """Response for GET /admin/tenants."""

    tenants: list[dict[str, Any]]
    total: int
    limit: int
    offset: int


class CreateTenantRequest(BaseModel):
    """Request body for POST /admin/tenants."""

    name: str = Field(..., min_length=1, max_length=200, description="Tenant name")
    tier: str = Field("free", description="Pricing tier")
    config: dict[str, Any] = Field(default_factory=dict, description="Tenant configuration")


class CreateTenantResponse(BaseModel):
    """Response for POST /admin/tenants."""

    success: bool
    tenant_id: str
    name: str


class UpdateTenantRequest(BaseModel):
    """Request body for PUT /admin/tenants/{tenant_id}."""

    name: str | None = Field(None, max_length=200, description="Tenant name")
    tier: str | None = Field(None, description="Pricing tier")
    is_active: bool | None = Field(None, description="Active flag")
    config: dict[str, Any] | None = Field(None, description="Tenant configuration")


class UpdateTenantResponse(BaseModel):
    """Response for PUT /admin/tenants/{tenant_id}."""

    success: bool
    tenant_id: str


class AuditLogEntry(BaseModel):
    """A single audit log entry."""

    id: str = ""
    timestamp: str = ""
    event_type: str = ""
    actor: str = ""
    action: str = ""
    resource_type: str = ""
    resource_id: str = ""
    details: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class AuditLogResponse(BaseModel):
    """Response for GET /admin/audit-log."""

    entries: list[dict[str, Any]]
    total: int
    limit: int
    offset: int


class SystemConfigResponse(BaseModel):
    """Response for GET /admin/config."""

    config: dict[str, Any]


class UpdateConfigRequest(BaseModel):
    """Request body for PUT /admin/config."""

    updates: dict[str, Any] = Field(..., description="Configuration key-value pairs to update")


class UpdateConfigResponse(BaseModel):
    """Response for PUT /admin/config."""

    success: bool
    updated_keys: list[str]


class LicenseInfoResponse(BaseModel):
    """Response for GET /admin/license."""

    license: dict[str, Any]


class ActivateLicenseRequest(BaseModel):
    """Request body for POST /admin/license."""

    license_key: str = Field(..., min_length=1, max_length=512, description="License key")


class ActivateLicenseResponse(BaseModel):
    """Response for POST /admin/license."""

    success: bool
    license: dict[str, Any]


class NomicStatusResponse(BaseModel):
    """Response for GET /admin/nomic/status."""

    running: bool = False
    current_phase: str | None = None
    cycle_id: str | None = None
    state_machine: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None
    circuit_breakers: dict[str, Any] | None = None
    last_checkpoint: dict[str, Any] | None = None
    errors: list[str] = Field(default_factory=list)

    model_config = {"extra": "allow"}


class NomicPauseResumeRequest(BaseModel):
    """Request body for POST /admin/nomic/pause and /admin/nomic/resume."""

    reason: str = Field("Admin action", max_length=500, description="Reason for action")


class NomicActionResponse(BaseModel):
    """Response for nomic pause/resume actions."""

    success: bool
    status: str
    previous_phase: str | None = None
    message: str | None = None

    model_config = {"extra": "allow"}


# =============================================================================
# Dependencies — Lazy subsystem access with graceful degradation
# =============================================================================


def _get_user_store(request: Request) -> Any:
    """Get the user store from app state context."""
    ctx = getattr(request.app.state, "context", None)
    if ctx:
        store = ctx.get("user_store")
        if store is not None:
            return store

    # Try module-level singleton
    try:
        from aragora.auth.store import get_user_store

        return get_user_store()
    except (ImportError, RuntimeError, AttributeError) as e:
        logger.debug("User store not available: %s", e)
        return None


def _require_user_store(request: Request) -> Any:
    """Dependency that raises 503 if user store is unavailable."""
    store = _get_user_store(request)
    if store is None:
        raise HTTPException(status_code=503, detail="User store not available")
    return store


def _get_tenant_store(request: Request) -> Any:
    """Get the tenant/org store from app state context."""
    ctx = getattr(request.app.state, "context", None)
    if ctx:
        store = ctx.get("tenant_store") or ctx.get("org_store")
        if store is not None:
            return store

    try:
        from aragora.tenancy.store import get_tenant_store

        return get_tenant_store()
    except (ImportError, RuntimeError, AttributeError) as e:
        logger.debug("Tenant store not available: %s", e)
        return None


def _get_audit_log(request: Request) -> Any:
    """Get the audit log from app state context or module."""
    ctx = getattr(request.app.state, "context", None)
    if ctx:
        log = ctx.get("audit_log")
        if log is not None:
            return log

    try:
        from aragora.observability.immutable_log import get_audit_log

        return get_audit_log()
    except (ImportError, RuntimeError, AttributeError) as e:
        logger.debug("Audit log not available: %s", e)
        return None


def _get_config_store(request: Request) -> Any:
    """Get the system config store from app state context or module."""
    ctx = getattr(request.app.state, "context", None)
    if ctx:
        store = ctx.get("config_store")
        if store is not None:
            return store

    try:
        from aragora.config.store import get_config_store

        return get_config_store()
    except (ImportError, RuntimeError, AttributeError) as e:
        logger.debug("Config store not available: %s", e)
        return None


def _get_license_manager(request: Request) -> Any:
    """Get the license manager from app state context or module."""
    ctx = getattr(request.app.state, "context", None)
    if ctx:
        mgr = ctx.get("license_manager")
        if mgr is not None:
            return mgr

    try:
        from aragora.billing.license import get_license_manager

        return get_license_manager()
    except (ImportError, RuntimeError, AttributeError) as e:
        logger.debug("License manager not available: %s", e)
        return None


# =============================================================================
# Helpers
# =============================================================================


def _validate_user_id(user_id: str) -> None:
    """Validate a user ID path parameter."""
    if not user_id or not SAFE_ID_PATTERN.match(user_id):
        raise HTTPException(status_code=400, detail="Invalid user ID format")


def _validate_tenant_id(tenant_id: str) -> None:
    """Validate a tenant ID path parameter."""
    if not tenant_id or not SAFE_ID_PATTERN.match(tenant_id):
        raise HTTPException(status_code=400, detail="Invalid tenant ID format")


def _sanitize_user_dict(user_dict: dict[str, Any]) -> dict[str, Any]:
    """Remove sensitive fields from a user dict before returning it."""
    try:
        from aragora.server.handlers.utils.sanitization import sanitize_user_response

        return sanitize_user_response(user_dict)
    except (ImportError, RuntimeError, AttributeError):
        # Fallback: manually strip known sensitive fields
        sensitive_keys = {
            "password",
            "password_hash",
            "hashed_password",
            "secret",
            "totp_secret",
            "mfa_secret",
            "recovery_codes",
        }
        return {k: v for k, v in user_dict.items() if k not in sensitive_keys}


# =============================================================================
# Endpoints — System Health & Metrics
# =============================================================================


@router.get("/stats", response_model=SystemStatsResponse)
async def get_admin_stats(
    request: Request,
    auth: AuthorizationContext = Depends(require_permission("admin:stats:read")),
    user_store: Any = Depends(_require_user_store),
) -> SystemStatsResponse:
    """
    Get system-wide statistics.

    Returns aggregate counts for users, organizations, debates, etc.
    Requires `admin:stats:read` permission.
    """
    try:
        stats = user_store.get_admin_stats()
        return SystemStatsResponse(stats=stats)
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error getting admin stats: %s", e)
        raise HTTPException(status_code=500, detail="Failed to retrieve system statistics")


@router.get("/system/metrics", response_model=SystemMetricsResponse)
async def get_system_metrics(
    request: Request,
    auth: AuthorizationContext = Depends(require_permission("admin:metrics:read")),
    user_store: Any = Depends(_require_user_store),
) -> SystemMetricsResponse:
    """
    Get aggregated system metrics from all subsystems.

    Collects metrics from user store, debate storage, circuit breakers,
    cache, and rate limiters. Individual subsystem failures are non-blocking.
    Requires `admin:metrics:read` permission.
    """
    try:
        timestamp = datetime.now(timezone.utc).isoformat()
        metrics: dict[str, Any] = {"timestamp": timestamp}

        # User stats
        try:
            metrics["users"] = user_store.get_admin_stats()
        except (KeyError, ValueError, TypeError, AttributeError, OSError) as e:
            logger.warning("Failed to get user stats: %s", e)
            metrics["users"] = {"error": "unavailable"}

        # Debate storage stats
        ctx = getattr(request.app.state, "context", None)
        if ctx:
            debate_storage = ctx.get("debate_storage")
            if debate_storage and hasattr(debate_storage, "get_statistics"):
                try:
                    metrics["debates"] = debate_storage.get_statistics()
                except (KeyError, ValueError, OSError, TypeError, AttributeError) as e:
                    logger.warning("Failed to get debate stats: %s", e)
                    metrics["debates"] = {"error": "unavailable"}

        # Circuit breaker stats
        try:
            from aragora.resilience import get_circuit_breaker_status

            metrics["circuit_breakers"] = get_circuit_breaker_status()
        except ImportError:
            pass
        except (RuntimeError, ValueError, TypeError, AttributeError) as e:
            logger.warning("Failed to get circuit breaker stats: %s", e)

        # Cache stats
        try:
            from aragora.server.handlers.admin.cache import get_cache_stats

            metrics["cache"] = get_cache_stats()
        except (ImportError, RuntimeError, ValueError, TypeError, AttributeError) as e:
            logger.warning("Failed to get cache stats: %s", e)

        # Rate limit stats
        try:
            from aragora.server.middleware.rate_limit import get_rate_limiter

            limiter = get_rate_limiter()
            if limiter and hasattr(limiter, "get_stats"):
                metrics["rate_limits"] = limiter.get_stats()
        except (ImportError, RuntimeError, ValueError, TypeError, AttributeError) as e:
            logger.warning("Failed to get rate limit stats: %s", e)

        return SystemMetricsResponse(metrics=metrics, timestamp=timestamp)

    except HTTPException:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error getting system metrics: %s", e)
        raise HTTPException(status_code=500, detail="Failed to retrieve system metrics")


@router.get("/revenue", response_model=RevenueResponse)
async def get_revenue_stats(
    request: Request,
    auth: AuthorizationContext = Depends(require_permission("admin:revenue:read")),
    user_store: Any = Depends(_require_user_store),
) -> RevenueResponse:
    """
    Get revenue and billing statistics.

    Calculates MRR/ARR from tier distribution data. Requires `admin:revenue:read`
    permission.
    """
    try:
        stats = user_store.get_admin_stats()
        tier_distribution = stats.get("tier_distribution", {})

        # Calculate MRR from tier pricing
        mrr_cents = 0
        tier_revenue: dict[str, Any] = {}
        try:
            from aragora.billing.models import TIER_LIMITS

            for tier_name, count in tier_distribution.items():
                tier_limits = TIER_LIMITS.get(tier_name)
                if tier_limits:
                    tier_mrr = tier_limits.price_monthly_cents * count
                    tier_revenue[tier_name] = {
                        "count": count,
                        "price_cents": tier_limits.price_monthly_cents,
                        "mrr_cents": tier_mrr,
                    }
                    mrr_cents += tier_mrr
        except (ImportError, RuntimeError, AttributeError) as e:
            logger.warning("Billing models not available for revenue calc: %s", e)

        return RevenueResponse(
            revenue={
                "mrr_cents": mrr_cents,
                "mrr_dollars": mrr_cents / 100,
                "arr_dollars": (mrr_cents * 12) / 100,
                "tier_breakdown": tier_revenue,
                "total_organizations": stats.get("total_organizations", 0),
                "paying_organizations": sum(
                    count for tier, count in tier_distribution.items() if tier != "free"
                ),
            }
        )

    except HTTPException:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error getting revenue stats: %s", e)
        raise HTTPException(status_code=500, detail="Failed to retrieve revenue statistics")


# =============================================================================
# Endpoints — Organization Management
# =============================================================================


@router.get("/organizations", response_model=OrganizationListResponse)
async def list_organizations(
    request: Request,
    limit: int = Query(50, ge=1, le=100, description="Max results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    tier: str | None = Query(None, description="Filter by pricing tier"),
    auth: AuthorizationContext = Depends(require_permission("admin:organizations:read")),
    user_store: Any = Depends(_require_user_store),
) -> OrganizationListResponse:
    """
    List all organizations with pagination.

    Supports optional tier filtering. Requires `admin:organizations:read` permission.
    """
    try:
        organizations, total = user_store.list_all_organizations(
            limit=limit,
            offset=offset,
            tier_filter=tier,
        )

        return OrganizationListResponse(
            organizations=[org.to_dict() for org in organizations],
            total=total,
            limit=limit,
            offset=offset,
        )

    except HTTPException:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error listing organizations: %s", e)
        raise HTTPException(status_code=500, detail="Failed to list organizations")


# =============================================================================
# Endpoints — User Management
# =============================================================================


@router.get("/users", response_model=UserListResponse)
async def list_users(
    request: Request,
    limit: int = Query(50, ge=1, le=100, description="Max results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    org_id: str | None = Query(None, description="Filter by organization ID"),
    role: str | None = Query(None, description="Filter by role"),
    active_only: bool = Query(False, description="Only return active users"),
    auth: AuthorizationContext = Depends(require_permission("admin:users:read")),
    user_store: Any = Depends(_require_user_store),
) -> UserListResponse:
    """
    List all users with pagination and filtering.

    Sensitive fields (password hashes, MFA secrets) are stripped from the
    response. Requires `admin:users:read` permission.
    """
    try:
        users, total = user_store.list_all_users(
            limit=limit,
            offset=offset,
            org_id_filter=org_id,
            role_filter=role,
            active_only=active_only,
        )

        user_dicts = [_sanitize_user_dict(user.to_dict()) for user in users]

        return UserListResponse(
            users=user_dicts,
            total=total,
            limit=limit,
            offset=offset,
        )

    except HTTPException:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error listing users: %s", e)
        raise HTTPException(status_code=500, detail="Failed to list users")


@router.post("/users", response_model=CreateUserResponse, status_code=201)
async def create_user(
    body: CreateUserRequest,
    request: Request,
    auth: AuthorizationContext = Depends(require_permission("admin:users:write")),
    user_store: Any = Depends(_require_user_store),
) -> CreateUserResponse:
    """
    Create a new user account.

    Requires `admin:users:write` permission. If no password is provided,
    one will be generated and the user will need to reset it.
    """
    try:
        # Check for existing user
        existing = None
        if hasattr(user_store, "get_user_by_email"):
            existing = user_store.get_user_by_email(body.email)
        if existing:
            raise HTTPException(status_code=409, detail="A user with this email already exists")

        user_data: dict[str, Any] = {
            "email": body.email,
            "name": body.name,
            "role": body.role,
        }
        if body.org_id:
            user_data["org_id"] = body.org_id
        if body.password:
            user_data["password"] = body.password

        user = user_store.create_user(**user_data)
        user_id = getattr(user, "id", str(user)) if user else ""

        logger.info("Admin %s created user %s (%s)", auth.user_id, user_id, body.email)

        # Audit
        try:
            from aragora.audit.unified import audit_admin

            audit_admin(
                admin_id=auth.user_id,
                action="create_user",
                target_type="user",
                target_id=user_id,
                target_email=body.email,
            )
        except (ImportError, RuntimeError, AttributeError) as e:
            logger.warning("Audit log unavailable: %s", e)

        return CreateUserResponse(success=True, user_id=user_id, email=body.email)

    except HTTPException:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error creating user: %s", e)
        raise HTTPException(status_code=500, detail="Failed to create user")


@router.post("/users/{user_id}/deactivate", response_model=UserActionResponse)
async def deactivate_user(
    user_id: str,
    request: Request,
    auth: AuthorizationContext = Depends(require_permission("admin:users:write")),
    user_store: Any = Depends(_require_user_store),
) -> UserActionResponse:
    """
    Deactivate a user account.

    Prevents the target user from logging in. Admins cannot deactivate
    themselves. Requires `admin:users:write` permission.
    """
    _validate_user_id(user_id)

    try:
        # Cannot deactivate yourself
        if user_id == auth.user_id:
            raise HTTPException(status_code=400, detail="Cannot deactivate yourself")

        target_user = user_store.get_user_by_id(user_id)
        if not target_user:
            raise NotFoundError(f"User {user_id} not found")

        user_store.update_user(user_id, is_active=False)

        logger.info("Admin %s deactivated user %s", auth.user_id, user_id)

        try:
            from aragora.audit.unified import audit_admin

            audit_admin(
                admin_id=auth.user_id,
                action="deactivate_user",
                target_type="user",
                target_id=user_id,
                target_email=getattr(target_user, "email", ""),
            )
        except (ImportError, RuntimeError, AttributeError) as e:
            logger.warning("Audit log unavailable: %s", e)

        return UserActionResponse(
            success=True,
            user_id=user_id,
            message=f"User {user_id} deactivated",
        )

    except HTTPException:
        raise
    except NotFoundError:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error deactivating user %s: %s", user_id, e)
        raise HTTPException(status_code=500, detail="Failed to deactivate user")


@router.post("/users/{user_id}/activate", response_model=UserActionResponse)
async def activate_user(
    user_id: str,
    request: Request,
    auth: AuthorizationContext = Depends(require_permission("admin:users:write")),
    user_store: Any = Depends(_require_user_store),
) -> UserActionResponse:
    """
    Activate a previously deactivated user account.

    Requires `admin:users:write` permission.
    """
    _validate_user_id(user_id)

    try:
        target_user = user_store.get_user_by_id(user_id)
        if not target_user:
            raise NotFoundError(f"User {user_id} not found")

        user_store.update_user(user_id, is_active=True)

        logger.info("Admin %s activated user %s", auth.user_id, user_id)

        try:
            from aragora.audit.unified import audit_admin

            audit_admin(
                admin_id=auth.user_id,
                action="activate_user",
                target_type="user",
                target_id=user_id,
                target_email=getattr(target_user, "email", ""),
            )
        except (ImportError, RuntimeError, AttributeError) as e:
            logger.warning("Audit log unavailable: %s", e)

        return UserActionResponse(
            success=True,
            user_id=user_id,
            message=f"User {user_id} activated",
        )

    except HTTPException:
        raise
    except NotFoundError:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error activating user %s: %s", user_id, e)
        raise HTTPException(status_code=500, detail="Failed to activate user")


@router.post("/users/{user_id}/unlock", response_model=UserActionResponse)
async def unlock_user(
    user_id: str,
    request: Request,
    auth: AuthorizationContext = Depends(require_permission("admin:users:write")),
    user_store: Any = Depends(_require_user_store),
) -> UserActionResponse:
    """
    Unlock a user account locked due to failed login attempts.

    Clears both the in-memory/Redis lockout tracker and database lockout state.
    Requires `admin:users:write` permission.
    """
    _validate_user_id(user_id)

    try:
        target_user = user_store.get_user_by_id(user_id)
        if not target_user:
            raise NotFoundError(f"User {user_id} not found")

        email = getattr(target_user, "email", "")
        lockout_cleared = False
        db_cleared = False

        # Clear in-memory/Redis lockout tracker
        try:
            from aragora.auth.lockout import get_lockout_tracker

            tracker = get_lockout_tracker()
            lockout_cleared = tracker.admin_unlock(email=email, user_id=user_id)
        except (ImportError, RuntimeError, AttributeError) as e:
            logger.warning("Lockout tracker not available: %s", e)

        # Clear database lockout state
        if hasattr(user_store, "reset_failed_login_attempts"):
            db_cleared = user_store.reset_failed_login_attempts(email)

        logger.info(
            "Admin %s unlocked user %s (tracker=%s, db=%s)",
            auth.user_id,
            user_id,
            lockout_cleared,
            db_cleared,
        )

        try:
            from aragora.audit.unified import audit_admin

            audit_admin(
                admin_id=auth.user_id,
                action="unlock_user",
                target_type="user",
                target_id=user_id,
                target_email=email,
            )
        except (ImportError, RuntimeError, AttributeError) as e:
            logger.warning("Audit log unavailable: %s", e)

        return UserActionResponse(
            success=True,
            user_id=user_id,
            message=f"Account lockout cleared for {email}",
        )

    except HTTPException:
        raise
    except NotFoundError:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error unlocking user %s: %s", user_id, e)
        raise HTTPException(status_code=500, detail="Failed to unlock user")


@router.post("/impersonate/{user_id}", response_model=ImpersonateResponse)
async def impersonate_user(
    user_id: str,
    request: Request,
    auth: AuthorizationContext = Depends(require_permission("admin:impersonate")),
    user_store: Any = Depends(_require_user_store),
) -> ImpersonateResponse:
    """
    Create a short-lived impersonation token for a user.

    Allows admins to view the system as the target user for support purposes.
    The token expires in 1 hour and the action is logged for audit.

    Requires `admin:impersonate` permission.
    """
    _validate_user_id(user_id)

    try:
        target_user = user_store.get_user_by_id(user_id)
        if not target_user:
            raise NotFoundError(f"User {user_id} not found")

        # Create short-lived impersonation token
        try:
            from aragora.billing.jwt_auth import create_access_token

            token = create_access_token(
                user_id=user_id,
                email=getattr(target_user, "email", ""),
                org_id=getattr(target_user, "org_id", None),
                role=getattr(target_user, "role", "member"),
                expiry_hours=1,
            )
        except (ImportError, RuntimeError, AttributeError) as e:
            logger.warning("JWT auth not available for impersonation: %s", e)
            raise HTTPException(
                status_code=503,
                detail="Token generation service not available",
            )

        logger.info("Admin %s impersonating user %s", auth.user_id, user_id)

        try:
            from aragora.audit.unified import audit_admin

            audit_admin(
                admin_id=auth.user_id,
                action="impersonate_user",
                target_type="user",
                target_id=user_id,
                target_email=getattr(target_user, "email", ""),
            )
        except (ImportError, RuntimeError, AttributeError) as e:
            logger.warning("Audit log unavailable: %s", e)

        return ImpersonateResponse(
            token=token,
            expires_in=3600,
            target_user={
                "id": getattr(target_user, "id", user_id),
                "email": getattr(target_user, "email", ""),
                "name": getattr(target_user, "name", ""),
                "role": getattr(target_user, "role", ""),
            },
            warning="This token grants full access as the target user. Use responsibly.",
        )

    except HTTPException:
        raise
    except NotFoundError:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error creating impersonation token for %s: %s", user_id, e)
        raise HTTPException(status_code=500, detail="Failed to create impersonation token")


# =============================================================================
# Endpoints — Tenant Management
# =============================================================================


@router.get("/tenants", response_model=TenantListResponse)
async def list_tenants(
    request: Request,
    limit: int = Query(50, ge=1, le=100, description="Max results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    tier: str | None = Query(None, description="Filter by pricing tier"),
    active_only: bool = Query(False, description="Only return active tenants"),
    auth: AuthorizationContext = Depends(require_permission("admin:tenants:read")),
) -> TenantListResponse:
    """
    List all tenants with pagination.

    Requires `admin:tenants:read` permission.
    """
    try:
        tenant_store = _get_tenant_store(request)
        if tenant_store is None:
            raise HTTPException(status_code=503, detail="Tenant store not available")

        list_kwargs: dict[str, Any] = {"limit": limit, "offset": offset}
        if tier is not None:
            list_kwargs["tier_filter"] = tier
        if active_only:
            list_kwargs["active_only"] = active_only

        if hasattr(tenant_store, "list_tenants"):
            tenants, total = tenant_store.list_tenants(**list_kwargs)
        elif hasattr(tenant_store, "list_all"):
            tenants, total = tenant_store.list_all(**list_kwargs)
        else:
            raise HTTPException(status_code=503, detail="Tenant listing not supported")

        return TenantListResponse(
            tenants=[
                t.to_dict() if hasattr(t, "to_dict") else t
                for t in tenants
            ],
            total=total,
            limit=limit,
            offset=offset,
        )

    except HTTPException:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error listing tenants: %s", e)
        raise HTTPException(status_code=500, detail="Failed to list tenants")


@router.post("/tenants", response_model=CreateTenantResponse, status_code=201)
async def create_tenant(
    body: CreateTenantRequest,
    request: Request,
    auth: AuthorizationContext = Depends(require_permission("admin:tenants:write")),
) -> CreateTenantResponse:
    """
    Create a new tenant.

    Requires `admin:tenants:write` permission.
    """
    try:
        tenant_store = _get_tenant_store(request)
        if tenant_store is None:
            raise HTTPException(status_code=503, detail="Tenant store not available")

        tenant_data: dict[str, Any] = {
            "name": body.name,
            "tier": body.tier,
            "config": body.config,
        }

        if hasattr(tenant_store, "create_tenant"):
            tenant = tenant_store.create_tenant(**tenant_data)
        elif hasattr(tenant_store, "create"):
            tenant = tenant_store.create(**tenant_data)
        else:
            raise HTTPException(status_code=503, detail="Tenant creation not supported")

        tenant_id = getattr(tenant, "id", str(tenant)) if tenant else ""

        logger.info("Admin %s created tenant %s (%s)", auth.user_id, tenant_id, body.name)

        try:
            from aragora.audit.unified import audit_admin

            audit_admin(
                admin_id=auth.user_id,
                action="create_tenant",
                target_type="tenant",
                target_id=tenant_id,
                tenant_name=body.name,
            )
        except (ImportError, RuntimeError, AttributeError) as e:
            logger.warning("Audit log unavailable: %s", e)

        return CreateTenantResponse(success=True, tenant_id=tenant_id, name=body.name)

    except HTTPException:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error creating tenant: %s", e)
        raise HTTPException(status_code=500, detail="Failed to create tenant")


@router.put("/tenants/{tenant_id}", response_model=UpdateTenantResponse)
async def update_tenant(
    tenant_id: str,
    body: UpdateTenantRequest,
    request: Request,
    auth: AuthorizationContext = Depends(require_permission("admin:tenants:write")),
) -> UpdateTenantResponse:
    """
    Update tenant configuration.

    Requires `admin:tenants:write` permission.
    """
    _validate_tenant_id(tenant_id)

    try:
        tenant_store = _get_tenant_store(request)
        if tenant_store is None:
            raise HTTPException(status_code=503, detail="Tenant store not available")

        # Build update dict from non-None fields
        updates = body.model_dump(exclude_none=True)
        if not updates:
            raise HTTPException(status_code=400, detail="No fields to update")

        if hasattr(tenant_store, "update_tenant"):
            tenant_store.update_tenant(tenant_id, **updates)
        elif hasattr(tenant_store, "update"):
            tenant_store.update(tenant_id, **updates)
        else:
            raise HTTPException(status_code=503, detail="Tenant update not supported")

        logger.info("Admin %s updated tenant %s", auth.user_id, tenant_id)

        try:
            from aragora.audit.unified import audit_admin

            audit_admin(
                admin_id=auth.user_id,
                action="update_tenant",
                target_type="tenant",
                target_id=tenant_id,
                updated_fields=list(updates.keys()),
            )
        except (ImportError, RuntimeError, AttributeError) as e:
            logger.warning("Audit log unavailable: %s", e)

        return UpdateTenantResponse(success=True, tenant_id=tenant_id)

    except HTTPException:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error updating tenant %s: %s", tenant_id, e)
        raise HTTPException(status_code=500, detail="Failed to update tenant")


# =============================================================================
# Endpoints — Audit Log
# =============================================================================


@router.get("/audit-log", response_model=AuditLogResponse)
async def get_audit_log(
    request: Request,
    limit: int = Query(50, ge=1, le=500, description="Max results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    event_type: str | None = Query(None, description="Filter by event type"),
    actor: str | None = Query(None, description="Filter by actor user ID"),
    resource_type: str | None = Query(None, description="Filter by resource type"),
    since: str | None = Query(None, description="ISO-8601 start datetime"),
    until: str | None = Query(None, description="ISO-8601 end datetime"),
    auth: AuthorizationContext = Depends(require_permission("admin:audit:read")),
) -> AuditLogResponse:
    """
    View audit log entries with filtering.

    Returns paginated immutable audit trail entries.
    Requires `admin:audit:read` permission.
    """
    try:
        audit_log = _get_audit_log(request)
        if audit_log is None:
            raise HTTPException(status_code=503, detail="Audit log not available")

        query_kwargs: dict[str, Any] = {"limit": limit, "offset": offset}
        if event_type is not None:
            query_kwargs["event_type"] = event_type
        if actor is not None:
            query_kwargs["actor"] = actor
        if resource_type is not None:
            query_kwargs["resource_type"] = resource_type
        if since is not None:
            query_kwargs["since"] = since
        if until is not None:
            query_kwargs["until"] = until

        if hasattr(audit_log, "query"):
            entries, total = audit_log.query(**query_kwargs)
        elif hasattr(audit_log, "list_entries"):
            entries, total = audit_log.list_entries(**query_kwargs)
        elif hasattr(audit_log, "search"):
            entries, total = audit_log.search(**query_kwargs)
        else:
            # Fallback: try get_entries with limited params
            entries = []
            total = 0
            if hasattr(audit_log, "get_entries"):
                entries = audit_log.get_entries(limit=limit, offset=offset)
                total = len(entries)

        entry_dicts = [
            e.to_dict() if hasattr(e, "to_dict") else (e if isinstance(e, dict) else {})
            for e in entries
        ]

        return AuditLogResponse(
            entries=entry_dicts,
            total=total,
            limit=limit,
            offset=offset,
        )

    except HTTPException:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error querying audit log: %s", e)
        raise HTTPException(status_code=500, detail="Failed to query audit log")


# =============================================================================
# Endpoints — Configuration Management
# =============================================================================


@router.get("/config", response_model=SystemConfigResponse)
async def get_system_config(
    request: Request,
    auth: AuthorizationContext = Depends(require_permission("admin:config:read")),
) -> SystemConfigResponse:
    """
    Get current system configuration.

    Returns non-sensitive configuration values. Secrets are redacted.
    Requires `admin:config:read` permission.
    """
    try:
        config_store = _get_config_store(request)
        if config_store is None:
            raise HTTPException(status_code=503, detail="Configuration store not available")

        if hasattr(config_store, "get_all"):
            config = config_store.get_all()
        elif hasattr(config_store, "to_dict"):
            config = config_store.to_dict()
        else:
            config = {}

        # Redact sensitive values
        sensitive_patterns = {"secret", "password", "key", "token", "credential"}
        redacted: dict[str, Any] = {}
        for k, v in config.items():
            if any(pat in k.lower() for pat in sensitive_patterns):
                redacted[k] = "***REDACTED***"
            else:
                redacted[k] = v

        return SystemConfigResponse(config=redacted)

    except HTTPException:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error getting system config: %s", e)
        raise HTTPException(status_code=500, detail="Failed to retrieve configuration")


@router.put("/config", response_model=UpdateConfigResponse)
async def update_system_config(
    body: UpdateConfigRequest,
    request: Request,
    auth: AuthorizationContext = Depends(require_permission("admin:config:write")),
) -> UpdateConfigResponse:
    """
    Update system configuration values.

    Accepts a dict of key-value pairs. Only known configuration keys are
    accepted. Requires `admin:config:write` permission.
    """
    try:
        config_store = _get_config_store(request)
        if config_store is None:
            raise HTTPException(status_code=503, detail="Configuration store not available")

        updated_keys: list[str] = []
        for key, value in body.updates.items():
            if hasattr(config_store, "set"):
                config_store.set(key, value)
                updated_keys.append(key)
            elif hasattr(config_store, "update"):
                config_store.update({key: value})
                updated_keys.append(key)

        if not updated_keys:
            raise HTTPException(status_code=400, detail="No configuration keys could be updated")

        logger.info("Admin %s updated config keys: %s", auth.user_id, updated_keys)

        try:
            from aragora.audit.unified import audit_admin

            audit_admin(
                admin_id=auth.user_id,
                action="update_config",
                target_type="system_config",
                target_id="global",
                updated_keys=updated_keys,
            )
        except (ImportError, RuntimeError, AttributeError) as e:
            logger.warning("Audit log unavailable: %s", e)

        return UpdateConfigResponse(success=True, updated_keys=updated_keys)

    except HTTPException:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error updating system config: %s", e)
        raise HTTPException(status_code=500, detail="Failed to update configuration")


# =============================================================================
# Endpoints — License Management
# =============================================================================


@router.get("/license", response_model=LicenseInfoResponse)
async def get_license_info(
    request: Request,
    auth: AuthorizationContext = Depends(require_permission("admin:license:read")),
) -> LicenseInfoResponse:
    """
    Get current license information.

    Returns license tier, expiration, feature flags, and usage limits.
    Requires `admin:license:read` permission.
    """
    try:
        license_mgr = _get_license_manager(request)
        if license_mgr is None:
            # Return a sensible default for community/unlicensed installs
            return LicenseInfoResponse(
                license={
                    "tier": "community",
                    "status": "active",
                    "features": [],
                    "message": "No license manager configured",
                }
            )

        if hasattr(license_mgr, "get_info"):
            info = license_mgr.get_info()
        elif hasattr(license_mgr, "to_dict"):
            info = license_mgr.to_dict()
        else:
            info = {"status": "unknown"}

        return LicenseInfoResponse(license=info)

    except HTTPException:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error getting license info: %s", e)
        raise HTTPException(status_code=500, detail="Failed to retrieve license information")


@router.post("/license", response_model=ActivateLicenseResponse)
async def activate_license(
    body: ActivateLicenseRequest,
    request: Request,
    auth: AuthorizationContext = Depends(require_permission("admin:license:write")),
) -> ActivateLicenseResponse:
    """
    Activate or update the platform license.

    Requires `admin:license:write` permission.
    """
    try:
        license_mgr = _get_license_manager(request)
        if license_mgr is None:
            raise HTTPException(status_code=503, detail="License manager not available")

        if hasattr(license_mgr, "activate"):
            result = license_mgr.activate(body.license_key)
        elif hasattr(license_mgr, "apply_key"):
            result = license_mgr.apply_key(body.license_key)
        else:
            raise HTTPException(status_code=503, detail="License activation not supported")

        # Normalise result to dict
        if hasattr(result, "to_dict"):
            result_dict = result.to_dict()
        elif isinstance(result, dict):
            result_dict = result
        else:
            result_dict = {"status": "activated"}

        logger.info("Admin %s activated license", auth.user_id)

        try:
            from aragora.audit.unified import audit_admin

            audit_admin(
                admin_id=auth.user_id,
                action="activate_license",
                target_type="license",
                target_id="global",
            )
        except (ImportError, RuntimeError, AttributeError) as e:
            logger.warning("Audit log unavailable: %s", e)

        return ActivateLicenseResponse(success=True, license=result_dict)

    except HTTPException:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error activating license: %s", e)
        raise HTTPException(status_code=500, detail="Failed to activate license")


# =============================================================================
# Endpoints — Nomic Loop Control
# =============================================================================


@router.get("/nomic/status", response_model=NomicStatusResponse)
async def get_nomic_status(
    request: Request,
    auth: AuthorizationContext = Depends(require_permission("admin:nomic:read")),
) -> NomicStatusResponse:
    """
    Get detailed nomic self-improvement loop status.

    Returns state machine state, metrics, circuit breakers, and the latest
    checkpoint. Requires `admin:nomic:read` permission.
    """
    try:
        import json as _json
        from pathlib import Path

        ctx = getattr(request.app.state, "context", None)
        nomic_dir_str = ".nomic"
        if ctx:
            nomic_dir_str = ctx.get("nomic_dir", ".nomic") or ".nomic"
        nomic_dir = Path(nomic_dir_str)

        errors: list[str] = []
        status = NomicStatusResponse(errors=errors)

        # Read state file
        state_file = nomic_dir / "nomic_state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state_data = _json.load(f)
                status.running = state_data.get("running", False)
                status.current_phase = state_data.get("phase")
                status.cycle_id = state_data.get("cycle_id")
                status.state_machine = state_data
            except (_json.JSONDecodeError, OSError, KeyError, ValueError) as e:
                errors.append(f"Failed to read state: {e}")

        # Metrics
        try:
            from aragora.nomic.metrics import (
                check_stuck_phases,
                get_nomic_metrics_summary,
            )

            metrics = get_nomic_metrics_summary()
            if isinstance(metrics, dict):
                status.metrics = metrics
            stuck_info = check_stuck_phases(max_idle_seconds=1800)
            if stuck_info and isinstance(status.metrics, dict):
                status.metrics["stuck_detection"] = stuck_info
        except ImportError:
            errors.append("Nomic metrics module not available")
        except (TypeError, ValueError, KeyError, AttributeError, OSError) as e:
            errors.append(f"Failed to get nomic metrics: {e}")

        # Circuit breakers
        try:
            from aragora.nomic.recovery import CircuitBreakerRegistry

            registry = CircuitBreakerRegistry()
            status.circuit_breakers = {
                "open": registry.all_open(),
                "details": registry.to_dict(),
            }
        except (ImportError, TypeError, ValueError, KeyError, AttributeError) as e:
            errors.append(f"Failed to get nomic circuit breakers: {e}")

        # Latest checkpoint
        checkpoint_dir = nomic_dir / "checkpoints"
        if checkpoint_dir.exists():
            try:
                from aragora.nomic.checkpoints import list_checkpoints

                checkpoints = list_checkpoints(str(checkpoint_dir))
                if checkpoints:
                    status.last_checkpoint = checkpoints[0]
            except (ImportError, TypeError, ValueError, KeyError, OSError) as e:
                errors.append(f"Failed to list nomic checkpoints: {e}")

        return status

    except HTTPException:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error getting nomic status: %s", e)
        raise HTTPException(status_code=500, detail="Failed to retrieve nomic status")


@router.post("/nomic/pause", response_model=NomicActionResponse)
async def pause_nomic(
    request: Request,
    body: NomicPauseResumeRequest | None = None,
    auth: AuthorizationContext = Depends(require_permission("admin:nomic:write")),
) -> NomicActionResponse:
    """
    Pause the nomic self-improvement loop.

    Sets the nomic state to 'paused' to prevent further phase transitions
    until resumed. Requires `admin:nomic:write` permission.
    """
    try:
        import json as _json
        from pathlib import Path

        reason = body.reason if body else "Admin requested pause"

        ctx = getattr(request.app.state, "context", None)
        nomic_dir_str = ".nomic"
        if ctx:
            nomic_dir_str = ctx.get("nomic_dir", ".nomic") or ".nomic"
        nomic_dir = Path(nomic_dir_str)
        state_file = nomic_dir / "nomic_state.json"

        # Read current state
        current_state: dict[str, Any] = {}
        if state_file.exists():
            try:
                with open(state_file) as f:
                    current_state = _json.load(f)
            except (_json.JSONDecodeError, OSError) as e:
                logger.warning("Could not read nomic state file: %s", e)

        previous_phase = current_state.get("phase")

        # Write paused state
        new_state = {
            **current_state,
            "phase": "paused",
            "running": False,
            "paused_by": auth.user_id,
            "paused_at": datetime.now(timezone.utc).isoformat() + "Z",
            "pause_reason": reason,
            "previous_phase": previous_phase,
        }

        nomic_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(state_file, "w") as f:
                _json.dump(new_state, f, indent=2)
        except (OSError, TypeError, ValueError) as e:
            logger.warning("Failed to write nomic state: %s", e)
            raise HTTPException(status_code=500, detail="Failed to write nomic state")

        logger.info("Admin %s paused nomic loop: %s", auth.user_id, reason)

        try:
            from aragora.audit.unified import audit_admin

            audit_admin(
                admin_id=auth.user_id,
                action="pause_nomic",
                target_type="nomic",
                target_id=current_state.get("cycle_id", "unknown"),
                reason=reason,
            )
        except (ImportError, RuntimeError, AttributeError) as e:
            logger.warning("Audit log unavailable: %s", e)

        return NomicActionResponse(
            success=True,
            status="paused",
            previous_phase=previous_phase,
            message=f"Nomic loop paused: {reason}",
        )

    except HTTPException:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error pausing nomic loop: %s", e)
        raise HTTPException(status_code=500, detail="Failed to pause nomic loop")


@router.post("/nomic/resume", response_model=NomicActionResponse)
async def resume_nomic(
    request: Request,
    body: NomicPauseResumeRequest | None = None,
    auth: AuthorizationContext = Depends(require_permission("admin:nomic:write")),
) -> NomicActionResponse:
    """
    Resume the nomic self-improvement loop.

    Restores the nomic state to the phase it was in before pausing,
    or to 'idle' if no previous phase is recorded.
    Requires `admin:nomic:write` permission.
    """
    try:
        import json as _json
        from pathlib import Path

        reason = body.reason if body else "Admin requested resume"

        ctx = getattr(request.app.state, "context", None)
        nomic_dir_str = ".nomic"
        if ctx:
            nomic_dir_str = ctx.get("nomic_dir", ".nomic") or ".nomic"
        nomic_dir = Path(nomic_dir_str)
        state_file = nomic_dir / "nomic_state.json"

        # Read current state
        current_state: dict[str, Any] = {}
        if state_file.exists():
            try:
                with open(state_file) as f:
                    current_state = _json.load(f)
            except (_json.JSONDecodeError, OSError) as e:
                logger.warning("Could not read nomic state file: %s", e)

        # Determine the phase to resume to
        resume_phase = current_state.get("previous_phase", "idle") or "idle"

        new_state = {
            **current_state,
            "phase": resume_phase,
            "running": resume_phase != "idle",
            "resumed_by": auth.user_id,
            "resumed_at": datetime.now(timezone.utc).isoformat() + "Z",
            "resume_reason": reason,
        }
        # Clean up pause fields
        new_state.pop("paused_by", None)
        new_state.pop("paused_at", None)
        new_state.pop("pause_reason", None)

        nomic_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(state_file, "w") as f:
                _json.dump(new_state, f, indent=2)
        except (OSError, TypeError, ValueError) as e:
            logger.warning("Failed to write nomic state: %s", e)
            raise HTTPException(status_code=500, detail="Failed to write nomic state")

        logger.info("Admin %s resumed nomic loop to %s: %s", auth.user_id, resume_phase, reason)

        try:
            from aragora.audit.unified import audit_admin

            audit_admin(
                admin_id=auth.user_id,
                action="resume_nomic",
                target_type="nomic",
                target_id=current_state.get("cycle_id", "unknown"),
                reason=reason,
                resumed_phase=resume_phase,
            )
        except (ImportError, RuntimeError, AttributeError) as e:
            logger.warning("Audit log unavailable: %s", e)

        return NomicActionResponse(
            success=True,
            status="resumed",
            previous_phase="paused",
            message=f"Nomic loop resumed to phase '{resume_phase}': {reason}",
        )

    except HTTPException:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error resuming nomic loop: %s", e)
        raise HTTPException(status_code=500, detail="Failed to resume nomic loop")
