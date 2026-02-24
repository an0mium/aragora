"""
Auth Endpoints (FastAPI v2).

Migrated from: aragora/server/handlers/auth/ (legacy handler)

Provides async authentication endpoints:
- POST /api/v2/auth/login   - Authenticate and get tokens
- POST /api/v2/auth/logout  - Invalidate current token
- GET  /api/v2/auth/me      - Get current user info
- POST /api/v2/auth/refresh - Refresh access token

Migration Notes:
    Delegates to existing auth logic in aragora.billing.jwt_auth and
    aragora.server.handlers.auth rather than reimplementing business logic.
    FastAPI dependency injection replaces the legacy handler auth patterns.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from aragora.rbac.models import AuthorizationContext

from ..dependencies.auth import require_authenticated
from ..middleware.error_handling import NotFoundError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/auth", tags=["Authentication"])

# =============================================================================
# Pydantic Models
# =============================================================================


class LoginRequest(BaseModel):
    """Request body for POST /auth/login."""

    email: str = Field(..., min_length=1, description="User email address")
    password: str = Field(..., min_length=1, description="User password")


class TokenResponse(BaseModel):
    """JWT token pair."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 3600

    model_config = {"extra": "allow"}


class LoginResponse(BaseModel):
    """Response for POST /auth/login."""

    user: dict[str, Any] | None = None
    tokens: TokenResponse | None = None
    mfa_required: bool = False
    pending_token: str | None = None
    message: str | None = None
    organization: dict[str, Any] | None = None
    organizations: list[dict[str, Any]] = Field(default_factory=list)


class LogoutResponse(BaseModel):
    """Response for POST /auth/logout."""

    message: str


class MeResponse(BaseModel):
    """Response for GET /auth/me."""

    user: dict[str, Any]
    organization: dict[str, Any] | None = None
    organizations: list[dict[str, Any]] = Field(default_factory=list)


class RefreshRequest(BaseModel):
    """Request body for POST /auth/refresh."""

    refresh_token: str = Field(..., min_length=1, description="Refresh token to exchange")


class RefreshResponse(BaseModel):
    """Response for POST /auth/refresh."""

    tokens: TokenResponse


# =============================================================================
# Dependencies
# =============================================================================


async def get_user_store(request: Request):
    """Dependency to get user store from app state."""
    ctx = getattr(request.app.state, "context", None)
    if ctx:
        store = ctx.get("user_store")
        if store:
            return store

    # Fall back to global user store
    try:
        from aragora.storage.user_store import get_user_store as _get_store

        return _get_store()
    except (ImportError, RuntimeError, OSError) as e:
        logger.warning("User store not available: %s", e)
        raise HTTPException(status_code=503, detail="Authentication service unavailable")


# =============================================================================
# Endpoints
# =============================================================================


AUTH_NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, private",
    "Pragma": "no-cache",
    "Expires": "0",
}


@router.post("/login", response_model=LoginResponse)
async def login(
    body: LoginRequest,
    request: Request,
    user_store=Depends(get_user_store),
) -> LoginResponse:
    """
    Authenticate user and obtain tokens.

    Accepts email and password. Returns JWT tokens or MFA challenge if enabled.
    """
    try:
        from aragora.billing.jwt_auth import create_mfa_pending_token, create_token_pair
        from aragora.auth.lockout import get_lockout_tracker
    except ImportError as e:
        logger.warning("Auth modules not available: %s", e)
        raise HTTPException(status_code=503, detail="Authentication service unavailable")

    email = body.email.strip().lower()
    password = body.password

    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password required")

    # Get client IP for lockout tracking
    client_ip = request.client.host if request.client else "unknown"

    # Check lockout
    try:
        lockout_tracker = get_lockout_tracker()
        if lockout_tracker.is_locked(email=email, ip=client_ip):
            remaining_seconds = lockout_tracker.get_remaining_time(email=email, ip=client_ip)
            remaining_minutes = max(1, remaining_seconds // 60)
            logger.warning("Login attempt on locked account/IP: ip=%s", client_ip)
            raise HTTPException(
                status_code=429,
                detail=f"Too many failed attempts. Try again in {remaining_minutes} minute(s).",
            )
    except HTTPException:
        raise
    except (RuntimeError, ValueError, TypeError, AttributeError) as e:
        logger.debug("Lockout check failed, proceeding: %s", e)
        lockout_tracker = None

    # Find user
    user = user_store.get_user_by_email(email)
    if not user:
        if lockout_tracker:
            lockout_tracker.record_failure(email=email, ip=client_ip)
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # Check if account is active
    if not user.is_active:
        raise HTTPException(status_code=401, detail="Account is disabled")

    # Verify password
    if not user.verify_password(password):
        if lockout_tracker:
            attempts, lockout_seconds = lockout_tracker.record_failure(email=email, ip=client_ip)
            if lockout_seconds:
                remaining_minutes = max(1, lockout_seconds // 60)
                raise HTTPException(
                    status_code=429,
                    detail=f"Too many failed attempts. Account locked for {remaining_minutes} minute(s).",
                )
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # Successful login - reset lockout
    if lockout_tracker:
        lockout_tracker.reset(email=email, ip=client_ip)

    # Check MFA
    if user.mfa_enabled and user.mfa_secret:
        pending_token = create_mfa_pending_token(user.id, user.email)
        return LoginResponse(
            mfa_required=True,
            pending_token=pending_token,
            message="MFA verification required",
        )

    # Create tokens
    tokens = create_token_pair(
        user_id=user.id,
        email=user.email,
        org_id=user.org_id,
        role=user.role,
    )

    logger.info("User logged in: user_id=%s", user.id)

    # Get organization data
    org_data = None
    org_membership: list[dict[str, Any]] = []
    if user.org_id:
        try:
            org = user_store.get_organization_by_id(user.org_id)
            if org:
                org_data = org.to_dict()
                joined_at = getattr(user, "created_at", None)
                org_membership = [
                    {
                        "user_id": user.id,
                        "org_id": org.id,
                        "organization": org_data,
                        "role": user.role or "member",
                        "is_default": True,
                        "joined_at": joined_at.isoformat() if joined_at else None,
                    }
                ]
        except (ValueError, AttributeError, RuntimeError) as e:
            logger.debug("Failed to get org data: %s", e)

    return LoginResponse(
        user=user.to_dict(),
        tokens=TokenResponse(**tokens.to_dict()),
        organization=org_data,
        organizations=org_membership,
    )


@router.post("/logout", response_model=LogoutResponse)
async def logout(
    request: Request,
    auth: AuthorizationContext = Depends(require_authenticated),
) -> LogoutResponse:
    """
    Logout the current user.

    Invalidates the current token by adding it to the blacklist.
    """
    try:
        from aragora.billing.jwt_auth import get_token_blacklist, revoke_token_persistent
    except ImportError:
        raise HTTPException(status_code=503, detail="Authentication service unavailable")

    # Extract token from Authorization header
    token = None
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]

    if token:
        # Persist first, then in-memory for atomic revocation
        try:
            revoke_token_persistent(token)
        except (OSError, ConnectionError, TimeoutError, RuntimeError) as e:
            logger.warning("Failed to persist token revocation: %s", e)

        try:
            blacklist = get_token_blacklist()
            blacklist.revoke_token(token)
        except (RuntimeError, ValueError, TypeError, AttributeError) as e:
            logger.warning("Failed to add token to in-memory blacklist: %s", e)

        logger.info("User logged out: user_id=%s", auth.user_id)
    else:
        logger.info("User logged out (no token to revoke): user_id=%s", auth.user_id)

    return LogoutResponse(message="Logged out successfully")


@router.get("/me", response_model=MeResponse)
async def get_me(
    request: Request,
    auth: AuthorizationContext = Depends(require_authenticated),
    user_store=Depends(get_user_store),
) -> MeResponse:
    """
    Get current user information.

    Returns the authenticated user's profile, organization, and memberships.
    """
    import asyncio

    try:
        # Get user by ID, with async support
        get_by_id_async = getattr(user_store, "get_user_by_id_async", None)
        if get_by_id_async and asyncio.iscoroutinefunction(get_by_id_async):
            user = await get_by_id_async(auth.user_id)
        else:
            get_user_by_id = getattr(user_store, "get_user_by_id", None)
            user = get_user_by_id(auth.user_id) if callable(get_user_by_id) else None

        # Fallback to email lookup if ID lookup fails
        if not user:
            email = getattr(auth, "email", None)
            if email:
                get_by_email_async = getattr(user_store, "get_user_by_email_async", None)
                if get_by_email_async and asyncio.iscoroutinefunction(get_by_email_async):
                    user = await get_by_email_async(email)
                else:
                    get_by_email = getattr(user_store, "get_user_by_email", None)
                    user = get_by_email(email) if callable(get_by_email) else None

        if not user:
            raise NotFoundError("User not found")

        # Get organization data
        org_data = None
        org_membership: list[dict[str, Any]] = []
        if user.org_id:
            try:
                get_org_async = getattr(user_store, "get_organization_by_id_async", None)
                if get_org_async and asyncio.iscoroutinefunction(get_org_async):
                    org = await get_org_async(user.org_id)
                else:
                    get_org_by_id = getattr(user_store, "get_organization_by_id", None)
                    org = get_org_by_id(user.org_id) if callable(get_org_by_id) else None
                if org:
                    org_data = org.to_dict()
                    joined_at = getattr(user, "created_at", None)
                    org_membership = [
                        {
                            "user_id": user.id,
                            "org_id": user.org_id,
                            "organization": org_data,
                            "role": user.role or "member",
                            "is_default": True,
                            "joined_at": joined_at.isoformat() if joined_at else None,
                        }
                    ]
            except (ValueError, AttributeError, RuntimeError) as e:
                logger.debug("Failed to get org data: %s", e)

        return MeResponse(
            user=user.to_dict(),
            organization=org_data,
            organizations=org_membership,
        )

    except NotFoundError:
        raise
    except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
        logger.exception("Error getting user info: %s", e)
        raise HTTPException(status_code=500, detail="Failed to get user info")


@router.post("/refresh", response_model=RefreshResponse)
async def refresh_token(
    body: RefreshRequest,
    request: Request,
    user_store=Depends(get_user_store),
) -> RefreshResponse:
    """
    Refresh access token.

    Exchanges a valid refresh token for a new token pair.
    The old refresh token is revoked to prevent reuse.
    """
    try:
        from aragora.billing.jwt_auth import (
            create_token_pair,
            get_token_blacklist,
            validate_refresh_token,
            revoke_token_persistent,
        )
    except ImportError:
        raise HTTPException(status_code=503, detail="Authentication service unavailable")

    refresh_token_str = body.refresh_token
    if not refresh_token_str:
        raise HTTPException(status_code=400, detail="Refresh token required")

    # Validate refresh token
    payload = validate_refresh_token(refresh_token_str)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")

    # Get user to ensure they still exist and are active
    get_user_by_id = getattr(user_store, "get_user_by_id", None)
    user = get_user_by_id(payload.user_id) if callable(get_user_by_id) else None

    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    if not user.is_active:
        raise HTTPException(status_code=401, detail="Account is disabled")

    # Revoke old refresh token (persistent first, then in-memory)
    try:
        revoke_token_persistent(refresh_token_str)
    except (OSError, ConnectionError, TimeoutError, RuntimeError) as e:
        logger.error("Failed to persist token revocation: %s", e)
        raise HTTPException(status_code=500, detail="Token revocation failed, please try again")

    try:
        blacklist = get_token_blacklist()
        blacklist.revoke_token(refresh_token_str)
    except (RuntimeError, ValueError, TypeError, AttributeError) as e:
        logger.warning("Failed to update in-memory blacklist: %s", e)

    # Create new token pair
    tokens = create_token_pair(
        user_id=user.id,
        email=user.email,
        org_id=user.org_id,
        role=user.role,
    )

    return RefreshResponse(tokens=TokenResponse(**tokens.to_dict()))
