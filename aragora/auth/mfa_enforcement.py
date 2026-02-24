"""
MFA Enforcement for Admin Roles.

Policy-based MFA enforcement middleware that integrates with the RBAC
AuthorizationContext to require multi-factor authentication for users
with administrative roles.

Implements GitHub issue #275: MFA enforcement for all admin roles.

SOC 2 Control: CC5-01 - Enforce MFA for administrative access.

Usage:
    from aragora.auth.mfa_enforcement import (
        MFAEnforcementMiddleware,
        MFAEnforcementPolicy,
        check_mfa_required,
        get_mfa_status,
    )

    policy = MFAEnforcementPolicy()
    middleware = MFAEnforcementMiddleware(policy)
    result = middleware.enforce(user_context, path="/api/v1/admin/users")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# Default roles that require MFA enforcement
DEFAULT_MFA_REQUIRED_ROLES: frozenset[str] = frozenset({
    "admin",
    "owner",
    "superadmin",
    "super_admin",
    "org_admin",
    "workspace_admin",
    "security_admin",
    "compliance_officer",
})

# Paths exempt from MFA enforcement (health, public, auth flows)
DEFAULT_EXEMPT_PATHS: frozenset[str] = frozenset({
    "/health",
    "/healthz",
    "/ready",
    "/metrics",
    "/api/docs",
    "/openapi.json",
    "/api/auth/login",
    "/api/auth/register",
    "/api/auth/callback",
    "/api/auth/refresh",
    "/api/auth/mfa/verify",
    "/api/auth/mfa/setup",
    "/api/auth/sso/login",
    "/api/auth/sso/callback",
})

# Default exempt path prefixes (matched with startswith)
DEFAULT_EXEMPT_PREFIXES: frozenset[str] = frozenset({
    "/api/auth/oauth/",
    "/api/auth/sso/",
    "/scim/v2/",
})


class MFAEnforcementResult(str, Enum):
    """Outcome of an MFA enforcement check."""

    ALLOWED = "allowed"
    DENIED = "denied"
    GRACE_PERIOD = "grace_period"
    EXEMPT_PATH = "exempt_path"
    NOT_REQUIRED = "not_required"


@dataclass(frozen=True)
class MFAEnforcementPolicy:
    """
    Configuration for MFA enforcement on admin roles.

    Attributes:
        required_roles: Roles that must have MFA enabled.
        grace_period_hours: Hours to allow after role promotion before enforcing.
        exempt_paths: Exact paths exempt from MFA enforcement.
        exempt_prefixes: Path prefixes exempt from MFA enforcement.
        enabled: Master switch to enable/disable enforcement.
    """

    required_roles: frozenset[str] = DEFAULT_MFA_REQUIRED_ROLES
    grace_period_hours: int = 48
    exempt_paths: frozenset[str] = DEFAULT_EXEMPT_PATHS
    exempt_prefixes: frozenset[str] = DEFAULT_EXEMPT_PREFIXES
    enabled: bool = True
    auto_disable_on_grace_expiry: bool = False


@dataclass
class MFAStatus:
    """
    MFA status for a user.

    Attributes:
        user_id: The user identifier.
        mfa_enabled: Whether MFA is currently enabled.
        mfa_verified_at: When MFA was last verified in the current session.
        role_assigned_at: When the admin role was assigned (for grace period).
        is_service_account: Whether the user is a service account.
        has_valid_bypass: Whether the user has a valid MFA bypass.
    """

    user_id: str
    mfa_enabled: bool = False
    mfa_verified_at: datetime | None = None
    role_assigned_at: datetime | None = None
    is_service_account: bool = False
    has_valid_bypass: bool = False


@dataclass
class MFAEnforcementDecision:
    """
    Result of an MFA enforcement check.

    Attributes:
        result: The enforcement outcome.
        allowed: Whether the request should proceed.
        reason: Human-readable explanation.
        grace_period_remaining_hours: Hours remaining in grace period (if applicable).
        user_id: The checked user ID.
        required_action: URL or instruction for remediation.
    """

    result: MFAEnforcementResult
    allowed: bool
    reason: str
    grace_period_remaining_hours: int | None = None
    user_id: str | None = None
    required_action: str | None = None


class MFAEnforcementMiddleware:
    """
    Middleware that enforces MFA for admin operations.

    Checks if the current user's role requires MFA and verifies that MFA
    was completed. Integrates with the RBAC AuthorizationContext.
    """

    def __init__(
        self,
        policy: MFAEnforcementPolicy | None = None,
        user_store: Any = None,
    ) -> None:
        self._policy = policy or MFAEnforcementPolicy()
        self._user_store = user_store

    @property
    def policy(self) -> MFAEnforcementPolicy:
        """Return the current enforcement policy."""
        return self._policy

    def enforce(
        self,
        user_context: Any,
        path: str = "",
        mfa_status: MFAStatus | None = None,
    ) -> MFAEnforcementDecision:
        """
        Check whether the request should be allowed based on MFA policy.

        Args:
            user_context: An AuthorizationContext or object with user_id and roles.
            path: The request path (for exempt path checking).
            mfa_status: Pre-fetched MFA status. If None, will be resolved
                        from the user_store or user_context metadata.

        Returns:
            MFAEnforcementDecision with the enforcement outcome.
        """
        if not self._policy.enabled:
            return MFAEnforcementDecision(
                result=MFAEnforcementResult.NOT_REQUIRED,
                allowed=True,
                reason="MFA enforcement is disabled",
            )

        # Check exempt paths
        if self._is_exempt_path(path):
            return MFAEnforcementDecision(
                result=MFAEnforcementResult.EXEMPT_PATH,
                allowed=True,
                reason="Path is exempt from MFA enforcement",
            )

        # Extract user information
        user_id = getattr(user_context, "user_id", None)
        raw_roles: Any = getattr(user_context, "roles", set())
        roles: set[str]
        if isinstance(raw_roles, (set, list, tuple)):
            roles = {str(role) for role in raw_roles}
        else:
            roles = set()

        # Check if user's role requires MFA
        if not self._role_requires_mfa(roles):
            return MFAEnforcementDecision(
                result=MFAEnforcementResult.NOT_REQUIRED,
                allowed=True,
                reason="User role does not require MFA",
                user_id=user_id,
            )

        # Resolve MFA status
        if mfa_status is None:
            mfa_status = self._resolve_mfa_status(user_id, user_context)

        # Service account with valid bypass
        if mfa_status.is_service_account and mfa_status.has_valid_bypass:
            logger.warning(
                "MFA enforcement: service account %s bypassing MFA requirement",
                user_id,
            )
            return MFAEnforcementDecision(
                result=MFAEnforcementResult.ALLOWED,
                allowed=True,
                reason="Service account with valid MFA bypass",
                user_id=user_id,
            )

        # MFA is enabled and verified -- allow
        if mfa_status.mfa_enabled:
            return MFAEnforcementDecision(
                result=MFAEnforcementResult.ALLOWED,
                allowed=True,
                reason="MFA is enabled for admin user",
                user_id=user_id,
            )

        # MFA not enabled -- check grace period for newly-promoted admins
        grace_decision = self._check_grace_period(mfa_status)
        if grace_decision is not None:
            return grace_decision

        # MFA not enabled, no grace period -- deny
        logger.warning(
            "MFA enforcement: admin user %s denied access without MFA (roles: %s, path: %s)",
            user_id,
            roles,
            path,
        )
        return MFAEnforcementDecision(
            result=MFAEnforcementResult.DENIED,
            allowed=False,
            reason="Administrative access requires MFA. Please enable MFA to continue.",
            user_id=user_id,
            required_action="/api/auth/mfa/setup",
        )

    def _is_exempt_path(self, path: str) -> bool:
        """Check if the path is exempt from MFA enforcement."""
        if not path:
            return False

        # Normalize: strip version prefix for matching
        normalized = path.rstrip("/")

        if normalized in self._policy.exempt_paths:
            return True

        for prefix in self._policy.exempt_prefixes:
            if normalized.startswith(prefix):
                return True

        return False

    def _role_requires_mfa(self, roles: set[str]) -> bool:
        """Check if any of the user's roles require MFA."""
        if not roles:
            return False
        # Normalize to lowercase for comparison
        normalized = {r.lower() for r in roles}
        return bool(normalized & self._policy.required_roles)

    def _resolve_mfa_status(
        self,
        user_id: str | None,
        user_context: Any,
    ) -> MFAStatus:
        """Resolve MFA status from user_store or context metadata."""
        if user_id is None:
            return MFAStatus(user_id="unknown")

        # Try user_store first
        if self._user_store is not None:
            return _fetch_mfa_status_from_store(user_id, self._user_store)

        # Fallback: extract from context metadata
        metadata = getattr(user_context, "metadata", {}) or {}
        return MFAStatus(
            user_id=user_id,
            mfa_enabled=bool(metadata.get("mfa_enabled", False)),
            mfa_verified_at=metadata.get("mfa_verified_at"),
            role_assigned_at=metadata.get("role_assigned_at"),
        )

    def _check_grace_period(
        self,
        mfa_status: MFAStatus,
    ) -> MFAEnforcementDecision | None:
        """
        Check if the user is within the grace period for MFA setup.

        Returns an ALLOWED decision with a warning if within the grace period,
        or None if the grace period has expired or is not applicable.
        """
        if self._policy.grace_period_hours <= 0:
            return None

        role_assigned_at = mfa_status.role_assigned_at
        if role_assigned_at is None:
            return None

        now = datetime.now(timezone.utc)

        # Ensure timezone-aware comparison
        if isinstance(role_assigned_at, str):
            try:
                role_assigned_at = datetime.fromisoformat(
                    role_assigned_at.replace("Z", "+00:00")
                )
            except ValueError:
                return None

        if role_assigned_at.tzinfo is None:
            role_assigned_at = role_assigned_at.replace(tzinfo=timezone.utc)

        grace_end = role_assigned_at + timedelta(hours=self._policy.grace_period_hours)

        if now < grace_end:
            remaining = grace_end - now
            remaining_hours = int(remaining.total_seconds() / 3600)
            logger.info(
                "MFA enforcement: admin user %s within grace period (%d hours remaining)",
                mfa_status.user_id,
                remaining_hours,
            )
            return MFAEnforcementDecision(
                result=MFAEnforcementResult.GRACE_PERIOD,
                allowed=True,
                reason=(
                    f"MFA not yet enabled. Grace period: {remaining_hours} hours remaining. "
                    "Please enable MFA before the grace period expires."
                ),
                grace_period_remaining_hours=remaining_hours,
                user_id=mfa_status.user_id,
                required_action="/api/auth/mfa/setup",
            )

        return None


def check_mfa_required(user_context: Any) -> bool:
    """
    Quick check whether MFA is required for the given user context.

    Args:
        user_context: An AuthorizationContext or object with ``roles``.

    Returns:
        True if the user holds a role that requires MFA.
    """
    raw_roles: Any = getattr(user_context, "roles", set())
    roles: set[str]
    if isinstance(raw_roles, (set, list, tuple)):
        roles = {str(role) for role in raw_roles}
    else:
        roles = set()
    normalized = {r.lower() for r in roles}
    return bool(normalized & DEFAULT_MFA_REQUIRED_ROLES)


def get_mfa_status(user_id: str, user_store: Any = None) -> MFAStatus:
    """
    Retrieve MFA status for a user.

    Args:
        user_id: The user identifier.
        user_store: Optional user storage backend. If None, returns
                    a default (unenrolled) status.

    Returns:
        MFAStatus with the user's current MFA state.
    """
    if user_store is None:
        return MFAStatus(user_id=user_id)
    return _fetch_mfa_status_from_store(user_id, user_store)


def _fetch_mfa_status_from_store(user_id: str, user_store: Any) -> MFAStatus:
    """Fetch MFA status from a user store backend."""
    try:
        user = user_store.get_user_by_id(user_id)
    except Exception:
        logger.warning("Failed to fetch user %s from store", user_id)
        return MFAStatus(user_id=user_id)

    if user is None:
        return MFAStatus(user_id=user_id)

    # Extract grace period start (set when user is promoted to admin)
    role_assigned_at = getattr(user, "mfa_grace_period_started_at", None)
    if role_assigned_at is None:
        role_assigned_at = getattr(user, "role_assigned_at", None)
    if role_assigned_at is None:
        role_assigned_at = getattr(user, "created_at", None)

    is_service = getattr(user, "is_service_account", False)

    has_bypass = False
    if is_service:
        if hasattr(user, "is_mfa_bypass_valid"):
            has_bypass = user.is_mfa_bypass_valid()
        else:
            has_bypass = bool(getattr(user, "mfa_bypass_approved_at", None))

    return MFAStatus(
        user_id=user_id,
        mfa_enabled=bool(getattr(user, "mfa_enabled", False)),
        mfa_verified_at=getattr(user, "mfa_verified_at", None),
        role_assigned_at=role_assigned_at,
        is_service_account=is_service,
        has_valid_bypass=has_bypass,
    )


__all__ = [
    "DEFAULT_MFA_REQUIRED_ROLES",
    "MFAEnforcementDecision",
    "MFAEnforcementMiddleware",
    "MFAEnforcementPolicy",
    "MFAEnforcementResult",
    "MFAStatus",
    "check_mfa_required",
    "get_mfa_status",
]
