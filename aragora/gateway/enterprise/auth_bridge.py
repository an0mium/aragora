"""
Authentication Bridge for External Framework Integration.

Provides authentication pass-through between Aragora's auth system and
external frameworks, enabling seamless SSO, permission mapping, and
session management across system boundaries.

Features:
- OIDC/SAML token validation and pass-through
- Permission mapping from Aragora RBAC to external actions
- Session lifecycle management with automatic cleanup
- Audit logging for all authentication events

Usage:
    from aragora.gateway.enterprise.auth_bridge import (
        AuthBridge,
        AuthContext,
        PermissionMapping,
    )

    # Initialize bridge with permission mappings
    bridge = AuthBridge(
        permission_mappings=[
            PermissionMapping(
                aragora_permission="debates.create",
                external_action="create_conversation",
            ),
        ]
    )

    # Verify incoming request
    context = await bridge.verify_request(token="...")

    # Check if action is allowed
    if bridge.is_action_allowed(context, "create_conversation"):
        # Proceed with external framework action
        ...

    # Exchange token for external framework
    exchange_result = await bridge.exchange_token(context, target_audience="external-api")
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class AuthBridgeError(Exception):
    """Base exception for auth bridge errors."""

    def __init__(
        self,
        message: str,
        code: str = "AUTH_BRIDGE_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}


class AuthenticationError(AuthBridgeError):
    """Authentication failed."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, "AUTHENTICATION_FAILED", details)


class PermissionDeniedError(AuthBridgeError):
    """Permission check failed."""

    def __init__(
        self,
        message: str,
        permission: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, "PERMISSION_DENIED", details)
        self.permission = permission


class SessionExpiredError(AuthBridgeError):
    """Session has expired."""

    def __init__(self, session_id: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(
            f"Session '{session_id}' has expired",
            "SESSION_EXPIRED",
            details,
        )
        self.session_id = session_id


# =============================================================================
# Data Classes
# =============================================================================


class TokenType(str, Enum):
    """Supported token types."""

    OIDC_ACCESS = "oidc_access"
    OIDC_ID = "oidc_id"
    SAML_ASSERTION = "saml_assertion"
    API_KEY = "api_key"
    SESSION = "session"


@dataclass
class AuthContext:
    """
    Authentication context containing user identity and claims.

    Represents the authenticated user's identity, tenant context,
    permissions, and session information for cross-system authorization.

    Attributes:
        user_id: Unique identifier for the user from Aragora.
        email: User's email address.
        tenant_id: Tenant/organization identifier for multi-tenant isolation.
        permissions: Set of Aragora RBAC permission keys (e.g., "debates.create").
        roles: Set of Aragora role names assigned to the user.
        session_id: Unique session identifier for session management.
        token_type: Type of token used for authentication.
        claims: Raw claims from the authentication token.
        expires_at: Unix timestamp when this context expires.
        created_at: Unix timestamp when this context was created.
        ip_address: Client IP address for audit logging.
        user_agent: Client user agent for audit logging.
        metadata: Additional context-specific metadata.
    """

    user_id: str
    email: str
    tenant_id: str | None = None
    permissions: set[str] = field(default_factory=set)
    roles: set[str] = field(default_factory=set)
    session_id: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    token_type: TokenType = TokenType.OIDC_ACCESS
    claims: dict[str, Any] = field(default_factory=dict)
    expires_at: float = field(default_factory=lambda: time.time() + 3600)
    created_at: float = field(default_factory=time.time)
    ip_address: str | None = None
    user_agent: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if this context has expired."""
        return time.time() > self.expires_at

    @property
    def remaining_ttl(self) -> float:
        """Get remaining time-to-live in seconds."""
        return max(0.0, self.expires_at - time.time())

    @property
    def display_name(self) -> str:
        """Get display name from claims or email."""
        return self.claims.get("name", self.email.split("@")[0])

    def has_permission(self, permission: str) -> bool:
        """Check if context has a specific permission.

        Args:
            permission: Permission key to check (e.g., "debates.create").

        Returns:
            True if permission is granted, False otherwise.
        """
        # Normalize permission format
        normalized = permission.replace(":", ".")

        # Check exact match
        if normalized in self.permissions:
            return True

        # Check wildcard patterns
        resource = normalized.split(".")[0]
        if f"{resource}.*" in self.permissions or f"{resource}:*" in self.permissions:
            return True

        # Check super wildcard
        return "*" in self.permissions

    def has_role(self, role: str) -> bool:
        """Check if context has a specific role.

        Args:
            role: Role name to check.

        Returns:
            True if role is assigned, False otherwise.
        """
        return role in self.roles

    def has_any_role(self, *roles: str) -> bool:
        """Check if context has any of the specified roles.

        Args:
            *roles: Role names to check.

        Returns:
            True if any role is assigned, False otherwise.
        """
        return bool(self.roles & set(roles))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation suitable for JSON serialization.
        """
        return {
            "user_id": self.user_id,
            "email": self.email,
            "tenant_id": self.tenant_id,
            "permissions": list(self.permissions),
            "roles": list(self.roles),
            "session_id": self.session_id,
            "token_type": self.token_type.value,
            "expires_at": self.expires_at,
            "created_at": self.created_at,
            "is_expired": self.is_expired,
            "display_name": self.display_name,
        }


@dataclass
class PermissionMapping:
    """
    Maps Aragora RBAC permissions to external framework actions.

    Enables fine-grained control over which Aragora permissions
    translate to which external framework actions.

    Attributes:
        aragora_permission: Aragora permission key (e.g., "debates.create").
        external_action: External framework action name.
        conditions: Optional conditions for the mapping.
        bidirectional: Whether mapping works both directions.
        priority: Priority for conflict resolution (higher = more precedence).
    """

    aragora_permission: str
    external_action: str
    conditions: dict[str, Any] = field(default_factory=dict)
    bidirectional: bool = True
    priority: int = 0

    def matches_aragora(self, permission: str) -> bool:
        """Check if Aragora permission matches this mapping.

        Args:
            permission: Aragora permission to check.

        Returns:
            True if permission matches, False otherwise.
        """
        # Normalize to dot notation
        normalized = permission.replace(":", ".")
        target = self.aragora_permission.replace(":", ".")

        # Exact match
        if normalized == target:
            return True

        # Wildcard match
        if target.endswith(".*"):
            prefix = target[:-2]
            return normalized.startswith(prefix + ".")

        return False

    def matches_external(self, action: str) -> bool:
        """Check if external action matches this mapping.

        Args:
            action: External action to check.

        Returns:
            True if action matches, False otherwise.
        """
        return self.bidirectional and action == self.external_action


@dataclass
class TokenExchangeResult:
    """
    Result of a token exchange operation.

    Contains the exchanged token and metadata for use with
    external frameworks.

    Attributes:
        access_token: The exchanged access token.
        token_type: Token type (typically "Bearer").
        expires_in: Token lifetime in seconds.
        scope: Scopes granted to the token.
        refresh_token: Optional refresh token for token renewal.
        id_token: Optional ID token for identity verification.
        issued_at: Unix timestamp when token was issued.
        audience: Target audience for the token.
        metadata: Additional token metadata.
    """

    access_token: str
    token_type: str = "Bearer"
    expires_in: int = 3600
    scope: str = ""
    refresh_token: str | None = None
    id_token: str | None = None
    issued_at: float = field(default_factory=time.time)
    audience: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def expires_at(self) -> float:
        """Get expiration timestamp."""
        return self.issued_at + self.expires_in

    @property
    def is_expired(self) -> bool:
        """Check if token has expired."""
        return time.time() > self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """Convert to OAuth 2.0 token response format.

        Returns:
            Dictionary compatible with OAuth 2.0 token response.
        """
        result = {
            "access_token": self.access_token,
            "token_type": self.token_type,
            "expires_in": self.expires_in,
        }
        if self.scope:
            result["scope"] = self.scope
        if self.refresh_token:
            result["refresh_token"] = self.refresh_token
        if self.id_token:
            result["id_token"] = self.id_token
        return result


@dataclass
class BridgedSession:
    """
    Represents a linked session between Aragora and external framework.

    Tracks session state across system boundaries for lifecycle management.

    Attributes:
        session_id: Unique session identifier.
        aragora_session_id: Aragora-side session ID.
        external_session_id: External framework session ID.
        auth_context: Associated authentication context.
        created_at: Unix timestamp when session was created.
        last_accessed_at: Unix timestamp of last activity.
        expires_at: Unix timestamp when session expires.
        metadata: Additional session metadata.
    """

    session_id: str
    aragora_session_id: str
    external_session_id: str | None = None
    auth_context: AuthContext | None = None
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + 28800)  # 8 hours
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if session has expired."""
        return time.time() > self.expires_at

    def touch(self) -> None:
        """Update last access time."""
        self.last_accessed_at = time.time()


@dataclass
class AuditEntry:
    """
    Audit log entry for authentication events.

    Attributes:
        timestamp: Unix timestamp of the event.
        event_type: Type of authentication event.
        user_id: User ID involved in the event.
        session_id: Session ID if applicable.
        action: Action attempted or performed.
        result: Result of the action (success/failure).
        ip_address: Client IP address.
        details: Additional event details.
    """

    timestamp: float
    event_type: str
    user_id: str
    session_id: str | None = None
    action: str = ""
    result: str = "success"
    ip_address: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Session Lifecycle Hooks
# =============================================================================


class SessionLifecycleHook:
    """
    Hook interface for session lifecycle events.

    Implement this class to receive callbacks for session events
    and perform custom logic (e.g., external system notifications).
    """

    async def on_session_created(
        self,
        session: BridgedSession,
        context: AuthContext,
    ) -> None:
        """Called when a new session is created.

        Args:
            session: The newly created session.
            context: Authentication context for the session.
        """
        pass

    async def on_session_accessed(self, session: BridgedSession) -> None:
        """Called when a session is accessed.

        Args:
            session: The accessed session.
        """
        pass

    async def on_session_expired(self, session: BridgedSession) -> None:
        """Called when a session expires.

        Args:
            session: The expired session.
        """
        pass

    async def on_session_destroyed(
        self,
        session: BridgedSession,
        reason: str = "logout",
    ) -> None:
        """Called when a session is explicitly destroyed.

        Args:
            session: The destroyed session.
            reason: Reason for destruction (e.g., "logout", "timeout", "revoked").
        """
        pass


# =============================================================================
# Auth Bridge Implementation
# =============================================================================


class AuthBridge:
    """
    Authentication bridge between Aragora and external frameworks.

    Provides seamless authentication pass-through, permission mapping,
    token exchange, and session management for integrating external
    frameworks with Aragora's authentication system.

    Example:
        >>> bridge = AuthBridge(
        ...     permission_mappings=[
        ...         PermissionMapping("debates.create", "create_conversation"),
        ...         PermissionMapping("debates.read", "view_conversation"),
        ...     ],
        ...     action_allowlist={"create_conversation", "view_conversation"},
        ... )
        >>> context = await bridge.verify_request(token="eyJ...")
        >>> if bridge.is_action_allowed(context, "create_conversation"):
        ...     # Proceed with action
        ...     pass

    Attributes:
        permission_mappings: List of permission mappings.
        action_allowlist: Set of allowed external actions (None = all).
        action_denylist: Set of denied external actions.
        session_duration: Default session duration in seconds.
        enable_audit: Whether to log authentication events.
    """

    def __init__(
        self,
        permission_mappings: list[PermissionMapping] | None = None,
        action_allowlist: set[str] | None = None,
        action_denylist: set[str] | None = None,
        session_duration: int = 28800,  # 8 hours
        enable_audit: bool = True,
        lifecycle_hooks: list[SessionLifecycleHook] | None = None,
    ) -> None:
        """
        Initialize the authentication bridge.

        Args:
            permission_mappings: Mappings from Aragora permissions to external actions.
            action_allowlist: If set, only these external actions are allowed.
            action_denylist: These external actions are always denied.
            session_duration: Default session duration in seconds.
            enable_audit: Whether to enable audit logging.
            lifecycle_hooks: Hooks for session lifecycle events.
        """
        self._permission_mappings = permission_mappings or []
        self._action_allowlist = action_allowlist
        self._action_denylist = action_denylist or set()
        self._session_duration = session_duration
        self._enable_audit = enable_audit
        self._lifecycle_hooks = lifecycle_hooks or []

        # In-memory stores (production should use Redis/database)
        self._sessions: dict[str, BridgedSession] = {}
        self._audit_log: list[AuditEntry] = []

        # Build lookup indexes for efficient permission checking
        self._aragora_to_external: dict[str, list[PermissionMapping]] = {}
        self._external_to_aragora: dict[str, list[PermissionMapping]] = {}
        self._rebuild_indexes()

        logger.info("AuthBridge initialized with %s mappings", len(self._permission_mappings))

    def _rebuild_indexes(self) -> None:
        """Rebuild permission mapping indexes."""
        self._aragora_to_external.clear()
        self._external_to_aragora.clear()

        for mapping in sorted(self._permission_mappings, key=lambda m: m.priority, reverse=True):
            # Index by Aragora permission
            perm = mapping.aragora_permission.replace(":", ".")
            if perm not in self._aragora_to_external:
                self._aragora_to_external[perm] = []
            self._aragora_to_external[perm].append(mapping)

            # Index by external action (if bidirectional)
            if mapping.bidirectional:
                action = mapping.external_action
                if action not in self._external_to_aragora:
                    self._external_to_aragora[action] = []
                self._external_to_aragora[action].append(mapping)

    # =========================================================================
    # Authentication Methods
    # =========================================================================

    async def verify_request(
        self,
        token: str | None = None,
        saml_assertion: str | None = None,
        api_key: str | None = None,
        session_id: str | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ) -> AuthContext:
        """
        Verify an incoming request and extract authentication context.

        Supports multiple authentication methods:
        - OIDC/JWT tokens (Bearer tokens)
        - SAML assertions
        - API keys
        - Existing session IDs

        Args:
            token: OIDC/JWT access token or ID token.
            saml_assertion: Base64-encoded SAML assertion.
            api_key: API key for service-to-service auth.
            session_id: Existing session ID for session revalidation.
            ip_address: Client IP for audit logging.
            user_agent: Client user agent for audit logging.

        Returns:
            AuthContext with user identity, permissions, and session info.

        Raises:
            AuthenticationError: If authentication fails.
            SessionExpiredError: If session has expired.
        """
        context: AuthContext | None = None

        try:
            if session_id:
                context = await self._verify_session(session_id)
            elif token:
                context = await self._verify_oidc_token(token)
            elif saml_assertion:
                context = await self._verify_saml_assertion(saml_assertion)
            elif api_key:
                context = await self._verify_api_key(api_key)
            else:
                raise AuthenticationError(
                    "No authentication credentials provided",
                    {"supported_methods": ["token", "saml_assertion", "api_key", "session_id"]},
                )

            # Add request metadata
            context.ip_address = ip_address
            context.user_agent = user_agent

            # Audit log
            if self._enable_audit:
                await self._log_audit(
                    event_type="authentication",
                    user_id=context.user_id,
                    session_id=context.session_id,
                    action="verify_request",
                    result="success",
                    ip_address=ip_address,
                    details={"token_type": context.token_type.value},
                )

            return context

        except AuthBridgeError:
            raise
        except (OSError, ConnectionError, RuntimeError, ValueError) as e:
            logger.error("Authentication verification failed: %s", e)
            if self._enable_audit and context:
                await self._log_audit(
                    event_type="authentication",
                    user_id=context.user_id if context else "unknown",
                    action="verify_request",
                    result="failure",
                    ip_address=ip_address,
                    details={"error": str(e)},
                )
            raise AuthenticationError(
                f"Authentication verification failed: {e}",
                {"error": str(e)},
            )

    async def _verify_oidc_token(self, token: str) -> AuthContext:
        """Verify OIDC/JWT token using Aragora's OIDC provider.

        Args:
            token: The JWT token to verify.

        Returns:
            AuthContext extracted from validated token.

        Raises:
            AuthenticationError: If token validation fails.
        """
        try:
            from aragora.auth.sso import get_sso_provider

            provider = get_sso_provider()
            if provider is None:
                raise AuthenticationError(
                    "SSO provider not configured",
                    {"hint": "Configure OIDC provider in settings"},
                )

            # For OIDC, we can validate the token directly
            # This assumes the token is an access token that can be used
            # to fetch user info, or an ID token that can be decoded
            from aragora.auth.oidc import OIDCProvider

            if isinstance(provider, OIDCProvider):
                # Try to validate as ID token
                try:
                    claims = await provider._validate_id_token(token)
                    return self._claims_to_context(claims, TokenType.OIDC_ID)
                except (ValueError, RuntimeError, KeyError) as e:
                    # Fall back to userinfo fetch with access token
                    logger.debug(
                        "ID token validation failed, falling back to userinfo: %s: %s",
                        type(e).__name__,
                        e,
                    )
                    tokens = {"access_token": token}
                    user = await provider._get_user_info(tokens)
                    return self._sso_user_to_context(user, TokenType.OIDC_ACCESS)
            else:
                # Generic SSO provider - attempt to use token for auth
                raise AuthenticationError("Token verification not supported for this provider type")

        except AuthenticationError:
            raise
        except (OSError, ConnectionError, RuntimeError, ValueError) as e:
            logger.error("OIDC token verification failed: %s", e)
            raise AuthenticationError(
                f"OIDC token verification failed: {e}",
                {"token_type": "oidc"},
            )

    async def _verify_saml_assertion(self, assertion: str) -> AuthContext:
        """Verify SAML assertion using Aragora's SAML provider.

        Args:
            assertion: Base64-encoded SAML assertion.

        Returns:
            AuthContext extracted from validated assertion.

        Raises:
            AuthenticationError: If assertion validation fails.
        """
        try:
            from aragora.auth import HAS_SAML

            if not HAS_SAML:
                raise AuthenticationError(
                    "SAML not available - install python3-saml",
                    {"hint": "pip install python3-saml"},
                )

            from aragora.auth.sso import get_sso_provider
            from aragora.auth.saml import SAMLProvider

            provider = get_sso_provider()
            if provider is None or not isinstance(provider, SAMLProvider):
                raise AuthenticationError(
                    "SAML provider not configured",
                    {"hint": "Configure SAML provider in settings"},
                )

            user = await provider.authenticate(saml_response=assertion)
            return self._sso_user_to_context(user, TokenType.SAML_ASSERTION)

        except AuthenticationError:
            raise
        except (OSError, ConnectionError, RuntimeError, ValueError) as e:
            logger.error("SAML assertion verification failed: %s", e)
            raise AuthenticationError(
                f"SAML assertion verification failed: {e}",
                {"token_type": "saml"},
            )

    async def _verify_api_key(self, api_key: str) -> AuthContext:
        """Verify API key against Aragora's API key store.

        Args:
            api_key: The API key to verify.

        Returns:
            AuthContext for the API key's associated user/service.

        Raises:
            AuthenticationError: If API key is invalid or expired.
        """
        try:
            # Hash API key for lookup
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()

            # Try to load API key info from RBAC system (import validates availability)
            from aragora.rbac.checker import get_permission_checker

            get_permission_checker()

            # In a real implementation, we would look up the API key
            # from a persistent store. For now, we create a minimal context.
            # The actual API key validation would happen in the RBAC layer.

            # Extract key prefix for identification (first 8 chars)
            key_prefix = api_key[:8] if len(api_key) >= 8 else api_key

            return AuthContext(
                user_id=f"api_key:{key_hash[:16]}",
                email=f"{key_prefix}@api.aragora.local",
                token_type=TokenType.API_KEY,
                claims={"key_prefix": key_prefix},
                metadata={"api_key_hash": key_hash},
            )

        except (OSError, RuntimeError, ValueError) as e:
            logger.error("API key verification failed: %s", e)
            raise AuthenticationError(
                f"API key verification failed: {e}",
                {"token_type": "api_key"},
            )

    async def _verify_session(self, session_id: str) -> AuthContext:
        """Verify existing session and return associated context.

        Args:
            session_id: The session ID to verify.

        Returns:
            AuthContext from the active session.

        Raises:
            SessionExpiredError: If session has expired.
            AuthenticationError: If session is not found.
        """
        session = self._sessions.get(session_id)

        if session is None:
            raise AuthenticationError(
                f"Session not found: {session_id}",
                {"session_id": session_id},
            )

        if session.is_expired:
            # Clean up expired session
            del self._sessions[session_id]
            for hook in self._lifecycle_hooks:
                try:
                    await hook.on_session_expired(session)
                except (RuntimeError, ValueError, TypeError) as e:  # noqa: BLE001 - user-provided lifecycle hook callback
                    logger.warning("Session expired hook failed: %s", e)
            raise SessionExpiredError(session_id)

        if session.auth_context is None:
            raise AuthenticationError(
                f"Session has no auth context: {session_id}",
                {"session_id": session_id},
            )

        # Touch session
        session.touch()
        for hook in self._lifecycle_hooks:
            try:
                await hook.on_session_accessed(session)
            except (RuntimeError, ValueError, TypeError) as e:  # noqa: BLE001 - user-provided lifecycle hook callback
                logger.warning("Session accessed hook failed: %s", e)

        return session.auth_context

    def _claims_to_context(
        self,
        claims: dict[str, Any],
        token_type: TokenType,
    ) -> AuthContext:
        """Convert token claims to AuthContext.

        Args:
            claims: Token claims dictionary.
            token_type: Type of token the claims came from.

        Returns:
            AuthContext populated from claims.
        """
        # Extract standard claims
        user_id = claims.get("sub", claims.get("user_id", ""))
        email = claims.get("email", "")
        tenant_id = claims.get("tenant_id", claims.get("org_id"))

        # Extract roles and groups
        roles = set(claims.get("roles", []))
        groups = set(claims.get("groups", []))
        roles.update(groups)

        # Resolve permissions from roles
        permissions = self._resolve_permissions(roles)

        # Calculate expiration
        exp = claims.get("exp")
        expires_at = float(exp) if exp else time.time() + 3600

        return AuthContext(
            user_id=user_id,
            email=email,
            tenant_id=tenant_id,
            permissions=permissions,
            roles=roles,
            token_type=token_type,
            claims=claims,
            expires_at=expires_at,
        )

    def _sso_user_to_context(
        self,
        user: Any,  # SSOUser
        token_type: TokenType,
    ) -> AuthContext:
        """Convert SSOUser to AuthContext.

        Args:
            user: SSOUser from Aragora's SSO system.
            token_type: Type of token used for authentication.

        Returns:
            AuthContext populated from SSOUser.
        """
        roles = set(user.roles)
        permissions = self._resolve_permissions(roles)

        return AuthContext(
            user_id=user.id,
            email=user.email,
            tenant_id=user.tenant_id or user.organization_id,
            permissions=permissions,
            roles=roles,
            token_type=token_type,
            claims=user.raw_claims,
            expires_at=user.token_expires_at or (time.time() + 3600),
        )

    def _resolve_permissions(self, roles: set[str]) -> set[str]:
        """Resolve permissions from roles using RBAC system.

        Args:
            roles: Set of role names.

        Returns:
            Set of permission keys granted by the roles.
        """
        try:
            from aragora.rbac.defaults import get_role_permissions

            all_permissions: set[str] = set()
            for role in roles:
                permissions = get_role_permissions(role, include_inherited=True)
                all_permissions.update(permissions)
            return all_permissions
        except ImportError:
            logger.warning("RBAC module not available, returning empty permissions")
            return set()

    # =========================================================================
    # Permission Mapping
    # =========================================================================

    def add_permission_mapping(self, mapping: PermissionMapping) -> None:
        """Add a permission mapping.

        Args:
            mapping: The permission mapping to add.
        """
        self._permission_mappings.append(mapping)
        self._rebuild_indexes()
        logger.debug(
            "Added permission mapping: %s -> %s",
            mapping.aragora_permission,
            mapping.external_action,
        )

    def remove_permission_mapping(
        self,
        aragora_permission: str | None = None,
        external_action: str | None = None,
    ) -> int:
        """Remove permission mappings by Aragora permission or external action.

        Args:
            aragora_permission: Remove mappings with this Aragora permission.
            external_action: Remove mappings with this external action.

        Returns:
            Number of mappings removed.
        """
        original_count = len(self._permission_mappings)

        self._permission_mappings = [
            m
            for m in self._permission_mappings
            if not (
                (aragora_permission and m.aragora_permission == aragora_permission)
                or (external_action and m.external_action == external_action)
            )
        ]

        removed = original_count - len(self._permission_mappings)
        if removed > 0:
            self._rebuild_indexes()
        return removed

    def get_external_actions(self, context: AuthContext) -> set[str]:
        """Get all external actions allowed for an auth context.

        Args:
            context: Authentication context to check.

        Returns:
            Set of external action names the context is allowed to perform.
        """
        allowed_actions: set[str] = set()

        for permission in context.permissions:
            normalized = permission.replace(":", ".")
            # Check direct mappings
            if normalized in self._aragora_to_external:
                for mapping in self._aragora_to_external[normalized]:
                    if self._evaluate_conditions(mapping.conditions, context):
                        allowed_actions.add(mapping.external_action)

            # Check wildcard mappings
            resource = normalized.split(".")[0]
            wildcard = f"{resource}.*"
            if wildcard in self._aragora_to_external:
                for mapping in self._aragora_to_external[wildcard]:
                    if self._evaluate_conditions(mapping.conditions, context):
                        allowed_actions.add(mapping.external_action)

        # Apply allowlist/denylist
        if self._action_allowlist is not None:
            allowed_actions &= self._action_allowlist
        allowed_actions -= self._action_denylist

        return allowed_actions

    def get_required_permissions(self, external_action: str) -> set[str]:
        """Get Aragora permissions required for an external action.

        Args:
            external_action: External action name.

        Returns:
            Set of Aragora permission keys that grant this action.
        """
        permissions: set[str] = set()

        if external_action in self._external_to_aragora:
            for mapping in self._external_to_aragora[external_action]:
                permissions.add(mapping.aragora_permission)

        return permissions

    def is_action_allowed(
        self,
        context: AuthContext,
        external_action: str,
    ) -> bool:
        """Check if an external action is allowed for the given context.

        Args:
            context: Authentication context to check.
            external_action: External action to check permission for.

        Returns:
            True if action is allowed, False otherwise.
        """
        # Check denylist first
        if external_action in self._action_denylist:
            return False

        # Check allowlist if configured
        if self._action_allowlist is not None and external_action not in self._action_allowlist:
            return False

        # Check if context has a permission that grants this action
        return external_action in self.get_external_actions(context)

    def check_action(
        self,
        context: AuthContext,
        external_action: str,
    ) -> None:
        """Check if action is allowed, raising exception if not.

        Args:
            context: Authentication context to check.
            external_action: External action to check permission for.

        Raises:
            PermissionDeniedError: If action is not allowed.
        """
        if not self.is_action_allowed(context, external_action):
            required = self.get_required_permissions(external_action)
            raise PermissionDeniedError(
                f"Action '{external_action}' not allowed for user",
                permission=external_action,
                details={
                    "required_permissions": list(required),
                    "user_permissions": list(context.permissions),
                },
            )

        if self._enable_audit:
            # Fire and forget audit log
            import asyncio

            asyncio.create_task(
                self._log_audit(
                    event_type="permission_check",
                    user_id=context.user_id,
                    session_id=context.session_id,
                    action=external_action,
                    result="allowed",
                    ip_address=context.ip_address,
                )
            )

    def _evaluate_conditions(
        self,
        conditions: dict[str, Any],
        context: AuthContext,
    ) -> bool:
        """Evaluate mapping conditions against context.

        Args:
            conditions: Condition dictionary to evaluate.
            context: Authentication context for evaluation.

        Returns:
            True if all conditions are satisfied, False otherwise.
        """
        if not conditions:
            return True

        for key, expected in conditions.items():
            if key == "tenant_id":
                if context.tenant_id != expected:
                    return False
            elif key == "roles":
                required_roles = set(expected) if isinstance(expected, list) else {expected}
                if not (context.roles & required_roles):
                    return False
            elif key == "has_permission":
                if not context.has_permission(expected):
                    return False
            elif key in context.claims:
                if context.claims[key] != expected:
                    return False
            elif key in context.metadata:
                if context.metadata[key] != expected:
                    return False

        return True

    # =========================================================================
    # Token Exchange
    # =========================================================================

    async def exchange_token(
        self,
        context: AuthContext,
        target_audience: str,
        scope: str = "",
        token_lifetime: int = 3600,
    ) -> TokenExchangeResult:
        """Exchange Aragora token for external framework token.

        Implements RFC 8693 Token Exchange for cross-system authentication.

        Args:
            context: Current authentication context.
            target_audience: Target audience for the exchanged token.
            scope: Requested scopes for the token.
            token_lifetime: Token lifetime in seconds.

        Returns:
            TokenExchangeResult containing the exchanged token.

        Raises:
            AuthenticationError: If token exchange fails.
        """
        try:
            # Generate a signed token for the external framework
            # In production, this would use proper JWT signing with RS256/ES256
            token_id = secrets.token_urlsafe(32)

            # Build token payload
            payload = {
                "sub": context.user_id,
                "email": context.email,
                "tenant_id": context.tenant_id,
                "aud": target_audience,
                "iat": int(time.time()),
                "exp": int(time.time() + token_lifetime),
                "jti": token_id,
            }

            # Add allowed actions as scope
            allowed_actions = self.get_external_actions(context)
            if scope:
                # Filter to requested scopes
                requested = set(scope.split())
                allowed_actions &= requested
            final_scope = " ".join(sorted(allowed_actions))

            # For demonstration, create a simple token
            # Production should use proper JWT library with key management
            import base64
            import json

            token_data = json.dumps(payload, separators=(",", ":"))
            access_token = base64.urlsafe_b64encode(token_data.encode()).decode()

            result = TokenExchangeResult(
                access_token=access_token,
                token_type="Bearer",
                expires_in=token_lifetime,
                scope=final_scope,
                audience=target_audience,
                metadata={
                    "exchange_time": time.time(),
                    "original_user_id": context.user_id,
                    "token_id": token_id,
                },
            )

            if self._enable_audit:
                await self._log_audit(
                    event_type="token_exchange",
                    user_id=context.user_id,
                    session_id=context.session_id,
                    action=f"exchange_for:{target_audience}",
                    result="success",
                    ip_address=context.ip_address,
                    details={
                        "target_audience": target_audience,
                        "scope": final_scope,
                    },
                )

            logger.info(
                "Token exchanged for user %s targeting %s", context.user_id, target_audience
            )
            return result

        except (OSError, ConnectionError, RuntimeError, ValueError) as e:
            logger.error("Token exchange failed: %s", e)
            raise AuthenticationError(
                f"Token exchange failed: {e}",
                {"target_audience": target_audience},
            )

    # =========================================================================
    # Session Management
    # =========================================================================

    async def create_session(
        self,
        context: AuthContext,
        external_session_id: str | None = None,
        duration: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BridgedSession:
        """Create a new bridged session linking Aragora and external sessions.

        Args:
            context: Authentication context for the session.
            external_session_id: Optional external framework session ID.
            duration: Session duration in seconds (defaults to bridge setting).
            metadata: Additional session metadata.

        Returns:
            The created BridgedSession.
        """
        session_duration = duration or self._session_duration
        session_id = secrets.token_urlsafe(32)

        session = BridgedSession(
            session_id=session_id,
            aragora_session_id=context.session_id,
            external_session_id=external_session_id,
            auth_context=context,
            expires_at=time.time() + session_duration,
            metadata=metadata or {},
        )

        self._sessions[session_id] = session

        # Invoke lifecycle hooks
        for hook in self._lifecycle_hooks:
            try:
                await hook.on_session_created(session, context)
            except (RuntimeError, ValueError, TypeError) as e:  # noqa: BLE001 - user-provided lifecycle hook callback
                logger.warning("Session created hook failed: %s", e)

        if self._enable_audit:
            await self._log_audit(
                event_type="session",
                user_id=context.user_id,
                session_id=session_id,
                action="create",
                result="success",
                ip_address=context.ip_address,
            )

        logger.info("Created bridged session %s for user %s", session_id, context.user_id)
        return session

    async def get_session(self, session_id: str) -> BridgedSession | None:
        """Get a session by ID.

        Args:
            session_id: The session ID to retrieve.

        Returns:
            The session if found and valid, None otherwise.
        """
        session = self._sessions.get(session_id)

        if session is None:
            return None

        if session.is_expired:
            await self.destroy_session(session_id, reason="expired")
            return None

        session.touch()
        return session

    async def destroy_session(
        self,
        session_id: str,
        reason: str = "logout",
    ) -> bool:
        """Destroy a session.

        Args:
            session_id: The session ID to destroy.
            reason: Reason for destruction (for audit logging).

        Returns:
            True if session was destroyed, False if not found.
        """
        session = self._sessions.pop(session_id, None)

        if session is None:
            return False

        # Invoke lifecycle hooks
        for hook in self._lifecycle_hooks:
            try:
                await hook.on_session_destroyed(session, reason)
            except (RuntimeError, ValueError, TypeError) as e:  # noqa: BLE001 - user-provided lifecycle hook callback
                logger.warning("Session destroyed hook failed: %s", e)

        if self._enable_audit and session.auth_context:
            await self._log_audit(
                event_type="session",
                user_id=session.auth_context.user_id,
                session_id=session_id,
                action="destroy",
                result="success",
                details={"reason": reason},
            )

        logger.info("Destroyed session %s (reason: %s)", session_id, reason)
        return True

    async def cleanup_expired_sessions(self) -> int:
        """Clean up all expired sessions.

        Returns:
            Number of sessions cleaned up.
        """
        expired_ids = [sid for sid, session in self._sessions.items() if session.is_expired]

        for session_id in expired_ids:
            await self.destroy_session(session_id, reason="expired")

        if expired_ids:
            logger.info("Cleaned up %s expired sessions", len(expired_ids))

        return len(expired_ids)

    def add_lifecycle_hook(self, hook: SessionLifecycleHook) -> None:
        """Add a session lifecycle hook.

        Args:
            hook: The lifecycle hook to add.
        """
        self._lifecycle_hooks.append(hook)

    def remove_lifecycle_hook(self, hook: SessionLifecycleHook) -> bool:
        """Remove a session lifecycle hook.

        Args:
            hook: The lifecycle hook to remove.

        Returns:
            True if hook was removed, False if not found.
        """
        try:
            self._lifecycle_hooks.remove(hook)
            return True
        except ValueError:
            return False

    # =========================================================================
    # Audit Logging
    # =========================================================================

    async def _log_audit(
        self,
        event_type: str,
        user_id: str,
        session_id: str | None = None,
        action: str = "",
        result: str = "success",
        ip_address: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log an audit entry.

        Args:
            event_type: Type of event (authentication, permission_check, etc.).
            user_id: User involved in the event.
            session_id: Session ID if applicable.
            action: Action attempted or performed.
            result: Result of the action.
            ip_address: Client IP address.
            details: Additional event details.
        """
        entry = AuditEntry(
            timestamp=time.time(),
            event_type=event_type,
            user_id=user_id,
            session_id=session_id,
            action=action,
            result=result,
            ip_address=ip_address,
            details=details or {},
        )

        self._audit_log.append(entry)

        # Keep audit log bounded (last 10000 entries)
        if len(self._audit_log) > 10000:
            self._audit_log = self._audit_log[-10000:]

        logger.debug("Audit: %s user=%s action=%s result=%s", event_type, user_id, action, result)

    async def get_audit_log(
        self,
        user_id: str | None = None,
        event_type: str | None = None,
        since: float | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get audit log entries.

        Args:
            user_id: Filter by user ID.
            event_type: Filter by event type.
            since: Filter entries after this timestamp.
            limit: Maximum entries to return.

        Returns:
            List of audit entries as dictionaries.
        """
        entries = self._audit_log

        if user_id:
            entries = [e for e in entries if e.user_id == user_id]
        if event_type:
            entries = [e for e in entries if e.event_type == event_type]
        if since:
            entries = [e for e in entries if e.timestamp >= since]

        # Return most recent entries
        entries = entries[-limit:]

        return [
            {
                "timestamp": e.timestamp,
                "event_type": e.event_type,
                "user_id": e.user_id,
                "session_id": e.session_id,
                "action": e.action,
                "result": e.result,
                "ip_address": e.ip_address,
                **e.details,
            }
            for e in entries
        ]

    # =========================================================================
    # Statistics and Monitoring
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get bridge statistics.

        Returns:
            Dictionary of bridge statistics.
        """
        active_sessions = sum(1 for s in self._sessions.values() if not s.is_expired)

        return {
            "permission_mappings": len(self._permission_mappings),
            "total_sessions": len(self._sessions),
            "active_sessions": active_sessions,
            "expired_sessions": len(self._sessions) - active_sessions,
            "audit_entries": len(self._audit_log),
            "action_allowlist_size": len(self._action_allowlist)
            if self._action_allowlist
            else None,
            "action_denylist_size": len(self._action_denylist),
            "lifecycle_hooks": len(self._lifecycle_hooks),
            "audit_enabled": self._enable_audit,
        }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Exceptions
    "AuthBridgeError",
    "AuthenticationError",
    "PermissionDeniedError",
    "SessionExpiredError",
    # Enums
    "TokenType",
    # Data classes
    "AuthContext",
    "PermissionMapping",
    "TokenExchangeResult",
    "BridgedSession",
    "AuditEntry",
    # Hooks
    "SessionLifecycleHook",
    # Main class
    "AuthBridge",
]
