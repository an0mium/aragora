"""
SSO Provider Abstraction for Aragora.

Provides a unified interface for SSO authentication providers
including SAML 2.0 and OpenID Connect (OIDC).

Usage:
    from aragora.auth.sso import get_sso_provider, SSOUser

    provider = get_sso_provider()
    if provider:
        # Get login URL
        auth_url = await provider.get_authorization_url(state="...")

        # Handle callback
        user = await provider.authenticate(code="...")
"""

from __future__ import annotations

import logging
import secrets
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SSOProviderType(str, Enum):
    """Supported SSO provider types."""

    SAML = "saml"
    OIDC = "oidc"
    AZURE_AD = "azure_ad"
    OKTA = "okta"
    GOOGLE = "google"
    GITHUB = "github"


class SSOError(Exception):
    """Base exception for SSO errors."""

    def __init__(self, message: str, code: str = "SSO_ERROR", details: Optional[Dict] = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}


class SSOAuthenticationError(SSOError):
    """Authentication failed."""

    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, "SSO_AUTH_FAILED", details)


class SSOConfigurationError(SSOError):
    """SSO is misconfigured."""

    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, "SSO_CONFIG_ERROR", details)


@dataclass
class SSOUser:
    """
    Authenticated user from SSO provider.

    Normalized representation of user data from any SSO provider.
    """

    # Core identity
    id: str  # Unique identifier from IdP (nameID, sub, etc.)
    email: str
    name: str = ""

    # Optional fields
    first_name: str = ""
    last_name: str = ""
    display_name: str = ""
    username: str = ""

    # Organization/tenant info
    organization_id: Optional[str] = None
    organization_name: Optional[str] = None
    tenant_id: Optional[str] = None

    # Roles and groups
    roles: List[str] = field(default_factory=list)
    groups: List[str] = field(default_factory=list)

    # Provider info
    provider_type: str = ""
    provider_id: str = ""

    # Tokens (for refresh/logout)
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    id_token: Optional[str] = None
    token_expires_at: Optional[float] = None

    # Raw claims from IdP
    raw_claims: Dict[str, Any] = field(default_factory=dict)

    # Timestamps
    authenticated_at: float = field(default_factory=time.time)

    @property
    def is_admin(self) -> bool:
        """Check if user has admin role."""
        admin_roles = {"admin", "administrator", "superadmin", "owner"}
        return bool(admin_roles & set(r.lower() for r in self.roles))

    @property
    def full_name(self) -> str:
        """Get full name, falling back to name or display_name."""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.name or self.display_name or self.email.split("@")[0]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "email": self.email,
            "name": self.full_name,
            "username": self.username or self.email.split("@")[0],
            "organization_id": self.organization_id,
            "organization_name": self.organization_name,
            "roles": self.roles,
            "groups": self.groups,
            "provider_type": self.provider_type,
            "is_admin": self.is_admin,
            "authenticated_at": self.authenticated_at,
        }


@dataclass
class SSOConfig:
    """
    Base SSO configuration.

    Extended by SAML and OIDC specific configs.
    """

    # Provider identification
    provider_type: SSOProviderType
    provider_id: str = ""

    # Enabled flag
    enabled: bool = False

    # Callback URL (where IdP redirects after auth)
    callback_url: str = ""

    # Entity ID / Client ID
    entity_id: str = ""

    # Logout URLs
    logout_url: str = ""
    post_logout_redirect_url: str = ""

    # Session settings
    session_duration_seconds: int = 3600 * 8  # 8 hours

    # Domain restrictions (optional)
    allowed_domains: List[str] = field(default_factory=list)

    # Role mapping (IdP role -> Aragora role)
    role_mapping: Dict[str, str] = field(default_factory=dict)

    # Group mapping (IdP group -> Aragora group)
    group_mapping: Dict[str, str] = field(default_factory=dict)

    # Auto-provision users on first login
    auto_provision: bool = True

    # Default role for new users
    default_role: str = "user"

    def validate(self) -> List[str]:
        """
        Validate configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if not self.entity_id:
            errors.append("entity_id is required")

        if not self.callback_url:
            errors.append("callback_url is required")

        return errors


class SSOProvider(ABC):
    """
    Abstract base class for SSO providers.

    Implementations must provide:
    - get_authorization_url: Generate login URL
    - authenticate: Validate response and return user
    - logout: Handle logout (optional)
    """

    def __init__(self, config: SSOConfig):
        self.config = config
        self._state_store: Dict[str, float] = {}  # state -> timestamp

    @property
    @abstractmethod
    def provider_type(self) -> SSOProviderType:
        """Return the provider type."""
        pass

    @abstractmethod
    async def get_authorization_url(
        self,
        state: Optional[str] = None,
        redirect_uri: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Generate the authorization URL to redirect user to IdP.

        Args:
            state: Optional state parameter for CSRF protection
            redirect_uri: Optional override for callback URL
            **kwargs: Provider-specific parameters

        Returns:
            URL to redirect user to
        """
        pass

    @abstractmethod
    async def authenticate(
        self,
        code: Optional[str] = None,
        saml_response: Optional[str] = None,
        **kwargs,
    ) -> SSOUser:
        """
        Authenticate user from IdP callback.

        Args:
            code: Authorization code (OIDC)
            saml_response: SAML assertion (SAML)
            **kwargs: Provider-specific parameters

        Returns:
            Authenticated user

        Raises:
            SSOAuthenticationError: If authentication fails
        """
        pass

    async def logout(self, user: SSOUser) -> Optional[str]:
        """
        Handle user logout.

        Args:
            user: User to logout

        Returns:
            Optional logout URL to redirect to (for IdP-initiated logout)
        """
        # Default: just return logout URL if configured
        return self.config.logout_url or None

    async def refresh_token(self, user: SSOUser) -> Optional[SSOUser]:
        """
        Refresh user's access token.

        Args:
            user: User with refresh token

        Returns:
            Updated user with new tokens, or None if refresh not supported
        """
        # Default: not supported
        return None

    def generate_state(self) -> str:
        """Generate secure state parameter for CSRF protection."""
        state = secrets.token_urlsafe(32)
        self._state_store[state] = time.time()
        return state

    def validate_state(self, state: str) -> bool:
        """Validate state parameter (must be used within 10 minutes)."""
        if state not in self._state_store:
            return False

        timestamp = self._state_store.pop(state)
        return time.time() - timestamp < 600  # 10 minute window

    def cleanup_expired_states(self) -> int:
        """Remove expired state entries."""
        now = time.time()
        expired = [k for k, v in self._state_store.items() if now - v > 600]
        for k in expired:
            del self._state_store[k]
        return len(expired)

    def map_roles(self, idp_roles: List[str]) -> List[str]:
        """Map IdP roles to Aragora roles."""
        mapped = []
        for role in idp_roles:
            if role in self.config.role_mapping:
                mapped.append(self.config.role_mapping[role])
            else:
                mapped.append(role)

        # Add default role if no roles mapped
        if not mapped:
            mapped.append(self.config.default_role)

        return list(set(mapped))  # Deduplicate

    def map_groups(self, idp_groups: List[str]) -> List[str]:
        """Map IdP groups to Aragora groups."""
        mapped = []
        for group in idp_groups:
            if group in self.config.group_mapping:
                mapped.append(self.config.group_mapping[group])
            else:
                mapped.append(group)
        return list(set(mapped))

    def is_domain_allowed(self, email: str) -> bool:
        """Check if email domain is in allowed list."""
        if not self.config.allowed_domains:
            return True  # No restrictions

        domain = email.split("@")[-1].lower()
        return domain in [d.lower() for d in self.config.allowed_domains]


# =============================================================================
# Global Provider Instance
# =============================================================================

_sso_provider: Optional[SSOProvider] = None
_sso_initialized: bool = False


def get_sso_provider() -> Optional[SSOProvider]:
    """
    Get the configured SSO provider.

    Returns:
        SSOProvider if configured, None otherwise
    """
    global _sso_provider, _sso_initialized

    if _sso_initialized:
        return _sso_provider

    try:
        from aragora.config.settings import get_settings

        settings = get_settings()

        # Check if SSO is configured
        if not hasattr(settings, "sso") or not settings.sso.enabled:
            _sso_initialized = True
            return None

        sso_settings = settings.sso

        # Create provider based on type
        if sso_settings.provider_type == "saml":
            from .saml import SAMLConfig, SAMLProvider

            saml_config = SAMLConfig(
                provider_type=SSOProviderType.SAML,
                enabled=True,
                callback_url=sso_settings.callback_url,
                entity_id=sso_settings.entity_id,
                idp_entity_id=sso_settings.idp_entity_id,
                idp_sso_url=sso_settings.idp_sso_url,
                idp_slo_url=sso_settings.idp_slo_url or "",
                idp_certificate=sso_settings.idp_certificate or "",
                sp_private_key=sso_settings.sp_private_key or "",
                sp_certificate=sso_settings.sp_certificate or "",
                allowed_domains=sso_settings.allowed_domains,
                auto_provision=sso_settings.auto_provision,
            )
            _sso_provider = SAMLProvider(saml_config)

        elif sso_settings.provider_type in ("oidc", "azure_ad", "okta", "google"):
            from .oidc import OIDCConfig, OIDCProvider

            oidc_config = OIDCConfig(
                provider_type=SSOProviderType(sso_settings.provider_type),
                enabled=True,
                callback_url=sso_settings.callback_url,
                entity_id=sso_settings.entity_id,
                client_id=sso_settings.client_id,
                client_secret=sso_settings.client_secret,
                issuer_url=sso_settings.issuer_url,
                authorization_endpoint=sso_settings.authorization_endpoint or "",
                token_endpoint=sso_settings.token_endpoint or "",
                userinfo_endpoint=sso_settings.userinfo_endpoint or "",
                jwks_uri=sso_settings.jwks_uri or "",
                scopes=sso_settings.scopes,
                allowed_domains=sso_settings.allowed_domains,
                auto_provision=sso_settings.auto_provision,
            )
            _sso_provider = OIDCProvider(oidc_config)

        else:
            logger.warning(f"Unknown SSO provider type: {sso_settings.provider_type}")

        _sso_initialized = True

    except Exception as e:
        logger.warning(f"SSO provider initialization failed: {e}")
        _sso_initialized = True

    return _sso_provider


def reset_sso_provider() -> None:
    """Reset SSO provider (for testing)."""
    global _sso_provider, _sso_initialized
    _sso_provider = None
    _sso_initialized = False


__all__ = [
    "SSOProviderType",
    "SSOError",
    "SSOAuthenticationError",
    "SSOConfigurationError",
    "SSOUser",
    "SSOConfig",
    "SSOProvider",
    "get_sso_provider",
    "reset_sso_provider",
]
