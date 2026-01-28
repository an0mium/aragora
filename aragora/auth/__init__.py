"""
Aragora Authentication Module.

Provides SSO/SAML/OIDC authentication for enterprise deployments,
and account lockout protection against brute-force attacks.

Usage:
    from aragora.auth import get_sso_provider, SSOProvider, SAMLProvider, OIDCProvider
    from aragora.auth.sso import SSOUser, SSOConfig

    # Get configured provider
    provider = get_sso_provider()
    if provider:
        auth_url = await provider.get_authorization_url(state="...")
        user = await provider.authenticate(code="...")

    # Lockout tracking
    from aragora.auth import get_lockout_tracker

    tracker = get_lockout_tracker()
    if tracker.is_locked(email=email, ip=client_ip):
        remaining = tracker.get_remaining_time(email=email, ip=client_ip)
        return error(f"Locked for {remaining} seconds")
"""

from .lockout import (
    LockoutEntry,
    LockoutTracker,
    get_lockout_tracker,
    reset_lockout_tracker,
)
from .oidc import (
    OIDCConfig,
    OIDCError,
    OIDCProvider,
)

# SAML requires python3-saml optional dependency
try:
    from .saml import (
        SAMLConfig,
        SAMLError,
        SAMLProvider,
    )

    HAS_SAML = True
except ImportError:
    # SAML unavailable - define placeholders
    SAMLConfig = None  # type: ignore[assignment, misc]
    SAMLError = None  # type: ignore[assignment, misc]
    SAMLProvider = None  # type: ignore[assignment, misc]
    HAS_SAML = False

from .sso import (
    SSOAuthenticationError,
    SSOConfig,
    SSOConfigurationError,
    SSOError,
    SSOProvider,
    SSOProviderType,
    SSOUser,
    get_sso_provider,
    reset_sso_provider,
)

__all__ = [
    # Base SSO
    "SSOProvider",
    "SSOProviderType",
    "SSOUser",
    "SSOConfig",
    "SSOError",
    "SSOAuthenticationError",
    "SSOConfigurationError",
    "get_sso_provider",
    "reset_sso_provider",
    # OIDC
    "OIDCProvider",
    "OIDCConfig",
    "OIDCError",
    # Lockout
    "LockoutTracker",
    "LockoutEntry",
    "get_lockout_tracker",
    "reset_lockout_tracker",
    # SAML availability flag
    "HAS_SAML",
]

# Add SAML exports only if available
if HAS_SAML:
    __all__.extend(["SAMLProvider", "SAMLConfig", "SAMLError"])
