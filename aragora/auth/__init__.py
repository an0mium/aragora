"""
Aragora Authentication Module.

Provides SSO/SAML/OIDC authentication for enterprise deployments.

Usage:
    from aragora.auth import get_sso_provider, SSOProvider, SAMLProvider, OIDCProvider
    from aragora.auth.sso import SSOUser, SSOConfig

    # Get configured provider
    provider = get_sso_provider()
    if provider:
        auth_url = await provider.get_authorization_url(state="...")
        user = await provider.authenticate(code="...")
"""

from .sso import (
    SSOProvider,
    SSOUser,
    SSOConfig,
    SSOError,
    SSOProviderType,
    SSOAuthenticationError,
    SSOConfigurationError,
    get_sso_provider,
    reset_sso_provider,
)
from .saml import (
    SAMLProvider,
    SAMLConfig,
    SAMLError,
)
from .oidc import (
    OIDCProvider,
    OIDCConfig,
    OIDCError,
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
    # SAML
    "SAMLProvider",
    "SAMLConfig",
    "SAMLError",
    # OIDC
    "OIDCProvider",
    "OIDCConfig",
    "OIDCError",
]
