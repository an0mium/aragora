"""
SSO (Single Sign-On) Namespace API

Provides methods for enterprise SSO authentication:
- Check SSO configuration status
- Initiate SSO login flow
- Handle SSO callback from identity provider
- Logout from SSO session
- Retrieve SAML/OIDC provider metadata
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

class SSOAPI:
    """
    Synchronous SSO API.

    Provides methods for Single Sign-On authentication operations:
    - Get SSO status and configuration
    - Initiate login with identity providers
    - Handle OAuth/SAML callbacks
    - Logout and session management
    - Provider metadata retrieval

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> status = client.sso.get_status()
        >>> if status.get("enabled"):
        ...     login = client.sso.login(provider="okta", redirect_uri="https://app.example.com/callback")
        ...     # Redirect user to login["redirect_url"]
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # ===========================================================================
    # SSO Status

class AsyncSSOAPI:
    """
    Asynchronous SSO API.

    Provides async methods for Single Sign-On authentication operations.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     status = await client.sso.get_status()
        ...     if status.get("enabled"):
        ...         login = await client.sso.login(
        ...             provider="okta",
        ...             redirect_uri="https://app.example.com/callback"
        ...         )
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # ===========================================================================
    # SSO Status

