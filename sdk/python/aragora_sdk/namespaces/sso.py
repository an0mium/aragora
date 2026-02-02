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

from typing import TYPE_CHECKING, Any

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
    # ===========================================================================

    def get_status(self) -> dict[str, Any]:
        """
        Get SSO configuration status.

        Returns whether SSO is enabled and configured for the current tenant.

        Returns:
            Dict with SSO status including:
            - enabled: Whether SSO is enabled
            - configured: Whether SSO is properly configured
            - provider_type: Type of SSO provider (saml, oidc, oauth2)
            - idp_url: Identity provider URL
            - entity_id: Service provider entity ID
        """
        return self._client.request("GET", "/api/v2/sso/status")

    # ===========================================================================
    # Authentication Flow
    # ===========================================================================

    def login(
        self,
        provider: str,
        redirect_uri: str,
    ) -> dict[str, Any]:
        """
        Initiate SSO login flow.

        Starts the authentication flow with the specified identity provider.
        Returns a redirect URL to send the user to for authentication.

        Args:
            provider: SSO provider identifier (e.g., "okta", "azure-ad", "google")
            redirect_uri: URL to redirect after successful authentication

        Returns:
            Dict with login flow details including:
            - redirect_url: URL to redirect user to for IdP authentication
            - state: CSRF state token for callback validation
            - nonce: OIDC nonce for token validation
            - provider: Provider type used
        """
        data: dict[str, Any] = {
            "provider": provider,
            "redirect_uri": redirect_uri,
        }
        return self._client.request("POST", "/api/v2/sso/login", json=data)

    def callback(
        self,
        provider: str,
        code: str,
        state: str,
    ) -> dict[str, Any]:
        """
        Handle SSO callback from identity provider.

        Processes the authorization callback from the IdP after user authentication.
        Exchanges the authorization code for tokens and creates a session.

        Args:
            provider: SSO provider identifier
            code: Authorization code from IdP callback
            state: State token for CSRF validation

        Returns:
            Dict with authentication result including:
            - success: Whether authentication succeeded
            - user: User information from IdP (id, email, name, groups, roles)
            - token: Access token for API authentication
            - refresh_token: Token for refreshing access
            - expires_in: Token expiration time in seconds
            - error: Error code if authentication failed
            - error_description: Detailed error message
        """
        data: dict[str, Any] = {
            "provider": provider,
            "code": code,
            "state": state,
        }
        return self._client.request("POST", "/api/v2/sso/callback", json=data)

    def logout(
        self,
        provider: str,
    ) -> dict[str, Any]:
        """
        Logout from SSO session.

        Terminates the current SSO session and optionally triggers
        Single Logout (SLO) with the identity provider.

        Args:
            provider: SSO provider identifier

        Returns:
            Dict with logout result including:
            - success: Whether logout succeeded
            - redirect_url: Optional URL to redirect for IdP logout
            - message: Status message
        """
        data: dict[str, Any] = {
            "provider": provider,
        }
        return self._client.request("POST", "/api/v2/sso/logout", json=data)

    # ===========================================================================
    # Provider Metadata
    # ===========================================================================

    def get_metadata(
        self,
        provider: str,
    ) -> dict[str, Any]:
        """
        Get SAML/OIDC provider metadata.

        Returns service provider metadata for configuring the identity provider.
        For SAML, this includes entity ID, ACS URL, and certificate.
        For OIDC, this includes client ID and redirect URIs.

        Args:
            provider: SSO provider identifier

        Returns:
            Dict with provider metadata including:
            - entity_id: Service provider entity ID
            - acs_url: Assertion Consumer Service URL (SAML)
            - slo_url: Single Logout URL (SAML)
            - certificate: SP signing certificate
            - metadata_xml: Raw SAML metadata XML
        """
        return self._client.request("GET", f"/api/v2/sso/{provider}/metadata")


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
    # ===========================================================================

    async def get_status(self) -> dict[str, Any]:
        """
        Get SSO configuration status.

        Returns whether SSO is enabled and configured for the current tenant.

        Returns:
            Dict with SSO status including enabled, configured, provider_type, etc.
        """
        return await self._client.request("GET", "/api/v2/sso/status")

    # ===========================================================================
    # Authentication Flow
    # ===========================================================================

    async def login(
        self,
        provider: str,
        redirect_uri: str,
    ) -> dict[str, Any]:
        """
        Initiate SSO login flow.

        Args:
            provider: SSO provider identifier (e.g., "okta", "azure-ad", "google")
            redirect_uri: URL to redirect after successful authentication

        Returns:
            Dict with redirect_url, state, nonce, and provider details
        """
        data: dict[str, Any] = {
            "provider": provider,
            "redirect_uri": redirect_uri,
        }
        return await self._client.request("POST", "/api/v2/sso/login", json=data)

    async def callback(
        self,
        provider: str,
        code: str,
        state: str,
    ) -> dict[str, Any]:
        """
        Handle SSO callback from identity provider.

        Args:
            provider: SSO provider identifier
            code: Authorization code from IdP callback
            state: State token for CSRF validation

        Returns:
            Dict with success, user, token, refresh_token, expires_in, and error details
        """
        data: dict[str, Any] = {
            "provider": provider,
            "code": code,
            "state": state,
        }
        return await self._client.request("POST", "/api/v2/sso/callback", json=data)

    async def logout(
        self,
        provider: str,
    ) -> dict[str, Any]:
        """
        Logout from SSO session.

        Args:
            provider: SSO provider identifier

        Returns:
            Dict with success, redirect_url, and message
        """
        data: dict[str, Any] = {
            "provider": provider,
        }
        return await self._client.request("POST", "/api/v2/sso/logout", json=data)

    # ===========================================================================
    # Provider Metadata
    # ===========================================================================

    async def get_metadata(
        self,
        provider: str,
    ) -> dict[str, Any]:
        """
        Get SAML/OIDC provider metadata.

        Args:
            provider: SSO provider identifier

        Returns:
            Dict with entity_id, acs_url, slo_url, certificate, and metadata_xml
        """
        return await self._client.request("GET", f"/api/v2/sso/{provider}/metadata")
