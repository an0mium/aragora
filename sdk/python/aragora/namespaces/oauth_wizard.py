"""
OAuth Wizard Namespace API

Unified API for discovering, configuring, and managing OAuth provider
integrations through a wizard interface designed for SME onboarding.

Provides methods for:
- Getting wizard configuration and status
- Listing and filtering OAuth providers
- Validating configuration and running preflight checks
- Managing provider connections (install, disconnect, refresh)
- Getting provider recommendations
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class OAuthWizardAPI:
    """
    Synchronous OAuth Wizard API.

    Provides methods for managing OAuth provider integrations through
    a wizard interface designed for SME onboarding.

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> config = client.oauth_wizard.get_config()
        >>> providers = client.oauth_wizard.list_providers(category="communication")
        >>> status = client.oauth_wizard.get_status()
        >>> validation = client.oauth_wizard.validate_config("slack")
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # ===========================================================================
    # Configuration
    # ===========================================================================

    def get_config(self) -> dict[str, Any]:
        """
        Get wizard configuration.

        Returns the current state of the integration wizard including
        available providers, configured providers, and recommended setup order.

        Returns:
            Dict with available_providers, configured_providers, recommended_order,
            total_setup_time_minutes, and completion_percent
        """
        return self._client.request("GET", "/api/v2/integrations/wizard")

    # ===========================================================================
    # Provider Discovery
    # ===========================================================================

    def list_providers(
        self,
        category: str | None = None,
        configured: bool | None = None,
    ) -> dict[str, Any]:
        """
        List all available OAuth providers.

        Args:
            category: Filter by provider category (communication, development,
                storage, crm, productivity, analytics)
            configured: Filter by configuration status (True for configured,
                False for not configured)

        Returns:
            Dict with providers array containing provider information
        """
        params: dict[str, Any] = {}
        if category is not None:
            params["category"] = category
        if configured is not None:
            params["configured"] = configured
        return self._client.request("GET", "/api/v2/integrations/wizard/providers", params=params)

    def get_provider(self, provider_id: str) -> dict[str, Any]:
        """
        Get a specific provider's details.

        Args:
            provider_id: Provider identifier (e.g., 'slack', 'github')

        Returns:
            Dict with provider details including id, name, description,
            category, features, required_env_vars, oauth_scopes, etc.
        """
        return self._client.request("GET", f"/api/v2/integrations/wizard/providers/{provider_id}")

    # ===========================================================================
    # Status
    # ===========================================================================

    def get_status(self) -> dict[str, Any]:
        """
        Get integration status for all providers.

        Returns the current connection and configuration status
        for all available integrations.

        Returns:
            Dict with total_providers, configured, connected, errors,
            providers array, health_score, and last_checked_at
        """
        return self._client.request("GET", "/api/v2/integrations/wizard/status")

    def get_provider_status(self, provider_id: str) -> dict[str, Any]:
        """
        Get status for a specific provider.

        Args:
            provider_id: Provider identifier

        Returns:
            Dict with provider_id, name, status, configured_at, missing_vars,
            optional_missing_vars, connected, last_sync_at, and error
        """
        return self._client.request("GET", f"/api/v2/integrations/wizard/status/{provider_id}")

    # ===========================================================================
    # Validation and Preflight
    # ===========================================================================

    def validate_config(self, provider_id: str) -> dict[str, Any]:
        """
        Validate provider configuration before connecting.

        Checks if all required environment variables are set
        and validates the configuration format.

        Args:
            provider_id: Provider identifier

        Returns:
            Dict with provider_id, valid, missing_required, missing_optional,
            warnings, errors, can_proceed, and suggested_actions
        """
        return self._client.request(
            "POST",
            "/api/v2/integrations/wizard/validate",
            json={"provider_id": provider_id},
        )

    def run_preflight_checks(self, provider_id: str) -> dict[str, Any]:
        """
        Run preflight checks before connecting to a provider.

        Performs connection tests, scope validation, and other
        checks to ensure successful integration.

        Args:
            provider_id: Provider identifier

        Returns:
            Dict with provider_id, checks array (each with name, status,
            message, required), can_connect, and estimated_connect_time_seconds
        """
        return self._client.request(
            "POST",
            "/api/v2/integrations/wizard/preflight",
            json={"provider_id": provider_id},
        )

    # ===========================================================================
    # Connection Management
    # ===========================================================================

    def get_install_url(
        self,
        provider_id: str,
        redirect_uri: str | None = None,
        scopes: list[str] | None = None,
        state: str | None = None,
    ) -> dict[str, Any]:
        """
        Get the install URL for a provider.

        Returns the OAuth authorization URL to initiate the connection flow.

        Args:
            provider_id: Provider identifier
            redirect_uri: Custom redirect URI (optional)
            scopes: Override default scopes (optional)
            state: Custom state parameter (optional)

        Returns:
            Dict with install_url, state, and expires_at
        """
        data: dict[str, Any] = {}
        if redirect_uri is not None:
            data["redirect_uri"] = redirect_uri
        if scopes is not None:
            data["scopes"] = scopes
        if state is not None:
            data["state"] = state
        return self._client.request(
            "POST",
            f"/api/v2/integrations/wizard/providers/{provider_id}/install",
            json=data,
        )

    def disconnect(self, provider_id: str) -> dict[str, Any]:
        """
        Disconnect a provider integration.

        Revokes OAuth tokens and removes the integration.

        Args:
            provider_id: Provider identifier

        Returns:
            Dict with disconnected (bool) and message
        """
        return self._client.request(
            "DELETE", f"/api/v2/integrations/wizard/providers/{provider_id}"
        )

    def refresh_tokens(self, provider_id: str) -> dict[str, Any]:
        """
        Refresh a provider's OAuth tokens.

        Args:
            provider_id: Provider identifier

        Returns:
            Dict with refreshed (bool) and expires_at
        """
        return self._client.request(
            "POST", f"/api/v2/integrations/wizard/providers/{provider_id}/refresh"
        )

    # ===========================================================================
    # Recommendations
    # ===========================================================================

    def get_recommendations(self) -> dict[str, Any]:
        """
        Get recommended providers based on current configuration.

        Returns providers that would benefit the user based on
        their current setup and usage patterns.

        Returns:
            Dict with recommendations array containing provider_id,
            reason, and priority (high, medium, low)
        """
        return self._client.request("GET", "/api/v2/integrations/wizard/recommendations")


class AsyncOAuthWizardAPI:
    """
    Asynchronous OAuth Wizard API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     config = await client.oauth_wizard.get_config()
        ...     providers = await client.oauth_wizard.list_providers(category="communication")
        ...     status = await client.oauth_wizard.get_status()
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # ===========================================================================
    # Configuration
    # ===========================================================================

    async def get_config(self) -> dict[str, Any]:
        """
        Get wizard configuration.

        Returns the current state of the integration wizard including
        available providers, configured providers, and recommended setup order.
        """
        return await self._client.request("GET", "/api/v2/integrations/wizard")

    # ===========================================================================
    # Provider Discovery
    # ===========================================================================

    async def list_providers(
        self,
        category: str | None = None,
        configured: bool | None = None,
    ) -> dict[str, Any]:
        """
        List all available OAuth providers.

        Args:
            category: Filter by provider category
            configured: Filter by configuration status
        """
        params: dict[str, Any] = {}
        if category is not None:
            params["category"] = category
        if configured is not None:
            params["configured"] = configured
        return await self._client.request(
            "GET", "/api/v2/integrations/wizard/providers", params=params
        )

    async def get_provider(self, provider_id: str) -> dict[str, Any]:
        """
        Get a specific provider's details.

        Args:
            provider_id: Provider identifier (e.g., 'slack', 'github')
        """
        return await self._client.request(
            "GET", f"/api/v2/integrations/wizard/providers/{provider_id}"
        )

    # ===========================================================================
    # Status
    # ===========================================================================

    async def get_status(self) -> dict[str, Any]:
        """
        Get integration status for all providers.

        Returns the current connection and configuration status
        for all available integrations.
        """
        return await self._client.request("GET", "/api/v2/integrations/wizard/status")

    async def get_provider_status(self, provider_id: str) -> dict[str, Any]:
        """
        Get status for a specific provider.

        Args:
            provider_id: Provider identifier
        """
        return await self._client.request(
            "GET", f"/api/v2/integrations/wizard/status/{provider_id}"
        )

    # ===========================================================================
    # Validation and Preflight
    # ===========================================================================

    async def validate_config(self, provider_id: str) -> dict[str, Any]:
        """
        Validate provider configuration before connecting.

        Checks if all required environment variables are set
        and validates the configuration format.

        Args:
            provider_id: Provider identifier
        """
        return await self._client.request(
            "POST",
            "/api/v2/integrations/wizard/validate",
            json={"provider_id": provider_id},
        )

    async def run_preflight_checks(self, provider_id: str) -> dict[str, Any]:
        """
        Run preflight checks before connecting to a provider.

        Performs connection tests, scope validation, and other
        checks to ensure successful integration.

        Args:
            provider_id: Provider identifier
        """
        return await self._client.request(
            "POST",
            "/api/v2/integrations/wizard/preflight",
            json={"provider_id": provider_id},
        )

    # ===========================================================================
    # Connection Management
    # ===========================================================================

    async def get_install_url(
        self,
        provider_id: str,
        redirect_uri: str | None = None,
        scopes: list[str] | None = None,
        state: str | None = None,
    ) -> dict[str, Any]:
        """
        Get the install URL for a provider.

        Returns the OAuth authorization URL to initiate the connection flow.

        Args:
            provider_id: Provider identifier
            redirect_uri: Custom redirect URI (optional)
            scopes: Override default scopes (optional)
            state: Custom state parameter (optional)
        """
        data: dict[str, Any] = {}
        if redirect_uri is not None:
            data["redirect_uri"] = redirect_uri
        if scopes is not None:
            data["scopes"] = scopes
        if state is not None:
            data["state"] = state
        return await self._client.request(
            "POST",
            f"/api/v2/integrations/wizard/providers/{provider_id}/install",
            json=data,
        )

    async def disconnect(self, provider_id: str) -> dict[str, Any]:
        """
        Disconnect a provider integration.

        Revokes OAuth tokens and removes the integration.

        Args:
            provider_id: Provider identifier
        """
        return await self._client.request(
            "DELETE", f"/api/v2/integrations/wizard/providers/{provider_id}"
        )

    async def refresh_tokens(self, provider_id: str) -> dict[str, Any]:
        """
        Refresh a provider's OAuth tokens.

        Args:
            provider_id: Provider identifier
        """
        return await self._client.request(
            "POST", f"/api/v2/integrations/wizard/providers/{provider_id}/refresh"
        )

    # ===========================================================================
    # Recommendations
    # ===========================================================================

    async def get_recommendations(self) -> dict[str, Any]:
        """
        Get recommended providers based on current configuration.

        Returns providers that would benefit the user based on
        their current setup and usage patterns.
        """
        return await self._client.request("GET", "/api/v2/integrations/wizard/recommendations")
