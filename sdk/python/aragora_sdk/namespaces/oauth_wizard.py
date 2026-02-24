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
        self, category: str | None = None, configured: bool | None = None
    ) -> dict[str, Any]:
        """
        List all available OAuth providers.

        Args:
            category: Filter by category (communication, development, storage, etc.)
            configured: Filter by configuration status

        Returns:
            Dict with providers list
        """
        params: dict[str, Any] = {}
        if category:
            params["category"] = category
        if configured is not None:
            params["configured"] = configured
        return self._client.request(
            "GET", "/api/v2/integrations/wizard/providers", params=params or None
        )

    def get_status(self) -> dict[str, Any]:
        """
        Get integration status for all providers.

        Returns:
            Dict with provider statuses, health score, and summary
        """
        return self._client.request("GET", "/api/v2/integrations/wizard/status")

    def validate_config(self, provider_id: str) -> dict[str, Any]:
        """
        Validate provider configuration before connecting.

        Args:
            provider_id: Provider identifier (e.g., 'slack', 'github')

        Returns:
            Validation result with missing vars, warnings, and errors
        """
        return self._client.request(
            "POST", "/api/v2/integrations/wizard/validate", json={"provider_id": provider_id}
        )

    def test_connection(self, provider_id: str) -> dict[str, Any]:
        """
        Test connection to a specific provider.

        Args:
            provider_id: Provider identifier

        Returns:
            Connection test results
        """
        return self._client.request("POST", f"/api/v2/integrations/wizard/{provider_id}/test")

    def list_workspaces(self, provider_id: str) -> dict[str, Any]:
        """
        List workspaces/tenants for a provider.

        Args:
            provider_id: Provider identifier

        Returns:
            Available workspaces for the provider
        """
        return self._client.request("GET", f"/api/v2/integrations/wizard/{provider_id}/workspaces")

    def disconnect_provider(self, provider_id: str) -> dict[str, Any]:
        """
        Disconnect a provider integration.

        Args:
            provider_id: Provider identifier

        Returns:
            Disconnection confirmation
        """
        return self._client.request("POST", f"/api/v2/integrations/wizard/{provider_id}/disconnect")


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
        self, category: str | None = None, configured: bool | None = None
    ) -> dict[str, Any]:
        """List all available OAuth providers."""
        params: dict[str, Any] = {}
        if category:
            params["category"] = category
        if configured is not None:
            params["configured"] = configured
        return await self._client.request(
            "GET", "/api/v2/integrations/wizard/providers", params=params or None
        )

    async def get_status(self) -> dict[str, Any]:
        """Get integration status for all providers."""
        return await self._client.request("GET", "/api/v2/integrations/wizard/status")

    async def validate_config(self, provider_id: str) -> dict[str, Any]:
        """Validate provider configuration before connecting."""
        return await self._client.request(
            "POST", "/api/v2/integrations/wizard/validate", json={"provider_id": provider_id}
        )

    async def test_connection(self, provider_id: str) -> dict[str, Any]:
        """Test connection to a specific provider."""
        return await self._client.request("POST", f"/api/v2/integrations/wizard/{provider_id}/test")

    async def list_workspaces(self, provider_id: str) -> dict[str, Any]:
        """List workspaces/tenants for a provider."""
        return await self._client.request(
            "GET", f"/api/v2/integrations/wizard/{provider_id}/workspaces"
        )

    async def disconnect_provider(self, provider_id: str) -> dict[str, Any]:
        """Disconnect a provider integration."""
        return await self._client.request(
            "POST", f"/api/v2/integrations/wizard/{provider_id}/disconnect"
        )
