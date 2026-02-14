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

