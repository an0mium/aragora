"""
Integrations namespace for external service connections.

Provides API access to manage integrations with external services
like Slack, Discord, GitHub, Jira, and other platforms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


_List = list  # Preserve builtin list for type annotations


class IntegrationsAPI:
    """Synchronous integrations API."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    def list(
        self,
        limit: int = 20,
        offset: int = 0,
        provider: str | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        """
        List integrations (v2).

        Args:
            limit: Maximum number of integrations to return
            offset: Number of integrations to skip
            provider: Filter by provider/type (slack, teams, discord, email)
            status: Filter by status (active, inactive)

        Returns:
            Integration listing with pagination
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if provider:
            params["type"] = provider
        if status:
            params["status"] = status

        return self._client.request("GET", "/api/v2/integrations", params=params)

    def get(self, integration_id: str, workspace_id: str | None = None) -> dict[str, Any]:
        """
        Get integration status (v2).

        Args:
            integration_id: Integration type (slack, teams, discord, email)
            workspace_id: Optional workspace/tenant ID

        Returns:
            Integration status
        """
        params = {"workspace_id": workspace_id} if workspace_id else None
        return self._client.request(
            "GET",
            f"/api/v2/integrations/{integration_id}",
            params=params,
        )

    def create(self, provider: str, config: dict[str, Any]) -> dict[str, Any]:
        """
        Configure a v1 integration by type.

        Args:
            provider: Integration type (slack, teams, discord, email)
            config: Provider-specific configuration

        Returns:
            Configuration result
        """
        return self._client.request(
            "PUT",
            f"/api/v1/integrations/{provider}",
            json=config,
        )

    def update(
        self,
        integration_id: str,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Update an integration configuration (v1).

        Args:
            integration_id: Integration type (slack, teams, discord, email)
            config: Updated configuration payload

        Returns:
            Updated integration record
        """
        return self._client.request(
            "PATCH",
            f"/api/v1/integrations/{integration_id}",
            json=config,
        )

    def delete(self, integration_id: str, workspace_id: str | None = None) -> dict[str, Any]:
        """
        Delete/disconnect an integration.

        Args:
            integration_id: Integration type
            workspace_id: Optional workspace to disconnect (v2)

        Returns:
            Deletion confirmation
        """
        if workspace_id:
            return self._client.request(
                "DELETE",
                f"/api/v2/integrations/{integration_id}",
                json={"workspace_id": workspace_id},
            )
        return self._client.request("DELETE", f"/api/v1/integrations/{integration_id}")

    def test(self, integration_id: str) -> dict[str, Any]:
        """
        Test an integration connection.

        Args:
            integration_id: Integration identifier

        Returns:
            Test results
        """
        return self._client.request("POST", f"/api/v1/integrations/{integration_id}/test")

    # =========================================================================
    # v1 Integration Management
    # =========================================================================

    def get_status(self) -> dict[str, Any]:
        """
        Get integration status overview.

        GET /api/v1/integrations/status

        Returns:
            Dict with status for all configured integrations
        """
        return self._client.request("GET", "/api/v1/integrations/status")

    def list_available(self) -> dict[str, Any]:
        """
        List available integration types.

        GET /api/v1/integrations/available

        Returns:
            Dict with available integration type names
        """
        return self._client.request("GET", "/api/v1/integrations/available")

    def list_configs(self) -> dict[str, Any]:
        """
        List integration configurations.

        GET /api/v1/integrations/config

        Returns:
            Dict with integration configuration entries
        """
        return self._client.request("GET", "/api/v1/integrations/config")

    def get_config(self, integration_type: str) -> dict[str, Any]:
        """
        Get configuration for a specific integration type.

        GET /api/v1/integrations/config/:integration_type

        Args:
            integration_type: Integration type (slack, teams, discord, email)

        Returns:
            Dict with integration configuration
        """
        return self._client.request("GET", f"/api/v1/integrations/config/{integration_type}")

    def sync_integration(self, integration_type: str) -> dict[str, Any]:
        """
        Trigger synchronization for an integration.

        POST /api/v1/integrations/:integration_type/sync

        Args:
            integration_type: Integration type to sync

        Returns:
            Dict with sync result
        """
        return self._client.request("POST", f"/api/v1/integrations/{integration_type}/sync")

    # =========================================================================
    # Bot Platform Status
    # =========================================================================

    def get_slack_status(self) -> dict[str, Any]:
        """Get Slack bot connection status."""
        return self._client.request("GET", "/api/v1/bots/slack/status")

    def get_telegram_status(self) -> dict[str, Any]:
        """Get Telegram bot connection status."""
        return self._client.request("GET", "/api/v1/bots/telegram/status")

    def get_whatsapp_status(self) -> dict[str, Any]:
        """Get WhatsApp bot connection status."""
        return self._client.request("GET", "/api/v1/bots/whatsapp/status")

    def get_discord_status(self) -> dict[str, Any]:
        """Get Discord bot connection status."""
        return self._client.request("GET", "/api/v1/bots/discord/status")

    def get_google_chat_status(self) -> dict[str, Any]:
        """Get Google Chat bot connection status."""
        return self._client.request("GET", "/api/v1/bots/google-chat/status")

    # =========================================================================
    # Teams Integration
    # =========================================================================

    def get_teams_status(self) -> dict[str, Any]:
        """Get Microsoft Teams integration status."""
        return self._client.request("GET", "/api/v1/integrations/teams/status")

    def notify_teams(
        self,
        channel_id: str,
        message: str,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send notification to Teams channel."""
        payload: dict[str, Any] = {"channel_id": channel_id, "message": message}
        if options:
            payload.update(options)
        return self._client.request("POST", "/api/v1/integrations/teams/notify", json=payload)

    # =========================================================================
    # Zapier Integration
    # =========================================================================

    def list_zapier_apps(self, workspace_id: str | None = None) -> dict[str, Any]:
        """List Zapier apps for workspace."""
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        return self._client.request(
            "GET", "/api/v1/integrations/zapier/apps", params=params or None
        )

    def create_zapier_app(self, workspace_id: str) -> dict[str, Any]:
        """Create new Zapier app (returns API credentials)."""
        return self._client.request(
            "POST", "/api/v1/integrations/zapier/apps", json={"workspace_id": workspace_id}
        )

    def list_zapier_trigger_types(self) -> dict[str, Any]:
        """Get available Zapier trigger and action types."""
        return self._client.request("GET", "/api/v1/integrations/zapier/triggers")

    def subscribe_zapier_trigger(
        self,
        app_id: str,
        trigger_type: str,
        webhook_url: str,
        workspace_id: str | None = None,
        debate_tags: _List[str] | None = None,
        min_confidence: float | None = None,
    ) -> dict[str, Any]:
        """Subscribe to Zapier trigger."""
        payload: dict[str, Any] = {
            "app_id": app_id,
            "trigger_type": trigger_type,
            "webhook_url": webhook_url,
        }
        if workspace_id:
            payload["workspace_id"] = workspace_id
        if debate_tags:
            payload["debate_tags"] = debate_tags
        if min_confidence is not None:
            payload["min_confidence"] = min_confidence
        return self._client.request("POST", "/api/v1/integrations/zapier/triggers", json=payload)

    # =========================================================================
    # Make (Integromat) Integration
    # =========================================================================

    def list_make_connections(self, workspace_id: str | None = None) -> dict[str, Any]:
        """List Make connections for workspace."""
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        return self._client.request(
            "GET", "/api/v1/integrations/make/connections", params=params or None
        )

    def create_make_connection(self, workspace_id: str) -> dict[str, Any]:
        """Create new Make connection (returns API key)."""
        return self._client.request(
            "POST", "/api/v1/integrations/make/connections", json={"workspace_id": workspace_id}
        )

    def list_make_modules(self) -> dict[str, Any]:
        """Get available Make module types."""
        return self._client.request("GET", "/api/v1/integrations/make/modules")

    def register_make_webhook(
        self,
        connection_id: str,
        module_type: str,
        webhook_url: str,
        workspace_id: str | None = None,
        event_filter: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Register Make webhook."""
        payload: dict[str, Any] = {
            "connection_id": connection_id,
            "module_type": module_type,
            "webhook_url": webhook_url,
        }
        if workspace_id:
            payload["workspace_id"] = workspace_id
        if event_filter:
            payload["event_filter"] = event_filter
        return self._client.request("POST", "/api/v1/integrations/make/webhooks", json=payload)

    # =========================================================================
    # n8n Integration
    # =========================================================================

    def list_n8n_credentials(self, workspace_id: str | None = None) -> dict[str, Any]:
        """List n8n credentials for workspace."""
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        return self._client.request(
            "GET", "/api/v1/integrations/n8n/credentials", params=params or None
        )

    def create_n8n_credential(
        self, workspace_id: str, api_url: str | None = None
    ) -> dict[str, Any]:
        """Create n8n credential (returns API key)."""
        payload: dict[str, Any] = {"workspace_id": workspace_id}
        if api_url:
            payload["api_url"] = api_url
        return self._client.request("POST", "/api/v1/integrations/n8n/credentials", json=payload)

    def get_n8n_nodes(self) -> dict[str, Any]:
        """Get n8n node, trigger, and credential definitions."""
        return self._client.request("GET", "/api/v1/integrations/n8n/nodes")

    def register_n8n_webhook(
        self,
        credential_id: str,
        events: _List[str],
        workflow_id: str | None = None,
        node_id: str | None = None,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        """Register n8n webhook."""
        payload: dict[str, Any] = {"credential_id": credential_id, "events": events}
        if workflow_id:
            payload["workflow_id"] = workflow_id
        if node_id:
            payload["node_id"] = node_id
        if workspace_id:
            payload["workspace_id"] = workspace_id
        return self._client.request("POST", "/api/v1/integrations/n8n/webhooks", json=payload)

    # =========================================================================
    # Integration Wizard (v2)
    # =========================================================================

    def start_wizard(self, integration_type: str) -> dict[str, Any]:
        """Start integration setup wizard."""
        return self._client.request(
            "POST", "/api/v2/integrations/wizard", json={"type": integration_type}
        )

    # =========================================================================
    # v2 Management
    # =========================================================================

    def list_v2(
        self,
        type: str | None = None,
        status: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """List integrations with advanced filtering (v2)."""
        params: dict[str, Any] = {}
        if type:
            params["type"] = type
        if status:
            params["status"] = status
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        return self._client.request("GET", "/api/v2/integrations", params=params or None)

    def get_by_type(self, type: str) -> dict[str, Any]:
        """Get integration by type (Slack, Teams, Discord, Email)."""
        return self._client.request("GET", f"/api/v2/integrations/{type}")

    def test_by_type(self, type: str) -> dict[str, Any]:
        """Test integration connectivity by type (v2)."""
        return self._client.request("POST", f"/api/v2/integrations/{type}/test", json={})

    def test_platform(self, platform: str) -> dict[str, Any]:
        """Test specific platform integration (v1)."""
        return self._client.request("POST", f"/api/v1/integrations/{platform}/test", json={})


class AsyncIntegrationsAPI:
    """Asynchronous integrations API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    async def list(
        self,
        limit: int = 20,
        offset: int = 0,
        provider: str | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        """List integrations (v2)."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if provider:
            params["type"] = provider
        if status:
            params["status"] = status

        return await self._client.request("GET", "/api/v2/integrations", params=params)

    async def get(self, integration_id: str, workspace_id: str | None = None) -> dict[str, Any]:
        """Get integration status (v2)."""
        params = {"workspace_id": workspace_id} if workspace_id else None
        return await self._client.request(
            "GET",
            f"/api/v2/integrations/{integration_id}",
            params=params,
        )

    async def create(self, provider: str, config: dict[str, Any]) -> dict[str, Any]:
        """Configure a v1 integration by type."""
        return await self._client.request(
            "PUT",
            f"/api/v1/integrations/{provider}",
            json=config,
        )

    async def update(self, integration_id: str, config: dict[str, Any]) -> dict[str, Any]:
        """Update an integration configuration (v1)."""
        return await self._client.request(
            "PATCH",
            f"/api/v1/integrations/{integration_id}",
            json=config,
        )

    async def delete(self, integration_id: str, workspace_id: str | None = None) -> dict[str, Any]:
        """Delete/disconnect an integration."""
        if workspace_id:
            return await self._client.request(
                "DELETE",
                f"/api/v2/integrations/{integration_id}",
                json={"workspace_id": workspace_id},
            )
        return await self._client.request("DELETE", f"/api/v1/integrations/{integration_id}")

    async def test(self, integration_id: str) -> dict[str, Any]:
        """Test an integration connection."""
        return await self._client.request("POST", f"/api/v1/integrations/{integration_id}/test")

    # =========================================================================
    # v1 Integration Management
    # =========================================================================

    async def get_status(self) -> dict[str, Any]:
        """Get integration status overview. GET /api/v1/integrations/status"""
        return await self._client.request("GET", "/api/v1/integrations/status")

    async def list_available(self) -> dict[str, Any]:
        """List available integration types. GET /api/v1/integrations/available"""
        return await self._client.request("GET", "/api/v1/integrations/available")

    async def list_configs(self) -> dict[str, Any]:
        """List integration configurations. GET /api/v1/integrations/config"""
        return await self._client.request("GET", "/api/v1/integrations/config")

    async def get_config(self, integration_type: str) -> dict[str, Any]:
        """Get config for an integration type. GET /api/v1/integrations/config/:integration_type"""
        return await self._client.request(
            "GET", f"/api/v1/integrations/config/{integration_type}"
        )

    async def sync_integration(self, integration_type: str) -> dict[str, Any]:
        """Trigger synchronization. POST /api/v1/integrations/:integration_type/sync"""
        return await self._client.request(
            "POST", f"/api/v1/integrations/{integration_type}/sync"
        )

    # =========================================================================
    # Bot Platform Status
    # =========================================================================

    async def get_slack_status(self) -> dict[str, Any]:
        """Get Slack bot connection status."""
        return await self._client.request("GET", "/api/v1/bots/slack/status")

    async def get_telegram_status(self) -> dict[str, Any]:
        """Get Telegram bot connection status."""
        return await self._client.request("GET", "/api/v1/bots/telegram/status")

    async def get_whatsapp_status(self) -> dict[str, Any]:
        """Get WhatsApp bot connection status."""
        return await self._client.request("GET", "/api/v1/bots/whatsapp/status")

    async def get_discord_status(self) -> dict[str, Any]:
        """Get Discord bot connection status."""
        return await self._client.request("GET", "/api/v1/bots/discord/status")

    async def get_google_chat_status(self) -> dict[str, Any]:
        """Get Google Chat bot connection status."""
        return await self._client.request("GET", "/api/v1/bots/google-chat/status")

    # =========================================================================
    # Teams Integration
    # =========================================================================

    async def get_teams_status(self) -> dict[str, Any]:
        """Get Microsoft Teams integration status."""
        return await self._client.request("GET", "/api/v1/integrations/teams/status")

    async def notify_teams(
        self,
        channel_id: str,
        message: str,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send notification to Teams channel."""
        payload: dict[str, Any] = {"channel_id": channel_id, "message": message}
        if options:
            payload.update(options)
        return await self._client.request(
            "POST", "/api/v1/integrations/teams/notify", json=payload
        )

    # =========================================================================
    # Zapier Integration
    # =========================================================================

    async def list_zapier_apps(self, workspace_id: str | None = None) -> dict[str, Any]:
        """List Zapier apps for workspace."""
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        return await self._client.request(
            "GET", "/api/v1/integrations/zapier/apps", params=params or None
        )

    async def create_zapier_app(self, workspace_id: str) -> dict[str, Any]:
        """Create new Zapier app (returns API credentials)."""
        return await self._client.request(
            "POST", "/api/v1/integrations/zapier/apps", json={"workspace_id": workspace_id}
        )

    async def list_zapier_trigger_types(self) -> dict[str, Any]:
        """Get available Zapier trigger and action types."""
        return await self._client.request("GET", "/api/v1/integrations/zapier/triggers")

    async def subscribe_zapier_trigger(
        self,
        app_id: str,
        trigger_type: str,
        webhook_url: str,
        workspace_id: str | None = None,
        debate_tags: _List[str] | None = None,
        min_confidence: float | None = None,
    ) -> dict[str, Any]:
        """Subscribe to Zapier trigger."""
        payload: dict[str, Any] = {
            "app_id": app_id,
            "trigger_type": trigger_type,
            "webhook_url": webhook_url,
        }
        if workspace_id:
            payload["workspace_id"] = workspace_id
        if debate_tags:
            payload["debate_tags"] = debate_tags
        if min_confidence is not None:
            payload["min_confidence"] = min_confidence
        return await self._client.request(
            "POST", "/api/v1/integrations/zapier/triggers", json=payload
        )

    # =========================================================================
    # Make (Integromat) Integration
    # =========================================================================

    async def list_make_connections(self, workspace_id: str | None = None) -> dict[str, Any]:
        """List Make connections for workspace."""
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        return await self._client.request(
            "GET", "/api/v1/integrations/make/connections", params=params or None
        )

    async def create_make_connection(self, workspace_id: str) -> dict[str, Any]:
        """Create new Make connection (returns API key)."""
        return await self._client.request(
            "POST", "/api/v1/integrations/make/connections", json={"workspace_id": workspace_id}
        )

    async def list_make_modules(self) -> dict[str, Any]:
        """Get available Make module types."""
        return await self._client.request("GET", "/api/v1/integrations/make/modules")

    async def register_make_webhook(
        self,
        connection_id: str,
        module_type: str,
        webhook_url: str,
        workspace_id: str | None = None,
        event_filter: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Register Make webhook."""
        payload: dict[str, Any] = {
            "connection_id": connection_id,
            "module_type": module_type,
            "webhook_url": webhook_url,
        }
        if workspace_id:
            payload["workspace_id"] = workspace_id
        if event_filter:
            payload["event_filter"] = event_filter
        return await self._client.request(
            "POST", "/api/v1/integrations/make/webhooks", json=payload
        )

    # =========================================================================
    # n8n Integration
    # =========================================================================

    async def list_n8n_credentials(self, workspace_id: str | None = None) -> dict[str, Any]:
        """List n8n credentials for workspace."""
        params: dict[str, Any] = {}
        if workspace_id:
            params["workspace_id"] = workspace_id
        return await self._client.request(
            "GET", "/api/v1/integrations/n8n/credentials", params=params or None
        )

    async def create_n8n_credential(
        self, workspace_id: str, api_url: str | None = None
    ) -> dict[str, Any]:
        """Create n8n credential (returns API key)."""
        payload: dict[str, Any] = {"workspace_id": workspace_id}
        if api_url:
            payload["api_url"] = api_url
        return await self._client.request(
            "POST", "/api/v1/integrations/n8n/credentials", json=payload
        )

    async def get_n8n_nodes(self) -> dict[str, Any]:
        """Get n8n node, trigger, and credential definitions."""
        return await self._client.request("GET", "/api/v1/integrations/n8n/nodes")

    async def register_n8n_webhook(
        self,
        credential_id: str,
        events: _List[str],
        workflow_id: str | None = None,
        node_id: str | None = None,
        workspace_id: str | None = None,
    ) -> dict[str, Any]:
        """Register n8n webhook."""
        payload: dict[str, Any] = {"credential_id": credential_id, "events": events}
        if workflow_id:
            payload["workflow_id"] = workflow_id
        if node_id:
            payload["node_id"] = node_id
        if workspace_id:
            payload["workspace_id"] = workspace_id
        return await self._client.request(
            "POST", "/api/v1/integrations/n8n/webhooks", json=payload
        )

    # =========================================================================
    # Integration Wizard (v2)
    # =========================================================================

    async def start_wizard(self, integration_type: str) -> dict[str, Any]:
        """Start integration setup wizard."""
        return await self._client.request(
            "POST", "/api/v2/integrations/wizard", json={"type": integration_type}
        )

    # =========================================================================
    # v2 Management
    # =========================================================================

    async def list_v2(
        self,
        type: str | None = None,
        status: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """List integrations with advanced filtering (v2)."""
        params: dict[str, Any] = {}
        if type:
            params["type"] = type
        if status:
            params["status"] = status
        if limit is not None:
            params["limit"] = limit
        if offset is not None:
            params["offset"] = offset
        return await self._client.request("GET", "/api/v2/integrations", params=params or None)

    async def get_by_type(self, type: str) -> dict[str, Any]:
        """Get integration by type (Slack, Teams, Discord, Email)."""
        return await self._client.request("GET", f"/api/v2/integrations/{type}")

    async def test_by_type(self, type: str) -> dict[str, Any]:
        """Test integration connectivity by type (v2)."""
        return await self._client.request("POST", f"/api/v2/integrations/{type}/test", json={})

    async def test_platform(self, platform: str) -> dict[str, Any]:
        """Test specific platform integration (v1)."""
        return await self._client.request("POST", f"/api/v1/integrations/{platform}/test", json={})
