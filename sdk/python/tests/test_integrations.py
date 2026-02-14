"""Tests for Integrations namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient

# =========================================================================
# Bot Platform Status
# =========================================================================


class TestIntegrationsBotStatus:
    """Tests for bot platform status methods."""

    def test_get_telegram_status(self) -> None:
        """Get Telegram bot status."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"connected": False}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.get_telegram_status()

            mock_request.assert_called_once_with("GET", "/api/v1/bots/telegram/status")
            client.close()

    def test_get_whatsapp_status(self) -> None:
        """Get WhatsApp bot status."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"connected": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.get_whatsapp_status()

            mock_request.assert_called_once_with("GET", "/api/v1/bots/whatsapp/status")
            client.close()

    def test_get_discord_status(self) -> None:
        """Get Discord bot status."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"connected": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.get_discord_status()

            mock_request.assert_called_once_with("GET", "/api/v1/bots/discord/status")
            client.close()

    def test_get_google_chat_status(self) -> None:
        """Get Google Chat bot status."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"connected": False}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.get_google_chat_status()

            mock_request.assert_called_once_with("GET", "/api/v1/bots/google-chat/status")
            client.close()


# =========================================================================
# Teams Integration
# =========================================================================


class TestIntegrationsTeams:
    """Tests for Teams integration methods."""

    def test_get_teams_status(self) -> None:
        """Get Teams integration status."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"status": "active"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.get_teams_status()

            mock_request.assert_called_once_with("GET", "/api/v1/integrations/teams/status")
            client.close()

    def test_notify_teams(self) -> None:
        """Send notification to Teams channel."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"sent": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.notify_teams("ch_1", "Hello Teams!")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/integrations/teams/notify",
                json={"channel_id": "ch_1", "message": "Hello Teams!"},
            )
            client.close()

    def test_notify_teams_with_options(self) -> None:
        """Send notification to Teams with extra options."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"sent": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.notify_teams("ch_1", "Hello!", options={"priority": "high"})

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/integrations/teams/notify",
                json={"channel_id": "ch_1", "message": "Hello!", "priority": "high"},
            )
            client.close()


# =========================================================================
# Zapier Integration
# =========================================================================


class TestIntegrationsZapier:
    """Tests for Zapier integration methods."""

    def test_list_zapier_apps(self) -> None:
        """List Zapier apps."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"apps": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.list_zapier_apps()

            mock_request.assert_called_once_with(
                "GET", "/api/v1/integrations/zapier/apps", params=None
            )
            client.close()

    def test_list_zapier_apps_with_workspace(self) -> None:
        """List Zapier apps filtered by workspace."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"apps": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.list_zapier_apps(workspace_id="ws_1")

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/integrations/zapier/apps",
                params={"workspace_id": "ws_1"},
            )
            client.close()

    def test_create_zapier_app(self) -> None:
        """Create Zapier app."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"app_id": "zap_1", "api_key": "ak_xxx"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.integrations.create_zapier_app("ws_1")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/integrations/zapier/apps",
                json={"workspace_id": "ws_1"},
            )
            assert "api_key" in result
            client.close()

    def test_list_zapier_trigger_types(self) -> None:
        """List Zapier trigger types."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"triggers": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.list_zapier_trigger_types()

            mock_request.assert_called_once_with("GET", "/api/v1/integrations/zapier/triggers")
            client.close()

    def test_subscribe_zapier_trigger(self) -> None:
        """Subscribe to Zapier trigger."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"trigger_id": "t_1"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.subscribe_zapier_trigger(
                "app_1", "debate_complete", "https://hooks.zapier.com/xxx"
            )

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/integrations/zapier/triggers",
                json={
                    "app_id": "app_1",
                    "trigger_type": "debate_complete",
                    "webhook_url": "https://hooks.zapier.com/xxx",
                },
            )
            client.close()

    def test_subscribe_zapier_trigger_with_options(self) -> None:
        """Subscribe to Zapier trigger with filtering options."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"trigger_id": "t_1"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.subscribe_zapier_trigger(
                "app_1",
                "debate_complete",
                "https://hooks.zapier.com/xxx",
                workspace_id="ws_1",
                debate_tags=["security"],
                min_confidence=0.8,
            )

            call_json = mock_request.call_args[1]["json"]
            assert call_json["workspace_id"] == "ws_1"
            assert call_json["debate_tags"] == ["security"]
            assert call_json["min_confidence"] == 0.8
            client.close()


# =========================================================================
# Make Integration
# =========================================================================


class TestIntegrationsMake:
    """Tests for Make (Integromat) integration methods."""

    def test_list_make_connections(self) -> None:
        """List Make connections."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"connections": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.list_make_connections()

            mock_request.assert_called_once_with(
                "GET", "/api/v1/integrations/make/connections", params=None
            )
            client.close()

    def test_create_make_connection(self) -> None:
        """Create Make connection."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"connection_id": "mc_1"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.create_make_connection("ws_1")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/integrations/make/connections",
                json={"workspace_id": "ws_1"},
            )
            client.close()

    def test_list_make_modules(self) -> None:
        """List Make modules."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"modules": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.list_make_modules()

            mock_request.assert_called_once_with("GET", "/api/v1/integrations/make/modules")
            client.close()

    def test_register_make_webhook(self) -> None:
        """Register Make webhook."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"webhook_id": "wh_1"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.register_make_webhook(
                "mc_1", "debate_result", "https://hook.make.com/xxx"
            )

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/integrations/make/webhooks",
                json={
                    "connection_id": "mc_1",
                    "module_type": "debate_result",
                    "webhook_url": "https://hook.make.com/xxx",
                },
            )
            client.close()


# =========================================================================
# n8n Integration
# =========================================================================


class TestIntegrationsN8n:
    """Tests for n8n integration methods."""

    def test_list_n8n_credentials(self) -> None:
        """List n8n credentials."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"credentials": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.list_n8n_credentials()

            mock_request.assert_called_once_with(
                "GET", "/api/v1/integrations/n8n/credentials", params=None
            )
            client.close()

    def test_create_n8n_credential(self) -> None:
        """Create n8n credential."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"credential_id": "nc_1"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.create_n8n_credential("ws_1")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/integrations/n8n/credentials",
                json={"workspace_id": "ws_1"},
            )
            client.close()

    def test_create_n8n_credential_with_url(self) -> None:
        """Create n8n credential with custom API URL."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"credential_id": "nc_1"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.create_n8n_credential("ws_1", api_url="https://n8n.local")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/integrations/n8n/credentials",
                json={"workspace_id": "ws_1", "api_url": "https://n8n.local"},
            )
            client.close()

    def test_get_n8n_nodes(self) -> None:
        """Get n8n node definitions."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"nodes": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.get_n8n_nodes()

            mock_request.assert_called_once_with("GET", "/api/v1/integrations/n8n/nodes")
            client.close()

    def test_register_n8n_webhook(self) -> None:
        """Register n8n webhook."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"webhook_id": "nwh_1"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.register_n8n_webhook("nc_1", ["debate.complete"])

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/integrations/n8n/webhooks",
                json={"credential_id": "nc_1", "events": ["debate.complete"]},
            )
            client.close()


# =========================================================================
# Integration Wizard
# =========================================================================


class TestIntegrationsWizard:
    """Tests for integration wizard methods."""

    def test_start_wizard(self) -> None:
        """Start integration wizard."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"session_id": "wiz_1", "step": "auth"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.integrations.start_wizard("slack")

            mock_request.assert_called_once_with(
                "POST", "/api/v2/integrations/wizard", json={"type": "slack"}
            )
            assert result["session_id"] == "wiz_1"
            client.close()


# =========================================================================
# v2 Management
# =========================================================================


class TestIntegrationsV2Management:
    """Tests for v2 management methods."""

    def test_list_v2_no_filters(self) -> None:
        """List v2 integrations without filters."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"integrations": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.list_v2()

            mock_request.assert_called_once_with("GET", "/api/v2/integrations", params=None)
            client.close()

    def test_list_v2_with_filters(self) -> None:
        """List v2 integrations with filters."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"integrations": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.list_v2(type="slack", status="active", limit=10)

            mock_request.assert_called_once_with(
                "GET",
                "/api/v2/integrations",
                params={"type": "slack", "status": "active", "limit": 10},
            )
            client.close()

    def test_get_by_type(self) -> None:
        """Get integration by type."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"type": "slack", "status": "active"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.get_by_type("slack")

            mock_request.assert_called_once_with("GET", "/api/v2/integrations/slack")
            client.close()

    def test_test_by_type(self) -> None:
        """Test integration by type."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"success": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.test_by_type("slack")

            mock_request.assert_called_once_with("POST", "/api/v2/integrations/slack/test", json={})
            client.close()

    def test_test_platform(self) -> None:
        """Test platform integration."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"success": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.test_platform("teams")

            mock_request.assert_called_once_with("POST", "/api/v1/integrations/teams/test", json={})
            client.close()


# =========================================================================
# Async Tests
# =========================================================================


class TestAsyncIntegrations:
    """Tests for async Integrations API."""

    @pytest.mark.asyncio
    async def test_async_list_zapier_apps(self) -> None:
        """List Zapier apps asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"apps": []}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                await client.integrations.list_zapier_apps(workspace_id="ws_1")

                mock_request.assert_called_once_with(
                    "GET",
                    "/api/v1/integrations/zapier/apps",
                    params={"workspace_id": "ws_1"},
                )

    @pytest.mark.asyncio
    async def test_async_create_make_connection(self) -> None:
        """Create Make connection asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"connection_id": "mc_1"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.integrations.create_make_connection("ws_1")

                mock_request.assert_called_once_with(
                    "POST",
                    "/api/v1/integrations/make/connections",
                    json={"workspace_id": "ws_1"},
                )
                assert result["connection_id"] == "mc_1"

    @pytest.mark.asyncio
    async def test_async_start_wizard(self) -> None:
        """Start wizard asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"session_id": "wiz_1"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.integrations.start_wizard("discord")

                mock_request.assert_called_once_with(
                    "POST", "/api/v2/integrations/wizard", json={"type": "discord"}
                )
                assert result["session_id"] == "wiz_1"

    @pytest.mark.asyncio
    async def test_async_register_n8n_webhook(self) -> None:
        """Register n8n webhook asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"webhook_id": "nwh_1"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                await client.integrations.register_n8n_webhook(
                    "nc_1", ["debate.complete", "consensus.reached"]
                )

                mock_request.assert_called_once_with(
                    "POST",
                    "/api/v1/integrations/n8n/webhooks",
                    json={
                        "credential_id": "nc_1",
                        "events": ["debate.complete", "consensus.reached"],
                    },
                )
