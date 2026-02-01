"""Tests for Integrations namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora.client import AragoraAsyncClient, AragoraClient

# =========================================================================
# Bot Platform Status
# =========================================================================


class TestIntegrationsBotStatus:
    """Tests for bot platform status methods."""

    def test_get_slack_status(self) -> None:
        """Get Slack bot status."""
        with patch.object(AragoraClient, "_request") as mock_request:
            mock_request.return_value = {"connected": True, "bot_id": "B123"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.integrations.get_slack_status()

            mock_request.assert_called_once_with("GET", "/api/v1/bots/slack/status")
            assert result["connected"] is True
            client.close()

    def test_get_telegram_status(self) -> None:
        """Get Telegram bot status."""
        with patch.object(AragoraClient, "_request") as mock_request:
            mock_request.return_value = {"connected": False}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.get_telegram_status()

            mock_request.assert_called_once_with("GET", "/api/v1/bots/telegram/status")
            client.close()

    def test_get_whatsapp_status(self) -> None:
        """Get WhatsApp bot status."""
        with patch.object(AragoraClient, "_request") as mock_request:
            mock_request.return_value = {"connected": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.get_whatsapp_status()

            mock_request.assert_called_once_with("GET", "/api/v1/bots/whatsapp/status")
            client.close()

    def test_get_discord_status(self) -> None:
        """Get Discord bot status."""
        with patch.object(AragoraClient, "_request") as mock_request:
            mock_request.return_value = {"connected": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.get_discord_status()

            mock_request.assert_called_once_with("GET", "/api/v1/bots/discord/status")
            client.close()

    def test_get_google_chat_status(self) -> None:
        """Get Google Chat bot status."""
        with patch.object(AragoraClient, "_request") as mock_request:
            mock_request.return_value = {"connected": False}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.get_google_chat_status()

            mock_request.assert_called_once_with("GET", "/api/v1/bots/google-chat/status")
            client.close()

    def test_get_email_status(self) -> None:
        """Get email integration status."""
        with patch.object(AragoraClient, "_request") as mock_request:
            mock_request.return_value = {"connected": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.get_email_status()

            mock_request.assert_called_once_with("GET", "/api/v1/bots/email/status")
            client.close()


# =========================================================================
# Teams Integration
# =========================================================================


class TestIntegrationsTeams:
    """Tests for Teams integration methods."""

    def test_get_teams_status(self) -> None:
        """Get Teams integration status."""
        with patch.object(AragoraClient, "_request") as mock_request:
            mock_request.return_value = {"status": "active"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.get_teams_status()

            mock_request.assert_called_once_with("GET", "/api/v1/integrations/teams/status")
            client.close()

    def test_install_teams(self) -> None:
        """Install Teams app."""
        with patch.object(AragoraClient, "_request") as mock_request:
            mock_request.return_value = {"install_url": "https://..."}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.install_teams("tenant_abc")

            mock_request.assert_called_once_with(
                "POST",
                "/api/integrations/teams/install",
                json={"tenant_id": "tenant_abc"},
            )
            client.close()

    def test_teams_callback(self) -> None:
        """Handle Teams OAuth callback."""
        with patch.object(AragoraClient, "_request") as mock_request:
            mock_request.return_value = {"success": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.teams_callback("code_123", "state_abc")

            mock_request.assert_called_once_with(
                "POST",
                "/api/integrations/teams/callback",
                json={"code": "code_123", "state": "state_abc"},
            )
            client.close()

    def test_refresh_teams_token(self) -> None:
        """Refresh Teams token."""
        with patch.object(AragoraClient, "_request") as mock_request:
            mock_request.return_value = {"refreshed": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.refresh_teams_token()

            mock_request.assert_called_once_with("POST", "/api/integrations/teams/refresh", json={})
            client.close()

    def test_notify_teams(self) -> None:
        """Send notification to Teams channel."""
        with patch.object(AragoraClient, "_request") as mock_request:
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
        with patch.object(AragoraClient, "_request") as mock_request:
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
# Discord Integration
# =========================================================================


class TestIntegrationsDiscord:
    """Tests for Discord integration methods."""

    def test_install_discord(self) -> None:
        """Install Discord bot."""
        with patch.object(AragoraClient, "_request") as mock_request:
            mock_request.return_value = {"installed": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.install_discord("guild_1")

            mock_request.assert_called_once_with(
                "POST",
                "/api/integrations/discord/install",
                json={"guild_id": "guild_1"},
            )
            client.close()

    def test_discord_callback(self) -> None:
        """Handle Discord OAuth callback."""
        with patch.object(AragoraClient, "_request") as mock_request:
            mock_request.return_value = {"success": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.discord_callback("code_1", "state_1")

            mock_request.assert_called_once_with(
                "POST",
                "/api/integrations/discord/callback",
                json={"code": "code_1", "state": "state_1"},
            )
            client.close()

    def test_uninstall_discord(self) -> None:
        """Uninstall Discord bot."""
        with patch.object(AragoraClient, "_request") as mock_request:
            mock_request.return_value = {"uninstalled": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.uninstall_discord("guild_1")

            mock_request.assert_called_once_with(
                "POST",
                "/api/integrations/discord/uninstall",
                json={"guild_id": "guild_1"},
            )
            client.close()


# =========================================================================
# Zapier Integration
# =========================================================================


class TestIntegrationsZapier:
    """Tests for Zapier integration methods."""

    def test_list_zapier_apps(self) -> None:
        """List Zapier apps."""
        with patch.object(AragoraClient, "_request") as mock_request:
            mock_request.return_value = {"apps": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.list_zapier_apps()

            mock_request.assert_called_once_with(
                "GET", "/api/v1/integrations/zapier/apps", params=None
            )
            client.close()

    def test_list_zapier_apps_with_workspace(self) -> None:
        """List Zapier apps filtered by workspace."""
        with patch.object(AragoraClient, "_request") as mock_request:
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
        with patch.object(AragoraClient, "_request") as mock_request:
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

    def test_delete_zapier_app(self) -> None:
        """Delete Zapier app."""
        with patch.object(AragoraClient, "_request") as mock_request:
            mock_request.return_value = {"deleted": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.delete_zapier_app("zap_1")

            mock_request.assert_called_once_with("DELETE", "/api/v1/integrations/zapier/apps/zap_1")
            client.close()

    def test_list_zapier_trigger_types(self) -> None:
        """List Zapier trigger types."""
        with patch.object(AragoraClient, "_request") as mock_request:
            mock_request.return_value = {"triggers": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.list_zapier_trigger_types()

            mock_request.assert_called_once_with("GET", "/api/v1/integrations/zapier/triggers")
            client.close()

    def test_subscribe_zapier_trigger(self) -> None:
        """Subscribe to Zapier trigger."""
        with patch.object(AragoraClient, "_request") as mock_request:
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
        with patch.object(AragoraClient, "_request") as mock_request:
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

    def test_unsubscribe_zapier_trigger(self) -> None:
        """Unsubscribe from Zapier trigger."""
        with patch.object(AragoraClient, "_request") as mock_request:
            mock_request.return_value = {"unsubscribed": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.unsubscribe_zapier_trigger("t_1", "app_1")

            mock_request.assert_called_once_with(
                "DELETE",
                "/api/v1/integrations/zapier/triggers/t_1",
                params={"app_id": "app_1"},
            )
            client.close()


# =========================================================================
# Make Integration
# =========================================================================


class TestIntegrationsMake:
    """Tests for Make (Integromat) integration methods."""

    def test_list_make_connections(self) -> None:
        """List Make connections."""
        with patch.object(AragoraClient, "_request") as mock_request:
            mock_request.return_value = {"connections": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.list_make_connections()

            mock_request.assert_called_once_with(
                "GET", "/api/v1/integrations/make/connections", params=None
            )
            client.close()

    def test_create_make_connection(self) -> None:
        """Create Make connection."""
        with patch.object(AragoraClient, "_request") as mock_request:
            mock_request.return_value = {"connection_id": "mc_1"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.create_make_connection("ws_1")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/integrations/make/connections",
                json={"workspace_id": "ws_1"},
            )
            client.close()

    def test_delete_make_connection(self) -> None:
        """Delete Make connection."""
        with patch.object(AragoraClient, "_request") as mock_request:
            mock_request.return_value = {"deleted": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.delete_make_connection("mc_1")

            mock_request.assert_called_once_with(
                "DELETE", "/api/v1/integrations/make/connections/mc_1"
            )
            client.close()

    def test_list_make_modules(self) -> None:
        """List Make modules."""
        with patch.object(AragoraClient, "_request") as mock_request:
            mock_request.return_value = {"modules": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.list_make_modules()

            mock_request.assert_called_once_with("GET", "/api/v1/integrations/make/modules")
            client.close()

    def test_register_make_webhook(self) -> None:
        """Register Make webhook."""
        with patch.object(AragoraClient, "_request") as mock_request:
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

    def test_unregister_make_webhook(self) -> None:
        """Unregister Make webhook."""
        with patch.object(AragoraClient, "_request") as mock_request:
            mock_request.return_value = {"unregistered": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.unregister_make_webhook("wh_1", "mc_1")

            mock_request.assert_called_once_with(
                "DELETE",
                "/api/v1/integrations/make/webhooks/wh_1",
                params={"connection_id": "mc_1"},
            )
            client.close()


# =========================================================================
# n8n Integration
# =========================================================================


class TestIntegrationsN8n:
    """Tests for n8n integration methods."""

    def test_list_n8n_credentials(self) -> None:
        """List n8n credentials."""
        with patch.object(AragoraClient, "_request") as mock_request:
            mock_request.return_value = {"credentials": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.list_n8n_credentials()

            mock_request.assert_called_once_with(
                "GET", "/api/v1/integrations/n8n/credentials", params=None
            )
            client.close()

    def test_create_n8n_credential(self) -> None:
        """Create n8n credential."""
        with patch.object(AragoraClient, "_request") as mock_request:
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
        with patch.object(AragoraClient, "_request") as mock_request:
            mock_request.return_value = {"credential_id": "nc_1"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.create_n8n_credential("ws_1", api_url="https://n8n.local")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/integrations/n8n/credentials",
                json={"workspace_id": "ws_1", "api_url": "https://n8n.local"},
            )
            client.close()

    def test_delete_n8n_credential(self) -> None:
        """Delete n8n credential."""
        with patch.object(AragoraClient, "_request") as mock_request:
            mock_request.return_value = {"deleted": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.delete_n8n_credential("nc_1")

            mock_request.assert_called_once_with(
                "DELETE", "/api/v1/integrations/n8n/credentials/nc_1"
            )
            client.close()

    def test_get_n8n_nodes(self) -> None:
        """Get n8n node definitions."""
        with patch.object(AragoraClient, "_request") as mock_request:
            mock_request.return_value = {"nodes": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.get_n8n_nodes()

            mock_request.assert_called_once_with("GET", "/api/v1/integrations/n8n/nodes")
            client.close()

    def test_register_n8n_webhook(self) -> None:
        """Register n8n webhook."""
        with patch.object(AragoraClient, "_request") as mock_request:
            mock_request.return_value = {"webhook_id": "nwh_1"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.register_n8n_webhook("nc_1", ["debate.complete"])

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/integrations/n8n/webhooks",
                json={"credential_id": "nc_1", "events": ["debate.complete"]},
            )
            client.close()

    def test_unregister_n8n_webhook(self) -> None:
        """Unregister n8n webhook."""
        with patch.object(AragoraClient, "_request") as mock_request:
            mock_request.return_value = {"unregistered": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.unregister_n8n_webhook("nwh_1", "nc_1")

            mock_request.assert_called_once_with(
                "DELETE",
                "/api/v1/integrations/n8n/webhooks/nwh_1",
                params={"credential_id": "nc_1"},
            )
            client.close()


# =========================================================================
# Integration Wizard
# =========================================================================


class TestIntegrationsWizard:
    """Tests for integration wizard methods."""

    def test_start_wizard(self) -> None:
        """Start integration wizard."""
        with patch.object(AragoraClient, "_request") as mock_request:
            mock_request.return_value = {"session_id": "wiz_1", "step": "auth"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.integrations.start_wizard("slack")

            mock_request.assert_called_once_with(
                "POST", "/api/v2/integrations/wizard", json={"type": "slack"}
            )
            assert result["session_id"] == "wiz_1"
            client.close()

    def test_get_wizard_status(self) -> None:
        """Get wizard status."""
        with patch.object(AragoraClient, "_request") as mock_request:
            mock_request.return_value = {"step": "configure", "progress": 50}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.get_wizard_status("wiz_1")

            mock_request.assert_called_once_with(
                "GET",
                "/api/v2/integrations/wizard/status",
                params={"session_id": "wiz_1"},
            )
            client.close()

    def test_validate_wizard_step(self) -> None:
        """Validate wizard step."""
        with patch.object(AragoraClient, "_request") as mock_request:
            mock_request.return_value = {"valid": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.validate_wizard_step("wiz_1", "auth", {"token": "xxx"})

            mock_request.assert_called_once_with(
                "POST",
                "/api/v2/integrations/wizard/validate",
                json={
                    "session_id": "wiz_1",
                    "step": "auth",
                    "data": {"token": "xxx"},
                },
            )
            client.close()


# =========================================================================
# v2 Management
# =========================================================================


class TestIntegrationsV2Management:
    """Tests for v2 management methods."""

    def test_list_v2_no_filters(self) -> None:
        """List v2 integrations without filters."""
        with patch.object(AragoraClient, "_request") as mock_request:
            mock_request.return_value = {"integrations": []}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.list_v2()

            mock_request.assert_called_once_with("GET", "/api/v2/integrations", params=None)
            client.close()

    def test_list_v2_with_filters(self) -> None:
        """List v2 integrations with filters."""
        with patch.object(AragoraClient, "_request") as mock_request:
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
        with patch.object(AragoraClient, "_request") as mock_request:
            mock_request.return_value = {"type": "slack", "status": "active"}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.get_by_type("slack")

            mock_request.assert_called_once_with("GET", "/api/v2/integrations/slack")
            client.close()

    def test_get_health(self) -> None:
        """Get integration health."""
        with patch.object(AragoraClient, "_request") as mock_request:
            mock_request.return_value = {"healthy": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.integrations.get_health("slack")

            mock_request.assert_called_once_with("GET", "/api/v2/integrations/slack/health")
            assert result["healthy"] is True
            client.close()

    def test_test_by_type(self) -> None:
        """Test integration by type."""
        with patch.object(AragoraClient, "_request") as mock_request:
            mock_request.return_value = {"success": True}

            client = AragoraClient(base_url="https://api.aragora.ai")
            client.integrations.test_by_type("slack")

            mock_request.assert_called_once_with("POST", "/api/v2/integrations/slack/test", json={})
            client.close()

    def test_get_stats(self) -> None:
        """Get integration stats."""
        with patch.object(AragoraClient, "_request") as mock_request:
            mock_request.return_value = {"total_syncs_24h": 150}

            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.integrations.get_stats()

            mock_request.assert_called_once_with("GET", "/api/v2/integrations/stats")
            assert result["total_syncs_24h"] == 150
            client.close()

    def test_test_platform(self) -> None:
        """Test platform integration."""
        with patch.object(AragoraClient, "_request") as mock_request:
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
    async def test_async_get_slack_status(self) -> None:
        """Get Slack status asynchronously."""
        with patch.object(AragoraAsyncClient, "_request") as mock_request:
            mock_request.return_value = {"connected": True}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.integrations.get_slack_status()

                mock_request.assert_called_once_with("GET", "/api/v1/bots/slack/status")
                assert result["connected"] is True

    @pytest.mark.asyncio
    async def test_async_install_teams(self) -> None:
        """Install Teams asynchronously."""
        with patch.object(AragoraAsyncClient, "_request") as mock_request:
            mock_request.return_value = {"install_url": "https://..."}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                await client.integrations.install_teams("tenant_1")

                mock_request.assert_called_once_with(
                    "POST",
                    "/api/integrations/teams/install",
                    json={"tenant_id": "tenant_1"},
                )

    @pytest.mark.asyncio
    async def test_async_list_zapier_apps(self) -> None:
        """List Zapier apps asynchronously."""
        with patch.object(AragoraAsyncClient, "_request") as mock_request:
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
        with patch.object(AragoraAsyncClient, "_request") as mock_request:
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
        with patch.object(AragoraAsyncClient, "_request") as mock_request:
            mock_request.return_value = {"session_id": "wiz_1"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.integrations.start_wizard("discord")

                mock_request.assert_called_once_with(
                    "POST", "/api/v2/integrations/wizard", json={"type": "discord"}
                )
                assert result["session_id"] == "wiz_1"

    @pytest.mark.asyncio
    async def test_async_get_stats(self) -> None:
        """Get stats asynchronously."""
        with patch.object(AragoraAsyncClient, "_request") as mock_request:
            mock_request.return_value = {"total_syncs_24h": 100}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.integrations.get_stats()

                mock_request.assert_called_once_with("GET", "/api/v2/integrations/stats")
                assert result["total_syncs_24h"] == 100

    @pytest.mark.asyncio
    async def test_async_register_n8n_webhook(self) -> None:
        """Register n8n webhook asynchronously."""
        with patch.object(AragoraAsyncClient, "_request") as mock_request:
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

    @pytest.mark.asyncio
    async def test_async_get_health(self) -> None:
        """Get integration health asynchronously."""
        with patch.object(AragoraAsyncClient, "_request") as mock_request:
            mock_request.return_value = {"healthy": True}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.integrations.get_health("teams")

                mock_request.assert_called_once_with("GET", "/api/v2/integrations/teams/health")
                assert result["healthy"] is True
