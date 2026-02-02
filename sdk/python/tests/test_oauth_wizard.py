"""Tests for OAuth Wizard namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient


class TestOAuthWizardConfiguration:
    """Tests for wizard configuration methods."""

    def test_get_config(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "available_providers": 12,
                "configured_providers": 3,
                "completion_percent": 25,
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.oauth_wizard.get_config()
            mock_request.assert_called_once_with("GET", "/api/v2/integrations/wizard")
            assert result["available_providers"] == 12
            assert result["completion_percent"] == 25
            client.close()


class TestOAuthWizardProviderDiscovery:
    """Tests for provider listing and discovery methods."""

    def test_list_providers_no_filters(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"providers": [{"id": "slack"}, {"id": "github"}]}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.oauth_wizard.list_providers()
            mock_request.assert_called_once_with(
                "GET", "/api/v2/integrations/wizard/providers", params={}
            )
            assert len(result["providers"]) == 2
            client.close()

    def test_list_providers_with_category(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "providers": [{"id": "slack", "category": "communication"}]
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.oauth_wizard.list_providers(category="communication")
            mock_request.assert_called_once_with(
                "GET",
                "/api/v2/integrations/wizard/providers",
                params={"category": "communication"},
            )
            client.close()

    def test_list_providers_with_configured_filter(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"providers": []}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.oauth_wizard.list_providers(category="development", configured=True)
            mock_request.assert_called_once_with(
                "GET",
                "/api/v2/integrations/wizard/providers",
                params={"category": "development", "configured": True},
            )
            client.close()

    def test_get_provider(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "id": "slack",
                "name": "Slack",
                "category": "communication",
                "oauth_scopes": ["channels:read", "chat:write"],
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.oauth_wizard.get_provider("slack")
            mock_request.assert_called_once_with(
                "GET", "/api/v2/integrations/wizard/providers/slack"
            )
            assert result["id"] == "slack"
            assert result["category"] == "communication"
            client.close()


class TestOAuthWizardStatus:
    """Tests for integration status methods."""

    def test_get_status(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "total_providers": 12,
                "configured": 3,
                "connected": 2,
                "health_score": 0.85,
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.oauth_wizard.get_status()
            mock_request.assert_called_once_with("GET", "/api/v2/integrations/wizard/status")
            assert result["health_score"] == 0.85
            client.close()

    def test_get_provider_status(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "provider_id": "github",
                "name": "GitHub",
                "status": "connected",
                "connected": True,
                "missing_vars": [],
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.oauth_wizard.get_provider_status("github")
            mock_request.assert_called_once_with("GET", "/api/v2/integrations/wizard/status/github")
            assert result["connected"] is True
            assert result["status"] == "connected"
            client.close()


class TestOAuthWizardValidation:
    """Tests for validation and preflight check methods."""

    def test_validate_config(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "provider_id": "slack",
                "valid": True,
                "can_proceed": True,
                "errors": [],
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.oauth_wizard.validate_config("slack")
            mock_request.assert_called_once_with(
                "POST",
                "/api/v2/integrations/wizard/validate",
                json={"provider_id": "slack"},
            )
            assert result["valid"] is True
            assert result["can_proceed"] is True
            client.close()

    def test_run_preflight_checks(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "provider_id": "github",
                "checks": [
                    {"name": "connection", "status": "passed", "required": True},
                    {"name": "scopes", "status": "passed", "required": True},
                ],
                "can_connect": True,
                "estimated_connect_time_seconds": 5,
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.oauth_wizard.run_preflight_checks("github")
            mock_request.assert_called_once_with(
                "POST",
                "/api/v2/integrations/wizard/preflight",
                json={"provider_id": "github"},
            )
            assert result["can_connect"] is True
            assert len(result["checks"]) == 2
            client.close()


class TestOAuthWizardConnectionManagement:
    """Tests for connection management methods."""

    def test_get_install_url_minimal(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "install_url": "https://slack.com/oauth/authorize?client_id=abc",
                "state": "xyz123",
                "expires_at": "2025-06-01T12:00:00Z",
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.oauth_wizard.get_install_url("slack")
            mock_request.assert_called_once_with(
                "POST",
                "/api/v2/integrations/wizard/providers/slack/install",
                json={},
            )
            assert "install_url" in result
            assert result["state"] == "xyz123"
            client.close()

    def test_get_install_url_with_all_options(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"install_url": "https://slack.com/oauth/authorize"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.oauth_wizard.get_install_url(
                "slack",
                redirect_uri="https://myapp.com/callback",
                scopes=["channels:read", "chat:write"],
                state="custom-state-token",
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v2/integrations/wizard/providers/slack/install",
                json={
                    "redirect_uri": "https://myapp.com/callback",
                    "scopes": ["channels:read", "chat:write"],
                    "state": "custom-state-token",
                },
            )
            client.close()

    def test_disconnect(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"disconnected": True, "message": "Provider disconnected"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.oauth_wizard.disconnect("slack")
            mock_request.assert_called_once_with(
                "DELETE", "/api/v2/integrations/wizard/providers/slack"
            )
            assert result["disconnected"] is True
            client.close()

    def test_refresh_tokens(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "refreshed": True,
                "expires_at": "2025-07-01T12:00:00Z",
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.oauth_wizard.refresh_tokens("github")
            mock_request.assert_called_once_with(
                "POST", "/api/v2/integrations/wizard/providers/github/refresh"
            )
            assert result["refreshed"] is True
            client.close()

    def test_get_recommendations(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "recommendations": [
                    {"provider_id": "jira", "reason": "Complements GitHub", "priority": "high"},
                    {
                        "provider_id": "notion",
                        "reason": "Knowledge management",
                        "priority": "medium",
                    },
                ]
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.oauth_wizard.get_recommendations()
            mock_request.assert_called_once_with(
                "GET", "/api/v2/integrations/wizard/recommendations"
            )
            assert len(result["recommendations"]) == 2
            assert result["recommendations"][0]["priority"] == "high"
            client.close()


class TestAsyncOAuthWizard:
    """Tests for async OAuth wizard methods."""

    @pytest.mark.asyncio
    async def test_get_config(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"available_providers": 12, "completion_percent": 25}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.oauth_wizard.get_config()
            mock_request.assert_called_once_with("GET", "/api/v2/integrations/wizard")
            assert result["available_providers"] == 12
            await client.close()

    @pytest.mark.asyncio
    async def test_list_providers_with_category(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"providers": [{"id": "slack"}]}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.oauth_wizard.list_providers(category="communication")
            mock_request.assert_called_once_with(
                "GET",
                "/api/v2/integrations/wizard/providers",
                params={"category": "communication"},
            )
            assert len(result["providers"]) == 1
            await client.close()

    @pytest.mark.asyncio
    async def test_validate_config(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"provider_id": "slack", "valid": True}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.oauth_wizard.validate_config("slack")
            mock_request.assert_called_once_with(
                "POST",
                "/api/v2/integrations/wizard/validate",
                json={"provider_id": "slack"},
            )
            assert result["valid"] is True
            await client.close()

    @pytest.mark.asyncio
    async def test_disconnect(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"disconnected": True}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.oauth_wizard.disconnect("github")
            mock_request.assert_called_once_with(
                "DELETE", "/api/v2/integrations/wizard/providers/github"
            )
            assert result["disconnected"] is True
            await client.close()
