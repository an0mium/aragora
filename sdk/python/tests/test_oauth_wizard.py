"""Tests for OAuth Wizard SDK namespace."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_client() -> MagicMock:
    """Create a mock client."""
    return MagicMock()


class TestOAuthWizardAPI:
    """Test synchronous OAuthWizardAPI."""

    def test_init(self, mock_client: MagicMock) -> None:
        """Test API initialization."""
        from aragora.namespaces.oauth_wizard import OAuthWizardAPI

        api = OAuthWizardAPI(mock_client)
        assert api._client is mock_client

    def test_get_config(self, mock_client: MagicMock) -> None:
        """Test get_config calls correct endpoint."""
        from aragora.namespaces.oauth_wizard import OAuthWizardAPI

        mock_client.request.return_value = {
            "available_providers": ["slack", "github", "google"],
            "configured_providers": ["slack"],
            "recommended_order": ["github", "google"],
            "completion_percent": 33,
        }

        api = OAuthWizardAPI(mock_client)
        result = api.get_config()

        mock_client.request.assert_called_once_with("GET", "/api/v2/integrations/wizard")
        assert result["completion_percent"] == 33
        assert "slack" in result["configured_providers"]

    def test_list_providers(self, mock_client: MagicMock) -> None:
        """Test list_providers calls correct endpoint."""
        from aragora.namespaces.oauth_wizard import OAuthWizardAPI

        mock_client.request.return_value = {
            "providers": [
                {"id": "slack", "name": "Slack", "category": "communication"},
                {"id": "teams", "name": "Microsoft Teams", "category": "communication"},
            ]
        }

        api = OAuthWizardAPI(mock_client)
        result = api.list_providers()

        mock_client.request.assert_called_once_with(
            "GET", "/api/v2/integrations/wizard/providers", params={}
        )
        assert len(result["providers"]) == 2

    def test_list_providers_with_filters(self, mock_client: MagicMock) -> None:
        """Test list_providers with category and configured filters."""
        from aragora.namespaces.oauth_wizard import OAuthWizardAPI

        mock_client.request.return_value = {"providers": []}

        api = OAuthWizardAPI(mock_client)
        api.list_providers(category="development", configured=True)

        mock_client.request.assert_called_once_with(
            "GET",
            "/api/v2/integrations/wizard/providers",
            params={"category": "development", "configured": True},
        )

    def test_get_provider(self, mock_client: MagicMock) -> None:
        """Test get_provider calls correct endpoint."""
        from aragora.namespaces.oauth_wizard import OAuthWizardAPI

        mock_client.request.return_value = {
            "id": "github",
            "name": "GitHub",
            "description": "GitHub integration",
            "category": "development",
            "features": ["repos", "issues", "prs"],
            "required_env_vars": ["GITHUB_CLIENT_ID", "GITHUB_CLIENT_SECRET"],
            "oauth_scopes": ["repo", "read:user"],
        }

        api = OAuthWizardAPI(mock_client)
        result = api.get_provider("github")

        mock_client.request.assert_called_once_with(
            "GET", "/api/v2/integrations/wizard/providers/github"
        )
        assert result["id"] == "github"
        assert "repos" in result["features"]

    def test_get_status(self, mock_client: MagicMock) -> None:
        """Test get_status calls correct endpoint."""
        from aragora.namespaces.oauth_wizard import OAuthWizardAPI

        mock_client.request.return_value = {
            "total_providers": 10,
            "configured": 3,
            "connected": 2,
            "errors": 1,
            "health_score": 0.8,
            "last_checked_at": "2024-01-01T00:00:00Z",
        }

        api = OAuthWizardAPI(mock_client)
        result = api.get_status()

        mock_client.request.assert_called_once_with("GET", "/api/v2/integrations/wizard/status")
        assert result["total_providers"] == 10
        assert result["health_score"] == 0.8

    def test_validate_config(self, mock_client: MagicMock) -> None:
        """Test validate_config calls correct endpoint."""
        from aragora.namespaces.oauth_wizard import OAuthWizardAPI

        mock_client.request.return_value = {
            "provider_id": "slack",
            "valid": True,
            "missing_required": [],
            "missing_optional": ["SLACK_SIGNING_SECRET"],
            "warnings": [],
            "errors": [],
            "can_proceed": True,
        }

        api = OAuthWizardAPI(mock_client)
        result = api.validate_config("slack")

        mock_client.request.assert_called_once_with(
            "POST",
            "/api/v2/integrations/wizard/validate",
            json={"provider_id": "slack"},
        )
        assert result["valid"] is True
        assert result["can_proceed"] is True

    def test_run_preflight_checks(self, mock_client: MagicMock) -> None:
        """Test run_preflight_checks calls correct endpoint."""
        from aragora.namespaces.oauth_wizard import OAuthWizardAPI

        mock_client.request.return_value = {
            "provider_id": "github",
            "checks": [
                {"name": "credentials", "status": "passed", "required": True},
                {"name": "scopes", "status": "passed", "required": True},
            ],
            "can_connect": True,
            "estimated_connect_time_seconds": 5,
        }

        api = OAuthWizardAPI(mock_client)
        result = api.run_preflight_checks("github")

        mock_client.request.assert_called_once_with(
            "POST",
            "/api/v2/integrations/wizard/preflight",
            json={"provider_id": "github"},
        )
        assert result["can_connect"] is True
        assert len(result["checks"]) == 2

    def test_get_install_url(self, mock_client: MagicMock) -> None:
        """Test get_install_url calls correct endpoint."""
        from aragora.namespaces.oauth_wizard import OAuthWizardAPI

        mock_client.request.return_value = {
            "install_url": "https://github.com/login/oauth/authorize?client_id=xxx",
            "state": "random_state_123",
            "expires_at": "2024-01-01T00:10:00Z",
        }

        api = OAuthWizardAPI(mock_client)
        result = api.get_install_url("github")

        mock_client.request.assert_called_once_with(
            "POST",
            "/api/v2/integrations/wizard/providers/github/install",
            json={},
        )
        assert "install_url" in result
        assert result["state"] == "random_state_123"

    def test_get_install_url_with_options(self, mock_client: MagicMock) -> None:
        """Test get_install_url with all options."""
        from aragora.namespaces.oauth_wizard import OAuthWizardAPI

        mock_client.request.return_value = {
            "install_url": "https://slack.com/oauth/authorize",
            "state": "custom_state",
        }

        api = OAuthWizardAPI(mock_client)
        api.get_install_url(
            provider_id="slack",
            redirect_uri="https://myapp.com/callback",
            scopes=["channels:read", "chat:write"],
            state="custom_state",
        )

        call_args = mock_client.request.call_args
        json_body = call_args[1]["json"]
        assert json_body["redirect_uri"] == "https://myapp.com/callback"
        assert json_body["scopes"] == ["channels:read", "chat:write"]
        assert json_body["state"] == "custom_state"

    def test_disconnect(self, mock_client: MagicMock) -> None:
        """Test disconnect calls correct endpoint."""
        from aragora.namespaces.oauth_wizard import OAuthWizardAPI

        mock_client.request.return_value = {
            "disconnected": True,
            "message": "Successfully disconnected Slack",
        }

        api = OAuthWizardAPI(mock_client)
        result = api.disconnect("slack")

        mock_client.request.assert_called_once_with(
            "DELETE", "/api/v2/integrations/wizard/providers/slack"
        )
        assert result["disconnected"] is True

    def test_refresh_tokens(self, mock_client: MagicMock) -> None:
        """Test refresh_tokens calls correct endpoint."""
        from aragora.namespaces.oauth_wizard import OAuthWizardAPI

        mock_client.request.return_value = {
            "refreshed": True,
            "expires_at": "2024-01-02T00:00:00Z",
        }

        api = OAuthWizardAPI(mock_client)
        result = api.refresh_tokens("github")

        mock_client.request.assert_called_once_with(
            "POST", "/api/v2/integrations/wizard/providers/github/refresh"
        )
        assert result["refreshed"] is True

    def test_get_recommendations(self, mock_client: MagicMock) -> None:
        """Test get_recommendations calls correct endpoint."""
        from aragora.namespaces.oauth_wizard import OAuthWizardAPI

        mock_client.request.return_value = {
            "recommendations": [
                {"provider_id": "github", "reason": "Popular with developers", "priority": "high"},
                {"provider_id": "google", "reason": "Enables calendar sync", "priority": "medium"},
            ]
        }

        api = OAuthWizardAPI(mock_client)
        result = api.get_recommendations()

        mock_client.request.assert_called_once_with(
            "GET", "/api/v2/integrations/wizard/recommendations"
        )
        assert len(result["recommendations"]) == 2
        assert result["recommendations"][0]["priority"] == "high"


@pytest.fixture
def mock_async_client() -> MagicMock:
    """Create a mock async client."""
    from unittest.mock import AsyncMock

    client = MagicMock()
    client.request = AsyncMock()
    return client


class TestAsyncOAuthWizardAPI:
    """Test asynchronous AsyncOAuthWizardAPI."""

    @pytest.mark.asyncio
    async def test_init(self, mock_async_client: MagicMock) -> None:
        """Test API initialization."""
        from aragora.namespaces.oauth_wizard import AsyncOAuthWizardAPI

        api = AsyncOAuthWizardAPI(mock_async_client)
        assert api._client is mock_async_client

    @pytest.mark.asyncio
    async def test_get_config(self, mock_async_client: MagicMock) -> None:
        """Test get_config calls correct endpoint."""
        from aragora.namespaces.oauth_wizard import AsyncOAuthWizardAPI

        mock_async_client.request.return_value = {
            "available_providers": ["slack"],
            "completion_percent": 50,
        }

        api = AsyncOAuthWizardAPI(mock_async_client)
        result = await api.get_config()

        mock_async_client.request.assert_called_once_with("GET", "/api/v2/integrations/wizard")
        assert result["completion_percent"] == 50

    @pytest.mark.asyncio
    async def test_list_providers(self, mock_async_client: MagicMock) -> None:
        """Test list_providers calls correct endpoint."""
        from aragora.namespaces.oauth_wizard import AsyncOAuthWizardAPI

        mock_async_client.request.return_value = {"providers": []}

        api = AsyncOAuthWizardAPI(mock_async_client)
        await api.list_providers(category="storage", configured=False)

        mock_async_client.request.assert_called_once_with(
            "GET",
            "/api/v2/integrations/wizard/providers",
            params={"category": "storage", "configured": False},
        )

    @pytest.mark.asyncio
    async def test_get_provider(self, mock_async_client: MagicMock) -> None:
        """Test get_provider calls correct endpoint."""
        from aragora.namespaces.oauth_wizard import AsyncOAuthWizardAPI

        mock_async_client.request.return_value = {"id": "dropbox", "name": "Dropbox"}

        api = AsyncOAuthWizardAPI(mock_async_client)
        result = await api.get_provider("dropbox")

        mock_async_client.request.assert_called_once_with(
            "GET", "/api/v2/integrations/wizard/providers/dropbox"
        )
        assert result["id"] == "dropbox"

    @pytest.mark.asyncio
    async def test_get_status(self, mock_async_client: MagicMock) -> None:
        """Test get_status calls correct endpoint."""
        from aragora.namespaces.oauth_wizard import AsyncOAuthWizardAPI

        mock_async_client.request.return_value = {
            "total_providers": 5,
            "connected": 3,
        }

        api = AsyncOAuthWizardAPI(mock_async_client)
        result = await api.get_status()

        mock_async_client.request.assert_called_once_with(
            "GET", "/api/v2/integrations/wizard/status"
        )
        assert result["connected"] == 3

    @pytest.mark.asyncio
    async def test_validate_config(self, mock_async_client: MagicMock) -> None:
        """Test validate_config calls correct endpoint."""
        from aragora.namespaces.oauth_wizard import AsyncOAuthWizardAPI

        mock_async_client.request.return_value = {
            "provider_id": "jira",
            "valid": False,
            "can_proceed": False,
        }

        api = AsyncOAuthWizardAPI(mock_async_client)
        result = await api.validate_config("jira")

        mock_async_client.request.assert_called_once_with(
            "POST",
            "/api/v2/integrations/wizard/validate",
            json={"provider_id": "jira"},
        )
        assert result["valid"] is False

    @pytest.mark.asyncio
    async def test_run_preflight_checks(self, mock_async_client: MagicMock) -> None:
        """Test run_preflight_checks calls correct endpoint."""
        from aragora.namespaces.oauth_wizard import AsyncOAuthWizardAPI

        mock_async_client.request.return_value = {
            "provider_id": "notion",
            "checks": [],
            "can_connect": True,
        }

        api = AsyncOAuthWizardAPI(mock_async_client)
        result = await api.run_preflight_checks("notion")

        mock_async_client.request.assert_called_once_with(
            "POST",
            "/api/v2/integrations/wizard/preflight",
            json={"provider_id": "notion"},
        )
        assert result["can_connect"] is True

    @pytest.mark.asyncio
    async def test_get_install_url(self, mock_async_client: MagicMock) -> None:
        """Test get_install_url calls correct endpoint."""
        from aragora.namespaces.oauth_wizard import AsyncOAuthWizardAPI

        mock_async_client.request.return_value = {
            "install_url": "https://auth.example.com",
            "state": "abc123",
        }

        api = AsyncOAuthWizardAPI(mock_async_client)
        result = await api.get_install_url(
            provider_id="custom",
            redirect_uri="https://callback.com",
            scopes=["read", "write"],
            state="abc123",
        )

        call_args = mock_async_client.request.call_args
        assert call_args[0] == ("POST", "/api/v2/integrations/wizard/providers/custom/install")
        json_body = call_args[1]["json"]
        assert json_body["redirect_uri"] == "https://callback.com"
        assert json_body["scopes"] == ["read", "write"]
        assert result["state"] == "abc123"

    @pytest.mark.asyncio
    async def test_disconnect(self, mock_async_client: MagicMock) -> None:
        """Test disconnect calls correct endpoint."""
        from aragora.namespaces.oauth_wizard import AsyncOAuthWizardAPI

        mock_async_client.request.return_value = {"disconnected": True}

        api = AsyncOAuthWizardAPI(mock_async_client)
        result = await api.disconnect("linear")

        mock_async_client.request.assert_called_once_with(
            "DELETE", "/api/v2/integrations/wizard/providers/linear"
        )
        assert result["disconnected"] is True

    @pytest.mark.asyncio
    async def test_refresh_tokens(self, mock_async_client: MagicMock) -> None:
        """Test refresh_tokens calls correct endpoint."""
        from aragora.namespaces.oauth_wizard import AsyncOAuthWizardAPI

        mock_async_client.request.return_value = {
            "refreshed": True,
            "expires_at": "2024-12-31T23:59:59Z",
        }

        api = AsyncOAuthWizardAPI(mock_async_client)
        result = await api.refresh_tokens("asana")

        mock_async_client.request.assert_called_once_with(
            "POST", "/api/v2/integrations/wizard/providers/asana/refresh"
        )
        assert result["refreshed"] is True

    @pytest.mark.asyncio
    async def test_get_recommendations(self, mock_async_client: MagicMock) -> None:
        """Test get_recommendations calls correct endpoint."""
        from aragora.namespaces.oauth_wizard import AsyncOAuthWizardAPI

        mock_async_client.request.return_value = {
            "recommendations": [{"provider_id": "trello", "priority": "low"}]
        }

        api = AsyncOAuthWizardAPI(mock_async_client)
        result = await api.get_recommendations()

        mock_async_client.request.assert_called_once_with(
            "GET", "/api/v2/integrations/wizard/recommendations"
        )
        assert len(result["recommendations"]) == 1
