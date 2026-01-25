"""Tests for the Channel Health Handler."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web

from aragora.server.handlers.social.channel_health import ChannelHealthHandler


@pytest.fixture
def handler():
    """Create a fresh handler instance."""
    return ChannelHealthHandler()


@pytest.fixture
def mock_request():
    """Create a mock request."""

    def _create_request(path: str = "/api/v1/channels/health", method: str = "GET"):
        request = MagicMock(spec=web.Request)
        request.path = path
        request.method = method
        request.match_info = {}

        # Extract channel from path for single channel health
        # Path: /api/v1/channels/{channel}/health
        # Parts: ['', 'api', 'v1', 'channels', '{channel}', 'health']
        parts = path.split("/")
        if len(parts) >= 6 and parts[5] == "health":
            request.match_info["channel"] = parts[4]

        return request

    return _create_request


class TestChannelHealthHandler:
    """Tests for ChannelHealthHandler."""

    def test_can_handle_all_channels_health(self, handler):
        """Test can_handle for all channels health endpoint."""
        assert handler.can_handle("GET", "/api/v1/channels/health") is True
        assert handler.can_handle("GET", "/api/v1/channels/health/") is True
        assert handler.can_handle("POST", "/api/v1/channels/health") is False
        assert handler.can_handle("GET", "/api/v1/other/health") is False

    def test_can_handle_single_channel_health(self, handler):
        """Test can_handle for single channel health endpoint."""
        assert handler.can_handle("GET", "/api/v1/channels/slack/health") is True
        assert handler.can_handle("GET", "/api/v1/channels/teams/health") is True
        assert handler.can_handle("GET", "/api/v1/channels/discord/health") is True
        assert handler.can_handle("POST", "/api/v1/channels/slack/health") is False

    @pytest.mark.asyncio
    async def test_get_all_channels_health_no_connectors(self, handler, mock_request):
        """Test health endpoint when no connectors are configured."""
        handler._initialized = True  # Skip connector initialization
        handler._connectors = {}

        request = mock_request()
        response = await handler.get_all_channels_health(request)

        assert response.status == 200
        data = response.body
        # The response is JSON bytes, need to decode
        import json

        data = json.loads(response.body)
        assert data["status"] == "no_channels_configured"
        assert data["health_score"] == 0
        assert data["summary"]["total"] == 0

    @pytest.mark.asyncio
    async def test_get_all_channels_health_with_healthy_connector(self, handler, mock_request):
        """Test health endpoint with a healthy connector."""
        handler._initialized = True

        # Create a mock connector with get_health method
        mock_connector = MagicMock()
        mock_connector.get_health = AsyncMock(
            return_value={
                "platform": "slack",
                "display_name": "Slack",
                "status": "healthy",
                "configured": True,
                "timestamp": 1234567890.0,
                "circuit_breaker": {"state": "closed", "enabled": True},
            }
        )
        handler._connectors = {"slack": mock_connector}

        request = mock_request()
        response = await handler.get_all_channels_health(request)

        import json

        data = json.loads(response.body)
        assert data["status"] == "healthy"
        assert data["health_score"] == 100
        assert data["summary"]["healthy"] == 1
        assert "slack" in data["channels"]

    @pytest.mark.asyncio
    async def test_get_all_channels_health_with_unhealthy_connector(self, handler, mock_request):
        """Test health endpoint with an unhealthy connector."""
        handler._initialized = True

        mock_connector = MagicMock()
        mock_connector.get_health = AsyncMock(
            return_value={
                "platform": "teams",
                "status": "unhealthy",
                "configured": True,
                "circuit_breaker": {"state": "open", "enabled": True},
            }
        )
        handler._connectors = {"teams": mock_connector}

        request = mock_request()
        response = await handler.get_all_channels_health(request)

        import json

        data = json.loads(response.body)
        assert data["status"] == "unhealthy"
        assert data["health_score"] == 0
        assert data["summary"]["unhealthy"] == 1

    @pytest.mark.asyncio
    async def test_get_all_channels_health_with_mixed_status(self, handler, mock_request):
        """Test health endpoint with mixed connector statuses."""
        handler._initialized = True

        healthy_connector = MagicMock()
        healthy_connector.get_health = AsyncMock(
            return_value={"platform": "slack", "status": "healthy"}
        )

        degraded_connector = MagicMock()
        degraded_connector.get_health = AsyncMock(
            return_value={"platform": "teams", "status": "degraded"}
        )

        handler._connectors = {
            "slack": healthy_connector,
            "teams": degraded_connector,
        }

        request = mock_request()
        response = await handler.get_all_channels_health(request)

        import json

        data = json.loads(response.body)
        assert data["status"] == "degraded"
        assert data["health_score"] == 75  # (100 + 50) / 2
        assert data["summary"]["healthy"] == 1
        assert data["summary"]["degraded"] == 1

    @pytest.mark.asyncio
    async def test_get_single_channel_health(self, handler, mock_request):
        """Test getting health for a specific channel."""
        handler._initialized = True

        mock_connector = MagicMock()
        mock_connector.get_health = AsyncMock(
            return_value={
                "platform": "slack",
                "status": "healthy",
                "circuit_breaker": {"state": "closed"},
            }
        )
        handler._connectors = {"slack": mock_connector}

        request = mock_request(path="/api/v1/channels/slack/health")
        response = await handler.get_channel_health(request)

        import json

        data = json.loads(response.body)
        assert data["platform"] == "slack"
        assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_get_unknown_channel_health(self, handler, mock_request):
        """Test getting health for an unknown channel."""
        handler._initialized = True
        handler._connectors = {"slack": MagicMock()}

        request = mock_request(path="/api/v1/channels/unknown/health")
        response = await handler.get_channel_health(request)

        assert response.status == 404
        import json

        data = json.loads(response.body)
        assert "error" in data
        assert "available_channels" in data

    @pytest.mark.asyncio
    async def test_get_basic_health_fallback(self, handler, mock_request):
        """Test fallback health check for connectors without get_health."""
        handler._initialized = True

        # Connector without get_health method
        mock_connector = MagicMock(spec=[])
        mock_connector.is_configured = True
        mock_connector.test_connection = AsyncMock(return_value={"success": True})

        handler._connectors = {"legacy": mock_connector}

        request = mock_request()
        response = await handler.get_all_channels_health(request)

        import json

        data = json.loads(response.body)
        assert "legacy" in data["channels"]
        assert data["channels"]["legacy"]["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_handle_routes_correctly(self, handler, mock_request):
        """Test that handle() routes to correct methods."""
        handler._initialized = True
        handler._connectors = {}

        # Test all channels health
        request = mock_request(path="/api/v1/channels/health")
        response = await handler.handle(request)
        assert response.status == 200

    @pytest.mark.asyncio
    async def test_connector_initialization_is_lazy(self, handler):
        """Test that connectors are lazily initialized."""
        assert handler._initialized is False
        assert handler._connectors == {}

        # Initialization happens when first request comes in
        handler._ensure_connectors_initialized()
        assert handler._initialized is True

    @pytest.mark.asyncio
    async def test_health_includes_timestamp(self, handler, mock_request):
        """Test that health responses include timestamps."""
        handler._initialized = True
        handler._connectors = {}

        request = mock_request()
        response = await handler.get_all_channels_health(request)

        import json

        data = json.loads(response.body)
        assert "timestamp" in data
        assert isinstance(data["timestamp"], float)
