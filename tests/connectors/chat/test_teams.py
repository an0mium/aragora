"""
Tests for Microsoft Teams Chat Connector.

Tests the core TeamsConnector functionality including:
- Initialization and configuration
- Platform properties
- Error classification
- Token management
- Message sending
- File operations
- Channel history
- Circuit breaker integration
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Error Classification Tests
# =============================================================================


class TestErrorClassification:
    """Test _classify_teams_error function."""

    def test_rate_limit_error_by_status_code(self):
        """Test rate limit detection by 429 status code."""
        from aragora.connectors.chat.teams import _classify_teams_error
        from aragora.connectors.exceptions import ConnectorRateLimitError

        error = _classify_teams_error("Some error", status_code=429)

        assert isinstance(error, ConnectorRateLimitError)
        assert error.connector_name == "teams"
        assert error.retry_after == 60.0

    def test_rate_limit_error_by_keyword(self):
        """Test rate limit detection by keyword."""
        from aragora.connectors.chat.teams import _classify_teams_error
        from aragora.connectors.exceptions import ConnectorRateLimitError

        error = _classify_teams_error("Request was throttled")

        assert isinstance(error, ConnectorRateLimitError)

    def test_rate_limit_with_retry_after(self):
        """Test rate limit with custom retry_after."""
        from aragora.connectors.chat.teams import _classify_teams_error
        from aragora.connectors.exceptions import ConnectorRateLimitError

        error = _classify_teams_error("Rate limited", retry_after=30.0)

        assert isinstance(error, ConnectorRateLimitError)
        assert error.retry_after == 30.0

    def test_auth_error_by_status_401(self):
        """Test auth error detection by 401 status."""
        from aragora.connectors.chat.teams import _classify_teams_error
        from aragora.connectors.exceptions import ConnectorAuthError

        error = _classify_teams_error("Unauthorized", status_code=401)

        assert isinstance(error, ConnectorAuthError)
        assert error.connector_name == "teams"

    def test_auth_error_by_status_403(self):
        """Test auth error detection by 403 status."""
        from aragora.connectors.chat.teams import _classify_teams_error
        from aragora.connectors.exceptions import ConnectorAuthError

        error = _classify_teams_error("Forbidden", status_code=403)

        assert isinstance(error, ConnectorAuthError)

    def test_auth_error_by_keyword(self):
        """Test auth error detection by keyword."""
        from aragora.connectors.chat.teams import _classify_teams_error
        from aragora.connectors.exceptions import ConnectorAuthError

        error = _classify_teams_error("Token expired, please re-authenticate")

        assert isinstance(error, ConnectorAuthError)

    def test_timeout_error(self):
        """Test timeout error detection."""
        from aragora.connectors.chat.teams import _classify_teams_error
        from aragora.connectors.exceptions import ConnectorTimeoutError

        error = _classify_teams_error("Request timed out")

        assert isinstance(error, ConnectorTimeoutError)

    def test_network_error(self):
        """Test network error detection."""
        from aragora.connectors.chat.teams import _classify_teams_error
        from aragora.connectors.exceptions import ConnectorNetworkError

        error = _classify_teams_error("Connection refused to host")

        assert isinstance(error, ConnectorNetworkError)

    def test_api_error_with_status_code(self):
        """Test API error with status code."""
        from aragora.connectors.chat.teams import _classify_teams_error
        from aragora.connectors.exceptions import ConnectorAPIError

        error = _classify_teams_error("Not found", status_code=404)

        assert isinstance(error, ConnectorAPIError)
        assert error.status_code == 404

    def test_default_api_error(self):
        """Test default fallback to API error."""
        from aragora.connectors.chat.teams import _classify_teams_error
        from aragora.connectors.exceptions import ConnectorAPIError

        error = _classify_teams_error("Unknown error")

        assert isinstance(error, ConnectorAPIError)


# =============================================================================
# TeamsConnector Initialization Tests
# =============================================================================


class TestTeamsConnectorInit:
    """Test TeamsConnector initialization."""

    def test_init_minimal(self):
        """Test minimal initialization."""
        from aragora.connectors.chat.teams import TeamsConnector

        connector = TeamsConnector()

        assert connector.platform_name == "teams"
        assert connector.platform_display_name == "Microsoft Teams"

    def test_init_with_credentials(self):
        """Test initialization with credentials."""
        from aragora.connectors.chat.teams import TeamsConnector

        connector = TeamsConnector(
            app_id="test-app-id",
            app_password="test-password",
            tenant_id="test-tenant",
        )

        assert connector.app_id == "test-app-id"
        assert connector.app_password == "test-password"
        assert connector.tenant_id == "test-tenant"

    def test_init_with_timeouts(self):
        """Test initialization with custom timeouts."""
        from aragora.connectors.chat.teams import TeamsConnector

        connector = TeamsConnector(
            request_timeout=45.0,
            upload_timeout=180.0,
        )

        assert connector._request_timeout == 45.0
        assert connector._upload_timeout == 180.0

    def test_init_token_cache_empty(self):
        """Test token caches start empty."""
        from aragora.connectors.chat.teams import TeamsConnector

        connector = TeamsConnector()

        assert connector._access_token is None
        assert connector._token_expires == 0
        assert connector._graph_token is None
        assert connector._graph_token_expires == 0

    def test_init_circuit_breaker_enabled_by_default(self):
        """Test circuit breaker enabled by default."""
        from aragora.connectors.chat.teams import TeamsConnector

        connector = TeamsConnector()

        assert connector._enable_circuit_breaker is True


# =============================================================================
# Platform Properties Tests
# =============================================================================


class TestPlatformProperties:
    """Test TeamsConnector platform properties."""

    def test_platform_name(self):
        """Test platform name."""
        from aragora.connectors.chat.teams import TeamsConnector

        connector = TeamsConnector()

        assert connector.platform_name == "teams"

    def test_platform_display_name(self):
        """Test platform display name."""
        from aragora.connectors.chat.teams import TeamsConnector

        connector = TeamsConnector()

        assert connector.platform_display_name == "Microsoft Teams"


# =============================================================================
# Token Management Tests
# =============================================================================


class TestTokenManagement:
    """Test token management functionality."""

    @pytest.fixture
    def connector(self):
        """Create connector for testing."""
        from aragora.connectors.chat.teams import TeamsConnector

        return TeamsConnector(
            app_id="test-app",
            app_password="test-secret",
        )

    @pytest.mark.asyncio
    async def test_get_access_token_cached(self, connector):
        """Test access token returns cached value."""
        connector._access_token = "cached-token"
        connector._token_expires = time.time() + 3600  # Valid for 1 hour

        token = await connector._get_access_token()

        assert token == "cached-token"

    @pytest.mark.asyncio
    async def test_get_access_token_expired_refreshes(self, connector):
        """Test expired token triggers refresh."""
        connector._access_token = "old-token"
        connector._token_expires = time.time() - 100  # Expired

        with patch.object(
            connector,
            "_http_request",
            new_callable=AsyncMock,
            return_value=(
                True,
                {"access_token": "new-token", "expires_in": 3600},
                None,
            ),
        ):
            token = await connector._get_access_token()

            assert token == "new-token"
            assert connector._access_token == "new-token"

    @pytest.mark.asyncio
    async def test_get_access_token_fails_gracefully(self, connector):
        """Test token failure raises error."""
        connector._access_token = None
        connector._token_expires = 0

        with patch.object(
            connector,
            "_http_request",
            new_callable=AsyncMock,
            return_value=(False, None, "Auth failed"),
        ):
            with pytest.raises(RuntimeError, match="Failed to get Bot Framework token"):
                await connector._get_access_token()

    @pytest.mark.asyncio
    async def test_get_graph_token_cached(self, connector):
        """Test Graph token returns cached value."""
        connector.tenant_id = "test-tenant"
        connector._graph_token = "cached-graph-token"
        connector._graph_token_expires = time.time() + 3600

        token = await connector._get_graph_token()

        assert token == "cached-graph-token"

    @pytest.mark.asyncio
    async def test_get_graph_token_requires_tenant(self, connector):
        """Test Graph token requires tenant ID."""
        connector.tenant_id = ""
        connector._graph_token = None

        with pytest.raises(RuntimeError, match="Tenant ID required"):
            await connector._get_graph_token()


# =============================================================================
# Send Message Tests
# =============================================================================


class TestSendMessage:
    """Test send_message functionality."""

    @pytest.fixture
    def connector(self):
        """Create connector for testing."""
        from aragora.connectors.chat.teams import TeamsConnector

        return TeamsConnector(
            app_id="test-app",
            app_password="test-secret",
        )

    @pytest.mark.asyncio
    async def test_send_message_httpx_not_available(self, connector):
        """Test send_message fails gracefully without httpx."""
        with patch(
            "aragora.connectors.chat.teams.HTTPX_AVAILABLE",
            False,
        ):
            response = await connector.send_message(
                channel_id="channel-1",
                text="Hello",
            )

            assert response.success is False
            assert "httpx not available" in response.error

    @pytest.mark.asyncio
    async def test_send_message_circuit_breaker_open(self, connector):
        """Test send_message blocked by open circuit breaker."""
        with (
            patch(
                "aragora.connectors.chat.teams.HTTPX_AVAILABLE",
                True,
            ),
            patch.object(
                connector,
                "_check_circuit_breaker",
                return_value=(False, "Circuit breaker open"),
            ),
        ):
            response = await connector.send_message(
                channel_id="channel-1",
                text="Hello",
            )

            assert response.success is False
            assert "Circuit breaker open" in response.error


# =============================================================================
# Graph API Tests
# =============================================================================


class TestGraphAPI:
    """Test Microsoft Graph API integration."""

    @pytest.fixture
    def connector(self):
        """Create connector for testing."""
        from aragora.connectors.chat.teams import TeamsConnector

        return TeamsConnector(
            app_id="test-app",
            app_password="test-secret",
            tenant_id="test-tenant",
        )

    @pytest.mark.asyncio
    async def test_graph_api_request_auth_failure(self, connector):
        """Test Graph API request with auth failure."""
        with patch.object(
            connector,
            "_get_graph_token",
            side_effect=Exception("Token error"),
        ):
            success, data, error = await connector._graph_api_request(
                endpoint="/me",
                operation="test_op",
            )

            assert success is False
            assert "Failed to get Graph token" in error

    @pytest.mark.asyncio
    async def test_graph_api_request_success(self, connector):
        """Test successful Graph API request."""
        with (
            patch.object(
                connector,
                "_get_graph_token",
                new_callable=AsyncMock,
                return_value="valid-token",
            ),
            patch.object(
                connector,
                "_http_request",
                new_callable=AsyncMock,
                return_value=(True, {"value": "result"}, None),
            ) as mock_request,
        ):
            success, data, error = await connector._graph_api_request(
                endpoint="/users",
                method="GET",
                operation="get_users",
            )

            assert success is True
            assert data == {"value": "result"}
            mock_request.assert_called_once()


# =============================================================================
# Module Export Tests
# =============================================================================


class TestModuleExports:
    """Test module exports are correct."""

    def test_teams_connector_exportable(self):
        """Test TeamsConnector can be imported."""
        from aragora.connectors.chat.teams import TeamsConnector

        assert TeamsConnector is not None

    def test_teams_thread_manager_exportable(self):
        """Test TeamsThreadManager in __all__."""
        from aragora.connectors.chat.teams import __all__

        assert "TeamsConnector" in __all__
        assert "TeamsThreadManager" in __all__


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Test module constants."""

    def test_bot_framework_auth_url(self):
        """Test Bot Framework auth URL."""
        from aragora.connectors.chat.teams import BOT_FRAMEWORK_AUTH_URL

        assert "microsoftonline.com" in BOT_FRAMEWORK_AUTH_URL
        assert "botframework.com" in BOT_FRAMEWORK_AUTH_URL

    def test_graph_api_base(self):
        """Test Graph API base URL."""
        from aragora.connectors.chat.teams import GRAPH_API_BASE

        assert GRAPH_API_BASE == "https://graph.microsoft.com/v1.0"

    def test_default_timeouts(self):
        """Test default timeout values."""
        from aragora.connectors.chat.teams import (
            TEAMS_REQUEST_TIMEOUT,
            TEAMS_UPLOAD_TIMEOUT,
        )

        assert TEAMS_REQUEST_TIMEOUT > 0
        assert TEAMS_UPLOAD_TIMEOUT > TEAMS_REQUEST_TIMEOUT


# =============================================================================
# Function Signature Tests
# =============================================================================


class TestFunctionSignatures:
    """Test function signatures are correct."""

    def test_classify_teams_error_signature(self):
        """Test _classify_teams_error signature."""
        import inspect

        from aragora.connectors.chat.teams import _classify_teams_error

        sig = inspect.signature(_classify_teams_error)
        params = list(sig.parameters.keys())

        assert "error_str" in params
        assert "status_code" in params
        assert "retry_after" in params

    def test_connector_init_signature(self):
        """Test TeamsConnector.__init__ signature."""
        import inspect

        from aragora.connectors.chat.teams import TeamsConnector

        sig = inspect.signature(TeamsConnector.__init__)
        params = list(sig.parameters.keys())

        assert "app_id" in params
        assert "app_password" in params
        assert "tenant_id" in params
        assert "request_timeout" in params
        assert "upload_timeout" in params

    def test_send_message_signature(self):
        """Test send_message signature."""
        import inspect

        from aragora.connectors.chat.teams import TeamsConnector

        sig = inspect.signature(TeamsConnector.send_message)
        params = list(sig.parameters.keys())

        assert "channel_id" in params
        assert "text" in params
        assert "blocks" in params
        assert "thread_id" in params
        assert "service_url" in params


# =============================================================================
# Integration-Style Tests
# =============================================================================


class TestIntegrationStyle:
    """Integration-style tests without external dependencies."""

    def test_connector_inherits_from_base(self):
        """Test TeamsConnector inherits from ChatPlatformConnector."""
        from aragora.connectors.chat.base import ChatPlatformConnector
        from aragora.connectors.chat.teams import TeamsConnector

        assert issubclass(TeamsConnector, ChatPlatformConnector)

    def test_connector_has_required_methods(self):
        """Test connector has required abstract methods."""
        from aragora.connectors.chat.teams import TeamsConnector

        connector = TeamsConnector()

        # Check required methods exist
        assert hasattr(connector, "send_message")
        assert hasattr(connector, "upload_file")
        assert hasattr(connector, "download_file")
        assert hasattr(connector, "get_channel_history")

        # Check they're callable
        assert callable(connector.send_message)
        assert callable(connector.upload_file)
        assert callable(connector.download_file)
        assert callable(connector.get_channel_history)

    def test_error_types_are_exceptions(self):
        """Test error types are proper exceptions."""
        from aragora.connectors.chat.teams import _classify_teams_error
        from aragora.connectors.exceptions import ConnectorError

        error = _classify_teams_error("test error")

        assert isinstance(error, ConnectorError)
        assert isinstance(error, Exception)
