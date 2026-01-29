"""Tests for Cloud Storage Handler."""

import sys
import types as _types_mod

# Pre-stub Slack modules to prevent import chain failures
_SLACK_ATTRS = [
    "SlackHandler",
    "get_slack_handler",
    "get_slack_integration",
    "get_workspace_store",
    "resolve_workspace",
    "create_tracked_task",
    "_validate_slack_url",
    "SLACK_SIGNING_SECRET",
    "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL",
    "SLACK_ALLOWED_DOMAINS",
    "SignatureVerifierMixin",
    "CommandsMixin",
    "EventsMixin",
    "init_slack_handler",
]
for _mod_name in (
    "aragora.server.handlers.social.slack.handler",
    "aragora.server.handlers.social.slack",
    "aragora.server.handlers.social._slack_impl",
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.server.handlers.features.cloud_storage import (
    CloudStorageHandler,
    PROVIDERS,
    CLOUD_READ_PERMISSION,
    CLOUD_WRITE_PERMISSION,
    get_provider_connector,
    get_provider_status,
    get_all_provider_status,
    get_auth_url,
    handle_auth_callback,
    list_files,
    download_file,
    _tokens,
)


@pytest.fixture(autouse=True)
def clear_tokens():
    """Clear tokens between tests."""
    _tokens.clear()
    yield


@pytest.fixture
def handler():
    """Create handler instance."""
    return CloudStorageHandler({})


class TestProviderConstants:
    """Tests for provider constants."""

    def test_providers_list(self):
        """Test providers list is defined."""
        assert "google_drive" in PROVIDERS
        assert "onedrive" in PROVIDERS
        assert "dropbox" in PROVIDERS
        assert "s3" in PROVIDERS

    def test_permissions_defined(self):
        """Test permission constants are defined."""
        assert CLOUD_READ_PERMISSION == "cloud:read"
        assert CLOUD_WRITE_PERMISSION == "cloud:write"


class TestCloudStorageHandler:
    """Tests for CloudStorageHandler class."""

    def test_handler_creation(self, handler):
        """Test creating handler instance."""
        assert handler is not None

    def test_handler_routes(self):
        """Test that handler has route definitions."""
        assert hasattr(CloudStorageHandler, "ROUTES")
        routes = CloudStorageHandler.ROUTES
        assert "/api/v1/cloud/status" in routes
        assert "/api/v1/cloud/google_drive/auth/url" in routes
        assert "/api/v1/cloud/google_drive/files" in routes

    def test_can_handle_cloud_routes(self, handler):
        """Test can_handle for cloud routes."""
        assert handler.can_handle("/api/v1/cloud/status") is True
        assert handler.can_handle("/api/v1/cloud/google_drive/auth/url") is True
        assert handler.can_handle("/api/v1/cloud/onedrive/files") is True

    def test_can_handle_invalid_routes(self, handler):
        """Test can_handle rejects invalid routes."""
        assert handler.can_handle("/api/v1/storage/") is False
        assert handler.can_handle("/api/v1/invalid/route") is False


class TestCloudStorageAuthentication:
    """Tests for cloud storage authentication."""

    @pytest.mark.asyncio
    async def test_handle_requires_authentication(self):
        """Test handle method requires authentication."""
        handler = CloudStorageHandler({})
        mock_handler = MagicMock()

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            from aragora.server.handlers.secure import UnauthorizedError

            mock_auth.side_effect = UnauthorizedError("Not authenticated")

            result = await handler.handle("/api/v1/cloud/status", {}, mock_handler)
            assert result is not None
            assert result.status == 401

    @pytest.mark.asyncio
    async def test_handle_checks_permission(self):
        """Test handle checks cloud:read permission."""
        handler = CloudStorageHandler({})
        mock_handler = MagicMock()

        with (
            patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth,
            patch.object(handler, "check_permission") as mock_check,
        ):
            from aragora.server.handlers.secure import ForbiddenError

            mock_auth.return_value = MagicMock()
            mock_check.side_effect = ForbiddenError("Permission denied")

            result = await handler.handle("/api/v1/cloud/status", {}, mock_handler)
            assert result is not None
            assert result.status == 403


class TestGetProviderConnector:
    """Tests for get_provider_connector function."""

    def test_get_google_drive_connector(self):
        """Test getting Google Drive connector."""
        with patch(
            "aragora.server.handlers.features.cloud_storage.GoogleDriveConnector",
            create=True,
        ) as MockConnector:
            mock_instance = MagicMock()
            MockConnector.return_value = mock_instance

            connector = get_provider_connector("google_drive")
            assert connector is not None

    def test_get_invalid_provider_connector(self):
        """Test getting connector for invalid provider returns None."""
        connector = get_provider_connector("invalid_provider")
        assert connector is None


class TestGetProviderStatus:
    """Tests for get_provider_status function."""

    def test_status_invalid_provider(self):
        """Test status for invalid provider."""
        status = get_provider_status("invalid_provider")
        assert status["connected"] is False
        assert status["configured"] is False

    def test_status_not_configured(self):
        """Test status when provider not configured."""
        with patch(
            "aragora.server.handlers.features.cloud_storage.get_provider_connector"
        ) as mock_get:
            mock_connector = MagicMock()
            mock_connector.is_configured.return_value = False
            mock_get.return_value = mock_connector

            status = get_provider_status("google_drive")
            assert status["connected"] is False
            assert status["configured"] is False

    def test_status_configured_with_token(self):
        """Test status when configured with token."""
        _tokens["google_drive"] = {"access_token": "test_token"}

        with patch(
            "aragora.server.handlers.features.cloud_storage.get_provider_connector"
        ) as mock_get:
            mock_connector = MagicMock()
            mock_connector.is_configured.return_value = True
            mock_get.return_value = mock_connector

            status = get_provider_status("google_drive")
            assert status["connected"] is True
            assert status["configured"] is True


class TestGetAllProviderStatus:
    """Tests for get_all_provider_status function."""

    def test_get_all_status(self):
        """Test getting status for all providers."""
        with patch(
            "aragora.server.handlers.features.cloud_storage.get_provider_status"
        ) as mock_status:
            mock_status.return_value = {"connected": False, "configured": False}

            all_status = get_all_provider_status()
            assert "google_drive" in all_status
            assert "onedrive" in all_status
            assert "dropbox" in all_status
            assert "s3" in all_status


class TestGetAuthUrl:
    """Tests for get_auth_url function."""

    @pytest.mark.asyncio
    async def test_auth_url_invalid_provider(self):
        """Test auth URL for invalid provider."""
        url = await get_auth_url("invalid_provider", "http://localhost/callback")
        assert url is None

    @pytest.mark.asyncio
    async def test_auth_url_success(self):
        """Test successful auth URL generation."""
        with patch(
            "aragora.server.handlers.features.cloud_storage.get_provider_connector"
        ) as mock_get:
            mock_connector = MagicMock()
            mock_connector.get_oauth_url.return_value = "https://oauth.provider.com"
            mock_get.return_value = mock_connector

            url = await get_auth_url("google_drive", "http://localhost/callback")
            assert url == "https://oauth.provider.com"


class TestHandleAuthCallback:
    """Tests for handle_auth_callback function."""

    @pytest.mark.asyncio
    async def test_callback_invalid_provider(self):
        """Test callback for invalid provider."""
        result = await handle_auth_callback("invalid", "code123", "http://localhost")
        assert result is False

    @pytest.mark.asyncio
    async def test_callback_success(self):
        """Test successful auth callback."""
        with patch(
            "aragora.server.handlers.features.cloud_storage.get_provider_connector"
        ) as mock_get:
            mock_connector = MagicMock()
            mock_connector.authenticate = AsyncMock(return_value=True)
            mock_connector._access_token = "access_token"
            mock_connector._refresh_token = "refresh_token"
            mock_get.return_value = mock_connector

            result = await handle_auth_callback("google_drive", "code123", "http://localhost")
            assert result is True
            assert "google_drive" in _tokens

    @pytest.mark.asyncio
    async def test_callback_failure(self):
        """Test failed auth callback."""
        with patch(
            "aragora.server.handlers.features.cloud_storage.get_provider_connector"
        ) as mock_get:
            mock_connector = MagicMock()
            mock_connector.authenticate = AsyncMock(return_value=False)
            mock_get.return_value = mock_connector

            result = await handle_auth_callback("google_drive", "code123", "http://localhost")
            assert result is False


class TestListFiles:
    """Tests for list_files function."""

    @pytest.mark.asyncio
    async def test_list_files_invalid_provider(self):
        """Test listing files for invalid provider."""
        files = await list_files("invalid_provider")
        assert files == []

    @pytest.mark.asyncio
    async def test_list_files_success(self):
        """Test successful file listing."""
        mock_file = MagicMock()
        mock_file.id = "file123"
        mock_file.name = "test.txt"
        mock_file.size = 1024
        mock_file.mime_type = "text/plain"
        mock_file.modified_time = "2024-01-01"

        async def mock_list_files(path):
            yield mock_file

        _tokens["google_drive"] = {"access_token": "test_token"}

        with patch(
            "aragora.server.handlers.features.cloud_storage.get_provider_connector"
        ) as mock_get:
            mock_connector = MagicMock()
            mock_connector.list_files = mock_list_files
            mock_connector.list_folders = AsyncMock(return_value=iter([]))
            mock_get.return_value = mock_connector

            files = await list_files("google_drive")
            assert len(files) == 1
            assert files[0]["id"] == "file123"


class TestDownloadFile:
    """Tests for download_file function."""

    @pytest.mark.asyncio
    async def test_download_invalid_provider(self):
        """Test download from invalid provider."""
        content = await download_file("invalid_provider", "file123")
        assert content is None

    @pytest.mark.asyncio
    async def test_download_success(self):
        """Test successful file download."""
        _tokens["google_drive"] = {"access_token": "test_token"}

        with patch(
            "aragora.server.handlers.features.cloud_storage.get_provider_connector"
        ) as mock_get:
            mock_connector = MagicMock()
            mock_connector.download_file = AsyncMock(return_value=b"file content")
            mock_get.return_value = mock_connector

            content = await download_file("google_drive", "file123")
            assert content == b"file content"


class TestCloudStorageHandlerEndpoints:
    """Tests for CloudStorageHandler endpoint methods."""

    @pytest.mark.asyncio
    async def test_get_status_endpoint(self):
        """Test cloud status endpoint."""
        handler = CloudStorageHandler({})
        mock_handler = MagicMock()

        with (
            patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth,
            patch.object(handler, "check_permission"),
            patch(
                "aragora.server.handlers.features.cloud_storage.get_all_provider_status"
            ) as mock_status,
        ):
            mock_auth.return_value = MagicMock()
            mock_status.return_value = {"google_drive": {"connected": False}}

            result = await handler.handle("/api/v1/cloud/status", {}, mock_handler)
            assert result.status == 200

    @pytest.mark.asyncio
    async def test_invalid_provider_endpoint(self):
        """Test endpoint with invalid provider."""
        handler = CloudStorageHandler({})
        mock_handler = MagicMock()

        with (
            patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth,
            patch.object(handler, "check_permission"),
        ):
            mock_auth.return_value = MagicMock()

            result = await handler.handle(
                "/api/v1/cloud/invalid_provider/auth/url", {}, mock_handler
            )
            assert result.status == 400

    @pytest.mark.asyncio
    async def test_auth_callback_missing_code(self):
        """Test auth callback with missing code."""
        handler = CloudStorageHandler({})
        mock_handler = MagicMock()
        mock_handler.command = "POST"

        with (
            patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth,
            patch.object(handler, "check_permission"),
        ):
            mock_auth.return_value = MagicMock()

            result = await handler.handle_post(
                "/api/v1/cloud/google_drive/auth/callback", {}, mock_handler
            )
            assert result.status == 400
