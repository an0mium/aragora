"""Tests for cloud storage handler.

Tests the cloud storage API endpoints including:
- GET  /api/cloud/status - Get connection status for all providers
- GET  /api/cloud/{provider}/auth/url - Get OAuth authorization URL
- POST /api/cloud/{provider}/auth/callback - Handle OAuth callback
- GET  /api/cloud/{provider}/files - List files in a folder
- POST /api/cloud/{provider}/download - Download file content

Note: The handler's internal path parsing uses parts[3] for provider extraction,
which corresponds to the version-stripped path format (/api/cloud/{provider}/...).
The can_handle() method and ROUTES reference /api/v1/cloud/ (versioned), but the
handle()/handle_post() dispatch receives the stripped path in production via the
handler registry's strip_version_prefix normalization.

Also tests the standalone utility functions:
- _validate_path() - path traversal prevention
- _validate_file_id() - file ID injection prevention
- get_provider_connector() - provider connector factory
- get_provider_status() - single provider status
- get_all_provider_status() - all providers status
- get_auth_url() - OAuth URL generation
- handle_auth_callback() - OAuth callback processing
- list_files() - file listing
- download_file() - file download with size limits
"""

import base64
import hashlib
import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.features.cloud_storage import (
    CLOUD_READ_PERMISSION,
    CLOUD_WRITE_PERMISSION,
    MAX_AUTH_CODE_LENGTH,
    MAX_DOWNLOAD_SIZE_BYTES,
    MAX_REDIRECT_URI_LENGTH,
    PROVIDERS,
    SAFE_FILE_ID_PATTERN,
    CloudStorageHandler,
    _tokens,
    _validate_file_id,
    _validate_path,
    download_file,
    get_all_provider_status,
    get_auth_url,
    get_provider_connector,
    get_provider_status,
    handle_auth_callback,
    list_files,
)
from aragora.server.handlers.base import HandlerResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _status(result: HandlerResult) -> int:
    """Extract status code from HandlerResult."""
    return result.status_code


def _body(result: HandlerResult) -> dict[str, Any]:
    """Extract parsed JSON body from HandlerResult."""
    return json.loads(result.body.decode("utf-8"))


@dataclass
class MockHTTPHandler:
    """Mock HTTP handler that mimics the real HTTP handler attributes."""

    path: str = "/"
    method: str = "GET"
    body: dict[str, Any] | None = None
    headers: dict[str, str] | None = None
    command: str = "GET"

    def __post_init__(self):
        if self.headers is None:
            self.headers = {"Content-Length": "0", "Content-Type": "application/json"}
        self.client_address = ("127.0.0.1", 12345)
        self.rfile = MagicMock()
        if self.body:
            body_bytes = json.dumps(self.body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers["Content-Length"] = str(len(body_bytes))
        else:
            self.rfile.read.return_value = b"{}"


# The handler's path parsing expects version-stripped paths (/api/cloud/...)
# because parts[3] maps to the provider. In production, the handler registry
# calls strip_version_prefix before dispatching.
# For testing handle()/handle_post(), we use stripped paths.
# For testing can_handle()/ROUTES, we use versioned paths since that's what
# the handler registry passes to can_handle().

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a CloudStorageHandler with minimal context."""
    ctx: dict[str, Any] = {}
    return CloudStorageHandler(ctx)


@pytest.fixture(autouse=True)
def reset_tokens():
    """Clear the in-memory token storage between tests."""
    _tokens.clear()
    yield
    _tokens.clear()


@pytest.fixture(autouse=True)
def reset_rate_limiters():
    """Reset rate limiters between tests."""
    try:
        from aragora.server.middleware.rate_limit.registry import (
            reset_rate_limiters as _reset,
        )

        _reset()
    except ImportError:
        pass
    yield
    try:
        from aragora.server.middleware.rate_limit.registry import (
            reset_rate_limiters as _reset,
        )

        _reset()
    except ImportError:
        pass


# ===========================================================================
# _validate_path tests
# ===========================================================================


class TestValidatePath:
    """Tests for the _validate_path helper."""

    def test_empty_path_returns_root(self):
        assert _validate_path("") == "/"

    def test_none_path_returns_root(self):
        assert _validate_path(None) == "/"

    def test_normal_path(self):
        assert _validate_path("/documents/reports") == "/documents/reports"

    def test_root_path(self):
        assert _validate_path("/") == "/"

    def test_rejects_dot_dot_traversal(self):
        assert _validate_path("/docs/../etc/passwd") is None

    def test_rejects_tilde(self):
        assert _validate_path("~/secret") is None

    def test_rejects_null_byte(self):
        assert _validate_path("/docs/file\x00.txt") is None

    def test_normalizes_repeated_slashes(self):
        result = _validate_path("/docs///reports//q1")
        assert result == "/docs/reports/q1"

    def test_single_directory(self):
        assert _validate_path("/mydir") == "/mydir"

    def test_deeply_nested_path(self):
        path = "/a/b/c/d/e/f/g"
        assert _validate_path(path) == path

    def test_path_with_spaces(self):
        assert _validate_path("/my folder/file name") == "/my folder/file name"

    def test_dot_dot_at_start(self):
        assert _validate_path("../etc/passwd") is None

    def test_dot_dot_at_end(self):
        assert _validate_path("/docs/..") is None

    def test_dot_dot_in_middle(self):
        assert _validate_path("/docs/../../../etc") is None

    def test_double_dot_dot(self):
        assert _validate_path("/docs/..../file") is None

    def test_tilde_in_middle(self):
        assert _validate_path("/home/~user/docs") is None


# ===========================================================================
# _validate_file_id tests
# ===========================================================================


class TestValidateFileId:
    """Tests for the _validate_file_id helper."""

    def test_valid_alphanumeric_id(self):
        assert _validate_file_id("abc123") is True

    def test_valid_id_with_hyphens(self):
        assert _validate_file_id("file-id-123") is True

    def test_valid_id_with_underscores(self):
        assert _validate_file_id("file_id_123") is True

    def test_valid_id_with_dots(self):
        assert _validate_file_id("file.id.123") is True

    def test_valid_id_with_colons(self):
        assert _validate_file_id("drive:file:123") is True

    def test_valid_id_with_equals_plus_slash(self):
        assert _validate_file_id("base64+encoded/id==") is True

    def test_empty_string_invalid(self):
        assert _validate_file_id("") is False

    def test_none_invalid(self):
        assert _validate_file_id(None) is False

    def test_non_string_invalid(self):
        assert _validate_file_id(123) is False

    def test_too_long_invalid(self):
        assert _validate_file_id("a" * 513) is False

    def test_max_length_valid(self):
        assert _validate_file_id("a" * 512) is True

    def test_special_chars_invalid(self):
        assert _validate_file_id("file;DROP TABLE") is False

    def test_newline_invalid(self):
        assert _validate_file_id("file\nid") is False

    def test_space_invalid(self):
        assert _validate_file_id("file id") is False

    def test_single_char_valid(self):
        assert _validate_file_id("x") is True

    def test_backslash_invalid(self):
        assert _validate_file_id("file\\id") is False

    def test_pipe_invalid(self):
        assert _validate_file_id("file|id") is False


# ===========================================================================
# get_provider_connector tests
# ===========================================================================


class TestGetProviderConnector:
    """Tests for get_provider_connector factory."""

    @patch(
        "aragora.connectors.enterprise.documents.gdrive.GoogleDriveConnector",
        new_callable=MagicMock,
    )
    def test_google_drive_connector(self, mock_cls):
        result = get_provider_connector("google_drive")
        assert result is not None

    @patch(
        "aragora.connectors.enterprise.documents.onedrive.OneDriveConnector",
        new_callable=MagicMock,
    )
    def test_onedrive_connector(self, mock_cls):
        result = get_provider_connector("onedrive")
        assert result is not None

    @patch(
        "aragora.connectors.enterprise.documents.dropbox.DropboxConnector",
        new_callable=MagicMock,
    )
    def test_dropbox_connector(self, mock_cls):
        result = get_provider_connector("dropbox")
        assert result is not None

    @patch(
        "aragora.connectors.enterprise.documents.s3.S3Connector",
        new_callable=MagicMock,
    )
    def test_s3_connector(self, mock_cls):
        result = get_provider_connector("s3")
        assert result is not None

    def test_unknown_provider_returns_none(self):
        assert get_provider_connector("azure_blob") is None

    def test_empty_provider_returns_none(self):
        assert get_provider_connector("") is None


# ===========================================================================
# get_provider_status tests
# ===========================================================================


class TestGetProviderStatus:
    """Tests for get_provider_status."""

    def test_unknown_provider(self):
        result = get_provider_status("unknown")
        assert result == {"connected": False, "configured": False}

    @patch("aragora.server.handlers.features.cloud_storage.get_provider_connector")
    def test_configured_with_token(self, mock_get_connector):
        mock_connector = MagicMock()
        mock_connector.is_configured = True
        mock_get_connector.return_value = mock_connector
        _tokens["test_provider"] = {"access_token": "tok123", "account_name": "Bob"}

        result = get_provider_status("test_provider")
        assert result["connected"] is True
        assert result["configured"] is True
        assert result["account_name"] == "Bob"

    @patch("aragora.server.handlers.features.cloud_storage.get_provider_connector")
    def test_configured_without_token(self, mock_get_connector):
        mock_connector = MagicMock()
        mock_connector.is_configured = True
        mock_get_connector.return_value = mock_connector

        result = get_provider_status("test_provider")
        assert result["connected"] is False
        assert result["configured"] is True

    @patch("aragora.server.handlers.features.cloud_storage.get_provider_connector")
    def test_not_configured(self, mock_get_connector):
        mock_connector = MagicMock()
        mock_connector.is_configured = False
        mock_get_connector.return_value = mock_connector

        result = get_provider_status("test_provider")
        assert result["connected"] is False
        assert result["configured"] is False

    @patch("aragora.server.handlers.features.cloud_storage.get_provider_connector")
    def test_is_configured_callable(self, mock_get_connector):
        """Test when is_configured is a method rather than a property."""
        mock_connector = MagicMock()
        mock_connector.is_configured = MagicMock(return_value=True)
        mock_get_connector.return_value = mock_connector
        _tokens["test_provider"] = {"access_token": "tok"}

        result = get_provider_status("test_provider")
        assert result["connected"] is True

    @patch("aragora.server.handlers.features.cloud_storage.get_provider_connector")
    def test_token_without_account_name(self, mock_get_connector):
        mock_connector = MagicMock()
        mock_connector.is_configured = True
        mock_get_connector.return_value = mock_connector
        _tokens["test_provider"] = {"access_token": "tok"}

        result = get_provider_status("test_provider")
        assert result["account_name"] is None

    @patch("aragora.server.handlers.features.cloud_storage.get_provider_connector")
    def test_configured_with_token_but_no_access_token_key(self, mock_get_connector):
        mock_connector = MagicMock()
        mock_connector.is_configured = True
        mock_get_connector.return_value = mock_connector
        _tokens["test_provider"] = {"refresh_token": "ref"}

        result = get_provider_status("test_provider")
        assert result["connected"] is False  # No access_token key


# ===========================================================================
# get_all_provider_status tests
# ===========================================================================


class TestGetAllProviderStatus:
    """Tests for get_all_provider_status."""

    @patch("aragora.server.handlers.features.cloud_storage.get_provider_status")
    def test_returns_all_providers(self, mock_status):
        mock_status.return_value = {"connected": False, "configured": False}
        result = get_all_provider_status()
        assert set(result.keys()) == set(PROVIDERS)
        assert mock_status.call_count == len(PROVIDERS)

    @patch("aragora.server.handlers.features.cloud_storage.get_provider_status")
    def test_providers_list(self, mock_status):
        mock_status.return_value = {"connected": False, "configured": False}
        result = get_all_provider_status()
        assert "google_drive" in result
        assert "onedrive" in result
        assert "dropbox" in result
        assert "s3" in result


# ===========================================================================
# get_auth_url tests
# ===========================================================================


class TestGetAuthUrl:
    """Tests for get_auth_url."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.get_provider_connector")
    async def test_returns_url(self, mock_get_connector):
        mock_connector = MagicMock()
        mock_connector.get_oauth_url.return_value = "https://auth.example.com/authorize"
        mock_get_connector.return_value = mock_connector

        url = await get_auth_url("google_drive", "https://localhost/callback", "state123")
        assert url == "https://auth.example.com/authorize"

    @pytest.mark.asyncio
    async def test_unknown_provider_returns_none(self):
        url = await get_auth_url("unknown_provider", "https://localhost/callback")
        assert url is None

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.get_provider_connector")
    async def test_no_oauth_method_returns_none(self, mock_get_connector):
        mock_connector = MagicMock(spec=[])  # No get_oauth_url
        mock_get_connector.return_value = mock_connector

        url = await get_auth_url("test_provider", "https://localhost/callback")
        assert url is None

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.get_provider_connector")
    async def test_passes_state_to_connector(self, mock_get_connector):
        mock_connector = MagicMock()
        mock_connector.get_oauth_url.return_value = "https://auth.example.com/"
        mock_get_connector.return_value = mock_connector

        await get_auth_url("google_drive", "https://localhost/cb", "my_state")
        mock_connector.get_oauth_url.assert_called_once_with("https://localhost/cb", "my_state")


# ===========================================================================
# handle_auth_callback tests
# ===========================================================================


class TestHandleAuthCallback:
    """Tests for handle_auth_callback."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.get_provider_connector")
    async def test_successful_callback(self, mock_get_connector):
        mock_connector = MagicMock()
        mock_connector.authenticate = AsyncMock(return_value=True)
        mock_connector._access_token = "access_tok"
        mock_connector._refresh_token = "refresh_tok"
        mock_connector.get_user_info = AsyncMock(return_value={"displayName": "Test User"})
        mock_get_connector.return_value = mock_connector

        result = await handle_auth_callback("google_drive", "auth_code", "https://localhost/cb")
        assert result is True
        assert _tokens["google_drive"]["access_token"] == "access_tok"
        assert _tokens["google_drive"]["refresh_token"] == "refresh_tok"
        assert _tokens["google_drive"]["account_name"] == "Test User"

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.get_provider_connector")
    async def test_callback_with_name_fallback(self, mock_get_connector):
        mock_connector = MagicMock()
        mock_connector.authenticate = AsyncMock(return_value=True)
        mock_connector._access_token = "tok"
        mock_connector._refresh_token = None
        mock_connector.get_user_info = AsyncMock(return_value={"name": "Bob"})
        mock_get_connector.return_value = mock_connector

        result = await handle_auth_callback("onedrive", "code", "https://localhost/cb")
        assert result is True
        assert _tokens["onedrive"]["account_name"] == "Bob"

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.get_provider_connector")
    async def test_callback_with_email_fallback(self, mock_get_connector):
        mock_connector = MagicMock()
        mock_connector.authenticate = AsyncMock(return_value=True)
        mock_connector._access_token = "tok"
        mock_connector._refresh_token = None
        mock_connector.get_user_info = AsyncMock(return_value={"email": "bob@example.com"})
        mock_get_connector.return_value = mock_connector

        result = await handle_auth_callback("dropbox", "code", "https://localhost/cb")
        assert result is True
        assert _tokens["dropbox"]["account_name"] == "bob@example.com"

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.get_provider_connector")
    async def test_callback_authentication_fails(self, mock_get_connector):
        mock_connector = MagicMock()
        mock_connector.authenticate = AsyncMock(return_value=False)
        mock_get_connector.return_value = mock_connector

        result = await handle_auth_callback("google_drive", "bad_code", "https://localhost/cb")
        assert result is False

    @pytest.mark.asyncio
    async def test_callback_unknown_provider(self):
        result = await handle_auth_callback("unknown", "code", "https://localhost/cb")
        assert result is False

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.get_provider_connector")
    async def test_callback_connection_error(self, mock_get_connector):
        mock_connector = MagicMock()
        mock_connector.authenticate = AsyncMock(side_effect=ConnectionError("refused"))
        mock_get_connector.return_value = mock_connector

        result = await handle_auth_callback("google_drive", "code", "https://localhost/cb")
        assert result is False

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.get_provider_connector")
    async def test_callback_timeout_error(self, mock_get_connector):
        mock_connector = MagicMock()
        mock_connector.authenticate = AsyncMock(side_effect=TimeoutError("timed out"))
        mock_get_connector.return_value = mock_connector

        result = await handle_auth_callback("google_drive", "code", "https://localhost/cb")
        assert result is False

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.get_provider_connector")
    async def test_callback_user_info_fails_gracefully(self, mock_get_connector):
        mock_connector = MagicMock()
        mock_connector.authenticate = AsyncMock(return_value=True)
        mock_connector._access_token = "tok"
        mock_connector._refresh_token = None
        mock_connector.get_user_info = AsyncMock(side_effect=AttributeError("no info"))
        mock_get_connector.return_value = mock_connector

        result = await handle_auth_callback("google_drive", "code", "https://localhost/cb")
        assert result is True
        # Token stored even though user info failed
        assert "google_drive" in _tokens

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.get_provider_connector")
    async def test_callback_no_get_user_info_method(self, mock_get_connector):
        mock_connector = MagicMock(spec=["authenticate", "_access_token", "_refresh_token"])
        mock_connector.authenticate = AsyncMock(return_value=True)
        mock_connector._access_token = "tok"
        mock_connector._refresh_token = None
        mock_get_connector.return_value = mock_connector

        result = await handle_auth_callback("google_drive", "code", "https://localhost/cb")
        assert result is True
        assert _tokens["google_drive"]["access_token"] == "tok"

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.get_provider_connector")
    async def test_callback_user_info_key_error(self, mock_get_connector):
        mock_connector = MagicMock()
        mock_connector.authenticate = AsyncMock(return_value=True)
        mock_connector._access_token = "tok"
        mock_connector._refresh_token = None
        mock_connector.get_user_info = AsyncMock(side_effect=KeyError("missing"))
        mock_get_connector.return_value = mock_connector

        result = await handle_auth_callback("google_drive", "code", "https://localhost/cb")
        assert result is True


# ===========================================================================
# list_files tests
# ===========================================================================


class TestListFiles:
    """Tests for list_files."""

    @pytest.mark.asyncio
    async def test_unknown_provider_returns_empty(self):
        files = await list_files("unknown", "/")
        assert files == []

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.get_provider_connector")
    async def test_list_files_basic(self, mock_get_connector):
        mock_file = MagicMock()
        mock_file.id = "f1"
        mock_file.name = "report.pdf"
        mock_file.path = "/docs/report.pdf"
        mock_file.size = 1024
        mock_file.mime_type = "application/pdf"
        mock_file.modified_time = "2026-01-01"
        mock_file.web_url = "https://drive.example.com/f1"

        mock_connector = MagicMock()

        async def list_files_gen(path):
            yield mock_file

        mock_connector.list_files = list_files_gen
        # Remove list_folders to prevent iteration
        del mock_connector.list_folders
        mock_get_connector.return_value = mock_connector

        files = await list_files("google_drive", "/docs")
        assert len(files) == 1
        assert files[0]["id"] == "f1"
        assert files[0]["name"] == "report.pdf"
        assert files[0]["is_folder"] is False
        assert files[0]["size"] == 1024
        assert files[0]["mime_type"] == "application/pdf"

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.get_provider_connector")
    async def test_list_files_with_folders(self, mock_get_connector):
        mock_folder = MagicMock()
        mock_folder.id = "d1"
        mock_folder.name = "Reports"
        mock_folder.path = "/Reports"

        mock_connector = MagicMock()

        async def list_files_gen(path):
            return
            yield  # Empty async generator

        async def list_folders_gen(path):
            yield mock_folder

        mock_connector.list_files = list_files_gen
        mock_connector.list_folders = list_folders_gen
        mock_get_connector.return_value = mock_connector

        files = await list_files("google_drive", "/")
        assert len(files) == 1
        assert files[0]["is_folder"] is True
        assert files[0]["name"] == "Reports"
        assert files[0]["mime_type"] == "application/vnd.google-apps.folder"

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.get_provider_connector")
    async def test_list_files_restores_tokens(self, mock_get_connector):
        _tokens["google_drive"] = {
            "access_token": "saved_token",
            "refresh_token": "refresh_tok",
        }

        mock_connector = MagicMock()
        mock_connector._access_token = None
        mock_connector._refresh_token = None

        async def list_files_gen(path):
            return
            yield

        mock_connector.list_files = list_files_gen

        async def list_folders_gen(path):
            return
            yield

        mock_connector.list_folders = list_folders_gen
        mock_get_connector.return_value = mock_connector

        await list_files("google_drive", "/")
        assert mock_connector._access_token == "saved_token"
        assert mock_connector._refresh_token == "refresh_tok"

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.get_provider_connector")
    async def test_list_files_page_size_limit(self, mock_get_connector):
        mock_connector = MagicMock()

        async def list_files_gen(path):
            for i in range(10):
                f = MagicMock()
                f.id = str(i)
                f.name = f"file{i}.txt"
                yield f

        mock_connector.list_files = list_files_gen
        del mock_connector.list_folders
        mock_get_connector.return_value = mock_connector

        files = await list_files("google_drive", "/", page_size=3)
        assert len(files) == 3

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.get_provider_connector")
    async def test_list_files_error_returns_empty(self, mock_get_connector):
        mock_connector = MagicMock()

        async def list_files_gen(path):
            raise ConnectionError("network failure")
            yield  # noqa: unreachable

        mock_connector.list_files = list_files_gen
        del mock_connector.list_folders
        mock_get_connector.return_value = mock_connector

        files = await list_files("google_drive", "/")
        assert files == []

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.get_provider_connector")
    async def test_list_files_no_list_files_method(self, mock_get_connector):
        mock_connector = MagicMock(spec=[])  # No methods
        mock_get_connector.return_value = mock_connector

        files = await list_files("s3", "/")
        assert files == []

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.get_provider_connector")
    async def test_list_files_file_path_fallback(self, mock_get_connector):
        """Test path fallback when file doesn't have path attribute."""
        mock_file = MagicMock(spec=["id", "name"])
        mock_file.id = "f1"
        mock_file.name = "test.txt"

        mock_connector = MagicMock()

        async def list_files_gen(path):
            yield mock_file

        mock_connector.list_files = list_files_gen
        del mock_connector.list_folders
        mock_get_connector.return_value = mock_connector

        files = await list_files("google_drive", "/docs")
        assert len(files) == 1
        assert files[0]["path"] == "/docs/test.txt"

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.get_provider_connector")
    async def test_list_files_no_tokens_stored(self, mock_get_connector):
        """Test that list_files works when no tokens stored for provider."""
        mock_connector = MagicMock()

        async def list_files_gen(path):
            return
            yield

        mock_connector.list_files = list_files_gen
        del mock_connector.list_folders
        mock_get_connector.return_value = mock_connector

        files = await list_files("google_drive", "/")
        assert files == []


# ===========================================================================
# download_file tests
# ===========================================================================


class TestDownloadFile:
    """Tests for download_file."""

    @pytest.mark.asyncio
    async def test_unknown_provider_returns_none(self):
        result = await download_file("unknown", "file123")
        assert result is None

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.get_provider_connector")
    async def test_successful_download(self, mock_get_connector):
        mock_connector = MagicMock()
        mock_connector.download_file = AsyncMock(return_value=b"file content")
        mock_get_connector.return_value = mock_connector

        result = await download_file("google_drive", "f1")
        assert result == b"file content"

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.get_provider_connector")
    async def test_download_restores_tokens(self, mock_get_connector):
        _tokens["google_drive"] = {
            "access_token": "tok",
            "refresh_token": "ref",
        }

        mock_connector = MagicMock()
        mock_connector._access_token = None
        mock_connector._refresh_token = None
        mock_connector.download_file = AsyncMock(return_value=b"data")
        mock_get_connector.return_value = mock_connector

        await download_file("google_drive", "f1")
        assert mock_connector._access_token == "tok"
        assert mock_connector._refresh_token == "ref"

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.get_provider_connector")
    async def test_download_exceeds_size_limit(self, mock_get_connector):
        mock_connector = MagicMock()
        oversized = b"x" * (MAX_DOWNLOAD_SIZE_BYTES + 1)
        mock_connector.download_file = AsyncMock(return_value=oversized)
        mock_get_connector.return_value = mock_connector

        result = await download_file("google_drive", "f1")
        assert result is None

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.get_provider_connector")
    async def test_download_exactly_at_size_limit(self, mock_get_connector):
        mock_connector = MagicMock()
        exact = b"x" * MAX_DOWNLOAD_SIZE_BYTES
        mock_connector.download_file = AsyncMock(return_value=exact)
        mock_get_connector.return_value = mock_connector

        result = await download_file("google_drive", "f1")
        assert result == exact

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.get_provider_connector")
    async def test_download_connection_error(self, mock_get_connector):
        mock_connector = MagicMock()
        mock_connector.download_file = AsyncMock(side_effect=ConnectionError("refused"))
        mock_get_connector.return_value = mock_connector

        result = await download_file("google_drive", "f1")
        assert result is None

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.get_provider_connector")
    async def test_download_timeout_error(self, mock_get_connector):
        mock_connector = MagicMock()
        mock_connector.download_file = AsyncMock(side_effect=TimeoutError("timed out"))
        mock_get_connector.return_value = mock_connector

        result = await download_file("google_drive", "f1")
        assert result is None

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.get_provider_connector")
    async def test_download_no_method(self, mock_get_connector):
        mock_connector = MagicMock(spec=[])  # No download_file method
        mock_get_connector.return_value = mock_connector

        result = await download_file("google_drive", "f1")
        assert result is None

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.get_provider_connector")
    async def test_download_returns_none(self, mock_get_connector):
        mock_connector = MagicMock()
        mock_connector.download_file = AsyncMock(return_value=None)
        mock_get_connector.return_value = mock_connector

        result = await download_file("google_drive", "f1")
        assert result is None

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.get_provider_connector")
    async def test_download_empty_file(self, mock_get_connector):
        mock_connector = MagicMock()
        mock_connector.download_file = AsyncMock(return_value=b"")
        mock_get_connector.return_value = mock_connector

        result = await download_file("google_drive", "f1")
        assert result == b""


# ===========================================================================
# CloudStorageHandler.can_handle tests
# ===========================================================================


class TestCanHandle:
    """Tests for the can_handle method.

    can_handle receives the original versioned path from the handler registry.
    """

    def test_cloud_status_path(self, handler):
        assert handler.can_handle("/api/v1/cloud/status") is True

    def test_cloud_provider_path(self, handler):
        assert handler.can_handle("/api/v1/cloud/google_drive/files") is True

    def test_cloud_auth_path(self, handler):
        assert handler.can_handle("/api/v1/cloud/onedrive/auth/url") is True

    def test_non_cloud_path(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_partial_prefix(self, handler):
        assert handler.can_handle("/api/v1/cloudwatch") is False

    def test_root_path(self, handler):
        assert handler.can_handle("/") is False


# ===========================================================================
# CloudStorageHandler.handle GET tests
#
# The handle()/handle_post() methods receive version-stripped paths in production.
# Path like /api/cloud/google_drive/auth/url splits as:
#   ['', 'api', 'cloud', 'google_drive', 'auth', 'url']
# parts[3] = 'google_drive' (the provider)
# ===========================================================================


class TestHandleGet:
    """Tests for GET request handling."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.get_all_provider_status")
    async def test_get_status(self, mock_all_status, handler):
        mock_all_status.return_value = {
            "google_drive": {"connected": True, "configured": True},
            "onedrive": {"connected": False, "configured": False},
        }
        mock_http = MockHTTPHandler(path="/api/v1/cloud/status")
        result = await handler.handle("/api/v1/cloud/status", {}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert "google_drive" in body

    @pytest.mark.asyncio
    async def test_unknown_provider(self, handler):
        mock_http = MockHTTPHandler()
        result = await handler.handle("/api/cloud/azure/files", {}, mock_http)
        assert _status(result) == 400
        assert "Unknown provider" in _body(result).get("error", "")

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.get_auth_url")
    async def test_get_auth_url_success(self, mock_auth_url, handler):
        mock_auth_url.return_value = "https://accounts.google.com/o/oauth2/auth?..."
        mock_http = MockHTTPHandler()
        result = await handler.handle(
            "/api/cloud/google_drive/auth/url",
            {"redirect_uri": "https://localhost/cb"},
            mock_http,
        )
        assert _status(result) == 200
        assert "url" in _body(result)

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.get_auth_url")
    async def test_get_auth_url_provider_not_configured(self, mock_auth_url, handler):
        mock_auth_url.return_value = None
        mock_http = MockHTTPHandler()
        result = await handler.handle(
            "/api/cloud/onedrive/auth/url",
            {},
            mock_http,
        )
        assert _status(result) == 400
        assert "not configured" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_get_auth_url_redirect_uri_too_long(self, handler):
        mock_http = MockHTTPHandler()
        result = await handler.handle(
            "/api/cloud/google_drive/auth/url",
            {"redirect_uri": "https://example.com/" + "a" * MAX_REDIRECT_URI_LENGTH},
            mock_http,
        )
        assert _status(result) == 400
        assert "too long" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_get_auth_url_invalid_scheme(self, handler):
        mock_http = MockHTTPHandler()
        result = await handler.handle(
            "/api/cloud/google_drive/auth/url",
            {"redirect_uri": "ftp://evil.com/callback"},
            mock_http,
        )
        assert _status(result) == 400
        assert "scheme" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.list_files")
    async def test_list_files_success(self, mock_list_files, handler):
        mock_list_files.return_value = [
            {"id": "f1", "name": "test.txt", "is_folder": False},
        ]
        mock_http = MockHTTPHandler()
        result = await handler.handle(
            "/api/cloud/google_drive/files",
            {"path": "/docs"},
            mock_http,
        )
        assert _status(result) == 200
        body = _body(result)
        assert len(body["files"]) == 1
        assert body["path"] == "/docs"

    @pytest.mark.asyncio
    async def test_list_files_path_traversal_blocked(self, handler):
        mock_http = MockHTTPHandler()
        result = await handler.handle(
            "/api/cloud/google_drive/files",
            {"path": "/docs/../etc/passwd"},
            mock_http,
        )
        assert _status(result) == 400
        assert "traversal" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.list_files")
    async def test_list_files_default_path(self, mock_list_files, handler):
        mock_list_files.return_value = []
        mock_http = MockHTTPHandler()
        result = await handler.handle(
            "/api/cloud/dropbox/files",
            {},
            mock_http,
        )
        assert _status(result) == 200
        assert _body(result)["path"] == "/"

    @pytest.mark.asyncio
    async def test_auth_callback_post_defers(self, handler):
        """POST to callback from handle() returns None (defers to handle_post)."""
        mock_http = MockHTTPHandler(command="POST")
        result = await handler.handle(
            "/api/cloud/google_drive/auth/callback",
            {},
            mock_http,
        )
        # command is POST -> returns None (defers to handle_post)
        assert result is None

    @pytest.mark.asyncio
    async def test_auth_callback_get_method_not_allowed(self, handler):
        """GET to callback with GET command returns 405."""
        mock_http = MockHTTPHandler(command="GET")
        result = await handler.handle(
            "/api/cloud/google_drive/auth/callback",
            {},
            mock_http,
        )
        assert _status(result) == 405

    @pytest.mark.asyncio
    async def test_not_found_action(self, handler):
        mock_http = MockHTTPHandler()
        result = await handler.handle(
            "/api/cloud/google_drive/nonexistent",
            {},
            mock_http,
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_provider_no_action(self, handler):
        mock_http = MockHTTPHandler()
        result = await handler.handle(
            "/api/cloud/google_drive",
            {},
            mock_http,
        )
        # action="" -> falls through to not_found
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_short_path_rejected(self, handler):
        mock_http = MockHTTPHandler()
        # /api/cloud splits to ['', 'api', 'cloud'] which has len 3 < 4
        result = await handler.handle("/api/cloud", {}, mock_http)
        assert _status(result) == 400

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.list_files")
    async def test_list_files_for_s3(self, mock_list_files, handler):
        mock_list_files.return_value = []
        mock_http = MockHTTPHandler()
        result = await handler.handle(
            "/api/cloud/s3/files",
            {},
            mock_http,
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.get_auth_url")
    async def test_get_auth_url_for_dropbox(self, mock_auth_url, handler):
        mock_auth_url.return_value = "https://dropbox.com/oauth2/authorize"
        mock_http = MockHTTPHandler()
        result = await handler.handle(
            "/api/cloud/dropbox/auth/url",
            {"redirect_uri": "https://localhost/cb"},
            mock_http,
        )
        assert _status(result) == 200


# ===========================================================================
# CloudStorageHandler.handle_post tests
# ===========================================================================


class TestHandlePost:
    """Tests for POST request handling."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.handle_auth_callback")
    async def test_auth_callback_success(self, mock_cb, handler):
        mock_cb.return_value = True
        mock_http = MockHTTPHandler(
            command="POST",
            body={"code": "auth_code_123", "redirect_uri": "https://localhost/cb"},
        )
        result = await handler.handle_post(
            "/api/cloud/google_drive/auth/callback",
            {},
            mock_http,
        )
        assert _status(result) == 200
        assert _body(result)["success"] is True

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.handle_auth_callback")
    async def test_auth_callback_failure(self, mock_cb, handler):
        mock_cb.return_value = False
        mock_http = MockHTTPHandler(
            command="POST",
            body={"code": "bad_code", "redirect_uri": "https://localhost/cb"},
        )
        result = await handler.handle_post(
            "/api/cloud/google_drive/auth/callback",
            {},
            mock_http,
        )
        assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_auth_callback_missing_code(self, handler):
        mock_http = MockHTTPHandler(
            command="POST",
            body={"redirect_uri": "https://localhost/cb"},
        )
        result = await handler.handle_post(
            "/api/cloud/google_drive/auth/callback",
            {},
            mock_http,
        )
        assert _status(result) == 400
        assert "Missing" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_auth_callback_code_too_long(self, handler):
        mock_http = MockHTTPHandler(
            command="POST",
            body={"code": "x" * (MAX_AUTH_CODE_LENGTH + 1), "redirect_uri": "https://localhost/cb"},
        )
        result = await handler.handle_post(
            "/api/cloud/google_drive/auth/callback",
            {},
            mock_http,
        )
        assert _status(result) == 400
        assert "Invalid authorization code" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_auth_callback_code_not_string(self, handler):
        mock_http = MockHTTPHandler(
            command="POST",
            body={"code": 12345, "redirect_uri": "https://localhost/cb"},
        )
        result = await handler.handle_post(
            "/api/cloud/google_drive/auth/callback",
            {},
            mock_http,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_auth_callback_redirect_uri_too_long(self, handler):
        mock_http = MockHTTPHandler(
            command="POST",
            body={
                "code": "valid",
                "redirect_uri": "https://x.com/" + "a" * MAX_REDIRECT_URI_LENGTH,
            },
        )
        result = await handler.handle_post(
            "/api/cloud/google_drive/auth/callback",
            {},
            mock_http,
        )
        assert _status(result) == 400
        assert "redirect_uri" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_auth_callback_redirect_uri_bad_scheme(self, handler):
        mock_http = MockHTTPHandler(
            command="POST",
            body={"code": "valid", "redirect_uri": "ftp://evil.com/cb"},
        )
        result = await handler.handle_post(
            "/api/cloud/google_drive/auth/callback",
            {},
            mock_http,
        )
        assert _status(result) == 400
        assert "scheme" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.download_file")
    async def test_download_success(self, mock_download, handler):
        content = b"Hello, World!"
        mock_download.return_value = content
        mock_http = MockHTTPHandler(
            command="POST",
            body={"file_id": "f123"},
        )
        result = await handler.handle_post(
            "/api/cloud/google_drive/download/file",
            {},
            mock_http,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["size"] == len(content)
        assert body["content"] == base64.b64encode(content).decode()
        expected_checksum = hashlib.sha256(content).hexdigest()
        assert body["checksum_sha256"] == expected_checksum

    @pytest.mark.asyncio
    async def test_download_missing_file_id(self, handler):
        mock_http = MockHTTPHandler(
            command="POST",
            body={},
        )
        result = await handler.handle_post(
            "/api/cloud/google_drive/download/file",
            {},
            mock_http,
        )
        assert _status(result) == 400
        assert "file_id" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_download_invalid_file_id(self, handler):
        mock_http = MockHTTPHandler(
            command="POST",
            body={"file_id": "invalid;DROP TABLE"},
        )
        result = await handler.handle_post(
            "/api/cloud/google_drive/download/file",
            {},
            mock_http,
        )
        assert _status(result) == 400
        assert "format" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.download_file")
    async def test_download_fails(self, mock_download, handler):
        mock_download.return_value = None
        mock_http = MockHTTPHandler(
            command="POST",
            body={"file_id": "f123"},
        )
        result = await handler.handle_post(
            "/api/cloud/google_drive/download/file",
            {},
            mock_http,
        )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_post_short_path(self, handler):
        mock_http = MockHTTPHandler(
            command="POST",
            body={"key": "value"},
        )
        result = await handler.handle_post(
            "/api/cloud",
            {},
            mock_http,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_post_unknown_provider(self, handler):
        mock_http = MockHTTPHandler(
            command="POST",
            body={"code": "test"},
        )
        result = await handler.handle_post(
            "/api/cloud/azure/auth/callback",
            {},
            mock_http,
        )
        assert _status(result) == 400
        assert "Unknown provider" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_post_unknown_action(self, handler):
        mock_http = MockHTTPHandler(
            command="POST",
            body={"key": "value"},
        )
        result = await handler.handle_post(
            "/api/cloud/google_drive/unknown/action",
            {},
            mock_http,
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_post_invalid_json_body(self, handler):
        mock_http = MockHTTPHandler(command="POST")
        # Set Content-Length to a non-zero value but provide invalid JSON
        mock_http.headers["Content-Length"] = "5"
        mock_http.rfile.read.return_value = b"notjs"
        result = await handler.handle_post(
            "/api/cloud/google_drive/auth/callback",
            {},
            mock_http,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.download_file")
    async def test_download_empty_file(self, mock_download, handler):
        content = b""
        mock_download.return_value = content
        mock_http = MockHTTPHandler(
            command="POST",
            body={"file_id": "empty_file"},
        )
        result = await handler.handle_post(
            "/api/cloud/google_drive/download/file",
            {},
            mock_http,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["size"] == 0
        assert body["content"] == ""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.handle_auth_callback")
    async def test_auth_callback_for_onedrive(self, mock_cb, handler):
        mock_cb.return_value = True
        mock_http = MockHTTPHandler(
            command="POST",
            body={"code": "test_code"},
        )
        result = await handler.handle_post(
            "/api/cloud/onedrive/auth/callback",
            {},
            mock_http,
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.download_file")
    async def test_download_for_dropbox(self, mock_download, handler):
        mock_download.return_value = b"dropbox-data"
        mock_http = MockHTTPHandler(
            command="POST",
            body={"file_id": "dbx123"},
        )
        result = await handler.handle_post(
            "/api/cloud/dropbox/download/file",
            {},
            mock_http,
        )
        assert _status(result) == 200


# ===========================================================================
# RBAC / Authorization tests
# ===========================================================================


class TestHandlerAuth:
    """Tests for authentication/authorization on handler methods."""

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_handle_get_unauthenticated(self, handler):
        from aragora.server.handlers.utils.auth import UnauthorizedError

        async def raise_unauth(self, request, require_auth=False):
            raise UnauthorizedError("no token")

        with patch.object(type(handler), "get_auth_context", raise_unauth):
            mock_http = MockHTTPHandler()
            result = await handler.handle("/api/v1/cloud/status", {}, mock_http)
            assert _status(result) == 401

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_handle_get_forbidden(self, handler):
        from aragora.rbac.models import AuthorizationContext
        from aragora.server.handlers.utils.auth import ForbiddenError

        async def mock_auth(self, request, require_auth=False):
            return AuthorizationContext(
                user_id="u1",
                user_email="u@test.com",
                org_id="o1",
                roles=set(),
                permissions=set(),
            )

        def mock_check_perm(self, auth_ctx, perm, resource_id=None):
            raise ForbiddenError("no permission", permission=perm)

        with (
            patch.object(type(handler), "get_auth_context", mock_auth),
            patch.object(type(handler), "check_permission", mock_check_perm),
        ):
            mock_http = MockHTTPHandler()
            result = await handler.handle("/api/v1/cloud/status", {}, mock_http)
            assert _status(result) == 403

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_handle_post_unauthenticated(self, handler):
        from aragora.server.handlers.utils.auth import UnauthorizedError

        async def raise_unauth(self, request, require_auth=False):
            raise UnauthorizedError("no token")

        with patch.object(type(handler), "get_auth_context", raise_unauth):
            mock_http = MockHTTPHandler(
                command="POST",
                body={"code": "test"},
            )
            result = await handler.handle_post(
                "/api/cloud/google_drive/auth/callback",
                {},
                mock_http,
            )
            assert _status(result) == 401

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_handle_post_forbidden(self, handler):
        from aragora.rbac.models import AuthorizationContext
        from aragora.server.handlers.utils.auth import ForbiddenError

        async def mock_auth(self, request, require_auth=False):
            return AuthorizationContext(
                user_id="u1",
                user_email="u@test.com",
                org_id="o1",
                roles=set(),
                permissions=set(),
            )

        def mock_check_perm(self, auth_ctx, perm, resource_id=None):
            raise ForbiddenError("no permission", permission=perm)

        with (
            patch.object(type(handler), "get_auth_context", mock_auth),
            patch.object(type(handler), "check_permission", mock_check_perm),
        ):
            mock_http = MockHTTPHandler(
                command="POST",
                body={"code": "test"},
            )
            result = await handler.handle_post(
                "/api/cloud/google_drive/auth/callback",
                {},
                mock_http,
            )
            assert _status(result) == 403


# ===========================================================================
# Constants / Configuration tests
# ===========================================================================


class TestConstants:
    """Tests for module-level constants and configuration."""

    def test_providers_list(self):
        assert "google_drive" in PROVIDERS
        assert "onedrive" in PROVIDERS
        assert "dropbox" in PROVIDERS
        assert "s3" in PROVIDERS

    def test_providers_count(self):
        assert len(PROVIDERS) == 4

    def test_cloud_permissions(self):
        assert CLOUD_READ_PERMISSION == "cloud:read"
        assert CLOUD_WRITE_PERMISSION == "cloud:write"

    def test_max_download_size_is_positive(self):
        assert MAX_DOWNLOAD_SIZE_BYTES > 0

    def test_max_download_size_default(self):
        # Default is 100MB
        assert MAX_DOWNLOAD_SIZE_BYTES == 100 * 1024 * 1024

    def test_max_auth_code_length(self):
        assert MAX_AUTH_CODE_LENGTH == 4096

    def test_max_redirect_uri_length(self):
        assert MAX_REDIRECT_URI_LENGTH == 2048

    def test_safe_file_id_pattern(self):
        assert SAFE_FILE_ID_PATTERN.match("abc123")
        assert not SAFE_FILE_ID_PATTERN.match("")
        assert not SAFE_FILE_ID_PATTERN.match("a b")

    def test_handler_routes_declared(self):
        assert "/api/v1/cloud/status" in CloudStorageHandler.ROUTES
        assert "/api/v1/cloud/google_drive/auth/url" in CloudStorageHandler.ROUTES
        assert "/api/v1/cloud/google_drive/files" in CloudStorageHandler.ROUTES

    def test_handler_routes_include_all_providers_auth(self):
        for provider in ["google_drive", "onedrive", "dropbox"]:
            assert f"/api/v1/cloud/{provider}/auth/url" in CloudStorageHandler.ROUTES
            assert f"/api/v1/cloud/{provider}/auth/callback" in CloudStorageHandler.ROUTES
            assert f"/api/v1/cloud/{provider}/files" in CloudStorageHandler.ROUTES


# ===========================================================================
# Edge cases and security tests
# ===========================================================================


class TestEdgeCases:
    """Edge case and security-focused tests."""

    def test_validate_path_null_byte_injection(self):
        assert _validate_path("/docs/\x00file") is None

    def test_validate_path_tilde_home(self):
        assert _validate_path("~root/.ssh") is None

    def test_validate_file_id_empty_string(self):
        assert _validate_file_id("") is False

    def test_validate_file_id_boundary_length(self):
        assert _validate_file_id("a" * 512) is True
        assert _validate_file_id("a" * 513) is False

    @pytest.mark.asyncio
    async def test_list_files_tilde_path_blocked(self, handler):
        mock_http = MockHTTPHandler()
        result = await handler.handle(
            "/api/cloud/google_drive/files",
            {"path": "~/private"},
            mock_http,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_list_files_null_byte_path_blocked(self, handler):
        mock_http = MockHTTPHandler()
        result = await handler.handle(
            "/api/cloud/google_drive/files",
            {"path": "/docs/\x00admin"},
            mock_http,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_download_file_id_sql_injection(self, handler):
        mock_http = MockHTTPHandler(
            command="POST",
            body={"file_id": "1; DROP TABLE files;--"},
        )
        result = await handler.handle_post(
            "/api/cloud/google_drive/download/file",
            {},
            mock_http,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_download_file_id_with_newlines(self, handler):
        mock_http = MockHTTPHandler(
            command="POST",
            body={"file_id": "file\nid"},
        )
        result = await handler.handle_post(
            "/api/cloud/google_drive/download/file",
            {},
            mock_http,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.download_file")
    async def test_download_checksum_integrity(self, mock_download, handler):
        content = b"test data for checksum"
        mock_download.return_value = content
        mock_http = MockHTTPHandler(
            command="POST",
            body={"file_id": "f1"},
        )
        result = await handler.handle_post(
            "/api/cloud/google_drive/download/file",
            {},
            mock_http,
        )
        body = _body(result)
        # Verify checksum matches
        expected = hashlib.sha256(content).hexdigest()
        assert body["checksum_sha256"] == expected
        # Verify decoded content matches
        decoded = base64.b64decode(body["content"])
        assert decoded == content

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.get_auth_url")
    async def test_auth_url_with_state(self, mock_auth_url, handler):
        mock_auth_url.return_value = "https://auth.example.com/?state=xyz"
        mock_http = MockHTTPHandler()
        result = await handler.handle(
            "/api/cloud/google_drive/auth/url",
            {"redirect_uri": "https://localhost/cb", "state": "xyz"},
            mock_http,
        )
        assert _status(result) == 200
        mock_auth_url.assert_called_once_with("google_drive", "https://localhost/cb", "xyz")

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.get_auth_url")
    async def test_auth_url_default_redirect(self, mock_auth_url, handler):
        mock_auth_url.return_value = "https://auth.example.com/"
        mock_http = MockHTTPHandler()
        result = await handler.handle(
            "/api/cloud/dropbox/auth/url",
            {},
            mock_http,
        )
        assert _status(result) == 200
        # Should use default redirect_uri
        mock_auth_url.assert_called_once_with(
            "dropbox",
            "http://localhost:3000/auth/callback",
            "",
        )

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.handle_auth_callback")
    async def test_auth_callback_default_redirect(self, mock_cb, handler):
        mock_cb.return_value = True
        mock_http = MockHTTPHandler(
            command="POST",
            body={"code": "test_code"},
        )
        result = await handler.handle_post(
            "/api/cloud/google_drive/auth/callback",
            {},
            mock_http,
        )
        assert _status(result) == 200
        mock_cb.assert_called_once_with(
            "google_drive",
            "test_code",
            "http://localhost:3000/auth/callback",
        )

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.list_files")
    async def test_list_files_with_custom_limit(self, mock_list_files, handler):
        mock_list_files.return_value = [{"id": "f1"}]
        mock_http = MockHTTPHandler()
        result = await handler.handle(
            "/api/cloud/google_drive/files",
            {"path": "/", "limit": "50"},
            mock_http,
        )
        assert _status(result) == 200
        # Verify page_size was passed correctly
        mock_list_files.assert_called_once()
        call_args = mock_list_files.call_args
        # list_files(provider, path, page_size)
        assert call_args[0][2] == 50

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.list_files")
    async def test_list_files_normalizes_slashes(self, mock_list_files, handler):
        mock_list_files.return_value = []
        mock_http = MockHTTPHandler()
        result = await handler.handle(
            "/api/cloud/google_drive/files",
            {"path": "/docs///reports"},
            mock_http,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["path"] == "/docs/reports"

    @pytest.mark.asyncio
    async def test_all_providers_for_auth_url(self, handler):
        """Ensure all OAuth-supporting providers work with the auth URL endpoint."""
        for provider in ["google_drive", "onedrive", "dropbox"]:
            with patch(
                "aragora.server.handlers.features.cloud_storage.get_auth_url",
                return_value=f"https://auth.{provider}.com/",
            ):
                mock_http = MockHTTPHandler()
                result = await handler.handle(
                    f"/api/cloud/{provider}/auth/url",
                    {"redirect_uri": "https://localhost/cb"},
                    mock_http,
                )
                assert _status(result) == 200, f"Failed for provider: {provider}"

    @pytest.mark.asyncio
    async def test_all_providers_for_files(self, handler):
        """Ensure all supported providers work with the files endpoint."""
        for provider in PROVIDERS:
            with patch(
                "aragora.server.handlers.features.cloud_storage.list_files",
                return_value=[],
            ):
                mock_http = MockHTTPHandler()
                result = await handler.handle(
                    f"/api/cloud/{provider}/files",
                    {},
                    mock_http,
                )
                assert _status(result) == 200, f"Failed for provider: {provider}"

    @pytest.mark.asyncio
    async def test_download_file_id_empty_string(self, handler):
        mock_http = MockHTTPHandler(
            command="POST",
            body={"file_id": ""},
        )
        result = await handler.handle_post(
            "/api/cloud/google_drive/download/file",
            {},
            mock_http,
        )
        # Empty file_id should fail ("Missing file_id" since empty string is falsy)
        assert _status(result) == 400

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.download_file")
    async def test_download_binary_content(self, mock_download, handler):
        content = bytes(range(256))
        mock_download.return_value = content
        mock_http = MockHTTPHandler(
            command="POST",
            body={"file_id": "binary_file"},
        )
        result = await handler.handle_post(
            "/api/cloud/google_drive/download/file",
            {},
            mock_http,
        )
        assert _status(result) == 200
        body = _body(result)
        decoded = base64.b64decode(body["content"])
        assert decoded == content
        assert body["size"] == 256

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.features.cloud_storage.list_files")
    async def test_list_files_returns_multiple(self, mock_list_files, handler):
        mock_list_files.return_value = [{"id": f"f{i}", "name": f"file{i}.txt"} for i in range(5)]
        mock_http = MockHTTPHandler()
        result = await handler.handle(
            "/api/cloud/google_drive/files",
            {},
            mock_http,
        )
        assert _status(result) == 200
        assert len(_body(result)["files"]) == 5

    @pytest.mark.asyncio
    async def test_auth_callback_redirect_uri_not_string(self, handler):
        mock_http = MockHTTPHandler(
            command="POST",
            body={"code": "valid_code", "redirect_uri": 12345},
        )
        result = await handler.handle_post(
            "/api/cloud/google_drive/auth/callback",
            {},
            mock_http,
        )
        assert _status(result) == 400
