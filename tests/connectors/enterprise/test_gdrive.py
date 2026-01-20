"""
Tests for Google Drive Enterprise Connector.

Tests the Google Drive integration including:
- OAuth2 authentication flow
- File and folder traversal
- Google Docs/Sheets/Slides export
- Incremental sync via Changes API
- Shared Drive support
"""

import base64
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.connectors.enterprise.base import SyncState, SyncStatus
from aragora.connectors.enterprise.documents.gdrive import (
    GoogleDriveConnector,
    DriveFile,
    DriveFolder,
    GOOGLE_WORKSPACE_MIMES,
    SUPPORTED_MIMES,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_credentials():
    """Mock credential provider with OAuth2 credentials."""
    from tests.connectors.enterprise.conftest import MockCredentialProvider

    return MockCredentialProvider(
        {
            "GDRIVE_CLIENT_ID": "test_client_id.apps.googleusercontent.com",
            "GDRIVE_CLIENT_SECRET": "test_client_secret",
            "GDRIVE_REFRESH_TOKEN": "test_refresh_token",
        }
    )


@pytest.fixture
def gdrive_connector(mock_credentials, tmp_path):
    """Create a Google Drive connector for testing."""
    return GoogleDriveConnector(
        folder_ids=None,
        include_shared_drives=True,
        include_trashed=False,
        export_google_docs=True,
        max_file_size_mb=100,
        credentials=mock_credentials,
        state_dir=tmp_path / "sync_state",
    )


@pytest.fixture
def sample_drive_files():
    """Sample Google Drive files."""
    return [
        {
            "id": "file1",
            "name": "Document.docx",
            "mimeType": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "size": "5000",
            "createdTime": "2024-01-15T10:00:00.000Z",
            "modifiedTime": "2024-01-16T14:30:00.000Z",
            "webViewLink": "https://drive.google.com/file/d/file1/view",
            "parents": ["folder1"],
            "owners": [{"displayName": "Alice Smith", "emailAddress": "alice@example.com"}],
            "shared": False,
        },
        {
            "id": "file2",
            "name": "Spreadsheet",
            "mimeType": "application/vnd.google-apps.spreadsheet",
            "size": "0",
            "createdTime": "2024-01-14T09:00:00.000Z",
            "modifiedTime": "2024-01-17T11:00:00.000Z",
            "webViewLink": "https://docs.google.com/spreadsheets/d/file2/edit",
            "parents": ["root"],
            "owners": [{"displayName": "Bob Jones"}],
            "shared": True,
        },
        {
            "id": "file3",
            "name": "Presentation.pdf",
            "mimeType": "application/pdf",
            "size": "120000",
            "createdTime": "2024-01-13T08:00:00.000Z",
            "modifiedTime": "2024-01-18T09:00:00.000Z",
            "webViewLink": "https://drive.google.com/file/d/file3/view",
            "parents": ["folder2"],
            "owners": [],
            "shared": True,
            "driveId": "shared_drive_1",
        },
    ]


@pytest.fixture
def sample_shared_drives():
    """Sample shared drives."""
    return [
        {"id": "shared_drive_1", "name": "Engineering"},
        {"id": "shared_drive_2", "name": "Marketing"},
    ]


@pytest.fixture
def sample_changes():
    """Sample changes from Changes API."""
    return [
        {
            "kind": "drive#change",
            "type": "file",
            "fileId": "file4",
            "removed": False,
            "file": {
                "id": "file4",
                "name": "NewDoc.md",
                "mimeType": "text/markdown",
                "size": "1500",
                "createdTime": "2024-01-19T10:00:00.000Z",
                "modifiedTime": "2024-01-19T10:00:00.000Z",
                "webViewLink": "https://drive.google.com/file/d/file4/view",
                "parents": ["root"],
                "owners": [{"displayName": "Charlie Brown"}],
            },
        },
        {
            "kind": "drive#change",
            "type": "file",
            "fileId": "file5",
            "removed": True,  # Deleted file
        },
    ]


# =============================================================================
# Initialization Tests
# =============================================================================


class TestGoogleDriveConnectorInit:
    """Test GoogleDriveConnector initialization."""

    def test_init_with_defaults(self, mock_credentials, tmp_path):
        """Test initialization with default options."""
        connector = GoogleDriveConnector(
            credentials=mock_credentials,
            state_dir=tmp_path,
        )
        assert connector.folder_ids is None
        assert connector.include_shared_drives is True
        assert connector.include_trashed is False
        assert connector.export_google_docs is True
        assert connector.max_file_size_bytes == 100 * 1024 * 1024
        assert connector.connector_id == "gdrive"

    def test_init_with_custom_options(self, mock_credentials, tmp_path):
        """Test initialization with custom options."""
        connector = GoogleDriveConnector(
            folder_ids=["folder1", "folder2"],
            include_shared_drives=False,
            include_trashed=True,
            export_google_docs=False,
            max_file_size_mb=50,
            exclude_patterns=["backup_", ".tmp"],
            credentials=mock_credentials,
            state_dir=tmp_path,
        )
        assert connector.folder_ids == ["folder1", "folder2"]
        assert connector.include_shared_drives is False
        assert connector.include_trashed is True
        assert connector.export_google_docs is False
        assert connector.max_file_size_bytes == 50 * 1024 * 1024
        assert "backup_" in connector.exclude_patterns
        assert ".tmp" in connector.exclude_patterns

    def test_source_type(self, gdrive_connector):
        """Test source type property."""
        from aragora.reasoning.provenance import SourceType

        assert gdrive_connector.source_type == SourceType.DOCUMENT

    def test_name_property(self, gdrive_connector):
        """Test name property."""
        assert gdrive_connector.name == "Google Drive"


# =============================================================================
# OAuth2 Access Token Tests
# =============================================================================


class TestAccessToken:
    """Test OAuth2 access token management."""

    @pytest.mark.asyncio
    async def test_get_access_token_fresh(self, gdrive_connector):
        """Test getting a fresh access token."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "new_access_token",
            "expires_in": 3600,
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            MockClient.return_value.__aenter__.return_value = mock_client

            token = await gdrive_connector._get_access_token()

            assert token == "new_access_token"
            assert gdrive_connector._access_token == "new_access_token"
            assert gdrive_connector._token_expiry is not None

    @pytest.mark.asyncio
    async def test_get_access_token_cached(self, gdrive_connector):
        """Test using cached access token."""
        gdrive_connector._access_token = "cached_token"
        gdrive_connector._token_expiry = datetime.now(timezone.utc) + timedelta(hours=1)

        token = await gdrive_connector._get_access_token()

        assert token == "cached_token"

    @pytest.mark.asyncio
    async def test_get_access_token_expired(self, gdrive_connector):
        """Test refreshing expired access token."""
        gdrive_connector._access_token = "expired_token"
        gdrive_connector._token_expiry = datetime.now(timezone.utc) - timedelta(hours=1)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "refreshed_token",
            "expires_in": 3600,
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            MockClient.return_value.__aenter__.return_value = mock_client

            token = await gdrive_connector._get_access_token()

            assert token == "refreshed_token"

    @pytest.mark.asyncio
    async def test_get_access_token_missing_credentials(self, tmp_path):
        """Test error when credentials are missing."""
        from tests.connectors.enterprise.conftest import MockCredentialProvider

        empty_credentials = MockCredentialProvider({})

        connector = GoogleDriveConnector(
            credentials=empty_credentials,
            state_dir=tmp_path,
        )

        with pytest.raises(ValueError, match="credentials not configured"):
            await connector._get_access_token()


# =============================================================================
# API Request Tests
# =============================================================================


class TestApiRequest:
    """Test Google Drive API requests."""

    @pytest.mark.asyncio
    async def test_api_request_success(self, gdrive_connector):
        """Test successful API request."""
        gdrive_connector._access_token = "test_token"
        gdrive_connector._token_expiry = datetime.now(timezone.utc) + timedelta(hours=1)

        mock_response = MagicMock()
        mock_response.json.return_value = {"files": []}
        mock_response.content = b'{"files": []}'
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.request.return_value = mock_response
            MockClient.return_value.__aenter__.return_value = mock_client

            result = await gdrive_connector._api_request("/files")

            assert result == {"files": []}
            mock_client.request.assert_called_once()

    @pytest.mark.asyncio
    async def test_api_request_with_params(self, gdrive_connector):
        """Test API request with query parameters."""
        gdrive_connector._access_token = "test_token"
        gdrive_connector._token_expiry = datetime.now(timezone.utc) + timedelta(hours=1)

        mock_response = MagicMock()
        mock_response.json.return_value = {"files": []}
        mock_response.content = b'{"files": []}'
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.request.return_value = mock_response
            MockClient.return_value.__aenter__.return_value = mock_client

            await gdrive_connector._api_request(
                "/files",
                params={"q": "name contains 'test'", "pageSize": 10},
            )

            call_kwargs = mock_client.request.call_args[1]
            assert call_kwargs["params"]["q"] == "name contains 'test'"
            assert call_kwargs["params"]["pageSize"] == 10


# =============================================================================
# File Download Tests
# =============================================================================


class TestDownloadFile:
    """Test file download functionality."""

    @pytest.mark.asyncio
    async def test_download_regular_file(self, gdrive_connector):
        """Test downloading a regular file."""
        gdrive_connector._access_token = "test_token"
        gdrive_connector._token_expiry = datetime.now(timezone.utc) + timedelta(hours=1)

        mock_response = MagicMock()
        mock_response.content = b"File content here"
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            MockClient.return_value.__aenter__.return_value = mock_client

            content = await gdrive_connector._download_file("file123")

            assert content == b"File content here"
            # Verify alt=media is used for regular files
            call_kwargs = mock_client.get.call_args[1]
            assert call_kwargs["params"]["alt"] == "media"

    @pytest.mark.asyncio
    async def test_download_google_doc_export(self, gdrive_connector):
        """Test exporting Google Doc to text."""
        gdrive_connector._access_token = "test_token"
        gdrive_connector._token_expiry = datetime.now(timezone.utc) + timedelta(hours=1)

        mock_response = MagicMock()
        mock_response.content = b"Exported document content"
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            MockClient.return_value.__aenter__.return_value = mock_client

            content = await gdrive_connector._download_file(
                "doc123",
                mime_type="application/vnd.google-apps.document",
            )

            assert content == b"Exported document content"
            # Verify export endpoint is used with correct MIME type
            call_args = mock_client.get.call_args
            assert "/export" in call_args[0][0]
            assert call_args[1]["params"]["mimeType"] == "text/plain"

    @pytest.mark.asyncio
    async def test_download_google_sheet_export(self, gdrive_connector):
        """Test exporting Google Sheet to CSV."""
        gdrive_connector._access_token = "test_token"
        gdrive_connector._token_expiry = datetime.now(timezone.utc) + timedelta(hours=1)

        mock_response = MagicMock()
        mock_response.content = b"col1,col2\nval1,val2"
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            MockClient.return_value.__aenter__.return_value = mock_client

            content = await gdrive_connector._download_file(
                "sheet123",
                mime_type="application/vnd.google-apps.spreadsheet",
            )

            # Verify CSV export MIME type
            call_args = mock_client.get.call_args
            assert call_args[1]["params"]["mimeType"] == "text/csv"


# =============================================================================
# File Listing Tests
# =============================================================================


class TestListFiles:
    """Test file listing functionality."""

    @pytest.mark.asyncio
    async def test_list_files_success(self, gdrive_connector, sample_drive_files):
        """Test listing files from Drive."""
        gdrive_connector._access_token = "test_token"
        gdrive_connector._token_expiry = datetime.now(timezone.utc) + timedelta(hours=1)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "files": sample_drive_files,
            "nextPageToken": None,
        }
        mock_response.content = b"{}"
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.request.return_value = mock_response
            MockClient.return_value.__aenter__.return_value = mock_client

            files, next_token = await gdrive_connector._list_files()

            assert len(files) == 3
            assert files[0].id == "file1"
            assert files[0].name == "Document.docx"
            assert files[1].mime_type == "application/vnd.google-apps.spreadsheet"
            assert files[2].drive_id == "shared_drive_1"
            assert next_token is None

    @pytest.mark.asyncio
    async def test_list_files_with_folder_filter(self, gdrive_connector):
        """Test listing files in a specific folder."""
        gdrive_connector._access_token = "test_token"
        gdrive_connector._token_expiry = datetime.now(timezone.utc) + timedelta(hours=1)

        mock_response = MagicMock()
        mock_response.json.return_value = {"files": [], "nextPageToken": None}
        mock_response.content = b"{}"
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.request.return_value = mock_response
            MockClient.return_value.__aenter__.return_value = mock_client

            await gdrive_connector._list_files(folder_id="folder123")

            call_kwargs = mock_client.request.call_args[1]
            assert "'folder123' in parents" in call_kwargs["params"]["q"]

    @pytest.mark.asyncio
    async def test_list_files_pagination(self, gdrive_connector):
        """Test file listing pagination."""
        gdrive_connector._access_token = "test_token"
        gdrive_connector._token_expiry = datetime.now(timezone.utc) + timedelta(hours=1)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "files": [{"id": "file1", "name": "Test", "mimeType": "text/plain"}],
            "nextPageToken": "next_page_123",
        }
        mock_response.content = b"{}"
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.request.return_value = mock_response
            MockClient.return_value.__aenter__.return_value = mock_client

            files, next_token = await gdrive_connector._list_files()

            assert next_token == "next_page_123"


# =============================================================================
# Shared Drives Tests
# =============================================================================


class TestSharedDrives:
    """Test Shared Drives functionality."""

    @pytest.mark.asyncio
    async def test_list_shared_drives(self, gdrive_connector, sample_shared_drives):
        """Test listing shared drives."""
        gdrive_connector._access_token = "test_token"
        gdrive_connector._token_expiry = datetime.now(timezone.utc) + timedelta(hours=1)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "drives": sample_shared_drives,
            "nextPageToken": None,
        }
        mock_response.content = b"{}"
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.request.return_value = mock_response
            MockClient.return_value.__aenter__.return_value = mock_client

            drives = await gdrive_connector._list_shared_drives()

            assert len(drives) == 2
            assert drives[0]["id"] == "shared_drive_1"
            assert drives[1]["name"] == "Marketing"

    @pytest.mark.asyncio
    async def test_list_shared_drives_disabled(self, mock_credentials, tmp_path):
        """Test shared drives listing when disabled."""
        connector = GoogleDriveConnector(
            include_shared_drives=False,
            credentials=mock_credentials,
            state_dir=tmp_path,
        )

        drives = await connector._list_shared_drives()
        assert drives == []


# =============================================================================
# Changes API Tests
# =============================================================================


class TestChangesApi:
    """Test Changes API for incremental sync."""

    @pytest.mark.asyncio
    async def test_get_changes(self, gdrive_connector, sample_changes):
        """Test getting changes since a token."""
        gdrive_connector._access_token = "test_token"
        gdrive_connector._token_expiry = datetime.now(timezone.utc) + timedelta(hours=1)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "changes": sample_changes,
            "newStartPageToken": "new_token_456",
        }
        mock_response.content = b"{}"
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.request.return_value = mock_response
            MockClient.return_value.__aenter__.return_value = mock_client

            files, new_token = await gdrive_connector._get_changes("start_token_123")

            # Should only get non-deleted files
            assert len(files) == 1
            assert files[0].id == "file4"
            assert files[0].name == "NewDoc.md"
            assert new_token == "new_token_456"

    @pytest.mark.asyncio
    async def test_get_start_page_token(self, gdrive_connector):
        """Test getting initial page token for changes."""
        gdrive_connector._access_token = "test_token"
        gdrive_connector._token_expiry = datetime.now(timezone.utc) + timedelta(hours=1)

        mock_response = MagicMock()
        mock_response.json.return_value = {"startPageToken": "initial_token_abc"}
        mock_response.content = b"{}"
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.request.return_value = mock_response
            MockClient.return_value.__aenter__.return_value = mock_client

            token = await gdrive_connector._get_start_page_token()

            assert token == "initial_token_abc"


# =============================================================================
# File Filtering Tests
# =============================================================================


class TestFileFiltering:
    """Test file filtering logic."""

    def test_should_skip_folder(self, gdrive_connector):
        """Test folders are skipped."""
        folder = DriveFile(
            id="folder1",
            name="My Folder",
            mime_type="application/vnd.google-apps.folder",
        )
        assert gdrive_connector._should_skip_file(folder) is True

    def test_should_skip_large_file(self, gdrive_connector):
        """Test files over size limit are skipped."""
        large_file = DriveFile(
            id="large1",
            name="BigFile.zip",
            mime_type="application/zip",
            size=200 * 1024 * 1024,  # 200MB
        )
        assert gdrive_connector._should_skip_file(large_file) is True

    def test_should_skip_excluded_pattern(self, mock_credentials, tmp_path):
        """Test excluded patterns are skipped."""
        connector = GoogleDriveConnector(
            exclude_patterns=["backup_", "_temp"],
            credentials=mock_credentials,
            state_dir=tmp_path,
        )

        backup_file = DriveFile(
            id="backup1",
            name="backup_2024_01.zip",
            mime_type="text/plain",
            size=1000,
        )
        assert connector._should_skip_file(backup_file) is True

        temp_file = DriveFile(
            id="temp1",
            name="data_temp.txt",
            mime_type="text/plain",
            size=1000,
        )
        assert connector._should_skip_file(temp_file) is True

    def test_should_skip_unsupported_mime(self, gdrive_connector):
        """Test unsupported MIME types are skipped."""
        binary_file = DriveFile(
            id="bin1",
            name="program.exe",
            mime_type="application/x-executable",
            size=5000,
        )
        assert gdrive_connector._should_skip_file(binary_file) is True

    def test_should_not_skip_supported_file(self, gdrive_connector):
        """Test supported files are not skipped."""
        text_file = DriveFile(
            id="txt1",
            name="document.txt",
            mime_type="text/plain",
            size=5000,
        )
        assert gdrive_connector._should_skip_file(text_file) is False

        google_doc = DriveFile(
            id="doc1",
            name="My Document",
            mime_type="application/vnd.google-apps.document",
            size=0,  # Google Docs have size 0
        )
        assert gdrive_connector._should_skip_file(google_doc) is False

    def test_should_not_skip_google_workspace_file(self, gdrive_connector):
        """Test Google Workspace files are not skipped."""
        for mime_type in GOOGLE_WORKSPACE_MIMES.keys():
            file = DriveFile(
                id="workspace1",
                name="Workspace File",
                mime_type=mime_type,
                size=0,
            )
            assert gdrive_connector._should_skip_file(file) is False


# =============================================================================
# Text Extraction Tests
# =============================================================================


class TestTextExtraction:
    """Test text extraction from files."""

    @pytest.mark.asyncio
    async def test_extract_text_plain(self, gdrive_connector):
        """Test extracting text from plain text file."""
        file = DriveFile(
            id="txt1",
            name="document.txt",
            mime_type="text/plain",
        )

        with patch.object(
            gdrive_connector,
            "_download_file",
            new_callable=AsyncMock,
            return_value=b"Plain text content",
        ):
            text = await gdrive_connector._extract_text(file)
            assert text == "Plain text content"

    @pytest.mark.asyncio
    async def test_extract_text_google_doc(self, gdrive_connector):
        """Test extracting text from Google Doc."""
        file = DriveFile(
            id="doc1",
            name="My Document",
            mime_type="application/vnd.google-apps.document",
        )

        with patch.object(
            gdrive_connector,
            "_download_file",
            new_callable=AsyncMock,
            return_value=b"Exported document text",
        ):
            text = await gdrive_connector._extract_text(file)
            assert text == "Exported document text"

    @pytest.mark.asyncio
    async def test_extract_text_pdf_placeholder(self, gdrive_connector):
        """Test PDF extraction returns placeholder."""
        file = DriveFile(
            id="pdf1",
            name="report.pdf",
            mime_type="application/pdf",
        )

        with patch.object(
            gdrive_connector,
            "_download_file",
            new_callable=AsyncMock,
            return_value=b"%PDF-1.4 binary content",
        ):
            text = await gdrive_connector._extract_text(file)
            assert "PDF content from report.pdf" in text

    @pytest.mark.asyncio
    async def test_extract_text_error_handling(self, gdrive_connector):
        """Test error handling during text extraction."""
        file = DriveFile(
            id="err1",
            name="error.txt",
            mime_type="text/plain",
        )

        with patch.object(
            gdrive_connector,
            "_download_file",
            new_callable=AsyncMock,
            side_effect=Exception("Download failed"),
        ):
            text = await gdrive_connector._extract_text(file)
            assert text == ""


# =============================================================================
# Sync Tests
# =============================================================================


class TestSync:
    """Test sync_items functionality."""

    @pytest.mark.asyncio
    async def test_sync_items_full_sync(self, gdrive_connector, sample_drive_files):
        """Test full sync when no cursor exists."""
        state = SyncState(connector_id="gdrive", status=SyncStatus.IDLE)

        # Mock API responses
        async def mock_api_request(endpoint, **kwargs):
            if "/changes/startPageToken" in endpoint:
                return {"startPageToken": "initial_token"}
            elif "/files" in endpoint:
                return {"files": sample_drive_files[:1], "nextPageToken": None}
            elif "/drives" in endpoint:
                return {"drives": [], "nextPageToken": None}
            return {}

        with patch.object(gdrive_connector, "_api_request", side_effect=mock_api_request):
            with patch.object(
                gdrive_connector,
                "_extract_text",
                new_callable=AsyncMock,
                return_value="Document content",
            ):
                items = []
                async for item in gdrive_connector.sync_items(state, batch_size=10):
                    items.append(item)

        assert len(items) >= 1
        assert state.cursor == "initial_token"

    @pytest.mark.asyncio
    async def test_sync_items_incremental(self, gdrive_connector, sample_changes):
        """Test incremental sync using Changes API."""
        state = SyncState(
            connector_id="gdrive",
            cursor="existing_token_123",
            status=SyncStatus.IDLE,
        )

        mock_response = {
            "changes": sample_changes,
            "newStartPageToken": "new_token_456",
        }

        with patch.object(gdrive_connector, "_api_request", return_value=mock_response):
            with patch.object(
                gdrive_connector,
                "_extract_text",
                new_callable=AsyncMock,
                return_value="Changed file content",
            ):
                items = []
                async for item in gdrive_connector.sync_items(state, batch_size=10):
                    items.append(item)

        assert len(items) == 1  # Only non-deleted change
        assert items[0].id == "gdrive-file4"
        assert state.cursor == "new_token_456"

    @pytest.mark.asyncio
    async def test_sync_items_metadata(self, gdrive_connector, sample_drive_files):
        """Test sync item metadata is correct."""
        state = SyncState(connector_id="gdrive", status=SyncStatus.IDLE)

        async def mock_api_request(endpoint, **kwargs):
            if "/changes/startPageToken" in endpoint:
                return {"startPageToken": "token"}
            elif "/files" in endpoint:
                return {"files": [sample_drive_files[0]], "nextPageToken": None}
            elif "/drives" in endpoint:
                return {"drives": [], "nextPageToken": None}
            return {}

        with patch.object(gdrive_connector, "_api_request", side_effect=mock_api_request):
            with patch.object(
                gdrive_connector,
                "_extract_text",
                new_callable=AsyncMock,
                return_value="Content",
            ):
                items = []
                async for item in gdrive_connector.sync_items(state, batch_size=10):
                    items.append(item)

        if items:
            item = items[0]
            assert item.source_type == "document"
            assert item.domain == "enterprise/gdrive"
            assert item.metadata.get("file_id") == "file1"
            assert "mime_type" in item.metadata


# =============================================================================
# Search Tests
# =============================================================================


class TestSearch:
    """Test search functionality."""

    @pytest.mark.asyncio
    async def test_search_success(self, gdrive_connector, sample_drive_files):
        """Test successful search."""
        gdrive_connector._access_token = "test_token"
        gdrive_connector._token_expiry = datetime.now(timezone.utc) + timedelta(hours=1)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "files": [
                {
                    "id": "file1",
                    "name": "Document.docx",
                    "mimeType": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "webViewLink": "https://drive.google.com/file/d/file1/view",
                }
            ]
        }
        mock_response.content = b"{}"
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.request.return_value = mock_response
            MockClient.return_value.__aenter__.return_value = mock_client

            results = await gdrive_connector.search("quarterly report", limit=5)

            assert len(results) == 1
            assert results[0].id == "gdrive-file1"
            assert results[0].title == "Document.docx"

    @pytest.mark.asyncio
    async def test_search_with_folder(self, gdrive_connector):
        """Test search within a specific folder."""
        gdrive_connector._access_token = "test_token"
        gdrive_connector._token_expiry = datetime.now(timezone.utc) + timedelta(hours=1)

        mock_response = MagicMock()
        mock_response.json.return_value = {"files": []}
        mock_response.content = b"{}"
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.request.return_value = mock_response
            MockClient.return_value.__aenter__.return_value = mock_client

            await gdrive_connector.search("query", folder_id="folder123")

            call_kwargs = mock_client.request.call_args[1]
            assert "'folder123' in parents" in call_kwargs["params"]["q"]

    @pytest.mark.asyncio
    async def test_search_error_handling(self, gdrive_connector):
        """Test search error handling."""
        gdrive_connector._access_token = "test_token"
        gdrive_connector._token_expiry = datetime.now(timezone.utc) + timedelta(hours=1)

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.request.side_effect = Exception("API Error")
            MockClient.return_value.__aenter__.return_value = mock_client

            results = await gdrive_connector.search("test")

            assert results == []


# =============================================================================
# Fetch Tests
# =============================================================================


class TestFetch:
    """Test fetch functionality."""

    @pytest.mark.asyncio
    async def test_fetch_success(self, gdrive_connector):
        """Test successful file fetch."""
        gdrive_connector._access_token = "test_token"
        gdrive_connector._token_expiry = datetime.now(timezone.utc) + timedelta(hours=1)

        file_data = {
            "id": "file123",
            "name": "test.txt",
            "mimeType": "text/plain",
            "size": "1000",
            "webViewLink": "https://drive.google.com/file/d/file123/view",
            "owners": [{"displayName": "Test User"}],
        }

        mock_response = MagicMock()
        mock_response.json.return_value = file_data
        mock_response.content = b"{}"
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.request.return_value = mock_response
            mock_client.get.return_value = MagicMock(content=b"File content")
            MockClient.return_value.__aenter__.return_value = mock_client

            with patch.object(
                gdrive_connector,
                "_extract_text",
                new_callable=AsyncMock,
                return_value="File content",
            ):
                evidence = await gdrive_connector.fetch("gdrive-file123")

                assert evidence is not None
                assert evidence.id == "gdrive-file123"
                assert evidence.title == "test.txt"
                assert evidence.content == "File content"

    @pytest.mark.asyncio
    async def test_fetch_with_raw_id(self, gdrive_connector):
        """Test fetch with raw file ID (no prefix)."""
        gdrive_connector._access_token = "test_token"
        gdrive_connector._token_expiry = datetime.now(timezone.utc) + timedelta(hours=1)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "rawid123",
            "name": "file.txt",
            "mimeType": "text/plain",
            "owners": [],
        }
        mock_response.content = b"{}"
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.request.return_value = mock_response
            MockClient.return_value.__aenter__.return_value = mock_client

            with patch.object(
                gdrive_connector,
                "_extract_text",
                new_callable=AsyncMock,
                return_value="Content",
            ):
                evidence = await gdrive_connector.fetch("rawid123")

                assert evidence is not None

    @pytest.mark.asyncio
    async def test_fetch_error(self, gdrive_connector):
        """Test fetch error handling."""
        gdrive_connector._access_token = "test_token"
        gdrive_connector._token_expiry = datetime.now(timezone.utc) + timedelta(hours=1)

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.request.side_effect = Exception("Not found")
            MockClient.return_value.__aenter__.return_value = mock_client

            evidence = await gdrive_connector.fetch("nonexistent")

            assert evidence is None


# =============================================================================
# Webhook Tests
# =============================================================================


class TestWebhook:
    """Test webhook handling."""

    @pytest.mark.asyncio
    async def test_handle_webhook_change(self, gdrive_connector):
        """Test handling change webhook."""
        payload = {
            "resourceState": "change",
            "resourceId": "file123",
        }

        with patch.object(gdrive_connector, "sync", new_callable=AsyncMock):
            result = await gdrive_connector.handle_webhook(payload)
            assert result is True

    @pytest.mark.asyncio
    async def test_handle_webhook_add(self, gdrive_connector):
        """Test handling add webhook."""
        payload = {
            "resourceState": "add",
            "resourceId": "newfile",
        }

        with patch.object(gdrive_connector, "sync", new_callable=AsyncMock):
            result = await gdrive_connector.handle_webhook(payload)
            assert result is True

    @pytest.mark.asyncio
    async def test_handle_webhook_unknown(self, gdrive_connector):
        """Test handling unknown webhook state."""
        payload = {
            "resourceState": "trash",
            "resourceId": "file123",
        }

        result = await gdrive_connector.handle_webhook(payload)
        assert result is False


# =============================================================================
# MIME Type Constants Tests
# =============================================================================


class TestMimeTypeConstants:
    """Test MIME type constants."""

    def test_google_workspace_mimes_mapping(self):
        """Test Google Workspace MIME type export mappings."""
        assert GOOGLE_WORKSPACE_MIMES["application/vnd.google-apps.document"] == "text/plain"
        assert GOOGLE_WORKSPACE_MIMES["application/vnd.google-apps.spreadsheet"] == "text/csv"
        assert GOOGLE_WORKSPACE_MIMES["application/vnd.google-apps.presentation"] == "text/plain"

    def test_supported_mimes_includes_common_types(self):
        """Test supported MIME types include common document types."""
        assert "text/plain" in SUPPORTED_MIMES
        assert "application/pdf" in SUPPORTED_MIMES
        assert "application/json" in SUPPORTED_MIMES
        assert (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            in SUPPORTED_MIMES
        )
