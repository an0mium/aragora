"""
Tests for TeamsFilesMixin - Microsoft Teams file operations.

Tests cover:
- File upload (small files via direct upload)
- File upload (large files via upload session)
- File download
- Error handling (timeout, connection, auth errors)
- Missing parameters handling
- Circuit breaker integration
- Graph API integration
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.connectors.chat.models import FileAttachment


class MockTeamsConnector:
    """Mock connector implementing the protocol required by TeamsFilesMixin."""

    def __init__(self):
        self._upload_timeout = 120.0
        self._graph_api_request_mock = AsyncMock()
        self._http_request_mock = AsyncMock()
        self._record_failure_mock = MagicMock()

    async def _graph_api_request(
        self,
        endpoint,
        method="GET",
        operation=None,
        json_data=None,
        data=None,
        content_type=None,
        params=None,
        use_full_url=False,
    ):
        return await self._graph_api_request_mock(
            endpoint=endpoint,
            method=method,
            operation=operation,
            json_data=json_data,
            data=data,
            content_type=content_type,
            params=params,
            use_full_url=use_full_url,
        )

    async def _http_request(
        self,
        method,
        url,
        headers=None,
        content=None,
        timeout=None,
        return_raw=False,
        operation=None,
    ):
        return await self._http_request_mock(
            method=method,
            url=url,
            headers=headers,
            content=content,
            timeout=timeout,
            return_raw=return_raw,
            operation=operation,
        )

    def _record_failure(self, error=None):
        self._record_failure_mock(error)


# =============================================================================
# Upload File Tests
# =============================================================================


class TestUploadFile:
    """Tests for upload_file method."""

    @pytest.mark.asyncio
    async def test_upload_small_file_success(self):
        """Should upload a small file (<4MB) via direct upload."""
        from aragora.connectors.chat.teams._files import TeamsFilesMixin

        class TestConnector(TeamsFilesMixin, MockTeamsConnector):
            pass

        connector = TestConnector()

        # Mock filesFolder response
        folder_data = {
            "id": "folder-123",
            "parentReference": {"driveId": "drive-456"},
        }

        # Mock upload response
        upload_data = {
            "id": "file-789",
            "webUrl": "https://sharepoint.com/files/test.txt",
        }

        connector._graph_api_request_mock.side_effect = [
            (True, folder_data, None),
            (True, upload_data, None),
        ]

        result = await connector.upload_file(
            channel_id="channel-abc",
            content=b"Hello, World!",
            filename="test.txt",
            content_type="text/plain",
            team_id="team-xyz",
        )

        assert result.id == "file-789"
        assert result.filename == "test.txt"
        assert result.content_type == "text/plain"
        assert result.size == 13
        assert result.url == "https://sharepoint.com/files/test.txt"
        assert result.metadata["drive_id"] == "drive-456"

    @pytest.mark.asyncio
    async def test_upload_small_file_with_kwargs_team_id(self):
        """Should extract team_id from kwargs if not provided directly."""
        from aragora.connectors.chat.teams._files import TeamsFilesMixin

        class TestConnector(TeamsFilesMixin, MockTeamsConnector):
            pass

        connector = TestConnector()

        folder_data = {
            "id": "folder-123",
            "parentReference": {"driveId": "drive-456"},
        }
        upload_data = {"id": "file-789", "webUrl": "https://sp.com/file"}

        connector._graph_api_request_mock.side_effect = [
            (True, folder_data, None),
            (True, upload_data, None),
        ]

        result = await connector.upload_file(
            channel_id="channel-abc",
            content=b"Content",
            filename="doc.txt",
            service_url="https://example.com",  # In kwargs for other purposes
            **{"team_id": "team-from-kwargs"},
        )

        assert result.id == "file-789"
        # Verify the endpoint used the team_id
        call_args = connector._graph_api_request_mock.call_args_list[0]
        assert "team-from-kwargs" in call_args.kwargs["endpoint"]

    @pytest.mark.asyncio
    async def test_upload_file_missing_team_id(self):
        """Should return error when team_id is not provided."""
        from aragora.connectors.chat.teams._files import TeamsFilesMixin

        class TestConnector(TeamsFilesMixin, MockTeamsConnector):
            pass

        connector = TestConnector()

        result = await connector.upload_file(
            channel_id="channel-abc",
            content=b"Hello",
            filename="test.txt",
            # team_id not provided
        )

        assert result.id == ""
        assert "team_id required" in result.metadata.get("error", "")
        assert result.content == b"Hello"

    @pytest.mark.asyncio
    async def test_upload_file_folder_fetch_failure(self):
        """Should return error when filesFolder fetch fails."""
        from aragora.connectors.chat.teams._files import TeamsFilesMixin

        class TestConnector(TeamsFilesMixin, MockTeamsConnector):
            pass

        connector = TestConnector()
        connector._graph_api_request_mock.return_value = (
            False,
            None,
            "Folder not found",
        )

        result = await connector.upload_file(
            channel_id="channel-abc",
            content=b"Content",
            filename="test.txt",
            team_id="team-xyz",
        )

        assert result.id == ""
        assert "Folder not found" in result.metadata.get("error", "")

    @pytest.mark.asyncio
    async def test_upload_file_missing_drive_id(self):
        """Should return error when drive ID is missing from folder response."""
        from aragora.connectors.chat.teams._files import TeamsFilesMixin

        class TestConnector(TeamsFilesMixin, MockTeamsConnector):
            pass

        connector = TestConnector()

        folder_data = {
            "id": "folder-123",
            "parentReference": {},  # driveId missing
        }
        connector._graph_api_request_mock.return_value = (True, folder_data, None)

        result = await connector.upload_file(
            channel_id="channel-abc",
            content=b"Content",
            filename="test.txt",
            team_id="team-xyz",
        )

        assert result.id == ""
        assert "Missing drive/folder IDs" in result.metadata.get("error", "")

    @pytest.mark.asyncio
    async def test_upload_file_direct_upload_failure(self):
        """Should return error when direct upload fails."""
        from aragora.connectors.chat.teams._files import TeamsFilesMixin

        class TestConnector(TeamsFilesMixin, MockTeamsConnector):
            pass

        connector = TestConnector()

        folder_data = {
            "id": "folder-123",
            "parentReference": {"driveId": "drive-456"},
        }

        connector._graph_api_request_mock.side_effect = [
            (True, folder_data, None),
            (False, None, "Upload failed"),
        ]

        result = await connector.upload_file(
            channel_id="channel-abc",
            content=b"Content",
            filename="test.txt",
            team_id="team-xyz",
        )

        assert result.id == ""
        assert "Upload failed" in result.metadata.get("error", "")

    @pytest.mark.asyncio
    async def test_upload_large_file_via_session(self):
        """Should upload large file (>=4MB) via upload session."""
        from aragora.connectors.chat.teams._files import TeamsFilesMixin

        class TestConnector(TeamsFilesMixin, MockTeamsConnector):
            pass

        connector = TestConnector()

        # Create a 5MB file content
        large_content = b"x" * (5 * 1024 * 1024)

        folder_data = {
            "id": "folder-123",
            "parentReference": {"driveId": "drive-456"},
        }
        session_data = {
            "uploadUrl": "https://graph.microsoft.com/upload-session/xyz",
        }
        upload_result = {
            "id": "large-file-789",
            "webUrl": "https://sp.com/large-file",
        }

        connector._graph_api_request_mock.side_effect = [
            (True, folder_data, None),  # Get folder
            (True, session_data, None),  # Create session
        ]
        connector._http_request_mock.return_value = (True, upload_result, None)

        result = await connector.upload_file(
            channel_id="channel-abc",
            content=large_content,
            filename="large.bin",
            content_type="application/octet-stream",
            team_id="team-xyz",
        )

        assert result.id == "large-file-789"
        assert result.size == 5 * 1024 * 1024
        # Verify upload session was used
        call_args = connector._http_request_mock.call_args
        assert call_args.kwargs["url"] == "https://graph.microsoft.com/upload-session/xyz"

    @pytest.mark.asyncio
    async def test_upload_large_file_session_creation_failure(self):
        """Should return error when upload session creation fails."""
        from aragora.connectors.chat.teams._files import TeamsFilesMixin

        class TestConnector(TeamsFilesMixin, MockTeamsConnector):
            pass

        connector = TestConnector()

        large_content = b"x" * (5 * 1024 * 1024)

        folder_data = {
            "id": "folder-123",
            "parentReference": {"driveId": "drive-456"},
        }

        connector._graph_api_request_mock.side_effect = [
            (True, folder_data, None),
            (False, None, "Session creation failed"),
        ]

        result = await connector.upload_file(
            channel_id="channel-abc",
            content=large_content,
            filename="large.bin",
            team_id="team-xyz",
        )

        assert result.id == ""
        assert "Session creation failed" in result.metadata.get("error", "")

    @pytest.mark.asyncio
    async def test_upload_large_file_upload_failure(self):
        """Should return error when large file upload fails."""
        from aragora.connectors.chat.teams._files import TeamsFilesMixin

        class TestConnector(TeamsFilesMixin, MockTeamsConnector):
            pass

        connector = TestConnector()

        large_content = b"x" * (5 * 1024 * 1024)

        folder_data = {
            "id": "folder-123",
            "parentReference": {"driveId": "drive-456"},
        }
        session_data = {
            "uploadUrl": "https://graph.microsoft.com/upload-session/xyz",
        }

        connector._graph_api_request_mock.side_effect = [
            (True, folder_data, None),
            (True, session_data, None),
        ]
        connector._http_request_mock.return_value = (False, None, "Upload failed")

        result = await connector.upload_file(
            channel_id="channel-abc",
            content=large_content,
            filename="large.bin",
            team_id="team-xyz",
        )

        assert result.id == ""
        assert "Upload failed" in result.metadata.get("error", "")

    @pytest.mark.asyncio
    async def test_upload_file_timeout_error(self):
        """Should handle timeout errors gracefully."""
        from aragora.connectors.chat.teams._files import TeamsFilesMixin

        class TestConnector(TeamsFilesMixin, MockTeamsConnector):
            pass

        connector = TestConnector()

        with patch("aragora.connectors.chat.teams._constants.HTTPX_AVAILABLE", True):
            import httpx

            connector._graph_api_request_mock.side_effect = httpx.TimeoutException(
                "Request timed out"
            )

            result = await connector.upload_file(
                channel_id="channel-abc",
                content=b"Content",
                filename="test.txt",
                team_id="team-xyz",
            )

            assert result.id == ""
            assert "timed out" in result.metadata.get("error", "").lower()
            connector._record_failure_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_file_connection_error(self):
        """Should handle connection errors gracefully."""
        from aragora.connectors.chat.teams._files import TeamsFilesMixin

        class TestConnector(TeamsFilesMixin, MockTeamsConnector):
            pass

        connector = TestConnector()

        with patch("aragora.connectors.chat.teams._constants.HTTPX_AVAILABLE", True):
            import httpx

            connector._graph_api_request_mock.side_effect = httpx.ConnectError("Connection refused")

            result = await connector.upload_file(
                channel_id="channel-abc",
                content=b"Content",
                filename="test.txt",
                team_id="team-xyz",
            )

            assert result.id == ""
            assert "connection" in result.metadata.get("error", "").lower() or "failed" in result.metadata.get("error", "").lower()
            connector._record_failure_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_file_json_decode_error(self):
        """Should handle JSON decode errors."""
        from aragora.connectors.chat.teams._files import TeamsFilesMixin

        class TestConnector(TeamsFilesMixin, MockTeamsConnector):
            pass

        connector = TestConnector()

        connector._graph_api_request_mock.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)

        result = await connector.upload_file(
            channel_id="channel-abc",
            content=b"Content",
            filename="test.txt",
            team_id="team-xyz",
        )

        assert result.id == ""
        connector._record_failure_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_file_key_error(self):
        """Should handle KeyError exceptions."""
        from aragora.connectors.chat.teams._files import TeamsFilesMixin

        class TestConnector(TeamsFilesMixin, MockTeamsConnector):
            pass

        connector = TestConnector()

        connector._graph_api_request_mock.side_effect = KeyError("missing_key")

        result = await connector.upload_file(
            channel_id="channel-abc",
            content=b"Content",
            filename="test.txt",
            team_id="team-xyz",
        )

        assert result.id == ""
        connector._record_failure_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_file_value_error(self):
        """Should handle ValueError exceptions."""
        from aragora.connectors.chat.teams._files import TeamsFilesMixin

        class TestConnector(TeamsFilesMixin, MockTeamsConnector):
            pass

        connector = TestConnector()

        connector._graph_api_request_mock.side_effect = ValueError("Invalid value")

        result = await connector.upload_file(
            channel_id="channel-abc",
            content=b"Content",
            filename="test.txt",
            team_id="team-xyz",
        )

        assert result.id == ""

    @pytest.mark.asyncio
    async def test_upload_file_os_error(self):
        """Should handle OSError exceptions."""
        from aragora.connectors.chat.teams._files import TeamsFilesMixin

        class TestConnector(TeamsFilesMixin, MockTeamsConnector):
            pass

        connector = TestConnector()

        connector._graph_api_request_mock.side_effect = OSError("Disk full")

        result = await connector.upload_file(
            channel_id="channel-abc",
            content=b"Content",
            filename="test.txt",
            team_id="team-xyz",
        )

        assert result.id == ""
        connector._record_failure_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_file_httpx_not_available(self):
        """Should return fallback when httpx is not available."""
        from aragora.connectors.chat.teams._files import TeamsFilesMixin

        class TestConnector(TeamsFilesMixin, MockTeamsConnector):
            pass

        connector = TestConnector()

        with patch("aragora.connectors.chat.teams._constants.HTTPX_AVAILABLE", False):
            result = await connector.upload_file(
                channel_id="channel-abc",
                content=b"Hello",
                filename="test.txt",
                content_type="text/plain",
                team_id="team-xyz",
            )

            # Returns basic FileAttachment without API call
            assert result.id == ""
            assert result.filename == "test.txt"
            assert result.content_type == "text/plain"
            assert result.size == 5
            assert result.content == b"Hello"

    @pytest.mark.asyncio
    async def test_upload_file_correct_endpoints(self):
        """Should construct correct Graph API endpoints."""
        from aragora.connectors.chat.teams._files import TeamsFilesMixin

        class TestConnector(TeamsFilesMixin, MockTeamsConnector):
            pass

        connector = TestConnector()

        folder_data = {
            "id": "folder-123",
            "parentReference": {"driveId": "drive-456"},
        }
        upload_data = {"id": "file-789"}

        connector._graph_api_request_mock.side_effect = [
            (True, folder_data, None),
            (True, upload_data, None),
        ]

        await connector.upload_file(
            channel_id="channel-abc",
            content=b"Test",
            filename="doc.txt",
            team_id="team-xyz",
        )

        # Check folder endpoint
        folder_call = connector._graph_api_request_mock.call_args_list[0]
        assert folder_call.kwargs["endpoint"] == "/teams/team-xyz/channels/channel-abc/filesFolder"
        assert folder_call.kwargs["method"] == "GET"

        # Check upload endpoint
        upload_call = connector._graph_api_request_mock.call_args_list[1]
        assert "/drives/drive-456/items/folder-123:/" in upload_call.kwargs["endpoint"]
        assert "doc.txt" in upload_call.kwargs["endpoint"]
        assert upload_call.kwargs["method"] == "PUT"

    @pytest.mark.asyncio
    async def test_upload_large_file_content_range_header(self):
        """Should set correct Content-Range header for large file upload."""
        from aragora.connectors.chat.teams._files import TeamsFilesMixin

        class TestConnector(TeamsFilesMixin, MockTeamsConnector):
            pass

        connector = TestConnector()

        large_content = b"x" * (5 * 1024 * 1024)  # 5MB
        file_size = len(large_content)

        folder_data = {
            "id": "folder-123",
            "parentReference": {"driveId": "drive-456"},
        }
        session_data = {"uploadUrl": "https://upload.url"}

        connector._graph_api_request_mock.side_effect = [
            (True, folder_data, None),
            (True, session_data, None),
        ]
        connector._http_request_mock.return_value = (True, {"id": "file"}, None)

        await connector.upload_file(
            channel_id="channel-abc",
            content=large_content,
            filename="large.bin",
            team_id="team-xyz",
        )

        call_args = connector._http_request_mock.call_args
        headers = call_args.kwargs["headers"]
        assert headers["Content-Length"] == str(file_size)
        assert headers["Content-Range"] == f"bytes 0-{file_size - 1}/{file_size}"


# =============================================================================
# Download File Tests
# =============================================================================


class TestDownloadFile:
    """Tests for download_file method."""

    @pytest.mark.asyncio
    async def test_download_file_success_with_drive_id(self):
        """Should download file successfully with drive_id."""
        from aragora.connectors.chat.teams._files import TeamsFilesMixin

        class TestConnector(TeamsFilesMixin, MockTeamsConnector):
            pass

        connector = TestConnector()

        meta_data = {
            "name": "document.pdf",
            "size": 1024,
            "file": {"mimeType": "application/pdf"},
            "@microsoft.graph.downloadUrl": "https://download.url/doc.pdf",
            "webUrl": "https://sp.com/doc.pdf",
        }

        connector._graph_api_request_mock.return_value = (True, meta_data, None)
        connector._http_request_mock.return_value = (True, b"PDF content here", None)

        result = await connector.download_file(
            file_id="file-123",
            drive_id="drive-456",
        )

        assert result.id == "file-123"
        assert result.filename == "document.pdf"
        assert result.content_type == "application/pdf"
        assert result.content == b"PDF content here"
        assert result.size == len(b"PDF content here")
        assert result.url == "https://sp.com/doc.pdf"
        assert result.metadata["drive_id"] == "drive-456"

    @pytest.mark.asyncio
    async def test_download_file_success_without_drive_id(self):
        """Should download file using full item ID when no drive_id."""
        from aragora.connectors.chat.teams._files import TeamsFilesMixin

        class TestConnector(TeamsFilesMixin, MockTeamsConnector):
            pass

        connector = TestConnector()

        meta_data = {
            "name": "doc.txt",
            "size": 100,
            "file": {"mimeType": "text/plain"},
            "@microsoft.graph.downloadUrl": "https://download.url",
        }

        connector._graph_api_request_mock.return_value = (True, meta_data, None)
        connector._http_request_mock.return_value = (True, b"Text content", None)

        result = await connector.download_file(file_id="full-item-path")

        assert result.filename == "doc.txt"
        # Check endpoint used
        call_args = connector._graph_api_request_mock.call_args
        assert "/drives/items/full-item-path" in call_args.kwargs["endpoint"]

    @pytest.mark.asyncio
    async def test_download_file_metadata_fetch_failure(self):
        """Should return error when metadata fetch fails."""
        from aragora.connectors.chat.teams._files import TeamsFilesMixin

        class TestConnector(TeamsFilesMixin, MockTeamsConnector):
            pass

        connector = TestConnector()
        connector._graph_api_request_mock.return_value = (False, None, "File not found")

        result = await connector.download_file(
            file_id="nonexistent-file",
            drive_id="drive-456",
        )

        assert result.id == "nonexistent-file"
        assert result.filename == ""
        assert "File not found" in result.metadata.get("error", "")

    @pytest.mark.asyncio
    async def test_download_file_no_download_url(self):
        """Should return error when no download URL in metadata."""
        from aragora.connectors.chat.teams._files import TeamsFilesMixin

        class TestConnector(TeamsFilesMixin, MockTeamsConnector):
            pass

        connector = TestConnector()

        meta_data = {
            "name": "doc.txt",
            "size": 100,
            "file": {"mimeType": "text/plain"},
            # No @microsoft.graph.downloadUrl
        }

        connector._graph_api_request_mock.return_value = (True, meta_data, None)

        result = await connector.download_file(
            file_id="file-123",
            drive_id="drive-456",
        )

        assert result.content is None
        assert "No download URL" in result.metadata.get("error", "")

    @pytest.mark.asyncio
    async def test_download_file_content_fetch_failure(self):
        """Should return error when content download fails."""
        from aragora.connectors.chat.teams._files import TeamsFilesMixin

        class TestConnector(TeamsFilesMixin, MockTeamsConnector):
            pass

        connector = TestConnector()

        meta_data = {
            "name": "doc.txt",
            "size": 100,
            "file": {"mimeType": "text/plain"},
            "@microsoft.graph.downloadUrl": "https://download.url",
        }

        connector._graph_api_request_mock.return_value = (True, meta_data, None)
        connector._http_request_mock.return_value = (False, None, "Download failed")

        result = await connector.download_file(
            file_id="file-123",
            drive_id="drive-456",
        )

        assert result.filename == "doc.txt"
        assert result.content is None
        assert "Download failed" in result.metadata.get("error", "")

    @pytest.mark.asyncio
    async def test_download_file_timeout_error(self):
        """Should handle timeout errors."""
        from aragora.connectors.chat.teams._files import TeamsFilesMixin

        class TestConnector(TeamsFilesMixin, MockTeamsConnector):
            pass

        connector = TestConnector()

        with patch("aragora.connectors.chat.teams._constants.HTTPX_AVAILABLE", True):
            import httpx

            connector._graph_api_request_mock.side_effect = httpx.TimeoutException("Timeout")

            result = await connector.download_file(
                file_id="file-123",
                drive_id="drive-456",
            )

            assert result.id == "file-123"
            assert "timed out" in result.metadata.get("error", "").lower()
            connector._record_failure_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_download_file_connection_error(self):
        """Should handle connection errors."""
        from aragora.connectors.chat.teams._files import TeamsFilesMixin

        class TestConnector(TeamsFilesMixin, MockTeamsConnector):
            pass

        connector = TestConnector()

        with patch("aragora.connectors.chat.teams._constants.HTTPX_AVAILABLE", True):
            import httpx

            connector._graph_api_request_mock.side_effect = httpx.ConnectError("Connection failed")

            result = await connector.download_file(
                file_id="file-123",
                drive_id="drive-456",
            )

            assert "Connection failed" in result.metadata.get("error", "")
            connector._record_failure_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_download_file_json_decode_error(self):
        """Should handle JSON decode errors."""
        from aragora.connectors.chat.teams._files import TeamsFilesMixin

        class TestConnector(TeamsFilesMixin, MockTeamsConnector):
            pass

        connector = TestConnector()

        connector._graph_api_request_mock.side_effect = json.JSONDecodeError("Invalid", "", 0)

        result = await connector.download_file(
            file_id="file-123",
            drive_id="drive-456",
        )

        assert result.id == "file-123"
        connector._record_failure_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_download_file_httpx_not_available(self):
        """Should return fallback when httpx is not available."""
        from aragora.connectors.chat.teams._files import TeamsFilesMixin

        class TestConnector(TeamsFilesMixin, MockTeamsConnector):
            pass

        connector = TestConnector()

        with patch("aragora.connectors.chat.teams._constants.HTTPX_AVAILABLE", False):
            result = await connector.download_file(
                file_id="file-123",
                drive_id="drive-456",
            )

            # Returns basic FileAttachment without API call
            assert result.id == "file-123"
            assert result.filename == ""
            assert result.size == 0

    @pytest.mark.asyncio
    async def test_download_file_correct_endpoint_with_drive(self):
        """Should construct correct endpoint with drive_id."""
        from aragora.connectors.chat.teams._files import TeamsFilesMixin

        class TestConnector(TeamsFilesMixin, MockTeamsConnector):
            pass

        connector = TestConnector()

        meta_data = {
            "name": "doc.txt",
            "@microsoft.graph.downloadUrl": "https://dl.url",
            "file": {},
        }
        connector._graph_api_request_mock.return_value = (True, meta_data, None)
        connector._http_request_mock.return_value = (True, b"", None)

        await connector.download_file(
            file_id="item-xyz",
            drive_id="drive-abc",
        )

        call_args = connector._graph_api_request_mock.call_args
        assert call_args.kwargs["endpoint"] == "/drives/drive-abc/items/item-xyz"

    @pytest.mark.asyncio
    async def test_download_file_uses_upload_timeout(self):
        """Should use upload_timeout for download request."""
        from aragora.connectors.chat.teams._files import TeamsFilesMixin

        class TestConnector(TeamsFilesMixin, MockTeamsConnector):
            pass

        connector = TestConnector()
        connector._upload_timeout = 180.0

        meta_data = {
            "name": "doc.txt",
            "@microsoft.graph.downloadUrl": "https://dl.url",
            "file": {},
        }
        connector._graph_api_request_mock.return_value = (True, meta_data, None)
        connector._http_request_mock.return_value = (True, b"content", None)

        await connector.download_file(
            file_id="file-123",
            drive_id="drive-456",
        )

        call_args = connector._http_request_mock.call_args
        assert call_args.kwargs["timeout"] == 180.0

    @pytest.mark.asyncio
    async def test_download_file_return_raw_flag(self):
        """Should set return_raw=True for content download."""
        from aragora.connectors.chat.teams._files import TeamsFilesMixin

        class TestConnector(TeamsFilesMixin, MockTeamsConnector):
            pass

        connector = TestConnector()

        meta_data = {
            "name": "doc.txt",
            "@microsoft.graph.downloadUrl": "https://dl.url",
            "file": {},
        }
        connector._graph_api_request_mock.return_value = (True, meta_data, None)
        connector._http_request_mock.return_value = (True, b"content", None)

        await connector.download_file(
            file_id="file-123",
            drive_id="drive-456",
        )

        call_args = connector._http_request_mock.call_args
        assert call_args.kwargs["return_raw"] is True

    @pytest.mark.asyncio
    async def test_download_file_default_mime_type(self):
        """Should use default mime type when not in metadata."""
        from aragora.connectors.chat.teams._files import TeamsFilesMixin

        class TestConnector(TeamsFilesMixin, MockTeamsConnector):
            pass

        connector = TestConnector()

        meta_data = {
            "name": "unknown-file",
            "@microsoft.graph.downloadUrl": "https://dl.url",
            "file": {},  # No mimeType
        }
        connector._graph_api_request_mock.return_value = (True, meta_data, None)
        connector._http_request_mock.return_value = (True, b"content", None)

        result = await connector.download_file(
            file_id="file-123",
            drive_id="drive-456",
        )

        assert result.content_type == "application/octet-stream"


# =============================================================================
# Error Classification Tests
# =============================================================================


class TestErrorClassification:
    """Tests for error classification in file operations."""

    @pytest.mark.asyncio
    async def test_upload_http_error_handling(self):
        """Should handle generic HTTPError."""
        from aragora.connectors.chat.teams._files import TeamsFilesMixin

        class TestConnector(TeamsFilesMixin, MockTeamsConnector):
            pass

        connector = TestConnector()

        with patch("aragora.connectors.chat.teams._constants.HTTPX_AVAILABLE", True):
            import httpx

            connector._graph_api_request_mock.side_effect = httpx.HTTPError("HTTP error")

            result = await connector.upload_file(
                channel_id="channel-abc",
                content=b"Content",
                filename="test.txt",
                team_id="team-xyz",
            )

            assert result.id == ""
            connector._record_failure_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_download_http_error_handling(self):
        """Should handle generic HTTPError on download."""
        from aragora.connectors.chat.teams._files import TeamsFilesMixin

        class TestConnector(TeamsFilesMixin, MockTeamsConnector):
            pass

        connector = TestConnector()

        with patch("aragora.connectors.chat.teams._constants.HTTPX_AVAILABLE", True):
            import httpx

            connector._graph_api_request_mock.side_effect = httpx.HTTPError("HTTP error")

            result = await connector.download_file(
                file_id="file-123",
                drive_id="drive-456",
            )

            assert result.filename == ""
            connector._record_failure_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_runtime_error(self):
        """Should handle RuntimeError."""
        from aragora.connectors.chat.teams._files import TeamsFilesMixin

        class TestConnector(TeamsFilesMixin, MockTeamsConnector):
            pass

        connector = TestConnector()

        connector._graph_api_request_mock.side_effect = RuntimeError("Runtime issue")

        result = await connector.upload_file(
            channel_id="channel-abc",
            content=b"Content",
            filename="test.txt",
            team_id="team-xyz",
        )

        assert result.id == ""
        assert "Runtime issue" in result.metadata.get("error", "")


# =============================================================================
# Metadata and Response Tests
# =============================================================================


class TestFileMetadata:
    """Tests for file metadata handling."""

    @pytest.mark.asyncio
    async def test_upload_file_preserves_content_on_failure(self):
        """Should preserve original content in FileAttachment on failure."""
        from aragora.connectors.chat.teams._files import TeamsFilesMixin

        class TestConnector(TeamsFilesMixin, MockTeamsConnector):
            pass

        connector = TestConnector()
        connector._graph_api_request_mock.return_value = (False, None, "Error")

        original_content = b"Original content bytes"

        result = await connector.upload_file(
            channel_id="channel-abc",
            content=original_content,
            filename="test.txt",
            team_id="team-xyz",
        )

        assert result.content == original_content

    @pytest.mark.asyncio
    async def test_upload_file_metadata_includes_web_url(self):
        """Should include webUrl in metadata on success."""
        from aragora.connectors.chat.teams._files import TeamsFilesMixin

        class TestConnector(TeamsFilesMixin, MockTeamsConnector):
            pass

        connector = TestConnector()

        folder_data = {
            "id": "folder",
            "parentReference": {"driveId": "drive"},
        }
        upload_data = {
            "id": "file-id",
            "webUrl": "https://sp.com/doc",
        }

        connector._graph_api_request_mock.side_effect = [
            (True, folder_data, None),
            (True, upload_data, None),
        ]

        result = await connector.upload_file(
            channel_id="channel",
            content=b"x",
            filename="f.txt",
            team_id="team",
        )

        assert result.metadata["web_url"] == "https://sp.com/doc"

    @pytest.mark.asyncio
    async def test_download_preserves_item_id_in_metadata(self):
        """Should include item_id in download result metadata."""
        from aragora.connectors.chat.teams._files import TeamsFilesMixin

        class TestConnector(TeamsFilesMixin, MockTeamsConnector):
            pass

        connector = TestConnector()

        meta_data = {
            "name": "doc.txt",
            "@microsoft.graph.downloadUrl": "https://dl.url",
            "file": {},
        }
        connector._graph_api_request_mock.return_value = (True, meta_data, None)
        connector._http_request_mock.return_value = (True, b"content", None)

        result = await connector.download_file(
            file_id="item-xyz",
            drive_id="drive-abc",
        )

        assert result.metadata["item_id"] == "item-xyz"
