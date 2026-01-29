"""Tests for Smart Upload Handler."""

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

from aragora.server.handlers.features.smart_upload import (
    SmartUploadHandler,
    FileCategory,
    ProcessingAction,
    UploadResult,
    detect_file_category,
    get_processing_action,
    smart_upload,
    get_upload_status,
    _upload_results,
)


@pytest.fixture(autouse=True)
def clear_uploads():
    """Clear upload results between tests."""
    _upload_results.clear()
    yield


@pytest.fixture
def handler():
    """Create handler instance."""
    return SmartUploadHandler({})


class TestFileCategory:
    """Tests for FileCategory enum."""

    def test_category_values(self):
        """Test category enum values."""
        assert FileCategory.CODE.value == "code"
        assert FileCategory.DOCUMENT.value == "document"
        assert FileCategory.AUDIO.value == "audio"
        assert FileCategory.VIDEO.value == "video"
        assert FileCategory.IMAGE.value == "image"
        assert FileCategory.DATA.value == "data"
        assert FileCategory.ARCHIVE.value == "archive"
        assert FileCategory.UNKNOWN.value == "unknown"


class TestProcessingAction:
    """Tests for ProcessingAction enum."""

    def test_action_values(self):
        """Test action enum values."""
        assert ProcessingAction.INDEX.value == "index"
        assert ProcessingAction.EXTRACT.value == "extract"
        assert ProcessingAction.TRANSCRIBE.value == "transcribe"
        assert ProcessingAction.OCR.value == "ocr"
        assert ProcessingAction.PARSE.value == "parse"
        assert ProcessingAction.EXPAND.value == "expand"
        assert ProcessingAction.SKIP.value == "skip"


class TestDetectFileCategory:
    """Tests for detect_file_category function."""

    def test_detect_python_file(self):
        """Test detecting Python files."""
        assert detect_file_category("script.py") == FileCategory.CODE

    def test_detect_javascript_file(self):
        """Test detecting JavaScript files."""
        assert detect_file_category("app.js") == FileCategory.CODE
        assert detect_file_category("component.tsx") == FileCategory.CODE

    def test_detect_pdf_file(self):
        """Test detecting PDF files."""
        assert detect_file_category("document.pdf") == FileCategory.DOCUMENT

    def test_detect_audio_file(self):
        """Test detecting audio files."""
        assert detect_file_category("audio.mp3") == FileCategory.AUDIO
        assert detect_file_category("sound.wav") == FileCategory.AUDIO

    def test_detect_video_file(self):
        """Test detecting video files."""
        assert detect_file_category("video.mp4") == FileCategory.VIDEO
        assert detect_file_category("movie.mov") == FileCategory.VIDEO

    def test_detect_image_file(self):
        """Test detecting image files."""
        assert detect_file_category("photo.png") == FileCategory.IMAGE
        assert detect_file_category("picture.jpg") == FileCategory.IMAGE

    def test_detect_data_file(self):
        """Test detecting data files."""
        assert detect_file_category("config.json") == FileCategory.DATA
        assert detect_file_category("data.csv") == FileCategory.DATA

    def test_detect_archive_file(self):
        """Test detecting archive files."""
        assert detect_file_category("archive.zip") == FileCategory.ARCHIVE
        assert detect_file_category("files.tar.gz") == FileCategory.ARCHIVE

    def test_detect_unknown_file(self):
        """Test detecting unknown files."""
        assert detect_file_category("file.xyz") == FileCategory.UNKNOWN

    def test_detect_with_mime_type(self):
        """Test detecting with MIME type hint."""
        assert detect_file_category("file", "audio/mpeg") == FileCategory.AUDIO


class TestGetProcessingAction:
    """Tests for get_processing_action function."""

    def test_code_action(self):
        """Test action for code files."""
        assert get_processing_action(FileCategory.CODE) == ProcessingAction.INDEX

    def test_document_action(self):
        """Test action for document files."""
        assert get_processing_action(FileCategory.DOCUMENT) == ProcessingAction.EXTRACT

    def test_audio_action(self):
        """Test action for audio files."""
        assert get_processing_action(FileCategory.AUDIO) == ProcessingAction.TRANSCRIBE

    def test_image_action(self):
        """Test action for image files."""
        assert get_processing_action(FileCategory.IMAGE) == ProcessingAction.OCR

    def test_unknown_action(self):
        """Test action for unknown files."""
        assert get_processing_action(FileCategory.UNKNOWN) == ProcessingAction.SKIP


class TestSmartUpload:
    """Tests for smart_upload function."""

    @pytest.mark.asyncio
    async def test_smart_upload_code_file(self):
        """Test smart upload of code file."""
        content = b"print('hello world')"

        result = await smart_upload(content, "script.py")

        assert result.filename == "script.py"
        assert result.category == FileCategory.CODE
        assert result.action == ProcessingAction.INDEX
        assert result.status_code == "completed"

    @pytest.mark.asyncio
    async def test_smart_upload_stores_result(self):
        """Test smart upload stores result."""
        content = b"test content"

        result = await smart_upload(content, "test.txt")

        assert result.id in _upload_results

    @pytest.mark.asyncio
    async def test_smart_upload_with_override_action(self):
        """Test smart upload with action override."""
        content = b"test content"

        result = await smart_upload(content, "test.txt", override_action=ProcessingAction.SKIP)

        assert result.action == ProcessingAction.SKIP


class TestGetUploadStatus:
    """Tests for get_upload_status function."""

    def test_get_status_not_found(self):
        """Test getting status for non-existent upload."""
        result = get_upload_status("invalid-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_status_success(self):
        """Test getting status for existing upload."""
        content = b"test content"
        upload_result = await smart_upload(content, "test.txt")

        result = get_upload_status(upload_result.id)
        assert result is not None
        assert result.id == upload_result.id


class TestSmartUploadHandler:
    """Tests for SmartUploadHandler class."""

    def test_handler_creation(self, handler):
        """Test creating handler instance."""
        assert handler is not None

    def test_handler_routes(self):
        """Test that handler has route definitions."""
        assert hasattr(SmartUploadHandler, "ROUTES")
        routes = SmartUploadHandler.ROUTES
        assert "/api/v1/upload/smart" in routes
        assert "/api/v1/upload/batch" in routes
        assert "/api/v1/upload/status" in routes

    def test_can_handle_upload_routes(self, handler):
        """Test can_handle for upload routes."""
        assert handler.can_handle("/api/v1/upload/smart") is True
        assert handler.can_handle("/api/v1/upload/batch") is True
        assert handler.can_handle("/api/v1/upload/status/abc123") is True

    def test_can_handle_invalid_routes(self, handler):
        """Test can_handle rejects invalid routes."""
        assert handler.can_handle("/api/v1/files/upload") is False


class TestSmartUploadEndpoints:
    """Tests for smart upload endpoints."""

    def test_get_status_not_found(self):
        """Test getting status for non-existent upload."""
        handler = SmartUploadHandler({})

        result = handler._get_status("invalid-id")
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_handle_smart_upload_missing_content(self):
        """Test smart upload requires content."""
        handler = SmartUploadHandler({})
        mock_handler = MagicMock()

        result = await handler._handle_smart_upload({"filename": "test.txt"}, mock_handler)
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_handle_smart_upload_success(self):
        """Test successful smart upload."""
        handler = SmartUploadHandler({})
        mock_handler = MagicMock()

        import base64

        content = base64.b64encode(b"test content").decode()

        result = await handler._handle_smart_upload(
            {"content": content, "filename": "test.txt"}, mock_handler
        )
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_batch_upload_missing_files(self):
        """Test batch upload requires files."""
        handler = SmartUploadHandler({})
        mock_handler = MagicMock()

        result = await handler._handle_batch_upload({}, mock_handler)
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_handle_batch_upload_success(self):
        """Test successful batch upload."""
        handler = SmartUploadHandler({})
        mock_handler = MagicMock()

        import base64

        content = base64.b64encode(b"test content").decode()

        result = await handler._handle_batch_upload(
            {
                "files": [
                    {"content": content, "filename": "test1.txt"},
                    {"content": content, "filename": "test2.txt"},
                ]
            },
            mock_handler,
        )
        assert result.status_code == 200

        import json

        body = json.loads(result.body)
        assert body["count"] == 2


class TestFileProcessing:
    """Tests for file processing functions."""

    @pytest.mark.asyncio
    async def test_extract_txt_document(self):
        """Test extracting text from txt file."""
        from aragora.server.handlers.features.smart_upload import _extract_document_text

        content = b"Hello, World!"
        result = await _extract_document_text(content, "test.txt", {})

        assert result["text"] == "Hello, World!"
        assert result["word_count"] == 2

    @pytest.mark.asyncio
    async def test_parse_json_data(self):
        """Test parsing JSON data file."""
        from aragora.server.handlers.features.smart_upload import _parse_data_file

        content = b'{"key": "value"}'
        result = await _parse_data_file(content, "data.json", {})

        assert result["parsed"] is True
        assert result["type"] == "json"

    @pytest.mark.asyncio
    async def test_parse_csv_data(self):
        """Test parsing CSV data file."""
        from aragora.server.handlers.features.smart_upload import _parse_data_file

        content = b"name,age\nAlice,30\nBob,25"
        result = await _parse_data_file(content, "data.csv", {})

        assert result["parsed"] is True
        assert result["type"] == "csv"
        assert result["record_count"] == 2

    @pytest.mark.asyncio
    async def test_index_code_file(self):
        """Test indexing code file."""
        from aragora.server.handlers.features.smart_upload import _index_code_file

        content = b"def hello():\n    print('hello')\n\nclass Foo:\n    pass"
        result = await _index_code_file(content, "script.py", {})

        assert result["indexed"] is True
        assert result["language"] == "python"
        assert len(result["symbols"]) > 0

    @pytest.mark.asyncio
    async def test_expand_zip_archive(self):
        """Test expanding zip archive."""
        from aragora.server.handlers.features.smart_upload import _expand_archive
        import io
        import zipfile

        # Create a small zip file
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr("test.txt", "hello")

        result = await _expand_archive(buffer.getvalue(), "archive.zip", {})

        assert result["expanded"] is True
        assert result["file_count"] == 1
