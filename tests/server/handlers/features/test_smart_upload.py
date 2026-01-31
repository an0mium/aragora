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
    validate_content_type,
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
        assert result.status == "completed"

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


class TestContentTypeValidation:
    """Tests for content-type validation using magic bytes."""

    def test_validate_png_content(self):
        """Test validating PNG file by magic bytes."""
        # PNG magic bytes
        png_content = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        result = validate_content_type(png_content, "image.png")

        assert result.valid is True
        assert result.detected_mime == "image/png"
        assert ".png" in result.expected_extensions
        assert result.is_dangerous is False

    def test_validate_jpeg_content(self):
        """Test validating JPEG file by magic bytes."""
        jpeg_content = b"\xff\xd8\xff" + b"\x00" * 100
        result = validate_content_type(jpeg_content, "photo.jpg")

        assert result.valid is True
        assert result.detected_mime == "image/jpeg"
        assert ".jpg" in result.expected_extensions

    def test_validate_gif_content(self):
        """Test validating GIF file by magic bytes."""
        gif87_content = b"GIF87a" + b"\x00" * 100
        result = validate_content_type(gif87_content, "animation.gif")

        assert result.valid is True
        assert result.detected_mime == "image/gif"

        gif89_content = b"GIF89a" + b"\x00" * 100
        result = validate_content_type(gif89_content, "animation.gif")

        assert result.valid is True
        assert result.detected_mime == "image/gif"

    def test_validate_pdf_content(self):
        """Test validating PDF file by magic bytes."""
        pdf_content = b"%PDF-1.4" + b"\x00" * 100
        result = validate_content_type(pdf_content, "document.pdf")

        assert result.valid is True
        assert result.detected_mime == "application/pdf"
        assert ".pdf" in result.expected_extensions

    def test_validate_zip_content(self):
        """Test validating ZIP file by magic bytes."""
        import io
        import zipfile

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr("test.txt", "hello")

        result = validate_content_type(buffer.getvalue(), "archive.zip")

        assert result.valid is True
        assert result.detected_mime == "application/zip"

    def test_block_exe_extension(self):
        """Test blocking executable by extension."""
        content = b"MZ" + b"\x00" * 100  # PE header
        result = validate_content_type(content, "malware.exe")

        assert result.valid is False
        assert result.is_dangerous is True
        assert "Dangerous file extension blocked" in result.mismatch_warning

    def test_block_dll_extension(self):
        """Test blocking DLL by extension."""
        content = b"\x00" * 100
        result = validate_content_type(content, "library.dll")

        assert result.valid is False
        assert result.is_dangerous is True

    def test_block_bat_extension(self):
        """Test blocking batch file by extension."""
        content = b"@echo off\r\ndel /s /q *"
        result = validate_content_type(content, "script.bat")

        assert result.valid is False
        assert result.is_dangerous is True

    def test_block_exe_by_content(self):
        """Test blocking executable by magic bytes even with safe extension."""
        # PE executable header with .txt extension
        pe_content = b"MZ" + b"\x00" * 100
        result = validate_content_type(pe_content, "safe.txt")

        assert result.valid is False
        assert result.is_dangerous is True
        assert result.detected_mime == "application/x-executable"

    def test_block_elf_by_content(self):
        """Test blocking ELF executable by magic bytes."""
        elf_content = b"\x7fELF" + b"\x00" * 100
        result = validate_content_type(elf_content, "binary.txt")

        assert result.valid is False
        assert result.is_dangerous is True

    def test_extension_mismatch_warning(self):
        """Test warning when extension doesn't match content."""
        # PNG content with .jpg extension
        png_content = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        result = validate_content_type(png_content, "photo.jpg")

        assert result.valid is True  # Not blocked, but warned
        assert result.mismatch_warning is not None
        assert "mismatch" in result.mismatch_warning.lower()

    def test_mime_type_mismatch_warning(self):
        """Test warning when claimed MIME doesn't match content."""
        png_content = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        result = validate_content_type(png_content, "image.png", claimed_mime="image/gif")

        assert result.valid is True
        assert result.mismatch_warning is not None
        assert "Claimed MIME type" in result.mismatch_warning

    def test_office_docs_as_zip_allowed(self):
        """Test that Office documents (ZIP-based) don't trigger mismatch warning."""
        import io
        import zipfile

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr("[Content_Types].xml", "<Types/>")

        result = validate_content_type(buffer.getvalue(), "document.docx")

        assert result.valid is True
        assert result.detected_mime == "application/zip"
        # No mismatch warning since .docx is expected for ZIP
        assert result.mismatch_warning is None

    def test_unknown_content_type(self):
        """Test handling of unknown/undetectable content."""
        unknown_content = b"some random text content"
        result = validate_content_type(unknown_content, "file.txt")

        assert result.valid is True
        assert result.detected_mime is None
        assert result.is_dangerous is False

    def test_mp3_id3_detection(self):
        """Test detecting MP3 by ID3 header."""
        mp3_content = b"ID3" + b"\x00" * 100
        result = validate_content_type(mp3_content, "song.mp3")

        assert result.valid is True
        assert result.detected_mime == "audio/mpeg"

    def test_gzip_detection(self):
        """Test detecting gzip compressed files."""
        gzip_content = b"\x1f\x8b" + b"\x00" * 100
        result = validate_content_type(gzip_content, "archive.gz")

        assert result.valid is True
        assert result.detected_mime == "application/gzip"


class TestSmartUploadWithValidation:
    """Tests for smart_upload with content validation integration."""

    @pytest.mark.asyncio
    async def test_smart_upload_blocks_exe(self):
        """Test that smart_upload blocks executable files."""
        exe_content = b"MZ" + b"\x00" * 100

        result = await smart_upload(exe_content, "malware.exe")

        assert result.status == "rejected"
        assert result.error is not None
        assert "Dangerous" in result.error

    @pytest.mark.asyncio
    async def test_smart_upload_blocks_disguised_exe(self):
        """Test that smart_upload blocks exe disguised as txt."""
        exe_content = b"MZ" + b"\x00" * 100

        result = await smart_upload(exe_content, "readme.txt")

        assert result.status == "rejected"
        assert "Dangerous" in (result.error or "")

    @pytest.mark.asyncio
    async def test_smart_upload_valid_png(self):
        """Test smart upload accepts valid PNG."""
        png_content = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        result = await smart_upload(png_content, "image.png")

        assert result.status in ("completed", "processing")
        assert result.error is None or result.error == ""

    @pytest.mark.asyncio
    async def test_smart_upload_warns_on_mismatch(self):
        """Test smart upload includes warning on content mismatch."""
        # PNG content with .jpg extension
        png_content = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        result = await smart_upload(png_content, "photo.jpg")

        assert result.status == "completed"
        # Warning should be in result
        if result.result:
            assert "content_type_warning" in result.result

    @pytest.mark.asyncio
    async def test_smart_upload_blocks_vbs_scripts(self):
        """Test blocking VBScript file extensions."""
        vbs_content = b'WScript.Echo "Hello"'

        result = await smart_upload(vbs_content, "script.vbs")

        # .vbs is in dangerous extensions
        assert result.status == "rejected"
        assert "Dangerous" in (result.error or "")

    @pytest.mark.asyncio
    async def test_smart_upload_blocks_ps1(self):
        """Test blocking PowerShell scripts."""
        ps1_content = b"Get-Process | Stop-Process"

        result = await smart_upload(ps1_content, "script.ps1")

        assert result.status == "rejected"
