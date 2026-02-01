"""
Security tests for aragora.server.handlers.transcription - File upload validation.

Tests cover:
- Magic byte validation for audio/video files
- File size verification (actual bytes vs Content-Length)
- Path traversal prevention (null bytes in filename)
- Double extension detection
- Empty file handling
- Unknown extension handling
"""

from __future__ import annotations

import json
from io import BytesIO
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock transcription availability before importing
with patch("aragora.transcription.get_available_backends", return_value=["mock"]):
    with patch(
        "aragora.transcription.whisper_backend.get_available_backends", return_value=["mock"]
    ):
        from aragora.server.handlers.transcription import (
            ALLOWED_AUDIO_SIGNATURES,
            ALLOWED_VIDEO_SIGNATURES,
            MAX_AUDIO_SIZE_MB,
            MAX_VIDEO_SIZE_MB,
            TranscriptionHandler,
            _validate_file_content,
            _validate_filename_security,
        )


# ===========================================================================
# Test Fixtures
# ===========================================================================


class MockHandler:
    """Mock HTTP handler for testing."""

    def __init__(
        self,
        headers: dict | None = None,
        body: bytes = b"",
        path: str = "/",
        method: str = "GET",
    ):
        self.headers = headers or {}
        self._body = body
        self.path = path
        self.command = method
        self.rfile = BytesIO(body)

    def send_response(self, code: int) -> None:
        self.response_code = code

    def send_header(self, key: str, value: str) -> None:
        pass

    def end_headers(self) -> None:
        pass


def create_multipart_body(
    filename: str, file_data: bytes, boundary: str = "----TestBoundary"
) -> bytes:
    """Create multipart form data body for testing."""
    body = (
        f"------{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode()
    body += file_data
    body += f"\r\n------{boundary}--\r\n".encode()
    return body


@pytest.fixture
def mock_server_context():
    """Create a mock server context for handler initialization."""
    return {
        "storage": MagicMock(),
        "user_store": MagicMock(),
        "elo_system": MagicMock(),
        "continuum_memory": MagicMock(),
        "critique_store": MagicMock(),
        "document_store": MagicMock(),
        "evidence_store": MagicMock(),
        "usage_tracker": MagicMock(),
    }


@pytest.fixture
def handler(mock_server_context):
    """Create a TranscriptionHandler instance."""
    return TranscriptionHandler(mock_server_context)


@pytest.fixture
def mock_transcription_available():
    """Mock the transcription availability check."""
    with patch(
        "aragora.server.handlers.transcription._check_transcription_available",
        return_value=(True, None),
    ):
        with patch(
            "aragora.transcription.get_available_backends",
            return_value=["mock"],
        ):
            with patch(
                "aragora.transcription.whisper_backend.get_available_backends",
                return_value=["mock"],
            ):
                yield


# ===========================================================================
# Direct _validate_file_content Tests
# ===========================================================================


class TestValidateFileContent:
    """Tests for _validate_file_content function."""

    def test_empty_file_returns_error(self):
        """Test that empty file data returns error."""
        valid, error = _validate_file_content(b"", ".mp3")
        assert valid is False
        assert "Empty file content" in error

    def test_unknown_extension_returns_error(self):
        """Test that unknown extension returns error."""
        valid, error = _validate_file_content(b"some data", ".xyz")
        assert valid is False
        assert "Unknown file extension" in error

    def test_mp3_with_id3_header_valid(self):
        """Test valid MP3 file with ID3 header."""
        # ID3v2 header
        file_data = b"ID3" + b"\x00" * 100
        valid, error = _validate_file_content(file_data, ".mp3")
        assert valid is True
        assert error is None

    def test_mp3_with_frame_sync_valid(self):
        """Test valid MP3 file with frame sync header."""
        # MP3 frame sync (0xff 0xfb)
        file_data = b"\xff\xfb" + b"\x00" * 100
        valid, error = _validate_file_content(file_data, ".mp3")
        assert valid is True
        assert error is None

    def test_mp3_with_wrong_magic_bytes_invalid(self):
        """Test MP3 file with wrong magic bytes is rejected."""
        # PDF magic bytes instead of MP3
        file_data = b"%PDF-1.4" + b"\x00" * 100
        valid, error = _validate_file_content(file_data, ".mp3")
        assert valid is False
        assert "does not match expected .mp3 format" in error

    def test_wav_valid_file(self):
        """Test valid WAV file with RIFF/WAVE header."""
        # RIFF....WAVE header
        file_data = b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 100
        valid, error = _validate_file_content(file_data, ".wav")
        assert valid is True
        assert error is None

    def test_wav_riff_without_wave_invalid(self):
        """Test RIFF file without WAVE marker is rejected."""
        # RIFF header but not WAVE format
        file_data = b"RIFF\x00\x00\x00\x00ABCD" + b"\x00" * 100
        valid, error = _validate_file_content(file_data, ".wav")
        assert valid is False
        assert "does not match" in error

    def test_flac_valid_file(self):
        """Test valid FLAC file."""
        file_data = b"fLaC" + b"\x00" * 100
        valid, error = _validate_file_content(file_data, ".flac")
        assert valid is True
        assert error is None

    def test_ogg_valid_file(self):
        """Test valid OGG file."""
        file_data = b"OggS" + b"\x00" * 100
        valid, error = _validate_file_content(file_data, ".ogg")
        assert valid is True
        assert error is None

    def test_aac_valid_file(self):
        """Test valid AAC file with ADTS header."""
        file_data = b"\xff\xf1" + b"\x00" * 100
        valid, error = _validate_file_content(file_data, ".aac")
        assert valid is True
        assert error is None

    def test_webm_audio_valid(self):
        """Test valid WebM audio file."""
        # EBML header
        file_data = b"\x1a\x45\xdf\xa3" + b"\x00" * 100
        valid, error = _validate_file_content(file_data, ".webm", is_video=False)
        assert valid is True
        assert error is None

    def test_m4a_valid_file(self):
        """Test valid M4A file with ftyp box."""
        # ISO base media file format (ftyp at offset 4)
        file_data = b"\x00\x00\x00\x00ftyp" + b"\x00" * 100
        valid, error = _validate_file_content(file_data, ".m4a")
        assert valid is True
        assert error is None

    def test_mp4_valid_file(self):
        """Test valid MP4 video file."""
        # ftyp at offset 4
        file_data = b"\x00\x00\x00\x00ftyp" + b"\x00" * 100
        valid, error = _validate_file_content(file_data, ".mp4", is_video=True)
        assert valid is True
        assert error is None

    def test_mkv_valid_file(self):
        """Test valid MKV video file."""
        # EBML header
        file_data = b"\x1a\x45\xdf\xa3" + b"\x00" * 100
        valid, error = _validate_file_content(file_data, ".mkv", is_video=True)
        assert valid is True
        assert error is None

    def test_avi_valid_file(self):
        """Test valid AVI video file."""
        # RIFF....AVI header
        file_data = b"RIFF\x00\x00\x00\x00AVI " + b"\x00" * 100
        valid, error = _validate_file_content(file_data, ".avi", is_video=True)
        assert valid is True
        assert error is None

    def test_avi_riff_without_avi_invalid(self):
        """Test RIFF file without AVI marker is rejected."""
        # RIFF header but not AVI format
        file_data = b"RIFF\x00\x00\x00\x00ABCD" + b"\x00" * 100
        valid, error = _validate_file_content(file_data, ".avi", is_video=True)
        assert valid is False
        assert "does not match" in error

    def test_mov_valid_file(self):
        """Test valid MOV video file."""
        # QuickTime with ftyp
        file_data = b"\x00\x00\x00\x00ftyp" + b"\x00" * 100
        valid, error = _validate_file_content(file_data, ".mov", is_video=True)
        assert valid is True
        assert error is None

    def test_file_too_short_for_signature(self):
        """Test file shorter than signature offset is rejected."""
        # File is only 3 bytes, but MP4 signature is at offset 4
        file_data = b"\x00\x00\x00"
        valid, error = _validate_file_content(file_data, ".mp4", is_video=True)
        assert valid is False
        assert "does not match" in error

    def test_executable_disguised_as_audio_rejected(self):
        """Test executable with audio extension is rejected."""
        # PE executable header (MZ)
        file_data = b"MZ" + b"\x00" * 100
        valid, error = _validate_file_content(file_data, ".mp3")
        assert valid is False
        assert "does not match" in error


# ===========================================================================
# Direct _validate_filename_security Tests
# ===========================================================================


class TestValidateFilenameSecurity:
    """Tests for _validate_filename_security function."""

    def test_valid_filename_passes(self):
        """Test normal filename passes validation."""
        valid, error = _validate_filename_security("audio.mp3")
        assert valid is True
        assert error is None

    def test_null_byte_in_filename_rejected(self):
        """Test filename with null byte is rejected."""
        valid, error = _validate_filename_security("audio\x00.mp3")
        assert valid is False
        assert "null bytes" in error

    def test_null_byte_path_traversal_rejected(self):
        """Test path traversal with null byte is rejected."""
        valid, error = _validate_filename_security("../../../etc/passwd\x00.mp3")
        assert valid is False
        assert "null bytes" in error

    def test_double_extension_mp3_exe_rejected(self):
        """Test double extension .mp3.exe is rejected."""
        valid, error = _validate_filename_security("audio.mp3.exe")
        assert valid is False
        assert "double extensions" in error

    def test_double_extension_wav_bat_rejected(self):
        """Test double extension .wav.bat is rejected."""
        valid, error = _validate_filename_security("sound.wav.bat")
        assert valid is False
        assert "double extensions" in error

    def test_double_extension_mp4_js_rejected(self):
        """Test double extension .mp4.js is rejected."""
        valid, error = _validate_filename_security("video.mp4.js")
        assert valid is False
        assert "double extensions" in error

    def test_double_extension_case_insensitive(self):
        """Test double extension detection is case insensitive."""
        valid, error = _validate_filename_security("audio.MP3.EXE")
        assert valid is False
        assert "double extensions" in error

    def test_triple_extension_only_checks_last_two(self):
        """Test only last two extensions matter for double extension check."""
        # .backup.mp3 is fine, .mp3.exe is not
        valid, error = _validate_filename_security("file.backup.mp3")
        assert valid is True

    def test_filename_with_spaces_valid(self):
        """Test filename with spaces is valid."""
        valid, error = _validate_filename_security("my audio file.mp3")
        assert valid is True
        assert error is None

    def test_unicode_filename_valid(self):
        """Test unicode filename is valid."""
        valid, error = _validate_filename_security("audio_cafe.mp3")
        assert valid is True
        assert error is None


# ===========================================================================
# Audio Transcription Handler Security Tests
# ===========================================================================


class TestAudioTranscriptionSecurity:
    """Security tests for audio transcription endpoint."""

    @pytest.mark.asyncio
    async def test_wrong_magic_bytes_rejected(self, handler, mock_transcription_available):
        """Test file with wrong magic bytes for declared extension is rejected."""
        # Create a "fake" MP3 that's actually a PDF
        file_data = b"%PDF-1.4" + b"\x00" * 100
        boundary = "----TestBoundary"
        body = create_multipart_body("audio.mp3", file_data, boundary)

        mock_http = MockHandler(
            headers={
                "Content-Type": f"multipart/form-data; boundary=----{boundary}",
                "Content-Length": str(len(body)),
            },
            body=body,
            method="POST",
        )

        result = await handler.handle_post("/api/v1/transcription/audio", {}, mock_http)

        assert result is not None
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "does not match" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_oversized_body_rejected(self, handler, mock_transcription_available):
        """Test oversized file body is rejected even with small Content-Length."""
        # Create a file larger than max size
        # Use valid MP3 header to pass magic byte check
        file_data = b"ID3" + b"\x00" * (MAX_AUDIO_SIZE_MB * 1024 * 1024 + 1000)
        boundary = "----TestBoundary"
        body = create_multipart_body("audio.mp3", file_data, boundary)

        mock_http = MockHandler(
            headers={
                "Content-Type": f"multipart/form-data; boundary=----{boundary}",
                # Lie about content length
                "Content-Length": "1000",
            },
            body=body,
            method="POST",
        )

        result = await handler.handle_post("/api/v1/transcription/audio", {}, mock_http)

        assert result is not None
        assert result.status_code == 413
        data = json.loads(result.body)
        assert "too large" in data.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_empty_file_rejected(self, handler, mock_transcription_available):
        """Test empty file upload returns appropriate error."""
        # Empty file data
        file_data = b""
        boundary = "----TestBoundary"
        body = create_multipart_body("audio.mp3", file_data, boundary)

        mock_http = MockHandler(
            headers={
                "Content-Type": f"multipart/form-data; boundary=----{boundary}",
                "Content-Length": str(len(body)),
            },
            body=body,
            method="POST",
        )

        result = await handler.handle_post("/api/v1/transcription/audio", {}, mock_http)

        assert result is not None
        assert result.status_code == 400
        data = json.loads(result.body)
        # Either "Empty file content" or "No file provided"
        assert (
            "empty" in data.get("error", "").lower() or "no file" in data.get("error", "").lower()
        )

    @pytest.mark.asyncio
    async def test_null_bytes_in_filename_rejected(self, handler, mock_transcription_available):
        """Test null bytes in filename are rejected."""
        file_data = b"ID3" + b"\x00" * 100  # Valid MP3 header
        boundary = "----TestBoundary"
        body = create_multipart_body("audio\x00.mp3", file_data, boundary)

        mock_http = MockHandler(
            headers={
                "Content-Type": f"multipart/form-data; boundary=----{boundary}",
                "Content-Length": str(len(body)),
            },
            body=body,
            method="POST",
        )

        result = await handler.handle_post("/api/v1/transcription/audio", {}, mock_http)

        assert result is not None
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "null bytes" in data.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_double_extension_rejected(self, handler, mock_transcription_available):
        """Test double extension filenames are rejected."""
        file_data = b"ID3" + b"\x00" * 100  # Valid MP3 header
        boundary = "----TestBoundary"
        body = create_multipart_body("audio.mp3.exe", file_data, boundary)

        mock_http = MockHandler(
            headers={
                "Content-Type": f"multipart/form-data; boundary=----{boundary}",
                "Content-Length": str(len(body)),
            },
            body=body,
            method="POST",
        )

        result = await handler.handle_post("/api/v1/transcription/audio", {}, mock_http)

        assert result is not None
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "double extensions" in data.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_valid_mp3_passes_validation(self, handler, mock_transcription_available):
        """Test valid audio file with correct magic bytes passes validation."""
        file_data = b"ID3" + b"\x00" * 100  # Valid MP3 header
        boundary = "----TestBoundary"
        body = create_multipart_body("audio.mp3", file_data, boundary)

        mock_http = MockHandler(
            headers={
                "Content-Type": f"multipart/form-data; boundary=----{boundary}",
                "Content-Length": str(len(body)),
            },
            body=body,
            method="POST",
        )

        # Mock the transcription to avoid actually processing
        with patch("aragora.server.handlers.transcription.transcribe_audio") as mock_transcribe:
            mock_result = MagicMock()
            mock_result.text = "Test transcription"
            mock_result.language = "en"
            mock_result.duration = 10.0
            mock_result.segments = []
            mock_result.backend = "mock"
            mock_result.processing_time = 1.0
            mock_result.to_dict.return_value = {}
            mock_transcribe.return_value = mock_result

            result = await handler.handle_post("/api/v1/transcription/audio", {}, mock_http)

            # Should succeed (200) or error for other reasons, not security validation
            assert result is not None
            # If it fails, it should not be due to magic byte validation
            if result.status_code == 400:
                data = json.loads(result.body)
                assert "does not match" not in data.get("error", "")

    @pytest.mark.asyncio
    async def test_unknown_extension_rejected(self, handler, mock_transcription_available):
        """Test unknown file extension is rejected."""
        file_data = b"some random data"
        boundary = "----TestBoundary"
        body = create_multipart_body("file.xyz", file_data, boundary)

        mock_http = MockHandler(
            headers={
                "Content-Type": f"multipart/form-data; boundary=----{boundary}",
                "Content-Length": str(len(body)),
            },
            body=body,
            method="POST",
        )

        result = await handler.handle_post("/api/v1/transcription/audio", {}, mock_http)

        assert result is not None
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "unsupported" in data.get("error", "").lower()


# ===========================================================================
# Video Transcription Handler Security Tests
# ===========================================================================


class TestVideoTranscriptionSecurity:
    """Security tests for video transcription endpoint."""

    @pytest.mark.asyncio
    async def test_wrong_magic_bytes_video_rejected(self, handler, mock_transcription_available):
        """Test video file with wrong magic bytes is rejected."""
        # Create a "fake" MP4 that's actually a JPEG
        file_data = b"\xff\xd8\xff\xe0" + b"\x00" * 100  # JPEG header
        boundary = "----TestBoundary"
        body = create_multipart_body("video.mp4", file_data, boundary)

        mock_http = MockHandler(
            headers={
                "Content-Type": f"multipart/form-data; boundary=----{boundary}",
                "Content-Length": str(len(body)),
            },
            body=body,
            method="POST",
        )

        result = await handler.handle_post("/api/v1/transcription/video", {}, mock_http)

        assert result is not None
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "does not match" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_valid_mp4_passes_validation(self, handler, mock_transcription_available):
        """Test valid video file with correct magic bytes passes validation."""
        # Valid MP4 with ftyp at offset 4
        file_data = b"\x00\x00\x00\x00ftyp" + b"\x00" * 100
        boundary = "----TestBoundary"
        body = create_multipart_body("video.mp4", file_data, boundary)

        mock_http = MockHandler(
            headers={
                "Content-Type": f"multipart/form-data; boundary=----{boundary}",
                "Content-Length": str(len(body)),
            },
            body=body,
            method="POST",
        )

        # Mock the transcription to avoid actually processing
        with patch("aragora.server.handlers.transcription.transcribe_video") as mock_transcribe:
            mock_result = MagicMock()
            mock_result.text = "Test transcription"
            mock_result.language = "en"
            mock_result.duration = 10.0
            mock_result.segments = []
            mock_result.backend = "mock"
            mock_result.processing_time = 1.0
            mock_result.to_dict.return_value = {}
            mock_transcribe.return_value = mock_result

            result = await handler.handle_post("/api/v1/transcription/video", {}, mock_http)

            # Should succeed (200) or error for other reasons, not security validation
            assert result is not None
            # If it fails, it should not be due to magic byte validation
            if result.status_code == 400:
                data = json.loads(result.body)
                assert "does not match" not in data.get("error", "")

    @pytest.mark.asyncio
    async def test_video_double_extension_rejected(self, handler, mock_transcription_available):
        """Test double extension in video filename is rejected."""
        file_data = b"\x00\x00\x00\x00ftyp" + b"\x00" * 100  # Valid MP4 header
        boundary = "----TestBoundary"
        body = create_multipart_body("video.mp4.bat", file_data, boundary)

        mock_http = MockHandler(
            headers={
                "Content-Type": f"multipart/form-data; boundary=----{boundary}",
                "Content-Length": str(len(body)),
            },
            body=body,
            method="POST",
        )

        result = await handler.handle_post("/api/v1/transcription/video", {}, mock_http)

        assert result is not None
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "double extensions" in data.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_video_null_bytes_rejected(self, handler, mock_transcription_available):
        """Test null bytes in video filename are rejected."""
        file_data = b"\x00\x00\x00\x00ftyp" + b"\x00" * 100  # Valid MP4 header
        boundary = "----TestBoundary"
        body = create_multipart_body("video\x00.mp4", file_data, boundary)

        mock_http = MockHandler(
            headers={
                "Content-Type": f"multipart/form-data; boundary=----{boundary}",
                "Content-Length": str(len(body)),
            },
            body=body,
            method="POST",
        )

        result = await handler.handle_post("/api/v1/transcription/video", {}, mock_http)

        assert result is not None
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "null bytes" in data.get("error", "").lower()


# ===========================================================================
# Signature Dictionary Tests
# ===========================================================================


class TestSignatureDictionaries:
    """Tests to verify signature dictionaries are properly configured."""

    def test_audio_signatures_cover_all_formats(self):
        """Test audio signature dict covers all audio formats."""
        from aragora.server.handlers.transcription import AUDIO_FORMATS

        for ext in AUDIO_FORMATS:
            assert ext in ALLOWED_AUDIO_SIGNATURES, f"Missing signatures for {ext}"
            assert len(ALLOWED_AUDIO_SIGNATURES[ext]) > 0, f"Empty signatures for {ext}"

    def test_video_signatures_cover_all_formats(self):
        """Test video signature dict covers all video formats."""
        from aragora.server.handlers.transcription import VIDEO_FORMATS

        for ext in VIDEO_FORMATS:
            assert ext in ALLOWED_VIDEO_SIGNATURES, f"Missing signatures for {ext}"
            assert len(ALLOWED_VIDEO_SIGNATURES[ext]) > 0, f"Empty signatures for {ext}"

    def test_signature_tuples_valid_format(self):
        """Test all signature tuples have valid format (offset, bytes)."""
        for ext, sigs in ALLOWED_AUDIO_SIGNATURES.items():
            for offset, sig_bytes in sigs:
                assert isinstance(offset, int), f"Invalid offset type for {ext}"
                assert offset >= 0, f"Negative offset for {ext}"
                assert isinstance(sig_bytes, bytes), f"Invalid signature type for {ext}"
                assert len(sig_bytes) > 0, f"Empty signature for {ext}"

        for ext, sigs in ALLOWED_VIDEO_SIGNATURES.items():
            for offset, sig_bytes in sigs:
                assert isinstance(offset, int), f"Invalid offset type for {ext}"
                assert offset >= 0, f"Negative offset for {ext}"
                assert isinstance(sig_bytes, bytes), f"Invalid signature type for {ext}"
                assert len(sig_bytes) > 0, f"Empty signature for {ext}"
