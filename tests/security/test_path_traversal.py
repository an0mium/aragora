"""
Path traversal security tests.

Verifies that path traversal attacks are prevented in file serving
and file upload/download operations.
"""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from aragora.utils.paths import PathTraversalError, safe_path, validate_path_component
from aragora.server.validation.entities import validate_debate_id, validate_gauntlet_id


# =============================================================================
# Path Validation Tests
# =============================================================================


class TestPathTraversalPrevention:
    """Test path traversal attack prevention."""

    def test_rejects_double_dot_traversal(self):
        """Should reject ../../../etc/passwd style attacks."""

        def is_safe_path(base_dir: str, requested_path: str) -> bool:
            """Check if requested path stays within base directory."""
            try:
                safe_path(base_dir, requested_path)
                return True
            except PathTraversalError:
                return False

        base_dir = "/var/www/static"

        # Safe paths
        assert is_safe_path(base_dir, "style.css") is True
        assert is_safe_path(base_dir, "js/app.js") is True
        assert is_safe_path(base_dir, "images/logo.png") is True

        # Traversal attempts
        assert is_safe_path(base_dir, "../etc/passwd") is False
        assert is_safe_path(base_dir, "../../etc/passwd") is False
        assert is_safe_path(base_dir, "../../../etc/passwd") is False
        assert is_safe_path(base_dir, "..\\..\\etc\\passwd") is False

    def test_rejects_encoded_traversal(self):
        """Should reject URL-encoded traversal attempts."""
        from urllib.parse import unquote

        def normalize_and_check(base_dir: str, path: str) -> bool:
            # Decode URL encoding (decode twice to catch double-encoding)
            decoded = unquote(unquote(path))
            try:
                safe_path(base_dir, decoded)
                return True
            except PathTraversalError:
                return False

        base_dir = "/var/www/static"

        # URL-encoded attacks
        assert normalize_and_check(base_dir, "%2e%2e/%2e%2e/etc/passwd") is False
        assert normalize_and_check(base_dir, "..%2f..%2fetc/passwd") is False
        # Double encoded: %252e%252e%252f decodes to %2e%2e%2f then to ../
        assert normalize_and_check(base_dir, "%252e%252e%252f%252e%252e%252fetc/passwd") is False

    def test_rejects_null_byte_injection(self):
        """Should reject null byte injection attempts."""
        with pytest.raises(PathTraversalError):
            validate_path_component("file.txt\x00.jpg")

        with pytest.raises(PathTraversalError):
            validate_path_component("../../etc/passwd\x00.txt")

    def test_rejects_absolute_paths(self):
        """Should reject attempts to use absolute paths."""

        def validate_relative_path(path: str) -> bool:
            """Ensure path is relative, not absolute."""
            try:
                safe_path("/var/www/static", path)
                return True
            except PathTraversalError:
                return False

        assert validate_relative_path("style.css") is True
        assert validate_relative_path("js/app.js") is True

        assert validate_relative_path("/etc/passwd") is False
        assert validate_relative_path("\\Windows\\System32") is False
        assert validate_relative_path("C:\\Windows\\System32") is False

    def test_rejects_symlink_escape(self):
        """Should reject symlinks that escape the base directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir) / "static"
            base_dir.mkdir()

            # Create a file in base
            safe_file = base_dir / "safe.txt"
            safe_file.write_text("safe content")

            # Create a symlink pointing outside
            outside_file = Path(tmpdir) / "outside.txt"
            outside_file.write_text("outside content")

            symlink = base_dir / "link"
            try:
                symlink.symlink_to(outside_file)

                # Symlink should be rejected
                safe_path(base_dir, "safe.txt")  # Should not raise
                with pytest.raises(PathTraversalError):
                    safe_path(base_dir, "link")

            except OSError:
                # Symlinks may not be supported on all systems
                pytest.skip("Symlinks not supported")


# =============================================================================
# File Upload Path Tests
# =============================================================================


class TestFileUploadPathSecurity:
    """Test file upload path security."""

    def test_sanitize_uploaded_filename(self):
        """Uploaded filenames should be sanitized."""
        # Normal filenames
        assert validate_path_component("document.pdf") == "document.pdf"
        assert validate_path_component("my-file_v2.txt") == "my-file_v2.txt"

        # Attack attempts
        with pytest.raises(PathTraversalError):
            validate_path_component("../../../etc/passwd")
        with pytest.raises(PathTraversalError):
            validate_path_component("/etc/passwd")
        with pytest.raises(PathTraversalError):
            validate_path_component("file.txt\x00.jpg")

    def test_unique_upload_paths(self):
        """Uploaded files should get unique paths to prevent overwriting."""
        import uuid

        def generate_upload_path(original_filename: str, upload_dir: str) -> str:
            """Generate unique upload path."""
            ext = Path(original_filename).suffix
            unique_name = f"{uuid.uuid4().hex}{ext}"
            return str(Path(upload_dir) / unique_name)

        path1 = generate_upload_path("document.pdf", "/uploads")
        path2 = generate_upload_path("document.pdf", "/uploads")

        # Same original filename should get different paths
        assert path1 != path2

    def test_upload_directory_stays_contained(self):
        """Uploads should stay in designated directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            upload_dir = Path(tmpdir) / "uploads"
            upload_dir.mkdir()

            def safe_upload(upload_dir: Path, filename: str, content: bytes) -> Path:
                safe_name = validate_path_component(Path(filename).name)
                target = safe_path(upload_dir, safe_name)

                target.write_bytes(content)
                return target

            # Safe upload
            result = safe_upload(upload_dir, "test.txt", b"content")
            # Compare resolved paths (macOS uses /private/var symlink)
            assert result.parent.resolve() == upload_dir.resolve()

            # Attack attempt - should be neutralized to basename
            result = safe_upload(upload_dir, "../../../etc/passwd", b"malicious")
            assert result.parent.resolve() == upload_dir.resolve()
            assert result.name == "passwd"


# =============================================================================
# Static File Serving Tests
# =============================================================================


class TestStaticFileServingSecurity:
    """Test static file serving security."""

    def test_static_files_restricted_to_directory(self):
        """Static file requests should be restricted to static directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            static_dir = Path(tmpdir) / "static"
            static_dir.mkdir()

            # Create a static file
            (static_dir / "style.css").write_text("body { color: red; }")

            # Create a secret file outside static
            secret = Path(tmpdir) / "secret.txt"
            secret.write_text("secret data")

            def serve_static(static_dir: Path, requested_path: str) -> tuple[int, str]:
                """Simulate static file serving."""
                try:
                    target = safe_path(static_dir, requested_path)
                except PathTraversalError:
                    return (403, "Forbidden")

                if not target.exists():
                    return (404, "Not Found")

                return (200, target.read_text())

            # Valid request
            status, content = serve_static(static_dir, "style.css")
            assert status == 200
            assert "color: red" in content

            # Traversal attempt
            status, content = serve_static(static_dir, "../secret.txt")
            assert status == 403

    def test_directory_listing_disabled(self):
        """Directory listing should be disabled."""

        def should_serve(path: Path) -> bool:
            """Only serve files, not directories."""
            return path.is_file()

        with tempfile.TemporaryDirectory() as tmpdir:
            static_dir = Path(tmpdir)
            (static_dir / "file.txt").write_text("content")

            assert should_serve(static_dir / "file.txt") is True
            assert should_serve(static_dir) is False

    def test_hidden_files_not_served(self):
        """Hidden files (starting with .) should not be served."""

        def is_hidden_file(filename: str) -> bool:
            return filename.startswith(".")

        assert is_hidden_file(".htaccess") is True
        assert is_hidden_file(".env") is True
        assert is_hidden_file(".git") is True
        assert is_hidden_file("normal.txt") is False


# =============================================================================
# Archive Extraction Tests
# =============================================================================


class TestArchiveExtractionSecurity:
    """Test security of archive extraction (zip bombs, path traversal)."""

    def test_zip_path_traversal_prevention(self):
        """Zip extraction should prevent path traversal."""
        import zipfile
        import io

        def safe_extract_member(zip_file: zipfile.ZipFile, member: str, target_dir: Path) -> bool:
            """Safely extract a single zip member."""
            try:
                safe_path(target_dir, member)
                return True
            except PathTraversalError:
                return False

        with tempfile.TemporaryDirectory() as tmpdir:
            target_dir = Path(tmpdir) / "extracted"
            target_dir.mkdir()

            # Create a malicious zip in memory
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                # Normal file
                zf.writestr("normal.txt", "safe content")
                # Malicious path
                zf.writestr("../../../etc/passwd", "malicious")

            zip_buffer.seek(0)

            with zipfile.ZipFile(zip_buffer, "r") as zf:
                for name in zf.namelist():
                    is_safe = safe_extract_member(zf, name, target_dir)
                    if name == "normal.txt":
                        assert is_safe is True
                    else:
                        assert is_safe is False

    def test_zip_bomb_size_limit(self):
        """Zip extraction should have size limits."""
        MAX_EXTRACTED_SIZE = 100 * 1024 * 1024  # 100MB
        MAX_FILE_COUNT = 1000

        def check_zip_limits(zip_info_list: list) -> bool:
            """Check if zip extraction would exceed limits."""
            total_size = sum(info.file_size for info in zip_info_list)
            file_count = len(zip_info_list)

            return total_size <= MAX_EXTRACTED_SIZE and file_count <= MAX_FILE_COUNT

        # Mock zip info
        class MockZipInfo:
            def __init__(self, size):
                self.file_size = size

        # Normal zip
        normal_files = [MockZipInfo(1024) for _ in range(10)]
        assert check_zip_limits(normal_files) is True

        # Zip bomb (huge uncompressed size)
        bomb_files = [MockZipInfo(1024 * 1024 * 1024) for _ in range(10)]  # 10GB
        assert check_zip_limits(bomb_files) is False

        # Too many files
        many_files = [MockZipInfo(100) for _ in range(10000)]
        assert check_zip_limits(many_files) is False


# =============================================================================
# API Path Parameter Tests
# =============================================================================


class TestAPIPathParameterSecurity:
    """Test API path parameter security."""

    def test_debate_id_validation(self):
        """Debate IDs should be validated."""
        # Valid IDs
        assert validate_debate_id("debate-123")[0] is True
        assert validate_debate_id("abc_xyz_456")[0] is True
        assert validate_debate_id("a" * 64)[0] is True

        # Invalid IDs
        assert validate_debate_id("../../../etc/passwd")[0] is False
        assert validate_debate_id("<script>")[0] is False
        assert validate_debate_id("")[0] is False
        assert validate_debate_id("a" * 129)[0] is False  # Too long
        assert validate_debate_id("id with spaces")[0] is False

    def test_gauntlet_id_validation(self):
        """Gauntlet IDs should be validated."""
        assert validate_gauntlet_id("gauntlet-abc123def456")[0] is True
        assert validate_gauntlet_id("gauntlet-12345678")[0] is True

        assert validate_gauntlet_id("../etc/passwd")[0] is False
        assert validate_gauntlet_id("gauntlet-<script>")[0] is False
        assert validate_gauntlet_id("notgauntlet-123")[0] is False
