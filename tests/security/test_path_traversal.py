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


# =============================================================================
# Path Validation Tests
# =============================================================================


class TestPathTraversalPrevention:
    """Test path traversal attack prevention."""

    def test_rejects_double_dot_traversal(self):
        """Should reject ../../../etc/passwd style attacks."""
        def is_safe_path(base_dir: str, requested_path: str) -> bool:
            """Check if requested path stays within base directory."""
            # Normalize backslashes to forward slashes for cross-platform
            requested_path = requested_path.replace("\\", "/")
            base = Path(base_dir).resolve()
            target = (base / requested_path).resolve()
            return str(target).startswith(str(base))

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
            # Check for traversal after decoding
            base = Path(base_dir).resolve()
            target = (base / decoded).resolve()
            return str(target).startswith(str(base))

        base_dir = "/var/www/static"

        # URL-encoded attacks
        assert normalize_and_check(base_dir, "%2e%2e/%2e%2e/etc/passwd") is False
        assert normalize_and_check(base_dir, "..%2f..%2fetc/passwd") is False
        # Double encoded: %252e%252e%252f decodes to %2e%2e%2f then to ../
        assert normalize_and_check(base_dir, "%252e%252e%252f%252e%252e%252fetc/passwd") is False

    def test_rejects_null_byte_injection(self):
        """Should reject null byte injection attempts."""
        def sanitize_path(path: str) -> str:
            """Remove null bytes and validate."""
            if "\x00" in path or "\0" in path:
                raise ValueError("Null byte in path")
            return path

        with pytest.raises(ValueError):
            sanitize_path("file.txt\x00.jpg")

        with pytest.raises(ValueError):
            sanitize_path("../../etc/passwd\x00.txt")

    def test_rejects_absolute_paths(self):
        """Should reject attempts to use absolute paths."""
        def validate_relative_path(path: str) -> bool:
            """Ensure path is relative, not absolute."""
            if path.startswith("/"):
                return False
            if path.startswith("\\"):
                return False
            # Windows drive letters
            if len(path) > 1 and path[1] == ":":
                return False
            return True

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

                def is_safe_with_symlink_check(base: Path, path: str) -> bool:
                    target = (base / path).resolve()
                    # resolve() follows symlinks
                    return str(target).startswith(str(base.resolve()))

                # Symlink should be rejected
                assert is_safe_with_symlink_check(base_dir, "safe.txt") is True
                assert is_safe_with_symlink_check(base_dir, "link") is False

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
        def sanitize_filename(filename: str) -> str:
            """Sanitize uploaded filename."""
            # Remove path separators
            filename = filename.replace("/", "_").replace("\\", "_")
            # Remove null bytes
            filename = filename.replace("\x00", "")
            # Remove leading dots (hidden files)
            filename = filename.lstrip(".")
            # Limit length
            if len(filename) > 255:
                filename = filename[:255]
            # Only allow alphanumeric, dash, underscore, dot
            import re
            filename = re.sub(r"[^a-zA-Z0-9._-]", "_", filename)
            return filename or "unnamed"

        # Normal filenames
        assert sanitize_filename("document.pdf") == "document.pdf"
        assert sanitize_filename("my-file_v2.txt") == "my-file_v2.txt"

        # Attack attempts
        # ../../../etc/passwd -> .._.._.._etc_passwd (4 slashes) -> _.._.._etc_passwd (lstrip dots)
        assert sanitize_filename("../../../etc/passwd") == "_.._.._etc_passwd"
        assert sanitize_filename("/etc/passwd") == "_etc_passwd"
        assert sanitize_filename(".htaccess") == "htaccess"
        assert sanitize_filename("file.txt\x00.jpg") == "file.txt.jpg"
        assert sanitize_filename("<script>alert(1)</script>.txt") == "_script_alert_1___script_.txt"

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
                # Reject filenames with path traversal attempts
                if ".." in filename or filename.startswith("/"):
                    raise ValueError("Path traversal attempt")

                # Sanitize filename - only keep the base name
                safe_name = Path(filename).name
                if not safe_name or safe_name.startswith("."):
                    raise ValueError("Invalid filename")

                target = (upload_dir / safe_name).resolve()

                # Verify target is within upload_dir
                if not str(target).startswith(str(upload_dir.resolve())):
                    raise ValueError("Path traversal attempt")

                target.write_bytes(content)
                return target

            # Safe upload
            result = safe_upload(upload_dir, "test.txt", b"content")
            # Compare resolved paths (macOS uses /private/var symlink)
            assert result.parent.resolve() == upload_dir.resolve()

            # Attack attempt
            with pytest.raises(ValueError):
                safe_upload(upload_dir, "../../../etc/passwd", b"malicious")


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
                # Normalize and validate path
                requested = Path(requested_path)

                # Remove leading slashes
                if requested.is_absolute():
                    return (403, "Forbidden")

                target = (static_dir / requested).resolve()

                # Must be within static_dir
                if not str(target).startswith(str(static_dir.resolve())):
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
            # Check for path traversal
            member_path = Path(member)
            if member_path.is_absolute():
                return False
            if ".." in member_path.parts:
                return False

            # Resolve and check
            target = (target_dir / member).resolve()
            if not str(target).startswith(str(target_dir.resolve())):
                return False

            return True

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
        import re

        def is_valid_debate_id(debate_id: str) -> bool:
            """Validate debate ID format."""
            # Only allow alphanumeric, dash, underscore
            pattern = r"^[a-zA-Z0-9_-]{1,64}$"
            return bool(re.match(pattern, debate_id))

        # Valid IDs
        assert is_valid_debate_id("debate-123") is True
        assert is_valid_debate_id("abc_xyz_456") is True
        assert is_valid_debate_id("a" * 64) is True

        # Invalid IDs
        assert is_valid_debate_id("../../../etc/passwd") is False
        assert is_valid_debate_id("<script>") is False
        assert is_valid_debate_id("") is False
        assert is_valid_debate_id("a" * 65) is False  # Too long
        assert is_valid_debate_id("id with spaces") is False

    def test_gauntlet_id_validation(self):
        """Gauntlet IDs should be validated."""
        import re

        def is_valid_gauntlet_id(gauntlet_id: str) -> bool:
            pattern = r"^gauntlet-[a-f0-9]{8,32}$"
            return bool(re.match(pattern, gauntlet_id))

        assert is_valid_gauntlet_id("gauntlet-abc123def456") is True
        assert is_valid_gauntlet_id("gauntlet-12345678") is True

        assert is_valid_gauntlet_id("../etc/passwd") is False
        assert is_valid_gauntlet_id("gauntlet-<script>") is False
        assert is_valid_gauntlet_id("notgauntlet-123") is False
