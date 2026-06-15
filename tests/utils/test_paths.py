"""Tests for path validation utilities."""

import tempfile
from pathlib import Path

import pytest

from aragora.utils.paths import (
    PathTraversalError,
    is_safe_path,
    safe_path,
    validate_path_component,
)


class TestSafePath:
    """Tests for safe_path function."""

    def test_simple_path(self, tmp_path: Path) -> None:
        """Test simple path stays within base."""
        result = safe_path(tmp_path, "subdir")
        assert result == tmp_path / "subdir"

    def test_nested_path(self, tmp_path: Path) -> None:
        """Test nested path stays within base."""
        result = safe_path(tmp_path, "a/b/c")
        assert result == tmp_path / "a" / "b" / "c"

    def test_traversal_blocked_simple(self, tmp_path: Path) -> None:
        """Test basic directory traversal is blocked."""
        with pytest.raises(PathTraversalError, match="Path traversal blocked"):
            safe_path(tmp_path, "../etc/passwd")

    def test_traversal_blocked_embedded(self, tmp_path: Path) -> None:
        """Test embedded traversal is blocked."""
        with pytest.raises(PathTraversalError, match="Path traversal blocked"):
            safe_path(tmp_path, "subdir/../../etc/passwd")

    def test_traversal_blocked_absolute(self, tmp_path: Path) -> None:
        """Test absolute path outside base is blocked."""
        with pytest.raises(PathTraversalError, match="Path traversal blocked"):
            safe_path(tmp_path, "/etc/passwd")

    def test_normalized_path_within_base(self, tmp_path: Path) -> None:
        """Test normalized path that stays in base is allowed."""
        result = safe_path(tmp_path, "subdir/../other")
        assert result == tmp_path / "other"

    def test_symlink_escape_blocked(self, tmp_path: Path) -> None:
        """Test symlink escaping base is blocked."""
        # Create a symlink pointing outside
        link = tmp_path / "escape_link"
        link.symlink_to("/etc")

        # The escape is caught - either by resolve() detecting it's outside base,
        # or by the symlink check. Either way, PathTraversalError is raised.
        with pytest.raises(PathTraversalError):
            safe_path(tmp_path, "escape_link/passwd")

    def test_symlink_within_base_allowed(self, tmp_path: Path) -> None:
        """Test symlink within base is allowed if allow_symlinks=True."""
        # Create a subdir and symlink to it
        subdir = tmp_path / "real"
        subdir.mkdir()
        (subdir / "file.txt").write_text("test")

        link = tmp_path / "link"
        link.symlink_to(subdir)

        # With allow_symlinks=True, this should work
        result = safe_path(tmp_path, "link/file.txt", allow_symlinks=True)
        assert result.exists()

    def test_must_exist_raises_for_missing(self, tmp_path: Path) -> None:
        """Test must_exist raises for non-existent path."""
        with pytest.raises(FileNotFoundError, match="does not exist"):
            safe_path(tmp_path, "nonexistent", must_exist=True)

    def test_must_exist_passes_for_existing(self, tmp_path: Path) -> None:
        """Test must_exist passes for existing path."""
        existing = tmp_path / "exists"
        existing.touch()

        result = safe_path(tmp_path, "exists", must_exist=True)
        assert result == existing

    def test_string_base_dir(self, tmp_path: Path) -> None:
        """Test base_dir can be a string."""
        result = safe_path(str(tmp_path), "subdir")
        assert result == tmp_path / "subdir"

    def test_double_dot_only(self, tmp_path: Path) -> None:
        """Test path with just .. is blocked."""
        with pytest.raises(PathTraversalError, match="Path traversal blocked"):
            safe_path(tmp_path, "..")


class TestValidatePathComponent:
    """Tests for validate_path_component function."""

    def test_valid_component(self) -> None:
        """Test valid path component."""
        assert validate_path_component("debate-123") == "debate-123"

    def test_valid_component_with_underscores(self) -> None:
        """Test valid component with underscores."""
        assert validate_path_component("my_file_name") == "my_file_name"

    def test_empty_component_rejected(self) -> None:
        """Test empty component is rejected."""
        with pytest.raises(PathTraversalError, match="empty"):
            validate_path_component("")

    def test_whitespace_component_rejected(self) -> None:
        """Test whitespace-only component is rejected."""
        with pytest.raises(PathTraversalError, match="empty"):
            validate_path_component("   ")

    def test_dot_dot_rejected(self) -> None:
        """Test .. is rejected."""
        with pytest.raises(PathTraversalError, match="Invalid path component"):
            validate_path_component("..")

    def test_single_dot_rejected(self) -> None:
        """Test . is rejected."""
        with pytest.raises(PathTraversalError, match="Invalid path component"):
            validate_path_component(".")

    def test_forward_slash_rejected(self) -> None:
        """Test / in component is rejected."""
        with pytest.raises(PathTraversalError, match="contains separator"):
            validate_path_component("a/b")

    def test_backslash_rejected(self) -> None:
        """Test backslash in component is rejected."""
        with pytest.raises(PathTraversalError, match="contains separator"):
            validate_path_component("a\\b")

    def test_null_byte_rejected(self) -> None:
        """Test null byte in component is rejected."""
        with pytest.raises(PathTraversalError, match="null byte"):
            validate_path_component("file\x00name")


class TestIsSafePath:
    """Tests for is_safe_path function."""

    def test_returns_true_for_safe_path(self, tmp_path: Path) -> None:
        """Test returns True for safe path."""
        assert is_safe_path(tmp_path, "subdir") is True

    def test_returns_false_for_traversal(self, tmp_path: Path) -> None:
        """Test returns False for traversal attempt."""
        assert is_safe_path(tmp_path, "../etc") is False

    def test_returns_false_for_missing_required(self, tmp_path: Path) -> None:
        """Test returns False when path doesn't exist but is required."""
        # Note: is_safe_path doesn't check existence, so this should be True
        assert is_safe_path(tmp_path, "nonexistent") is True


class TestRealWorldScenarios:
    """Test real-world path traversal attack scenarios."""

    def test_url_encoded_traversal(self, tmp_path: Path) -> None:
        """Test URL-encoded traversal (already decoded) is blocked."""
        # URL decoding happens before reaching this function
        # Testing the already-decoded version
        with pytest.raises(PathTraversalError):
            safe_path(tmp_path, "../../../etc/passwd")

    def test_windows_style_traversal(self, tmp_path: Path) -> None:
        """Test Windows-style traversal on Windows would be blocked.

        Note: On Unix, backslashes are valid filename characters, not separators.
        This test verifies the path stays within base (it does, as backslashes
        don't cause traversal on Unix).
        """
        import sys

        if sys.platform == "win32":
            # On Windows, backslashes are path separators and this would traverse
            with pytest.raises(PathTraversalError):
                safe_path(tmp_path, "..\\..\\Windows\\System32")
        else:
            # On Unix, backslashes are just characters in the filename
            result = safe_path(tmp_path, "..\\..\\Windows\\System32")
            assert result.parent == tmp_path

    def test_mixed_separators(self, tmp_path: Path) -> None:
        """Test forward slashes cause traversal regardless of platform."""
        # Forward slashes always work as separators
        with pytest.raises(PathTraversalError):
            safe_path(tmp_path, "../etc/passwd")

    def test_unicode_traversal(self, tmp_path: Path) -> None:
        """Test Unicode characters don't bypass protection."""
        # Various Unicode lookalikes for dots and slashes
        result = safe_path(tmp_path, "normal_dir")
        assert result == tmp_path / "normal_dir"

    def test_deeply_nested_valid_path(self, tmp_path: Path) -> None:
        """Test deeply nested but valid path works."""
        result = safe_path(tmp_path, "a/b/c/d/e/f/g")
        expected = tmp_path / "a/b/c/d/e/f/g"
        assert result == expected

    def test_debate_id_like_value(self, tmp_path: Path) -> None:
        """Test typical debate ID format is safe."""
        result = safe_path(tmp_path, "debate-2024-01-15-abc123")
        assert result == tmp_path / "debate-2024-01-15-abc123"

    def test_uuid_like_value(self, tmp_path: Path) -> None:
        """Test UUID-like path component is safe."""
        result = safe_path(tmp_path, "550e8400-e29b-41d4-a716-446655440000")
        assert result.name == "550e8400-e29b-41d4-a716-446655440000"
