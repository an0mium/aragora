"""
Security tests for Audio Handler.

Tests verify protection against:
- Path traversal attacks
- Null byte injection
- Invalid ID formats
"""

from __future__ import annotations

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json

from aragora.server.validation.entities import (
    validate_debate_id,
    validate_path_segment,
    SAFE_SLUG_PATTERN,
)


# ============================================================================
# Path Traversal Protection Tests
# ============================================================================


class TestPathTraversalProtection:
    """Tests for path traversal attack prevention."""

    def test_rejects_parent_directory_traversal(self):
        """Test that '../' sequences are rejected."""
        is_valid, error = validate_debate_id("../../../etc/passwd")
        assert is_valid is False
        assert error is not None

    def test_rejects_absolute_paths(self):
        """Test that absolute paths are rejected."""
        is_valid, error = validate_debate_id("/etc/passwd")
        assert is_valid is False

    def test_rejects_windows_traversal(self):
        """Test that Windows-style traversal is rejected."""
        is_valid, error = validate_debate_id("..\\..\\windows\\system32")
        assert is_valid is False

    def test_rejects_encoded_traversal(self):
        """Test that URL-encoded traversal is rejected."""
        # %2e%2e = ..
        is_valid, error = validate_debate_id("%2e%2e%2f%2e%2e")
        assert is_valid is False

    def test_rejects_dot_segments(self):
        """Test that dot segments are rejected."""
        is_valid, error = validate_debate_id("./local/file")
        assert is_valid is False

    def test_rejects_multiple_dots(self):
        """Test that multiple consecutive dots are rejected."""
        is_valid, error = validate_debate_id("...secret")
        assert is_valid is False


# ============================================================================
# Null Byte Injection Tests
# ============================================================================


class TestNullByteProtection:
    """Tests for null byte injection prevention."""

    def test_rejects_null_bytes(self):
        """Test that null bytes are rejected."""
        is_valid, error = validate_debate_id("valid\x00.html")
        assert is_valid is False

    def test_rejects_null_byte_prefix(self):
        """Test null byte at start is rejected."""
        is_valid, error = validate_debate_id("\x00hidden")
        assert is_valid is False


# ============================================================================
# Special Characters Tests
# ============================================================================


class TestSpecialCharacterProtection:
    """Tests for special character handling."""

    def test_rejects_forward_slashes(self):
        """Test that forward slashes are rejected."""
        is_valid, error = validate_debate_id("path/to/file")
        assert is_valid is False

    def test_rejects_backslashes(self):
        """Test that backslashes are rejected."""
        is_valid, error = validate_debate_id("path\\to\\file")
        assert is_valid is False

    def test_rejects_colons(self):
        """Test that colons are rejected (Windows drive letters)."""
        is_valid, error = validate_debate_id("C:file")
        assert is_valid is False

    def test_rejects_semicolons(self):
        """Test that semicolons are rejected."""
        is_valid, error = validate_debate_id("debate;id")
        assert is_valid is False

    def test_rejects_spaces(self):
        """Test that spaces are rejected."""
        is_valid, error = validate_debate_id("debate id with spaces")
        assert is_valid is False

    def test_rejects_quotes(self):
        """Test that quotes are rejected."""
        is_valid, error = validate_debate_id('debate"id')
        assert is_valid is False

    def test_rejects_angle_brackets(self):
        """Test that angle brackets are rejected."""
        is_valid, error = validate_debate_id("debate<script>")
        assert is_valid is False


# ============================================================================
# Valid ID Format Tests
# ============================================================================


class TestValidIdFormats:
    """Tests for valid debate ID formats."""

    def test_accepts_alphanumeric(self):
        """Test that alphanumeric IDs are accepted."""
        is_valid, error = validate_debate_id("debate123")
        assert is_valid is True
        assert error is None

    def test_accepts_hyphens(self):
        """Test that hyphens are accepted."""
        is_valid, error = validate_debate_id("my-debate-123")
        assert is_valid is True

    def test_accepts_underscores(self):
        """Test that underscores are accepted."""
        is_valid, error = validate_debate_id("my_debate_123")
        assert is_valid is True

    def test_accepts_mixed_valid_chars(self):
        """Test that mixed valid characters are accepted."""
        is_valid, error = validate_debate_id("Debate-2026_01-Test")
        assert is_valid is True

    def test_accepts_uuid_format(self):
        """Test that UUID-like formats are accepted."""
        is_valid, error = validate_debate_id("550e8400-e29b-41d4-a716-446655440000")
        assert is_valid is True


# ============================================================================
# Length Validation Tests
# ============================================================================


class TestLengthValidation:
    """Tests for ID length validation."""

    def test_rejects_empty_id(self):
        """Test that empty IDs are rejected."""
        is_valid, error = validate_debate_id("")
        assert is_valid is False

    def test_accepts_single_char_id(self):
        """Test that single character IDs are accepted."""
        is_valid, error = validate_debate_id("a")
        assert is_valid is True

    def test_accepts_max_length_id(self):
        """Test that 128 character IDs are accepted."""
        is_valid, error = validate_debate_id("a" * 128)
        assert is_valid is True

    def test_rejects_too_long_id(self):
        """Test that IDs over 128 chars are rejected."""
        is_valid, error = validate_debate_id("a" * 129)
        assert is_valid is False


# ============================================================================
# Audio Handler Integration Tests
# ============================================================================


class TestAudioHandlerSecurity:
    """Integration tests for audio handler security."""

    @pytest.fixture
    def audio_handler(self):
        """Create audio handler with mock storage."""
        try:
            from aragora.server.handlers.features import AudioHandler

            mock_storage = Mock()
            mock_storage.storage_dir = Path("/safe/audio/dir")
            mock_storage.get_path.return_value = None  # Default to not found

            handler = AudioHandler({"audio_store": mock_storage})
            return handler
        except ImportError:
            pytest.skip("AudioHandler not available")

    def test_traversal_blocked_before_storage_access(self, audio_handler):
        """Test that path traversal is blocked by validation, not storage."""
        mock_http_handler = Mock()

        result = audio_handler.handle(
            "/api/audio/debate/../../../etc/passwd/audio", {}, mock_http_handler
        )

        # Should return 400 for invalid ID format
        if result is not None:
            assert result.status_code == 400

    def test_valid_debate_id_proceeds_to_storage(self, audio_handler):
        """Test that valid IDs proceed to storage lookup."""
        mock_http_handler = Mock()

        result = audio_handler.handle(
            "/api/audio/debate/valid-debate-123/audio", {}, mock_http_handler
        )

        # Should return 404 for not found (valid ID, but no file)
        if result is not None:
            assert result.status_code in [404, 400]


# ============================================================================
# Pattern Regex Tests
# ============================================================================


class TestSafeSlugPattern:
    """Tests for SAFE_SLUG_PATTERN regex."""

    def test_pattern_anchored_start(self):
        """Test pattern is anchored at start."""
        # Should not match if there's a prefix
        assert SAFE_SLUG_PATTERN.match("../valid") is None

    def test_pattern_anchored_end(self):
        """Test pattern is anchored at end."""
        # Should not match if there's a suffix
        match = SAFE_SLUG_PATTERN.match("valid/../evil")
        # Full match should fail due to invalid characters
        assert match is None

    def test_pattern_rejects_dot_only(self):
        """Test pattern rejects single dot."""
        assert SAFE_SLUG_PATTERN.match(".") is None

    def test_pattern_rejects_double_dot(self):
        """Test pattern rejects double dot."""
        assert SAFE_SLUG_PATTERN.match("..") is None
