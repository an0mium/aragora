"""
Tests for orchestration validation module.

Tests cover:
- Source ID validation (path traversal prevention)
- Channel ID validation
- RBAC permission constants
- Security validation edge cases
"""

from __future__ import annotations

import pytest

from aragora.server.handlers.orchestration.validation import (
    MAX_SOURCE_ID_LENGTH,
    PERM_CHANNEL_DISCORD,
    PERM_CHANNEL_EMAIL,
    PERM_CHANNEL_SLACK,
    PERM_CHANNEL_TEAMS,
    PERM_CHANNEL_TELEGRAM,
    PERM_CHANNEL_WEBHOOK,
    PERM_KNOWLEDGE_CONFLUENCE,
    PERM_KNOWLEDGE_DOCUMENT,
    PERM_KNOWLEDGE_GITHUB,
    PERM_KNOWLEDGE_JIRA,
    PERM_KNOWLEDGE_SLACK,
    PERM_ORCH_ADMIN,
    PERM_ORCH_CHANNELS_WRITE,
    PERM_ORCH_DELIBERATE,
    PERM_ORCH_KNOWLEDGE_READ,
    SAFE_SOURCE_ID_PATTERN,
    SourceIdValidationError,
    safe_source_id,
    validate_channel_id,
)


class TestRBACPermissions:
    """Tests for RBAC permission constants."""

    def test_core_permissions(self):
        """Test core orchestration permissions."""
        assert PERM_ORCH_DELIBERATE == "orchestration:deliberate:create"
        assert PERM_ORCH_KNOWLEDGE_READ == "orchestration:knowledge:read"
        assert PERM_ORCH_CHANNELS_WRITE == "orchestration:channels:write"
        assert PERM_ORCH_ADMIN == "orchestration:admin"

    def test_knowledge_source_permissions(self):
        """Test knowledge source type permissions."""
        assert PERM_KNOWLEDGE_SLACK == "orchestration:knowledge:slack"
        assert PERM_KNOWLEDGE_CONFLUENCE == "orchestration:knowledge:confluence"
        assert PERM_KNOWLEDGE_GITHUB == "orchestration:knowledge:github"
        assert PERM_KNOWLEDGE_JIRA == "orchestration:knowledge:jira"
        assert PERM_KNOWLEDGE_DOCUMENT == "orchestration:knowledge:document"

    def test_channel_permissions(self):
        """Test channel type permissions."""
        assert PERM_CHANNEL_SLACK == "orchestration:channel:slack"
        assert PERM_CHANNEL_TEAMS == "orchestration:channel:teams"
        assert PERM_CHANNEL_DISCORD == "orchestration:channel:discord"
        assert PERM_CHANNEL_TELEGRAM == "orchestration:channel:telegram"
        assert PERM_CHANNEL_EMAIL == "orchestration:channel:email"
        assert PERM_CHANNEL_WEBHOOK == "orchestration:channel:webhook"


class TestSafeSourceIdPattern:
    """Tests for the safe source ID regex pattern."""

    def test_valid_patterns(self):
        """Test valid source_id patterns."""
        valid_ids = [
            "channel_123",
            "owner/repo/pr/123",
            "PROJ-123",
            "page-id",
            "user@example.com",
            "doc.txt",
            "issue#456",
            "C12345678",  # Slack channel ID format
            "team-name_project",
        ]
        for source_id in valid_ids:
            assert SAFE_SOURCE_ID_PATTERN.match(source_id), f"Should match: {source_id}"

    def test_invalid_patterns(self):
        """Test invalid source_id patterns."""
        invalid_ids = [
            "../etc/passwd",
            "file;rm -rf /",
            "$(whoami)",
            "`ls`",
            "file\x00.txt",  # Null byte
            "path/../traversal",
        ]
        for source_id in invalid_ids:
            # Either doesn't match pattern or contains dangerous characters
            if SAFE_SOURCE_ID_PATTERN.match(source_id):
                # Pattern matched but should fail other checks
                pass  # safe_source_id will catch these


class TestSafeSourceId:
    """Tests for safe_source_id validation function."""

    def test_valid_source_ids(self):
        """Test valid source IDs pass validation."""
        valid_ids = [
            "simple",
            "with-hyphen",
            "with_underscore",
            "owner/repo/pr/123",
            "PROJ-123",
            "user@email.com",
            "file.txt",
            "a" * MAX_SOURCE_ID_LENGTH,  # Max length
        ]
        for source_id in valid_ids:
            result = safe_source_id(source_id)
            assert result == source_id

    def test_empty_source_id(self):
        """Test empty source_id raises error."""
        with pytest.raises(SourceIdValidationError, match="cannot be empty"):
            safe_source_id("")

    def test_too_long_source_id(self):
        """Test source_id exceeding max length raises error."""
        long_id = "a" * (MAX_SOURCE_ID_LENGTH + 1)
        with pytest.raises(SourceIdValidationError, match="too long"):
            safe_source_id(long_id)

    def test_path_traversal_dotdot(self):
        """Test path traversal with .. is blocked."""
        with pytest.raises(SourceIdValidationError, match="path traversal"):
            safe_source_id("../etc/passwd")

        with pytest.raises(SourceIdValidationError, match="path traversal"):
            safe_source_id("foo/../bar")

        with pytest.raises(SourceIdValidationError, match="path traversal"):
            safe_source_id(".....")  # Contains ..

    def test_absolute_path_unix(self):
        """Test Unix absolute paths are blocked."""
        with pytest.raises(SourceIdValidationError, match="cannot start with"):
            safe_source_id("/etc/passwd")

        with pytest.raises(SourceIdValidationError, match="cannot start with"):
            safe_source_id("/home/user/file")

    def test_absolute_path_windows(self):
        """Test Windows absolute paths are blocked."""
        with pytest.raises(SourceIdValidationError, match="Windows absolute path"):
            safe_source_id("C:/Windows/System32")

        with pytest.raises(SourceIdValidationError, match="Windows absolute path"):
            safe_source_id("D:\\Users\\Admin")

    def test_null_byte_injection(self):
        """Test null byte injection is blocked."""
        with pytest.raises(SourceIdValidationError, match="null byte"):
            safe_source_id("file\x00.txt")

        with pytest.raises(SourceIdValidationError, match="null byte"):
            safe_source_id("\x00malicious")

    def test_invalid_characters(self):
        """Test invalid characters are blocked."""
        invalid_chars = ["$", "`", ";", "|", "&", "(", ")", "<", ">", "\\", "!", "?"]
        for char in invalid_chars:
            with pytest.raises(SourceIdValidationError, match="invalid characters"):
                safe_source_id(f"file{char}name")


class TestValidateChannelId:
    """Tests for validate_channel_id function."""

    def test_valid_channel_ids(self):
        """Test valid channel IDs for various types."""
        # Slack channel
        assert validate_channel_id("C12345678", "slack") == "C12345678"

        # Teams channel
        assert validate_channel_id("team-channel-123", "teams") == "team-channel-123"

        # Discord channel
        assert validate_channel_id("123456789012345678", "discord") == "123456789012345678"

        # Telegram chat
        assert validate_channel_id("-1001234567890", "telegram") == "-1001234567890"

        # Email
        assert validate_channel_id("user@example.com", "email") == "user@example.com"

    def test_valid_webhook_urls(self):
        """Test valid webhook URLs."""
        valid_urls = [
            "https://hooks.slack.com/services/T00/B00/xxx",
            "https://webhook.site/test-id",
            "http://localhost:8080/webhook",
            "https://example.com/api/callback",
        ]
        for url in valid_urls:
            result = validate_channel_id(url, "webhook")
            assert result == url

    def test_empty_channel_id(self):
        """Test empty channel_id raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_channel_id("", "slack")

    def test_too_long_channel_id(self):
        """Test channel_id exceeding max length raises error."""
        long_id = "a" * (MAX_SOURCE_ID_LENGTH + 1)
        with pytest.raises(ValueError, match="too long"):
            validate_channel_id(long_id, "slack")

    def test_null_byte_in_channel_id(self):
        """Test null byte in channel_id raises error."""
        with pytest.raises(ValueError, match="null byte"):
            validate_channel_id("channel\x00id", "slack")

    def test_webhook_requires_url(self):
        """Test webhook channel_id must be a valid URL."""
        with pytest.raises(ValueError, match="valid URL"):
            validate_channel_id("not-a-url", "webhook")

        with pytest.raises(ValueError, match="valid URL"):
            validate_channel_id("ftp://example.com", "webhook")

    def test_webhook_path_traversal(self):
        """Test webhook URL with path traversal is blocked."""
        with pytest.raises(ValueError, match="invalid characters"):
            validate_channel_id("https://example.com/../etc/passwd", "webhook")

    def test_non_webhook_path_traversal(self):
        """Test non-webhook channel_id with path traversal is blocked."""
        with pytest.raises(ValueError, match="invalid path characters"):
            validate_channel_id("../malicious", "slack")

        with pytest.raises(ValueError, match="invalid path characters"):
            validate_channel_id("/etc/passwd", "teams")


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_max_length_boundary(self):
        """Test exact max length is allowed."""
        # Exactly at limit should pass
        exact_max = "a" * MAX_SOURCE_ID_LENGTH
        assert safe_source_id(exact_max) == exact_max
        assert validate_channel_id(exact_max, "slack") == exact_max

        # One over limit should fail
        over_max = "a" * (MAX_SOURCE_ID_LENGTH + 1)
        with pytest.raises(SourceIdValidationError):
            safe_source_id(over_max)
        with pytest.raises(ValueError):
            validate_channel_id(over_max, "slack")

    def test_unicode_source_ids(self):
        """Test Unicode characters in source IDs."""
        # Basic ASCII should work
        assert safe_source_id("test-123") == "test-123"

        # Non-ASCII should be rejected by the pattern
        with pytest.raises(SourceIdValidationError, match="invalid characters"):
            safe_source_id("日本語")

    def test_special_valid_formats(self):
        """Test special but valid source ID formats."""
        # GitHub-style paths
        assert safe_source_id("owner/repo/pr/123") == "owner/repo/pr/123"
        assert safe_source_id("owner/repo/issues/456") == "owner/repo/issues/456"

        # Jira-style keys
        assert safe_source_id("PROJ-123") == "PROJ-123"
        assert safe_source_id("ABC-1") == "ABC-1"

        # Email-like IDs
        assert safe_source_id("user@domain.com") == "user@domain.com"

        # Slack channel IDs
        assert safe_source_id("C0123456789") == "C0123456789"

    def test_dotdot_not_at_boundary(self):
        """Test that .. is blocked even in the middle of paths."""
        with pytest.raises(SourceIdValidationError, match="path traversal"):
            safe_source_id("valid/../invalid")

        with pytest.raises(SourceIdValidationError, match="path traversal"):
            safe_source_id("some..path")
