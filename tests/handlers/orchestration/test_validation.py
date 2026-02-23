"""Tests for orchestration validation module.

Covers all public functions, constants, error handling, and edge cases in
aragora/server/handlers/orchestration/validation.py:

- RBAC permission constants (existence and naming convention)
- SAFE_SOURCE_ID_PATTERN regex correctness
- MAX_SOURCE_ID_LENGTH constant
- SourceIdValidationError exception class
- safe_source_id() -- valid IDs, empty, too long, path traversal, absolute paths,
  Windows paths, null bytes, invalid characters, boundary cases
- validate_channel_id() -- valid IDs, empty, too long, null bytes, webhook URLs,
  non-webhook path traversal, boundary cases
"""

from __future__ import annotations

import logging
import string
from unittest.mock import patch

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


# =============================================================================
# Permission constant tests
# =============================================================================


class TestPermissionConstants:
    """Verify RBAC permission constants exist with correct values."""

    def test_core_deliberate_permission(self):
        assert PERM_ORCH_DELIBERATE == "orchestration:deliberate:create"

    def test_core_knowledge_read_permission(self):
        assert PERM_ORCH_KNOWLEDGE_READ == "orchestration:knowledge:read"

    def test_core_channels_write_permission(self):
        assert PERM_ORCH_CHANNELS_WRITE == "orchestration:channels:write"

    def test_core_admin_permission(self):
        assert PERM_ORCH_ADMIN == "orchestration:admin"

    def test_knowledge_source_permissions(self):
        assert PERM_KNOWLEDGE_SLACK == "orchestration:knowledge:slack"
        assert PERM_KNOWLEDGE_CONFLUENCE == "orchestration:knowledge:confluence"
        assert PERM_KNOWLEDGE_GITHUB == "orchestration:knowledge:github"
        assert PERM_KNOWLEDGE_JIRA == "orchestration:knowledge:jira"
        assert PERM_KNOWLEDGE_DOCUMENT == "orchestration:knowledge:document"

    def test_channel_type_permissions(self):
        assert PERM_CHANNEL_SLACK == "orchestration:channel:slack"
        assert PERM_CHANNEL_TEAMS == "orchestration:channel:teams"
        assert PERM_CHANNEL_DISCORD == "orchestration:channel:discord"
        assert PERM_CHANNEL_TELEGRAM == "orchestration:channel:telegram"
        assert PERM_CHANNEL_EMAIL == "orchestration:channel:email"
        assert PERM_CHANNEL_WEBHOOK == "orchestration:channel:webhook"

    def test_all_permissions_are_strings(self):
        perms = [
            PERM_ORCH_DELIBERATE,
            PERM_ORCH_KNOWLEDGE_READ,
            PERM_ORCH_CHANNELS_WRITE,
            PERM_ORCH_ADMIN,
            PERM_KNOWLEDGE_SLACK,
            PERM_KNOWLEDGE_CONFLUENCE,
            PERM_KNOWLEDGE_GITHUB,
            PERM_KNOWLEDGE_JIRA,
            PERM_KNOWLEDGE_DOCUMENT,
            PERM_CHANNEL_SLACK,
            PERM_CHANNEL_TEAMS,
            PERM_CHANNEL_DISCORD,
            PERM_CHANNEL_TELEGRAM,
            PERM_CHANNEL_EMAIL,
            PERM_CHANNEL_WEBHOOK,
        ]
        for perm in perms:
            assert isinstance(perm, str)

    def test_all_permissions_start_with_orchestration(self):
        perms = [
            PERM_ORCH_DELIBERATE,
            PERM_ORCH_KNOWLEDGE_READ,
            PERM_ORCH_CHANNELS_WRITE,
            PERM_ORCH_ADMIN,
            PERM_KNOWLEDGE_SLACK,
            PERM_KNOWLEDGE_CONFLUENCE,
            PERM_KNOWLEDGE_GITHUB,
            PERM_KNOWLEDGE_JIRA,
            PERM_KNOWLEDGE_DOCUMENT,
            PERM_CHANNEL_SLACK,
            PERM_CHANNEL_TEAMS,
            PERM_CHANNEL_DISCORD,
            PERM_CHANNEL_TELEGRAM,
            PERM_CHANNEL_EMAIL,
            PERM_CHANNEL_WEBHOOK,
        ]
        for perm in perms:
            assert perm.startswith("orchestration:"), f"{perm} missing prefix"


# =============================================================================
# Pattern and constant tests
# =============================================================================


class TestPatternAndConstants:
    """Verify the compiled regex and length constant."""

    def test_max_source_id_length_value(self):
        assert MAX_SOURCE_ID_LENGTH == 256

    def test_safe_pattern_allows_alphanumeric(self):
        assert SAFE_SOURCE_ID_PATTERN.match("abc123")

    def test_safe_pattern_allows_hyphens(self):
        assert SAFE_SOURCE_ID_PATTERN.match("my-source")

    def test_safe_pattern_allows_underscores(self):
        assert SAFE_SOURCE_ID_PATTERN.match("my_source")

    def test_safe_pattern_allows_colons(self):
        assert SAFE_SOURCE_ID_PATTERN.match("prefix:value")

    def test_safe_pattern_allows_slashes(self):
        assert SAFE_SOURCE_ID_PATTERN.match("owner/repo/pr/123")

    def test_safe_pattern_allows_at(self):
        assert SAFE_SOURCE_ID_PATTERN.match("user@domain")

    def test_safe_pattern_allows_dots(self):
        assert SAFE_SOURCE_ID_PATTERN.match("file.txt")

    def test_safe_pattern_allows_hash(self):
        assert SAFE_SOURCE_ID_PATTERN.match("channel#general")

    def test_safe_pattern_rejects_spaces(self):
        assert not SAFE_SOURCE_ID_PATTERN.match("has space")

    def test_safe_pattern_rejects_newlines(self):
        assert not SAFE_SOURCE_ID_PATTERN.match("has\nnewline")

    def test_safe_pattern_rejects_backticks(self):
        assert not SAFE_SOURCE_ID_PATTERN.match("has`backtick")

    def test_safe_pattern_rejects_shell_metacharacters(self):
        for ch in "$;|&<>`!{}[]()":
            assert not SAFE_SOURCE_ID_PATTERN.match(f"bad{ch}id"), f"should reject '{ch}'"

    def test_safe_pattern_rejects_empty(self):
        assert not SAFE_SOURCE_ID_PATTERN.match("")


# =============================================================================
# SourceIdValidationError tests
# =============================================================================


class TestSourceIdValidationError:
    """Verify custom exception type."""

    def test_is_value_error_subclass(self):
        assert issubclass(SourceIdValidationError, ValueError)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(SourceIdValidationError):
            raise SourceIdValidationError("test")

    def test_caught_by_value_error(self):
        with pytest.raises(ValueError):
            raise SourceIdValidationError("test")

    def test_message_preserved(self):
        with pytest.raises(SourceIdValidationError, match="custom msg"):
            raise SourceIdValidationError("custom msg")


# =============================================================================
# safe_source_id() tests
# =============================================================================


class TestSafeSourceId:
    """Comprehensive tests for the safe_source_id validator."""

    # -- Valid IDs --

    def test_simple_alphanumeric(self):
        assert safe_source_id("abc123") == "abc123"

    def test_github_style_path(self):
        assert safe_source_id("owner/repo/pr/123") == "owner/repo/pr/123"

    def test_jira_style_key(self):
        assert safe_source_id("PROJ-123") == "PROJ-123"

    def test_channel_id_with_underscores(self):
        assert safe_source_id("channel_id_42") == "channel_id_42"

    def test_source_with_at_sign(self):
        assert safe_source_id("user@domain.com") == "user@domain.com"

    def test_source_with_hash(self):
        assert safe_source_id("slack#general") == "slack#general"

    def test_source_with_colon(self):
        assert safe_source_id("prefix:value:sub") == "prefix:value:sub"

    def test_source_single_char(self):
        assert safe_source_id("x") == "x"

    def test_source_at_max_length(self):
        source_id = "a" * MAX_SOURCE_ID_LENGTH
        assert safe_source_id(source_id) == source_id

    # -- Empty --

    def test_empty_string_raises(self):
        with pytest.raises(SourceIdValidationError, match="cannot be empty"):
            safe_source_id("")

    # -- Too long --

    def test_exceeds_max_length(self):
        source_id = "a" * (MAX_SOURCE_ID_LENGTH + 1)
        with pytest.raises(SourceIdValidationError, match="too long"):
            safe_source_id(source_id)

    def test_exactly_one_over_max_length(self):
        source_id = "b" * 257
        with pytest.raises(SourceIdValidationError, match="too long"):
            safe_source_id(source_id)

    # -- Path traversal --

    def test_dot_dot_slash(self):
        with pytest.raises(SourceIdValidationError, match="path traversal"):
            safe_source_id("../etc/passwd")

    def test_dot_dot_in_middle(self):
        with pytest.raises(SourceIdValidationError, match="path traversal"):
            safe_source_id("foo/../bar")

    def test_dot_dot_at_end(self):
        with pytest.raises(SourceIdValidationError, match="path traversal"):
            safe_source_id("foo/..")

    def test_double_dot_dot(self):
        with pytest.raises(SourceIdValidationError, match="path traversal"):
            safe_source_id("../../secret")

    def test_path_traversal_logs_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            with pytest.raises(SourceIdValidationError):
                safe_source_id("../attack")
        assert "[SECURITY]" in caplog.text

    # -- Absolute paths --

    def test_unix_absolute_path(self):
        with pytest.raises(SourceIdValidationError, match="cannot start with /"):
            safe_source_id("/etc/passwd")

    def test_single_slash(self):
        with pytest.raises(SourceIdValidationError, match="cannot start with /"):
            safe_source_id("/")

    def test_absolute_path_logs_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            with pytest.raises(SourceIdValidationError):
                safe_source_id("/root")
        assert "[SECURITY]" in caplog.text

    # -- Windows absolute paths --

    def test_windows_c_drive(self):
        with pytest.raises(SourceIdValidationError, match="Windows absolute path"):
            safe_source_id("C:\\windows\\system32")

    def test_windows_d_drive(self):
        with pytest.raises(SourceIdValidationError, match="Windows absolute path"):
            safe_source_id("D:\\data")

    def test_windows_lowercase_drive(self):
        with pytest.raises(SourceIdValidationError, match="Windows absolute path"):
            safe_source_id("c:\\users")

    def test_windows_path_logs_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            with pytest.raises(SourceIdValidationError):
                safe_source_id("C:\\bad")
        assert "[SECURITY]" in caplog.text

    # -- Null bytes --

    def test_null_byte_at_start(self):
        with pytest.raises(SourceIdValidationError, match="null byte"):
            safe_source_id("\x00abc")

    def test_null_byte_in_middle(self):
        with pytest.raises(SourceIdValidationError, match="null byte"):
            safe_source_id("abc\x00def")

    def test_null_byte_at_end(self):
        with pytest.raises(SourceIdValidationError, match="null byte"):
            safe_source_id("abc\x00")

    def test_null_byte_logs_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            with pytest.raises(SourceIdValidationError):
                safe_source_id("x\x00y")
        assert "[SECURITY]" in caplog.text

    # -- Invalid characters --

    def test_space_rejected(self):
        with pytest.raises(SourceIdValidationError, match="invalid characters"):
            safe_source_id("has space")

    def test_semicolon_rejected(self):
        with pytest.raises(SourceIdValidationError, match="invalid characters"):
            safe_source_id("cmd;injection")

    def test_pipe_rejected(self):
        with pytest.raises(SourceIdValidationError, match="invalid characters"):
            safe_source_id("cmd|pipe")

    def test_ampersand_rejected(self):
        with pytest.raises(SourceIdValidationError, match="invalid characters"):
            safe_source_id("a&b")

    def test_backtick_rejected(self):
        with pytest.raises(SourceIdValidationError, match="invalid characters"):
            safe_source_id("cmd`whoami`")

    def test_dollar_rejected(self):
        with pytest.raises(SourceIdValidationError, match="invalid characters"):
            safe_source_id("$HOME")

    def test_curly_brace_rejected(self):
        with pytest.raises(SourceIdValidationError, match="invalid characters"):
            safe_source_id("{bad}")

    def test_square_bracket_rejected(self):
        with pytest.raises(SourceIdValidationError, match="invalid characters"):
            safe_source_id("[bad]")

    def test_newline_rejected(self):
        with pytest.raises(SourceIdValidationError, match="invalid characters"):
            safe_source_id("line\nnew")

    def test_tab_rejected(self):
        with pytest.raises(SourceIdValidationError, match="invalid characters"):
            safe_source_id("tab\there")

    def test_invalid_chars_log_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            with pytest.raises(SourceIdValidationError):
                safe_source_id("bad char!")
        assert "[SECURITY]" in caplog.text

    # -- Validation order (early exit) --

    def test_empty_checked_before_length(self):
        """Empty string error takes precedence over any other check."""
        with pytest.raises(SourceIdValidationError, match="cannot be empty"):
            safe_source_id("")

    def test_length_checked_before_traversal(self):
        """Too-long string with traversal sequences should fail on length."""
        long_id = ".." + "a" * MAX_SOURCE_ID_LENGTH
        with pytest.raises(SourceIdValidationError, match="too long"):
            safe_source_id(long_id)

    def test_traversal_checked_before_pattern(self):
        """Traversal detection should happen before regex check."""
        # '../x' has '..' AND starts with '.', but error should mention traversal
        with pytest.raises(SourceIdValidationError, match="path traversal"):
            safe_source_id("../x")

    # -- Single dot is valid (not path traversal) --

    def test_single_dot_is_valid(self):
        assert safe_source_id("file.txt") == "file.txt"

    def test_dots_without_dotdot_valid(self):
        assert safe_source_id("a.b.c.d") == "a.b.c.d"


# =============================================================================
# validate_channel_id() tests
# =============================================================================


class TestValidateChannelId:
    """Comprehensive tests for the validate_channel_id function."""

    # -- Valid non-webhook IDs --

    def test_valid_slack_channel(self):
        assert validate_channel_id("C12345678", "slack") == "C12345678"

    def test_valid_teams_channel(self):
        assert validate_channel_id("general", "teams") == "general"

    def test_valid_discord_channel(self):
        assert validate_channel_id("123456789012345678", "discord") == "123456789012345678"

    def test_valid_telegram_channel(self):
        assert validate_channel_id("-100123456789", "telegram") == "-100123456789"

    def test_valid_email_channel(self):
        assert validate_channel_id("team@example.com", "email") == "team@example.com"

    # -- Valid webhook URLs --

    def test_valid_https_webhook(self):
        url = "https://hooks.slack.com/services/T00/B00/xxxx"
        assert validate_channel_id(url, "webhook") == url

    def test_valid_http_webhook(self):
        url = "http://localhost:8080/webhook"
        assert validate_channel_id(url, "webhook") == url

    # -- Empty --

    def test_empty_channel_id_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_channel_id("", "slack")

    def test_empty_webhook_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_channel_id("", "webhook")

    # -- Too long --

    def test_too_long_channel_id(self):
        long_id = "a" * (MAX_SOURCE_ID_LENGTH + 1)
        with pytest.raises(ValueError, match="too long"):
            validate_channel_id(long_id, "slack")

    def test_too_long_webhook(self):
        long_url = "https://example.com/" + "a" * MAX_SOURCE_ID_LENGTH
        with pytest.raises(ValueError, match="too long"):
            validate_channel_id(long_url, "webhook")

    # -- Null bytes --

    def test_null_byte_in_channel_id(self):
        with pytest.raises(ValueError, match="null byte"):
            validate_channel_id("abc\x00def", "slack")

    def test_null_byte_in_webhook(self):
        with pytest.raises(ValueError, match="null byte"):
            validate_channel_id("https://example.com/\x00hook", "webhook")

    # -- Webhook-specific validation --

    def test_webhook_missing_scheme(self):
        with pytest.raises(ValueError, match="valid URL"):
            validate_channel_id("hooks.slack.com/services/T00/B00", "webhook")

    def test_webhook_ftp_scheme_rejected(self):
        with pytest.raises(ValueError, match="valid URL"):
            validate_channel_id("ftp://example.com/webhook", "webhook")

    def test_webhook_with_dotdot_rejected(self):
        with pytest.raises(ValueError, match="invalid characters"):
            validate_channel_id("https://example.com/../admin", "webhook")

    def test_webhook_with_backslash_rejected(self):
        with pytest.raises(ValueError, match="invalid characters"):
            validate_channel_id("https://example.com\\admin", "webhook")

    # -- Non-webhook path traversal --

    def test_slack_with_dotdot_rejected(self):
        with pytest.raises(ValueError, match="invalid path characters"):
            validate_channel_id("../secret", "slack")

    def test_teams_with_leading_slash_rejected(self):
        with pytest.raises(ValueError, match="invalid path characters"):
            validate_channel_id("/etc/passwd", "teams")

    def test_discord_with_dotdot_middle(self):
        with pytest.raises(ValueError, match="invalid path characters"):
            validate_channel_id("foo/../bar", "discord")

    # -- Boundary: at max length --

    def test_channel_id_at_exact_max_length(self):
        channel_id = "a" * MAX_SOURCE_ID_LENGTH
        assert validate_channel_id(channel_id, "slack") == channel_id

    def test_webhook_at_exact_max_length(self):
        # Build a webhook URL exactly at max length
        prefix = "https://x.co/"
        padding = "a" * (MAX_SOURCE_ID_LENGTH - len(prefix))
        url = prefix + padding
        assert validate_channel_id(url, "webhook") == url

    # -- Unknown channel type treated as non-webhook --

    def test_unknown_channel_type_valid(self):
        assert validate_channel_id("my-channel", "custom") == "my-channel"

    def test_unknown_channel_type_rejects_traversal(self):
        with pytest.raises(ValueError, match="invalid path characters"):
            validate_channel_id("../x", "custom")


# =============================================================================
# Integration / combination edge-case tests
# =============================================================================


class TestEdgeCases:
    """Edge cases and combinations that stress boundary conditions."""

    def test_source_id_all_allowed_special_chars(self):
        """A single ID using every allowed special character."""
        sid = "a-b_c:d/e@f.g#h"
        assert safe_source_id(sid) == sid

    def test_source_id_only_digits(self):
        assert safe_source_id("1234567890") == "1234567890"

    def test_source_id_only_uppercase(self):
        assert safe_source_id("ABCDEFG") == "ABCDEFG"

    def test_source_id_mixed_case(self):
        assert safe_source_id("AbCdEf") == "AbCdEf"

    def test_channel_id_with_hyphen_and_numbers(self):
        assert validate_channel_id("C-12345-AB", "slack") == "C-12345-AB"

    def test_numeric_colon_prefix_not_windows_path(self):
        """A digit followed by colon is NOT a Windows drive letter."""
        assert safe_source_id("9:metric") == "9:metric"

    def test_colon_at_position_1_non_alpha_first(self):
        """source_id like '1:value' - colon at [1] but [0] is not alpha."""
        assert safe_source_id("1:value") == "1:value"

    def test_two_char_id_with_colon(self):
        """'a:' triggers Windows path check since [0] is alpha and [1] is ':'."""
        with pytest.raises(SourceIdValidationError, match="Windows absolute path"):
            safe_source_id("a:")

    def test_webhook_both_dotdot_and_backslash(self):
        with pytest.raises(ValueError, match="invalid characters"):
            validate_channel_id("https://example.com/..\\admin", "webhook")

    def test_source_id_log_truncates_long_input(self, caplog):
        """Security log truncates the ID to 50 chars for safety."""
        long_traversal = "../" + "x" * 200
        with caplog.at_level(logging.WARNING):
            with pytest.raises(SourceIdValidationError):
                safe_source_id(long_traversal)
        # The logged value should be truncated (source code uses [:50])
        assert "[SECURITY]" in caplog.text
