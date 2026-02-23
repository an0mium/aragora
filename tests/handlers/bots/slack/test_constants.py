"""
Tests for Slack Bot Handler Constants and Validation.

Covers all constants, patterns, and validation functions in
aragora.server.handlers.bots.slack.constants:

- RBAC permission constants:
  - All six permission strings defined and prefixed correctly

- RBAC optional imports:
  - RBAC_AVAILABLE flag is bool
  - Graceful degradation when RBAC unavailable

- Dangerous input patterns:
  - SQL injection characters
  - XSS script tags
  - Template injection (${...} and {{...}})
  - Shell injection characters
  - Hex escape sequences
  - Unicode escape sequences

- Length constants:
  - MAX_TOPIC_LENGTH, MAX_COMMAND_LENGTH, MAX_USER_ID_LENGTH, MAX_CHANNEL_ID_LENGTH

- Regex patterns:
  - COMMAND_PATTERN: /aragora <command> [args]
  - TOPIC_PATTERN: quoted and unquoted topics

- Agent display names:
  - All 10 entries present and mapped correctly

- validate_slack_input():
  - Valid input passes
  - Empty input rejected / allowed
  - Exceeds max_length rejected
  - Dangerous patterns rejected (each category)
  - Custom field_name in error messages

- validate_slack_user_id():
  - Valid IDs (U/W/B prefixed)
  - Empty / too long / lowercase rejected

- validate_slack_channel_id():
  - Valid IDs (C/D/G prefixed)
  - Empty / too long / lowercase rejected

- validate_slack_team_id():
  - Valid team IDs (T prefixed)
  - Empty / too long / wrong prefix rejected

- Backward compatibility aliases:
  - _validate_slack_input is validate_slack_input
  - _validate_slack_user_id is validate_slack_user_id
  - _validate_slack_channel_id is validate_slack_channel_id
  - _validate_slack_team_id is validate_slack_team_id

- __all__ exports:
  - All expected names present
"""

from __future__ import annotations

import pytest

from aragora.server.handlers.bots.slack.constants import (
    AGENT_DISPLAY_NAMES,
    COMMAND_PATTERN,
    MAX_CHANNEL_ID_LENGTH,
    MAX_COMMAND_LENGTH,
    MAX_TOPIC_LENGTH,
    MAX_USER_ID_LENGTH,
    PERM_SLACK_ADMIN,
    PERM_SLACK_COMMANDS_EXECUTE,
    PERM_SLACK_COMMANDS_READ,
    PERM_SLACK_DEBATES_CREATE,
    PERM_SLACK_INTERACTIVE,
    PERM_SLACK_VOTES_RECORD,
    RBAC_AVAILABLE,
    TOPIC_PATTERN,
    _validate_slack_channel_id,
    _validate_slack_input,
    _validate_slack_team_id,
    _validate_slack_user_id,
    validate_slack_channel_id,
    validate_slack_input,
    validate_slack_team_id,
    validate_slack_user_id,
)


# =============================================================================
# RBAC Permission Constants
# =============================================================================


class TestRBACPermissionConstants:
    """Tests for RBAC permission string constants."""

    def test_perm_slack_commands_read(self):
        assert PERM_SLACK_COMMANDS_READ == "slack.commands.read"

    def test_perm_slack_commands_execute(self):
        assert PERM_SLACK_COMMANDS_EXECUTE == "slack.commands.execute"

    def test_perm_slack_debates_create(self):
        assert PERM_SLACK_DEBATES_CREATE == "slack.debates.create"

    def test_perm_slack_votes_record(self):
        assert PERM_SLACK_VOTES_RECORD == "slack.votes.record"

    def test_perm_slack_interactive(self):
        assert PERM_SLACK_INTERACTIVE == "slack.interactive.respond"

    def test_perm_slack_admin(self):
        assert PERM_SLACK_ADMIN == "slack.admin"

    def test_all_permissions_have_slack_prefix(self):
        perms = [
            PERM_SLACK_COMMANDS_READ,
            PERM_SLACK_COMMANDS_EXECUTE,
            PERM_SLACK_DEBATES_CREATE,
            PERM_SLACK_VOTES_RECORD,
            PERM_SLACK_INTERACTIVE,
            PERM_SLACK_ADMIN,
        ]
        for perm in perms:
            assert perm.startswith("slack."), f"Permission {perm!r} missing slack. prefix"

    def test_permissions_are_all_strings(self):
        perms = [
            PERM_SLACK_COMMANDS_READ,
            PERM_SLACK_COMMANDS_EXECUTE,
            PERM_SLACK_DEBATES_CREATE,
            PERM_SLACK_VOTES_RECORD,
            PERM_SLACK_INTERACTIVE,
            PERM_SLACK_ADMIN,
        ]
        for perm in perms:
            assert isinstance(perm, str)


# =============================================================================
# RBAC Availability Flag
# =============================================================================


class TestRBACAvailability:
    """Tests for RBAC_AVAILABLE flag and optional imports."""

    def test_rbac_available_is_bool(self):
        assert isinstance(RBAC_AVAILABLE, bool)

    def test_rbac_available_consistent_with_check_permission(self):
        from aragora.server.handlers.bots.slack.constants import check_permission

        if RBAC_AVAILABLE:
            assert check_permission is not None
        else:
            assert check_permission is None


# =============================================================================
# Length Constants
# =============================================================================


class TestLengthConstants:
    """Tests for maximum length constants."""

    def test_max_topic_length(self):
        assert MAX_TOPIC_LENGTH == 2000

    def test_max_command_length(self):
        assert MAX_COMMAND_LENGTH == 500

    def test_max_user_id_length(self):
        assert MAX_USER_ID_LENGTH == 100

    def test_max_channel_id_length(self):
        assert MAX_CHANNEL_ID_LENGTH == 100

    def test_all_length_constants_are_positive_ints(self):
        for val in (
            MAX_TOPIC_LENGTH,
            MAX_COMMAND_LENGTH,
            MAX_USER_ID_LENGTH,
            MAX_CHANNEL_ID_LENGTH,
        ):
            assert isinstance(val, int)
            assert val > 0


# =============================================================================
# COMMAND_PATTERN
# =============================================================================


class TestCommandPattern:
    """Tests for the COMMAND_PATTERN regex."""

    def test_basic_command_with_args(self):
        m = COMMAND_PATTERN.match("/aragora debate Should we use Rust?")
        assert m is not None
        assert m.group(1) == "debate"
        assert m.group(2) == "Should we use Rust?"

    def test_command_without_args(self):
        m = COMMAND_PATTERN.match("/aragora help")
        assert m is not None
        assert m.group(1) == "help"
        assert m.group(2) is None

    def test_case_insensitive(self):
        m = COMMAND_PATTERN.match("/ARAGORA STATUS")
        assert m is not None
        assert m.group(1) == "STATUS"

    def test_mixed_case(self):
        m = COMMAND_PATTERN.match("/Aragora Vote yes")
        assert m is not None
        assert m.group(1) == "Vote"

    def test_extra_whitespace_between_command_and_args(self):
        m = COMMAND_PATTERN.match("/aragora debate   extra spaces")
        assert m is not None
        assert m.group(1) == "debate"
        # The \s+ in the pattern consumes leading whitespace after the command
        assert m.group(2) == "extra spaces"

    def test_no_match_wrong_prefix(self):
        m = COMMAND_PATTERN.match("/slackbot debate topic")
        assert m is None

    def test_no_match_missing_command(self):
        m = COMMAND_PATTERN.match("/aragora")
        assert m is None

    def test_no_match_bare_slash(self):
        m = COMMAND_PATTERN.match("/")
        assert m is None


# =============================================================================
# TOPIC_PATTERN
# =============================================================================


class TestTopicPattern:
    """Tests for the TOPIC_PATTERN regex."""

    def test_unquoted_topic(self):
        m = TOPIC_PATTERN.match("Should we use microservices?")
        assert m is not None
        assert m.group(1) == "Should we use microservices?"

    def test_double_quoted_topic(self):
        m = TOPIC_PATTERN.match('"Is Python the best language?"')
        assert m is not None
        assert m.group(1) == "Is Python the best language?"

    def test_single_quoted_topic(self):
        m = TOPIC_PATTERN.match("'Should we refactor the monolith?'")
        assert m is not None
        assert m.group(1) == "Should we refactor the monolith?"

    def test_empty_string_no_match(self):
        m = TOPIC_PATTERN.match("")
        assert m is None


# =============================================================================
# Agent Display Names
# =============================================================================


class TestAgentDisplayNames:
    """Tests for the AGENT_DISPLAY_NAMES mapping."""

    def test_contains_expected_agents(self):
        expected_keys = {
            "claude",
            "gpt4",
            "gemini",
            "mistral",
            "deepseek",
            "grok",
            "qwen",
            "kimi",
            "anthropic-api",
            "openai-api",
        }
        assert set(AGENT_DISPLAY_NAMES.keys()) == expected_keys

    def test_display_name_values_are_strings(self):
        for key, val in AGENT_DISPLAY_NAMES.items():
            assert isinstance(val, str), f"Display name for {key!r} is not a string"

    def test_claude_display_name(self):
        assert AGENT_DISPLAY_NAMES["claude"] == "Claude"

    def test_api_aliases_match_base_agents(self):
        assert AGENT_DISPLAY_NAMES["anthropic-api"] == AGENT_DISPLAY_NAMES["claude"]
        assert AGENT_DISPLAY_NAMES["openai-api"] == AGENT_DISPLAY_NAMES["gpt4"]

    def test_agent_count(self):
        assert len(AGENT_DISPLAY_NAMES) == 10


# =============================================================================
# validate_slack_input
# =============================================================================


class TestValidateSlackInput:
    """Tests for the validate_slack_input function."""

    def test_valid_simple_input(self):
        valid, err = validate_slack_input("hello world", "topic")
        assert valid is True
        assert err is None

    def test_empty_input_rejected_by_default(self):
        valid, err = validate_slack_input("", "topic")
        assert valid is False
        assert "required" in err

    def test_empty_input_allowed_when_flag_set(self):
        valid, err = validate_slack_input("", "topic", allow_empty=True)
        assert valid is True
        assert err is None

    def test_exceeds_max_length(self):
        long_input = "a" * (MAX_COMMAND_LENGTH + 1)
        valid, err = validate_slack_input(long_input, "command")
        assert valid is False
        assert "maximum length" in err

    def test_at_max_length_passes(self):
        exact = "a" * MAX_COMMAND_LENGTH
        valid, err = validate_slack_input(exact, "command")
        assert valid is True
        assert err is None

    def test_custom_max_length(self):
        valid, err = validate_slack_input("abcdef", "field", max_length=5)
        assert valid is False
        assert "maximum length" in err

    def test_field_name_in_error_message(self):
        valid, err = validate_slack_input("", "my_field")
        assert valid is False
        assert "my_field" in err

    # Dangerous pattern tests

    def test_rejects_sql_injection_single_quote(self):
        valid, err = validate_slack_input("Robert'; DROP TABLE users;--", "name")
        assert valid is False
        assert "invalid characters" in err

    def test_rejects_sql_injection_double_quote(self):
        valid, err = validate_slack_input('value "OR 1=1', "field")
        assert valid is False
        assert "invalid characters" in err

    def test_rejects_xss_script_tag(self):
        valid, err = validate_slack_input("<script>alert(1)</script>", "topic")
        assert valid is False
        assert "invalid characters" in err

    def test_rejects_template_injection_dollar_brace(self):
        valid, err = validate_slack_input("${7*7}", "expr")
        assert valid is False
        assert "invalid characters" in err

    def test_rejects_template_injection_jinja(self):
        valid, err = validate_slack_input("{{config}}", "tpl")
        assert valid is False
        assert "invalid characters" in err

    def test_rejects_shell_injection_pipe(self):
        valid, err = validate_slack_input("ls | rm -rf /", "cmd")
        assert valid is False
        assert "invalid characters" in err

    def test_rejects_shell_injection_backtick(self):
        valid, err = validate_slack_input("`whoami`", "cmd")
        assert valid is False
        assert "invalid characters" in err

    def test_rejects_shell_injection_ampersand(self):
        valid, err = validate_slack_input("echo hi & rm -rf /", "cmd")
        assert valid is False
        assert "invalid characters" in err

    def test_rejects_shell_injection_semicolon(self):
        valid, err = validate_slack_input("cmd; evil", "input")
        assert valid is False
        assert "invalid characters" in err

    def test_rejects_shell_injection_dollar(self):
        valid, err = validate_slack_input("$HOME", "path")
        assert valid is False
        assert "invalid characters" in err

    def test_rejects_hex_escape(self):
        valid, err = validate_slack_input("test\\x41value", "data")
        assert valid is False
        assert "invalid characters" in err

    def test_rejects_unicode_escape(self):
        valid, err = validate_slack_input("test\\u0041value", "data")
        assert valid is False
        assert "invalid characters" in err

    def test_allows_safe_unicode_characters(self):
        # Actual unicode chars (not escape sequences) should be fine
        valid, err = validate_slack_input("caf\u00e9 meeting", "topic")
        assert valid is True
        assert err is None

    def test_none_input_rejected(self):
        valid, err = validate_slack_input(None, "field")
        assert valid is False
        assert "required" in err


# =============================================================================
# validate_slack_user_id
# =============================================================================


class TestValidateSlackUserId:
    """Tests for the validate_slack_user_id function."""

    def test_valid_user_id_u_prefix(self):
        valid, err = validate_slack_user_id("U024BE7LH")
        assert valid is True
        assert err is None

    def test_valid_user_id_w_prefix(self):
        valid, err = validate_slack_user_id("W012A3CDE")
        assert valid is True
        assert err is None

    def test_valid_user_id_b_prefix(self):
        valid, err = validate_slack_user_id("B01234ABCD")
        assert valid is True
        assert err is None

    def test_empty_user_id_rejected(self):
        valid, err = validate_slack_user_id("")
        assert valid is False
        assert "required" in err

    def test_too_long_user_id_rejected(self):
        valid, err = validate_slack_user_id("U" + "A" * MAX_USER_ID_LENGTH)
        assert valid is False
        assert "too long" in err.lower()

    def test_lowercase_rejected(self):
        valid, err = validate_slack_user_id("u024be7lh")
        assert valid is False
        assert "Invalid" in err

    def test_special_chars_rejected(self):
        valid, err = validate_slack_user_id("U024-BE7LH")
        assert valid is False
        assert "Invalid" in err


# =============================================================================
# validate_slack_channel_id
# =============================================================================


class TestValidateSlackChannelId:
    """Tests for the validate_slack_channel_id function."""

    def test_valid_channel_id_c_prefix(self):
        valid, err = validate_slack_channel_id("C024BE7LH")
        assert valid is True
        assert err is None

    def test_valid_channel_id_d_prefix(self):
        valid, err = validate_slack_channel_id("D012A3CDE")
        assert valid is True
        assert err is None

    def test_valid_channel_id_g_prefix(self):
        valid, err = validate_slack_channel_id("G01234ABCD")
        assert valid is True
        assert err is None

    def test_empty_channel_id_rejected(self):
        valid, err = validate_slack_channel_id("")
        assert valid is False
        assert "required" in err

    def test_too_long_channel_id_rejected(self):
        valid, err = validate_slack_channel_id("C" + "A" * MAX_CHANNEL_ID_LENGTH)
        assert valid is False
        assert "too long" in err.lower()

    def test_lowercase_rejected(self):
        valid, err = validate_slack_channel_id("c024be7lh")
        assert valid is False
        assert "Invalid" in err


# =============================================================================
# validate_slack_team_id
# =============================================================================


class TestValidateSlackTeamId:
    """Tests for the validate_slack_team_id function."""

    def test_valid_team_id(self):
        valid, err = validate_slack_team_id("T024BE7LH")
        assert valid is True
        assert err is None

    def test_empty_team_id_rejected(self):
        valid, err = validate_slack_team_id("")
        assert valid is False
        assert "required" in err

    def test_too_long_team_id_rejected(self):
        valid, err = validate_slack_team_id("T" + "A" * MAX_USER_ID_LENGTH)
        assert valid is False
        assert "too long" in err.lower()

    def test_wrong_prefix_rejected(self):
        """Team IDs must start with T."""
        valid, err = validate_slack_team_id("U024BE7LH")
        assert valid is False
        assert "Invalid" in err

    def test_lowercase_rejected(self):
        valid, err = validate_slack_team_id("t024be7lh")
        assert valid is False
        assert "Invalid" in err

    def test_numeric_only_without_t_prefix_rejected(self):
        valid, err = validate_slack_team_id("1234567890")
        assert valid is False
        assert "Invalid" in err


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================


class TestBackwardCompatibilityAliases:
    """Tests that underscore-prefixed aliases point to the same functions."""

    def test_validate_slack_input_alias(self):
        assert _validate_slack_input is validate_slack_input

    def test_validate_slack_user_id_alias(self):
        assert _validate_slack_user_id is validate_slack_user_id

    def test_validate_slack_channel_id_alias(self):
        assert _validate_slack_channel_id is validate_slack_channel_id

    def test_validate_slack_team_id_alias(self):
        assert _validate_slack_team_id is validate_slack_team_id


# =============================================================================
# __all__ Exports
# =============================================================================


class TestAllExports:
    """Tests for __all__ completeness."""

    def test_all_permissions_exported(self):
        from aragora.server.handlers.bots.slack import constants

        for name in (
            "PERM_SLACK_COMMANDS_READ",
            "PERM_SLACK_COMMANDS_EXECUTE",
            "PERM_SLACK_DEBATES_CREATE",
            "PERM_SLACK_VOTES_RECORD",
            "PERM_SLACK_INTERACTIVE",
            "PERM_SLACK_ADMIN",
        ):
            assert name in constants.__all__, f"{name} missing from __all__"

    def test_all_validation_functions_exported(self):
        from aragora.server.handlers.bots.slack import constants

        for name in (
            "validate_slack_input",
            "validate_slack_user_id",
            "validate_slack_channel_id",
            "validate_slack_team_id",
        ):
            assert name in constants.__all__, f"{name} missing from __all__"

    def test_all_backward_compat_aliases_exported(self):
        from aragora.server.handlers.bots.slack import constants

        for name in (
            "_validate_slack_input",
            "_validate_slack_user_id",
            "_validate_slack_channel_id",
            "_validate_slack_team_id",
        ):
            assert name in constants.__all__, f"{name} missing from __all__"

    def test_all_constants_exported(self):
        from aragora.server.handlers.bots.slack import constants

        for name in (
            "MAX_TOPIC_LENGTH",
            "MAX_COMMAND_LENGTH",
            "MAX_USER_ID_LENGTH",
            "MAX_CHANNEL_ID_LENGTH",
            "COMMAND_PATTERN",
            "TOPIC_PATTERN",
            "AGENT_DISPLAY_NAMES",
            "RBAC_AVAILABLE",
            "SLACK_SIGNING_SECRET",
            "SLACK_BOT_TOKEN",
        ):
            assert name in constants.__all__, f"{name} missing from __all__"

    def test_all_exports_count(self):
        from aragora.server.handlers.bots.slack import constants

        # 6 perms + 4 RBAC + 6 validation/patterns + 2 env + 1 agent names
        # + 4 validation funcs + 4 aliases = 27
        assert len(constants.__all__) == 27
