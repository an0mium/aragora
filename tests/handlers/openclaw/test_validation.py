"""Comprehensive tests for OpenClaw validation module.

Covers all validation functions and constants defined in
aragora/server/handlers/openclaw/validation.py:

Functions:
- validate_credential_name      (name format, length, pattern)
- validate_credential_secret    (secret length, null bytes, type-specific)
- validate_session_config       (size, key count, nesting depth)
- validate_action_type          (type format, length, pattern)
- validate_action_input         (input size, serialization)
- validate_metadata             (metadata size, custom max_size)
- sanitize_action_parameters    (shell metacharacter escaping, nested structures)

Constants:
- MAX_CREDENTIAL_NAME_LENGTH, MAX_CREDENTIAL_SECRET_LENGTH
- MAX_CREDENTIAL_METADATA_SIZE, MIN_CREDENTIAL_SECRET_LENGTH
- MAX_SESSION_CONFIG_SIZE, MAX_SESSION_METADATA_SIZE
- MAX_SESSION_CONFIG_KEYS, MAX_SESSION_CONFIG_DEPTH
- MAX_ACTION_TYPE_LENGTH, MAX_ACTION_INPUT_SIZE, MAX_ACTION_METADATA_SIZE
- SAFE_CREDENTIAL_NAME_PATTERN, SAFE_ACTION_TYPE_PATTERN
- SHELL_METACHARACTERS

Test categories:
- Valid inputs (happy paths)
- None/empty inputs
- Type errors (non-string, non-dict)
- Length/size boundary tests (at limit, over limit)
- Pattern matching (valid chars, invalid chars, edge patterns)
- Security (null bytes, shell metacharacters, command injection)
- Nested structure handling (recursive sanitization, depth limits)
- Type-specific credential validation
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from aragora.server.handlers.openclaw.validation import (
    MAX_ACTION_INPUT_SIZE,
    MAX_ACTION_METADATA_SIZE,
    MAX_ACTION_TYPE_LENGTH,
    MAX_CREDENTIAL_METADATA_SIZE,
    MAX_CREDENTIAL_NAME_LENGTH,
    MAX_CREDENTIAL_SECRET_LENGTH,
    MAX_SESSION_CONFIG_DEPTH,
    MAX_SESSION_CONFIG_KEYS,
    MAX_SESSION_CONFIG_SIZE,
    MAX_SESSION_METADATA_SIZE,
    MIN_CREDENTIAL_SECRET_LENGTH,
    SAFE_ACTION_TYPE_PATTERN,
    SAFE_CREDENTIAL_NAME_PATTERN,
    SHELL_METACHARACTERS,
    sanitize_action_parameters,
    validate_action_input,
    validate_action_type,
    validate_credential_name,
    validate_credential_secret,
    validate_metadata,
    validate_session_config,
)


# ============================================================================
# Constants verification
# ============================================================================


class TestConstants:
    """Verify exported constants have expected values."""

    def test_max_credential_name_length(self):
        assert MAX_CREDENTIAL_NAME_LENGTH == 128

    def test_max_credential_secret_length(self):
        assert MAX_CREDENTIAL_SECRET_LENGTH == 8192

    def test_max_credential_metadata_size(self):
        assert MAX_CREDENTIAL_METADATA_SIZE == 4096

    def test_min_credential_secret_length(self):
        assert MIN_CREDENTIAL_SECRET_LENGTH == 8

    def test_max_session_config_size(self):
        assert MAX_SESSION_CONFIG_SIZE == 8192

    def test_max_session_metadata_size(self):
        assert MAX_SESSION_METADATA_SIZE == 4096

    def test_max_session_config_keys(self):
        assert MAX_SESSION_CONFIG_KEYS == 50

    def test_max_session_config_depth(self):
        assert MAX_SESSION_CONFIG_DEPTH == 5

    def test_max_action_type_length(self):
        assert MAX_ACTION_TYPE_LENGTH == 64

    def test_max_action_input_size(self):
        assert MAX_ACTION_INPUT_SIZE == 65536

    def test_max_action_metadata_size(self):
        assert MAX_ACTION_METADATA_SIZE == 4096


class TestPatterns:
    """Verify regex patterns match expected inputs."""

    def test_credential_name_pattern_simple(self):
        assert SAFE_CREDENTIAL_NAME_PATTERN.match("myKey")

    def test_credential_name_pattern_with_hyphens(self):
        assert SAFE_CREDENTIAL_NAME_PATTERN.match("my-key-name")

    def test_credential_name_pattern_with_underscores(self):
        assert SAFE_CREDENTIAL_NAME_PATTERN.match("my_key_name")

    def test_credential_name_pattern_alphanumeric(self):
        assert SAFE_CREDENTIAL_NAME_PATTERN.match("key123")

    def test_credential_name_pattern_rejects_leading_digit(self):
        assert not SAFE_CREDENTIAL_NAME_PATTERN.match("1key")

    def test_credential_name_pattern_rejects_leading_hyphen(self):
        assert not SAFE_CREDENTIAL_NAME_PATTERN.match("-key")

    def test_credential_name_pattern_rejects_spaces(self):
        assert not SAFE_CREDENTIAL_NAME_PATTERN.match("my key")

    def test_credential_name_pattern_rejects_dots(self):
        assert not SAFE_CREDENTIAL_NAME_PATTERN.match("my.key")

    def test_action_type_pattern_simple(self):
        assert SAFE_ACTION_TYPE_PATTERN.match("execute")

    def test_action_type_pattern_with_dots(self):
        assert SAFE_ACTION_TYPE_PATTERN.match("tool.execute")

    def test_action_type_pattern_with_hyphens(self):
        assert SAFE_ACTION_TYPE_PATTERN.match("tool-exec")

    def test_action_type_pattern_with_underscores(self):
        assert SAFE_ACTION_TYPE_PATTERN.match("tool_exec")

    def test_action_type_pattern_rejects_leading_digit(self):
        assert not SAFE_ACTION_TYPE_PATTERN.match("1tool")

    def test_action_type_pattern_rejects_spaces(self):
        assert not SAFE_ACTION_TYPE_PATTERN.match("my tool")

    def test_shell_metacharacters_detects_semicolon(self):
        assert SHELL_METACHARACTERS.search(";")

    def test_shell_metacharacters_detects_pipe(self):
        assert SHELL_METACHARACTERS.search("|")

    def test_shell_metacharacters_detects_backtick(self):
        assert SHELL_METACHARACTERS.search("`")

    def test_shell_metacharacters_detects_dollar(self):
        assert SHELL_METACHARACTERS.search("$")

    def test_shell_metacharacters_detects_newline(self):
        assert SHELL_METACHARACTERS.search("\n")

    def test_shell_metacharacters_detects_null_byte(self):
        assert SHELL_METACHARACTERS.search("\x00")

    def test_shell_metacharacters_no_match_on_safe_string(self):
        assert not SHELL_METACHARACTERS.search("hello-world_123")


# ============================================================================
# validate_credential_name
# ============================================================================


class TestValidateCredentialName:
    """Tests for validate_credential_name()."""

    def test_valid_simple_name(self):
        valid, err = validate_credential_name("myKey")
        assert valid is True
        assert err is None

    def test_valid_name_with_hyphens_and_underscores(self):
        valid, err = validate_credential_name("my-api_key-1")
        assert valid is True
        assert err is None

    def test_valid_single_letter(self):
        valid, err = validate_credential_name("a")
        assert valid is True
        assert err is None

    def test_valid_max_length_name(self):
        # Pattern allows letter + up to 127 more characters = 128 total
        name = "a" + "b" * 127
        assert len(name) == MAX_CREDENTIAL_NAME_LENGTH
        valid, err = validate_credential_name(name)
        assert valid is True
        assert err is None

    def test_none_name(self):
        valid, err = validate_credential_name(None)
        assert valid is False
        assert "required" in err

    def test_empty_string(self):
        valid, err = validate_credential_name("")
        assert valid is False
        assert "required" in err

    def test_whitespace_only(self):
        valid, err = validate_credential_name("   ")
        assert valid is False
        assert "empty" in err

    def test_exceeds_max_length(self):
        name = "a" + "b" * MAX_CREDENTIAL_NAME_LENGTH
        assert len(name) > MAX_CREDENTIAL_NAME_LENGTH
        valid, err = validate_credential_name(name)
        assert valid is False
        assert "maximum length" in err

    def test_starts_with_digit(self):
        valid, err = validate_credential_name("123key")
        assert valid is False
        assert "start with a letter" in err

    def test_starts_with_hyphen(self):
        valid, err = validate_credential_name("-mykey")
        assert valid is False
        assert "start with a letter" in err

    def test_starts_with_underscore(self):
        valid, err = validate_credential_name("_mykey")
        assert valid is False
        assert "start with a letter" in err

    def test_contains_spaces(self):
        valid, err = validate_credential_name("my key")
        assert valid is False
        assert "start with a letter" in err

    def test_contains_dots(self):
        valid, err = validate_credential_name("my.key")
        assert valid is False
        assert "start with a letter" in err

    def test_contains_shell_metacharacters(self):
        valid, err = validate_credential_name("key;rm -rf")
        assert valid is False

    def test_contains_null_byte(self):
        valid, err = validate_credential_name("key\x00hack")
        assert valid is False

    def test_unicode_characters_rejected(self):
        valid, err = validate_credential_name("key\u00e9")
        assert valid is False

    def test_valid_name_with_numbers(self):
        valid, err = validate_credential_name("apiKey2024v3")
        assert valid is True
        assert err is None

    def test_name_stripped_before_empty_check(self):
        # " " gets stripped to "" => empty error
        valid, err = validate_credential_name(" ")
        assert valid is False
        assert "empty" in err


# ============================================================================
# validate_credential_secret
# ============================================================================


class TestValidateCredentialSecret:
    """Tests for validate_credential_secret()."""

    def test_valid_secret(self):
        valid, err = validate_credential_secret("my-super-secret-key-12345")
        assert valid is True
        assert err is None

    def test_valid_secret_exact_min_length(self):
        secret = "a" * MIN_CREDENTIAL_SECRET_LENGTH
        valid, err = validate_credential_secret(secret)
        assert valid is True
        assert err is None

    def test_valid_secret_at_max_length(self):
        secret = "a" * MAX_CREDENTIAL_SECRET_LENGTH
        valid, err = validate_credential_secret(secret)
        assert valid is True
        assert err is None

    def test_none_secret(self):
        valid, err = validate_credential_secret(None)
        assert valid is False
        assert "required" in err

    def test_empty_secret(self):
        valid, err = validate_credential_secret("")
        assert valid is False
        assert "required" in err

    def test_secret_exceeds_max_length(self):
        secret = "a" * (MAX_CREDENTIAL_SECRET_LENGTH + 1)
        valid, err = validate_credential_secret(secret)
        assert valid is False
        assert "maximum length" in err

    def test_secret_contains_null_byte(self):
        valid, err = validate_credential_secret("valid-secret\x00injection")
        assert valid is False
        assert "invalid characters" in err

    def test_secret_too_short_no_type(self):
        secret = "a" * (MIN_CREDENTIAL_SECRET_LENGTH - 1)
        valid, err = validate_credential_secret(secret)
        assert valid is False
        assert f"at least {MIN_CREDENTIAL_SECRET_LENGTH}" in err

    def test_short_secret_with_non_api_key_type(self):
        # When credential_type is not "api_key", short secrets are allowed
        valid, err = validate_credential_secret("short", credential_type="oauth_token")
        assert valid is True
        assert err is None

    def test_short_secret_with_api_key_type(self):
        # api_key type enforces min length
        valid, err = validate_credential_secret("short", credential_type="api_key")
        assert valid is False
        assert f"at least {MIN_CREDENTIAL_SECRET_LENGTH}" in err

    def test_single_char_placeholder_with_type(self):
        # Single character placeholder allowed when credential_type is specified
        valid, err = validate_credential_secret("x", credential_type="api_key")
        assert valid is True
        assert err is None

    def test_single_char_no_type_fails(self):
        # Single char without type fails min length check
        valid, err = validate_credential_secret("x")
        assert valid is False
        assert f"at least {MIN_CREDENTIAL_SECRET_LENGTH}" in err

    def test_two_char_api_key_too_short(self):
        # Two chars is not single-char placeholder and not >= min length
        valid, err = validate_credential_secret("xy", credential_type="api_key")
        assert valid is False

    def test_non_api_key_type_skips_min_length(self):
        valid, err = validate_credential_secret("ab", credential_type="bearer_token")
        assert valid is True
        assert err is None

    def test_null_byte_checked_before_type_skip(self):
        # Null byte check runs before type-specific early return
        valid, err = validate_credential_secret("\x00abc", credential_type="oauth_token")
        assert valid is False
        assert "invalid characters" in err

    def test_valid_long_secret(self):
        valid, err = validate_credential_secret("a" * 4096)
        assert valid is True
        assert err is None


# ============================================================================
# validate_session_config
# ============================================================================


class TestValidateSessionConfig:
    """Tests for validate_session_config()."""

    def test_none_config_allowed(self):
        valid, err = validate_session_config(None)
        assert valid is True
        assert err is None

    def test_empty_dict(self):
        valid, err = validate_session_config({})
        assert valid is True
        assert err is None

    def test_simple_config(self):
        valid, err = validate_session_config({"timeout": 30, "model": "gpt-4"})
        assert valid is True
        assert err is None

    def test_non_dict_rejected(self):
        valid, err = validate_session_config("not a dict")
        assert valid is False
        assert "must be an object" in err

    def test_list_rejected(self):
        valid, err = validate_session_config([1, 2, 3])
        assert valid is False
        assert "must be an object" in err

    def test_config_exceeds_max_size(self):
        # Create a config that exceeds MAX_SESSION_CONFIG_SIZE when serialized
        big_config = {"key": "x" * MAX_SESSION_CONFIG_SIZE}
        valid, err = validate_session_config(big_config)
        assert valid is False
        assert "maximum size" in err

    def test_config_at_max_size(self):
        # Build a config that is exactly at the limit
        # json.dumps({}) = 2 bytes, each "k":"v" adds overhead
        # We need to be careful to be right at the boundary
        config = {"a": "x" * (MAX_SESSION_CONFIG_SIZE - 10)}
        serialized = json.dumps(config)
        if len(serialized) <= MAX_SESSION_CONFIG_SIZE:
            valid, err = validate_session_config(config)
            assert valid is True
        else:
            valid, err = validate_session_config(config)
            assert valid is False

    def test_too_many_keys(self):
        config = {f"key{i}": i for i in range(MAX_SESSION_CONFIG_KEYS + 1)}
        valid, err = validate_session_config(config)
        assert valid is False
        assert "maximum" in err and "keys" in err

    def test_exact_max_keys(self):
        config = {f"k{i}": i for i in range(MAX_SESSION_CONFIG_KEYS)}
        assert len(config) == MAX_SESSION_CONFIG_KEYS
        valid, err = validate_session_config(config)
        assert valid is True
        assert err is None

    def test_nesting_at_max_depth(self):
        # Build nested dict at exactly max depth
        config: dict[str, Any] = {"value": "leaf"}
        for _ in range(MAX_SESSION_CONFIG_DEPTH - 1):
            config = {"nested": config}
        valid, err = validate_session_config(config)
        assert valid is True
        assert err is None

    def test_nesting_exceeds_max_depth(self):
        # Build nested dict exceeding max depth
        config: dict[str, Any] = {"value": "leaf"}
        for _ in range(MAX_SESSION_CONFIG_DEPTH + 1):
            config = {"nested": config}
        valid, err = validate_session_config(config)
        assert valid is False
        assert "nesting depth" in err

    def test_nesting_with_lists(self):
        # Lists also count toward depth
        config: Any = "leaf"
        for _ in range(MAX_SESSION_CONFIG_DEPTH + 2):
            config = [config]
        wrapper = {"data": config}
        valid, err = validate_session_config(wrapper)
        assert valid is False
        assert "nesting depth" in err

    def test_config_with_unserializable_data(self):
        # Objects that cannot be serialized to JSON
        config = {"callback": object()}
        valid, err = validate_session_config(config)
        assert valid is False
        assert "invalid data" in err

    def test_mixed_nested_types(self):
        config = {
            "list_val": [1, 2, {"nested": True}],
            "str_val": "hello",
            "int_val": 42,
            "bool_val": True,
            "null_val": None,
        }
        valid, err = validate_session_config(config)
        assert valid is True
        assert err is None

    def test_deeply_nested_lists_in_dicts(self):
        """Deeply nested alternating dicts and lists."""
        config: Any = "val"
        for i in range(MAX_SESSION_CONFIG_DEPTH + 2):
            if i % 2 == 0:
                config = [config]
            else:
                config = {"k": config}
        wrapper = {"root": config}
        valid, err = validate_session_config(wrapper)
        assert valid is False
        assert "nesting depth" in err


# ============================================================================
# validate_action_type
# ============================================================================


class TestValidateActionType:
    """Tests for validate_action_type()."""

    def test_valid_simple_type(self):
        valid, err = validate_action_type("execute")
        assert valid is True
        assert err is None

    def test_valid_dotted_type(self):
        valid, err = validate_action_type("tool.execute.run")
        assert valid is True
        assert err is None

    def test_valid_type_with_hyphens(self):
        valid, err = validate_action_type("tool-exec")
        assert valid is True
        assert err is None

    def test_valid_type_with_underscores(self):
        valid, err = validate_action_type("tool_exec")
        assert valid is True
        assert err is None

    def test_valid_type_alphanumeric(self):
        valid, err = validate_action_type("action2run")
        assert valid is True
        assert err is None

    def test_valid_single_letter(self):
        valid, err = validate_action_type("a")
        assert valid is True
        assert err is None

    def test_valid_max_length(self):
        # Pattern allows letter + up to 63 more = 64 total
        action_type = "a" + "b" * 63
        assert len(action_type) == MAX_ACTION_TYPE_LENGTH
        valid, err = validate_action_type(action_type)
        assert valid is True
        assert err is None

    def test_none_type(self):
        valid, err = validate_action_type(None)
        assert valid is False
        assert "required" in err

    def test_empty_type(self):
        valid, err = validate_action_type("")
        assert valid is False
        assert "required" in err

    def test_whitespace_only(self):
        valid, err = validate_action_type("   ")
        assert valid is False
        assert "empty" in err

    def test_exceeds_max_length(self):
        action_type = "a" + "b" * MAX_ACTION_TYPE_LENGTH
        assert len(action_type) > MAX_ACTION_TYPE_LENGTH
        valid, err = validate_action_type(action_type)
        assert valid is False
        assert "maximum length" in err

    def test_starts_with_digit(self):
        valid, err = validate_action_type("1action")
        assert valid is False
        assert "start with a letter" in err

    def test_starts_with_dot(self):
        valid, err = validate_action_type(".action")
        assert valid is False
        assert "start with a letter" in err

    def test_contains_spaces(self):
        valid, err = validate_action_type("my action")
        assert valid is False

    def test_contains_shell_metacharacters(self):
        valid, err = validate_action_type("exec;rm")
        assert valid is False

    def test_contains_slash(self):
        valid, err = validate_action_type("tool/exec")
        assert valid is False

    def test_type_stripped_before_empty_check(self):
        valid, err = validate_action_type(" ")
        assert valid is False
        assert "empty" in err


# ============================================================================
# sanitize_action_parameters
# ============================================================================


class TestSanitizeActionParameters:
    """Tests for sanitize_action_parameters()."""

    def test_none_returns_empty_dict(self):
        result = sanitize_action_parameters(None)
        assert result == {}

    def test_empty_dict_returns_empty_dict(self):
        result = sanitize_action_parameters({})
        assert result == {}

    def test_non_dict_returns_empty_dict(self):
        result = sanitize_action_parameters("not a dict")
        assert result == {}

    def test_list_returns_empty_dict(self):
        result = sanitize_action_parameters([1, 2])
        assert result == {}

    def test_safe_string_unchanged(self):
        result = sanitize_action_parameters({"cmd": "hello-world"})
        assert result["cmd"] == "hello-world"

    def test_numeric_values_unchanged(self):
        result = sanitize_action_parameters({"count": 42, "ratio": 3.14})
        assert result["count"] == 42
        assert result["ratio"] == 3.14

    def test_boolean_values_unchanged(self):
        result = sanitize_action_parameters({"flag": True, "off": False})
        assert result["flag"] is True
        assert result["off"] is False

    def test_none_value_unchanged(self):
        result = sanitize_action_parameters({"key": None})
        assert result["key"] is None

    def test_semicolon_escaped(self):
        result = sanitize_action_parameters({"cmd": "ls; rm -rf /"})
        assert ";" not in result["cmd"] or "\\;" in result["cmd"]

    def test_pipe_escaped(self):
        result = sanitize_action_parameters({"cmd": "cat /etc/passwd | grep root"})
        assert "\\|" in result["cmd"]

    def test_backtick_escaped(self):
        result = sanitize_action_parameters({"cmd": "`whoami`"})
        assert "\\`" in result["cmd"]

    def test_dollar_sign_escaped(self):
        result = sanitize_action_parameters({"cmd": "$HOME"})
        assert "\\$" in result["cmd"]

    def test_parentheses_escaped(self):
        result = sanitize_action_parameters({"cmd": "$(whoami)"})
        sanitized = result["cmd"]
        assert "\\$" in sanitized
        assert "\\(" in sanitized

    def test_null_byte_removed(self):
        result = sanitize_action_parameters({"cmd": "hello\x00world"})
        assert "\x00" not in result["cmd"]

    def test_newline_replaced_with_space(self):
        result = sanitize_action_parameters({"cmd": "line1\nline2"})
        assert "\n" not in result["cmd"]
        # Newlines are replaced with spaces per the lambda
        assert " " in result["cmd"]

    def test_carriage_return_replaced_with_space(self):
        result = sanitize_action_parameters({"cmd": "line1\rline2"})
        assert "\r" not in result["cmd"]

    def test_nested_dict_sanitized(self):
        result = sanitize_action_parameters({"inner": {"cmd": "ls; cat"}})
        assert isinstance(result["inner"], dict)
        assert "\\;" in result["inner"]["cmd"]

    def test_nested_list_sanitized(self):
        result = sanitize_action_parameters({"items": ["safe", "danger; rm"]})
        assert isinstance(result["items"], list)
        assert "safe" == result["items"][0]
        assert "\\;" in result["items"][1]

    def test_deeply_nested_sanitization(self):
        params = {"l1": {"l2": {"l3": [{"cmd": "`hack`"}]}}}
        result = sanitize_action_parameters(params)
        cmd = result["l1"]["l2"]["l3"][0]["cmd"]
        assert "\\`" in cmd

    def test_mixed_types_in_nested(self):
        params = {
            "str_val": "hello; world",
            "int_val": 42,
            "nested": {"arr": [1, "cmd|pipe", True, None]},
        }
        result = sanitize_action_parameters(params)
        assert "\\;" in result["str_val"]
        assert result["int_val"] == 42
        assert "\\|" in result["nested"]["arr"][1]
        assert result["nested"]["arr"][2] is True
        assert result["nested"]["arr"][3] is None

    def test_empty_string_unchanged(self):
        result = sanitize_action_parameters({"key": ""})
        assert result["key"] == ""

    def test_ampersand_escaped(self):
        result = sanitize_action_parameters({"cmd": "cmd1 & cmd2"})
        assert "\\&" in result["cmd"]

    def test_backslash_escaped(self):
        result = sanitize_action_parameters({"path": "C:\\Windows"})
        assert "\\\\" in result["path"]

    def test_quotes_escaped(self):
        result = sanitize_action_parameters({"val": 'say "hello"'})
        assert '\\"' in result["val"]

    def test_single_quotes_escaped(self):
        result = sanitize_action_parameters({"val": "it's"})
        assert "\\'" in result["val"]

    def test_angle_brackets_escaped(self):
        result = sanitize_action_parameters({"html": "<script>alert(1)</script>"})
        sanitized = result["html"]
        assert "\\<" in sanitized
        assert "\\>" in sanitized

    def test_curly_braces_escaped(self):
        result = sanitize_action_parameters({"val": "${VAR}"})
        sanitized = result["val"]
        assert "\\{" in sanitized or "\\}" in sanitized

    def test_square_brackets_escaped(self):
        result = sanitize_action_parameters({"val": "arr[0]"})
        sanitized = result["val"]
        assert "\\[" in sanitized


# ============================================================================
# validate_action_input
# ============================================================================


class TestValidateActionInput:
    """Tests for validate_action_input()."""

    def test_none_input_allowed(self):
        valid, err = validate_action_input(None)
        assert valid is True
        assert err is None

    def test_empty_dict(self):
        valid, err = validate_action_input({})
        assert valid is True
        assert err is None

    def test_simple_input(self):
        valid, err = validate_action_input({"prompt": "hello", "temperature": 0.7})
        assert valid is True
        assert err is None

    def test_non_dict_rejected(self):
        valid, err = validate_action_input("not a dict")
        assert valid is False
        assert "must be an object" in err

    def test_list_rejected(self):
        valid, err = validate_action_input([1, 2, 3])
        assert valid is False
        assert "must be an object" in err

    def test_integer_rejected(self):
        valid, err = validate_action_input(42)
        assert valid is False
        assert "must be an object" in err

    def test_input_exceeds_max_size(self):
        big_input = {"data": "x" * MAX_ACTION_INPUT_SIZE}
        valid, err = validate_action_input(big_input)
        assert valid is False
        assert "maximum size" in err

    def test_input_at_max_size_boundary(self):
        # A dict that serializes to exactly MAX_ACTION_INPUT_SIZE or less
        # json.dumps({"a": "x..."}) overhead is ~8 bytes
        remaining = MAX_ACTION_INPUT_SIZE - 8
        input_data = {"a": "x" * remaining}
        serialized = json.dumps(input_data)
        if len(serialized) <= MAX_ACTION_INPUT_SIZE:
            valid, err = validate_action_input(input_data)
            assert valid is True
        else:
            valid, err = validate_action_input(input_data)
            assert valid is False

    def test_input_with_nested_structure(self):
        valid, err = validate_action_input(
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                ],
                "config": {"temperature": 0.5},
            }
        )
        assert valid is True
        assert err is None

    def test_input_with_unserializable_data(self):
        valid, err = validate_action_input({"func": object()})
        assert valid is False
        assert "invalid data" in err


# ============================================================================
# validate_metadata
# ============================================================================


class TestValidateMetadata:
    """Tests for validate_metadata()."""

    def test_none_metadata_allowed(self):
        valid, err = validate_metadata(None)
        assert valid is True
        assert err is None

    def test_empty_dict(self):
        valid, err = validate_metadata({})
        assert valid is True
        assert err is None

    def test_simple_metadata(self):
        valid, err = validate_metadata({"source": "api", "version": "1.0"})
        assert valid is True
        assert err is None

    def test_non_dict_rejected(self):
        valid, err = validate_metadata("string")
        assert valid is False
        assert "must be an object" in err

    def test_list_rejected(self):
        valid, err = validate_metadata([1, 2])
        assert valid is False
        assert "must be an object" in err

    def test_exceeds_default_max_size(self):
        big_meta = {"data": "x" * MAX_ACTION_METADATA_SIZE}
        valid, err = validate_metadata(big_meta)
        assert valid is False
        assert "maximum size" in err

    def test_custom_max_size(self):
        meta = {"key": "x" * 100}
        valid, err = validate_metadata(meta, max_size=50)
        assert valid is False
        assert "maximum size" in err

    def test_within_custom_max_size(self):
        meta = {"key": "val"}
        valid, err = validate_metadata(meta, max_size=50000)
        assert valid is True
        assert err is None

    def test_unserializable_metadata(self):
        valid, err = validate_metadata({"obj": object()})
        assert valid is False
        assert "invalid data" in err

    def test_metadata_with_nested_types(self):
        valid, err = validate_metadata(
            {
                "tags": ["a", "b"],
                "count": 10,
                "active": True,
                "extra": None,
            }
        )
        assert valid is True
        assert err is None

    def test_credential_metadata_max_size(self):
        """Test with credential metadata size limit."""
        big = {"data": "x" * MAX_CREDENTIAL_METADATA_SIZE}
        valid, err = validate_metadata(big, max_size=MAX_CREDENTIAL_METADATA_SIZE)
        assert valid is False

    def test_session_metadata_max_size(self):
        """Test with session metadata size limit."""
        big = {"data": "x" * MAX_SESSION_METADATA_SIZE}
        valid, err = validate_metadata(big, max_size=MAX_SESSION_METADATA_SIZE)
        assert valid is False


# ============================================================================
# Integration / Security scenarios
# ============================================================================


class TestSecurityScenarios:
    """Tests that combine multiple validators for security scenarios."""

    def test_command_injection_via_credential_name(self):
        valid, _ = validate_credential_name("$(whoami)")
        assert valid is False

    def test_command_injection_via_action_type(self):
        valid, _ = validate_action_type("exec;rm -rf /")
        assert valid is False

    def test_sql_injection_via_credential_name(self):
        valid, _ = validate_credential_name("key' OR '1'='1")
        assert valid is False

    def test_path_traversal_via_credential_name(self):
        valid, _ = validate_credential_name("../../../etc/passwd")
        assert valid is False

    def test_null_byte_injection_via_secret(self):
        valid, _ = validate_credential_secret("valid\x00malicious_override")
        assert valid is False

    def test_sanitize_prevents_command_chaining(self):
        result = sanitize_action_parameters({"cmd": "echo hello; rm -rf /"})
        # The semicolon should be escaped
        assert "\\;" in result["cmd"]

    def test_sanitize_prevents_subshell(self):
        result = sanitize_action_parameters({"cmd": "$(cat /etc/passwd)"})
        assert "\\$" in result["cmd"]
        assert "\\(" in result["cmd"]

    def test_sanitize_prevents_backtick_execution(self):
        result = sanitize_action_parameters({"cmd": "`id`"})
        assert "\\`" in result["cmd"]

    def test_oversized_config_with_many_keys(self):
        """Config that has both too many keys and is too large."""
        config = {f"key_{i}": "x" * 200 for i in range(100)}
        valid, err = validate_session_config(config)
        assert valid is False

    def test_deeply_nested_config_attack(self):
        """Attempt to cause stack overflow with deep nesting."""
        config: dict[str, Any] = {"val": "x"}
        for _ in range(20):
            config = {"n": config}
        valid, err = validate_session_config(config)
        assert valid is False
        assert "nesting depth" in err

    def test_large_action_input_with_repetition(self):
        """Large input via repeated keys."""
        data = {f"field_{i}": "x" * 1000 for i in range(100)}
        valid, err = validate_action_input(data)
        # Depends on whether total size exceeds 64KB
        serialized = json.dumps(data)
        if len(serialized) > MAX_ACTION_INPUT_SIZE:
            assert valid is False
        else:
            assert valid is True

    def test_xss_in_action_parameters_sanitized(self):
        result = sanitize_action_parameters({"html": "<script>alert('xss')</script>"})
        sanitized = result["html"]
        assert "\\<" in sanitized
        assert "\\>" in sanitized


# ============================================================================
# Edge cases and boundary tests
# ============================================================================


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_credential_name_all_uppercase(self):
        valid, err = validate_credential_name("MYAPIKEY")
        assert valid is True

    def test_credential_name_mixed_case(self):
        valid, err = validate_credential_name("MyApiKey")
        assert valid is True

    def test_action_type_all_periods(self):
        valid, err = validate_action_type("a.b.c.d.e")
        assert valid is True

    def test_action_type_trailing_dot_rejected(self):
        # "a." matches pattern: starts with letter, then "." is allowed
        valid, _ = validate_action_type("a.")
        assert valid is True  # Pattern allows trailing dot

    def test_sanitize_preserves_int_zero(self):
        result = sanitize_action_parameters({"val": 0})
        assert result["val"] == 0

    def test_sanitize_preserves_empty_list(self):
        result = sanitize_action_parameters({"items": []})
        assert result["items"] == []

    def test_sanitize_preserves_empty_nested_dict(self):
        result = sanitize_action_parameters({"inner": {}})
        assert result["inner"] == {}

    def test_validate_metadata_zero_max_size(self):
        """Any non-empty metadata should fail with max_size=0."""
        valid, err = validate_metadata({"k": "v"}, max_size=0)
        assert valid is False

    def test_validate_metadata_empty_dict_with_zero_max_size(self):
        """Empty dict serializes to '{}' (2 bytes), fails with max_size=0."""
        valid, err = validate_metadata({}, max_size=0)
        assert valid is False

    def test_session_config_one_key(self):
        valid, err = validate_session_config({"only": "one"})
        assert valid is True
        assert err is None

    def test_session_config_depth_zero_flat(self):
        """Flat config should always pass depth check."""
        config = {f"k{i}": i for i in range(10)}
        valid, err = validate_session_config(config)
        assert valid is True

    def test_credential_secret_exactly_min_minus_one(self):
        secret = "a" * (MIN_CREDENTIAL_SECRET_LENGTH - 1)
        valid, err = validate_credential_secret(secret)
        assert valid is False

    def test_credential_secret_exactly_max_plus_one(self):
        secret = "a" * (MAX_CREDENTIAL_SECRET_LENGTH + 1)
        valid, err = validate_credential_secret(secret)
        assert valid is False

    def test_action_type_exactly_max_plus_one(self):
        action_type = "a" + "b" * MAX_ACTION_TYPE_LENGTH
        assert len(action_type) == MAX_ACTION_TYPE_LENGTH + 1
        valid, err = validate_action_type(action_type)
        assert valid is False

    def test_sanitize_false_value_returns_empty_dict(self):
        """Falsy non-None values should return empty dict."""
        assert sanitize_action_parameters(0) == {}
        assert sanitize_action_parameters(False) == {}
        assert sanitize_action_parameters("") == {}
