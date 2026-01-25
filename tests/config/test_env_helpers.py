"""Tests for environment variable helper utilities."""

import os
import pytest
from unittest.mock import patch


class TestEnvHelpers:
    """Tests for env_helpers module."""

    def test_env_str_returns_value(self):
        """Test env_str returns environment value."""
        from aragora.config.env_helpers import env_str

        with patch.dict(os.environ, {"TEST_VAR": "hello"}):
            assert env_str("TEST_VAR", "default") == "hello"

    def test_env_str_returns_default(self):
        """Test env_str returns default when not set."""
        from aragora.config.env_helpers import env_str

        # Ensure the var is not set
        os.environ.pop("NONEXISTENT_VAR", None)
        assert env_str("NONEXISTENT_VAR", "default") == "default"

    def test_env_int_returns_value(self):
        """Test env_int parses integer value."""
        from aragora.config.env_helpers import env_int

        with patch.dict(os.environ, {"TEST_INT": "42"}):
            assert env_int("TEST_INT", 0) == 42

    def test_env_int_returns_default_on_invalid(self):
        """Test env_int returns default on invalid value."""
        from aragora.config.env_helpers import env_int

        with patch.dict(os.environ, {"TEST_INT": "not_a_number"}):
            assert env_int("TEST_INT", 99) == 99

    def test_env_int_returns_default_when_not_set(self):
        """Test env_int returns default when not set."""
        from aragora.config.env_helpers import env_int

        os.environ.pop("NONEXISTENT_INT", None)
        assert env_int("NONEXISTENT_INT", 123) == 123

    def test_env_float_returns_value(self):
        """Test env_float parses float value."""
        from aragora.config.env_helpers import env_float

        with patch.dict(os.environ, {"TEST_FLOAT": "3.14"}):
            assert env_float("TEST_FLOAT", 0.0) == 3.14

    def test_env_float_returns_default_on_invalid(self):
        """Test env_float returns default on invalid value."""
        from aragora.config.env_helpers import env_float

        with patch.dict(os.environ, {"TEST_FLOAT": "not_a_float"}):
            assert env_float("TEST_FLOAT", 2.5) == 2.5

    def test_env_bool_true_values(self):
        """Test env_bool recognizes true values."""
        from aragora.config.env_helpers import env_bool

        for val in ["true", "TRUE", "True", "1", "yes", "YES", "on", "ON"]:
            with patch.dict(os.environ, {"TEST_BOOL": val}):
                assert env_bool("TEST_BOOL", False) is True, f"Failed for {val}"

    def test_env_bool_false_values(self):
        """Test env_bool returns False for other values."""
        from aragora.config.env_helpers import env_bool

        for val in ["false", "FALSE", "0", "no", "off", "anything"]:
            with patch.dict(os.environ, {"TEST_BOOL": val}):
                assert env_bool("TEST_BOOL", True) is False, f"Failed for {val}"

    def test_env_bool_returns_default(self):
        """Test env_bool returns default when not set."""
        from aragora.config.env_helpers import env_bool

        os.environ.pop("NONEXISTENT_BOOL", None)
        assert env_bool("NONEXISTENT_BOOL", True) is True
        assert env_bool("NONEXISTENT_BOOL", False) is False

    def test_env_list_returns_parsed_list(self):
        """Test env_list parses comma-separated values."""
        from aragora.config.env_helpers import env_list

        with patch.dict(os.environ, {"TEST_LIST": "a, b, c"}):
            result = env_list("TEST_LIST")
            assert result == ["a", "b", "c"]

    def test_env_list_handles_extra_spaces(self):
        """Test env_list strips whitespace."""
        from aragora.config.env_helpers import env_list

        with patch.dict(os.environ, {"TEST_LIST": "  item1  ,  item2  ,  item3  "}):
            result = env_list("TEST_LIST")
            assert result == ["item1", "item2", "item3"]

    def test_env_list_returns_default_when_empty(self):
        """Test env_list returns default for empty value."""
        from aragora.config.env_helpers import env_list

        with patch.dict(os.environ, {"TEST_LIST": ""}):
            result = env_list("TEST_LIST", default=["default"])
            assert result == ["default"]

    def test_env_list_returns_default_when_not_set(self):
        """Test env_list returns default when not set."""
        from aragora.config.env_helpers import env_list

        os.environ.pop("NONEXISTENT_LIST", None)
        result = env_list("NONEXISTENT_LIST", default=["a", "b"])
        assert result == ["a", "b"]

    def test_env_list_custom_separator(self):
        """Test env_list with custom separator."""
        from aragora.config.env_helpers import env_list

        with patch.dict(os.environ, {"TEST_LIST": "a;b;c"}):
            result = env_list("TEST_LIST", separator=";")
            assert result == ["a", "b", "c"]


class TestBackwardCompatibility:
    """Test backward compatibility aliases."""

    def test_private_aliases_available(self):
        """Test that private aliases are exported."""
        from aragora.config.env_helpers import (
            _env_str,
            _env_int,
            _env_float,
            _env_bool,
        )

        assert callable(_env_str)
        assert callable(_env_int)
        assert callable(_env_float)
        assert callable(_env_bool)

    def test_private_aliases_work_correctly(self):
        """Test that private aliases work the same as public functions."""
        from aragora.config.env_helpers import (
            env_str,
            env_int,
            _env_str,
            _env_int,
        )

        with patch.dict(os.environ, {"TEST_VAR": "42"}):
            assert _env_str("TEST_VAR", "default") == env_str("TEST_VAR", "default")
            assert _env_int("TEST_VAR", 0) == env_int("TEST_VAR", 0)
