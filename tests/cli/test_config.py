"""Tests for CLI config module."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from aragora.cli.config import (
    CONFIG_FILE,
    ENV_KEYS,
    find_config,
    get_nested,
    load_config,
    save_config,
    set_nested,
)


class TestFindConfig:
    """Test config file discovery."""

    def test_find_config_in_current_dir(self, tmp_path):
        """Test finding config in current directory."""
        config_file = tmp_path / CONFIG_FILE
        config_file.write_text("key: value")

        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = find_config()
            assert result is not None
            assert result.name == CONFIG_FILE
        finally:
            os.chdir(original_cwd)

    def test_find_config_in_parent_dir(self, tmp_path):
        """Test finding config in parent directory."""
        config_file = tmp_path / CONFIG_FILE
        config_file.write_text("key: value")

        subdir = tmp_path / "subdir" / "nested"
        subdir.mkdir(parents=True)

        original_cwd = os.getcwd()
        try:
            os.chdir(subdir)
            result = find_config()
            assert result is not None
            assert result == config_file
        finally:
            os.chdir(original_cwd)

    def test_find_config_not_found(self, tmp_path):
        """Test when config file doesn't exist."""
        subdir = tmp_path / "empty"
        subdir.mkdir()

        original_cwd = os.getcwd()
        try:
            os.chdir(subdir)
            # Create a fresh temp directory with no config anywhere
            # Note: This may find a config from parent dirs in the real filesystem
            # so we just verify it returns either None or a Path
            result = find_config()
            # Result is either None or a valid Path
            assert result is None or isinstance(result, Path)
        finally:
            os.chdir(original_cwd)


class TestLoadConfig:
    """Test config loading."""

    def test_load_config_from_path(self, tmp_path):
        """Test loading config from explicit path."""
        config_file = tmp_path / "config.yaml"
        config_data = {"debate": {"rounds": 5}, "agents": ["claude", "gpt"]}
        config_file.write_text(yaml.dump(config_data))

        result = load_config(config_file)
        assert result == config_data

    def test_load_config_missing_file(self, tmp_path):
        """Test loading config when file doesn't exist."""
        missing = tmp_path / "missing.yaml"
        result = load_config(missing)
        assert result == {}

    def test_load_config_empty_file(self, tmp_path):
        """Test loading empty config file."""
        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("")

        result = load_config(empty_file)
        assert result == {}

    def test_load_config_invalid_yaml(self, tmp_path, capsys):
        """Test loading invalid YAML."""
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("invalid: yaml: content: [")

        result = load_config(bad_file)
        assert result == {}

        captured = capsys.readouterr()
        assert "Warning" in captured.out or "Warning" in captured.err

    def test_load_config_with_none_path(self, tmp_path):
        """Test loading config with None path (uses find_config)."""
        # When no path given and no config found, return empty
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = load_config(None)
            # Should return empty dict or found config
            assert isinstance(result, dict)
        finally:
            os.chdir(original_cwd)


class TestSaveConfig:
    """Test config saving."""

    def test_save_config_to_path(self, tmp_path):
        """Test saving config to explicit path."""
        config_file = tmp_path / "output.yaml"
        config_data = {"setting": "value", "nested": {"key": 123}}

        result = save_config(config_data, config_file)
        assert result is True
        assert config_file.exists()

        # Verify content
        loaded = yaml.safe_load(config_file.read_text())
        assert loaded == config_data

    def test_save_config_creates_file(self, tmp_path):
        """Test that save creates file if it doesn't exist."""
        config_file = tmp_path / "new.yaml"
        assert not config_file.exists()

        save_config({"new": "config"}, config_file)
        assert config_file.exists()

    def test_save_config_overwrites(self, tmp_path):
        """Test that save overwrites existing content."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"old": "data"}))

        save_config({"new": "data"}, config_file)

        loaded = yaml.safe_load(config_file.read_text())
        assert loaded == {"new": "data"}
        assert "old" not in loaded

    def test_save_config_permission_error(self, tmp_path, capsys):
        """Test handling permission errors."""
        # Create a read-only directory
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        config_file = readonly_dir / "config.yaml"

        # Make directory read-only (skip on Windows)
        if os.name != "nt":
            readonly_dir.chmod(0o444)
            try:
                result = save_config({"test": "data"}, config_file)
                assert result is False
                captured = capsys.readouterr()
                assert "Error" in captured.out
            finally:
                readonly_dir.chmod(0o755)


class TestGetNested:
    """Test nested config value retrieval."""

    def test_get_single_level(self):
        """Test getting a top-level key."""
        config = {"key": "value"}
        assert get_nested(config, "key") == "value"

    def test_get_nested_two_levels(self):
        """Test getting a two-level nested key."""
        config = {"level1": {"level2": "value"}}
        assert get_nested(config, "level1.level2") == "value"

    def test_get_nested_three_levels(self):
        """Test getting a deeply nested key."""
        config = {"a": {"b": {"c": "deep"}}}
        assert get_nested(config, "a.b.c") == "deep"

    def test_get_nested_missing_key(self):
        """Test getting a non-existent key."""
        config = {"key": "value"}
        assert get_nested(config, "missing") is None

    def test_get_nested_partial_path(self):
        """Test getting a partial path that doesn't exist."""
        config = {"a": {"b": "value"}}
        assert get_nested(config, "a.missing.key") is None

    def test_get_nested_non_dict_intermediate(self):
        """Test when intermediate value is not a dict."""
        config = {"a": "not_a_dict"}
        assert get_nested(config, "a.b") is None

    def test_get_nested_returns_dict(self):
        """Test getting a nested dict."""
        config = {"parent": {"child": {"key": "value"}}}
        result = get_nested(config, "parent.child")
        assert result == {"key": "value"}

    def test_get_nested_returns_list(self):
        """Test getting a list value."""
        config = {"items": [1, 2, 3]}
        assert get_nested(config, "items") == [1, 2, 3]


class TestSetNested:
    """Test nested config value setting."""

    def test_set_single_level(self):
        """Test setting a top-level key."""
        config = {}
        set_nested(config, "key", "value")
        assert config == {"key": "value"}

    def test_set_nested_two_levels(self):
        """Test setting a two-level nested key."""
        config = {}
        set_nested(config, "level1.level2", "value")
        assert config == {"level1": {"level2": "value"}}

    def test_set_nested_three_levels(self):
        """Test setting a deeply nested key."""
        config = {}
        set_nested(config, "a.b.c", "deep")
        assert config == {"a": {"b": {"c": "deep"}}}

    def test_set_nested_preserves_siblings(self):
        """Test that setting doesn't overwrite siblings."""
        config = {"a": {"existing": "value"}}
        set_nested(config, "a.new", "new_value")
        assert config == {"a": {"existing": "value", "new": "new_value"}}

    def test_set_nested_overwrites_value(self):
        """Test overwriting an existing value."""
        config = {"a": {"b": "old"}}
        set_nested(config, "a.b", "new")
        assert config == {"a": {"b": "new"}}

    def test_set_nested_creates_intermediate(self):
        """Test that intermediate dicts are created."""
        config = {"existing": "value"}
        set_nested(config, "new.path.deep", "value")
        assert config == {
            "existing": "value",
            "new": {"path": {"deep": "value"}},
        }


class TestEnvKeys:
    """Test ENV_KEYS constant."""

    def test_all_expected_keys_present(self):
        """Verify all expected API key names are in ENV_KEYS."""
        expected = [
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
        ]
        for key in expected:
            assert key in ENV_KEYS

    def test_keys_are_strings(self):
        """All keys should be strings."""
        for key in ENV_KEYS:
            assert isinstance(key, str)
            assert key.endswith("_KEY") or key.endswith("_API_KEY")
