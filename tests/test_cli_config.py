"""Tests for CLI config command."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from aragora.cli.config import (
    CONFIG_FILE,
    ENV_KEYS,
    cmd_config,
    find_config,
    get_nested,
    load_config,
    save_config,
    set_nested,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for config tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def config_file(temp_dir):
    """Create a temporary config file."""
    config_path = temp_dir / CONFIG_FILE
    config_data = {
        "debate": {
            "rounds": 3,
            "agents": ["anthropic", "openai"],
        },
        "server": {
            "port": 8080,
            "host": "localhost",
        },
        "enabled": True,
    }
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    return config_path


class TestFindConfig:
    """Tests for find_config function."""

    def test_finds_config_in_current_dir(self, temp_dir, config_file):
        """Find config in current directory."""
        with patch("aragora.cli.config.Path.cwd", return_value=temp_dir):
            result = find_config()
            assert result == config_file

    def test_finds_config_in_parent_dir(self, temp_dir, config_file):
        """Find config in parent directory."""
        child_dir = temp_dir / "child" / "grandchild"
        child_dir.mkdir(parents=True)

        with patch("aragora.cli.config.Path.cwd", return_value=child_dir):
            result = find_config()
            assert result == config_file

    def test_returns_none_when_not_found(self, temp_dir):
        """Return None when no config file exists."""
        with patch("aragora.cli.config.Path.cwd", return_value=temp_dir):
            result = find_config()
            assert result is None


class TestLoadConfig:
    """Tests for load_config function."""

    def test_loads_valid_config(self, config_file):
        """Load a valid config file."""
        config = load_config(config_file)

        assert config["debate"]["rounds"] == 3
        assert config["server"]["port"] == 8080
        assert config["enabled"] is True

    def test_returns_empty_for_missing_file(self, temp_dir):
        """Return empty dict for missing file."""
        missing_path = temp_dir / "nonexistent.yaml"
        config = load_config(missing_path)
        assert config == {}

    def test_returns_empty_for_none_path(self, temp_dir):
        """Return empty dict when path is None."""
        with patch("aragora.cli.config.find_config", return_value=None):
            config = load_config()
            assert config == {}

    def test_handles_invalid_yaml(self, temp_dir):
        """Handle invalid YAML gracefully."""
        bad_config = temp_dir / CONFIG_FILE
        with open(bad_config, "w") as f:
            f.write("invalid: yaml: content: [")

        config = load_config(bad_config)
        assert config == {}

    def test_handles_empty_file(self, temp_dir):
        """Handle empty config file."""
        empty_config = temp_dir / CONFIG_FILE
        empty_config.touch()

        config = load_config(empty_config)
        assert config == {}


class TestSaveConfig:
    """Tests for save_config function."""

    def test_saves_config(self, temp_dir):
        """Save a config file."""
        config_path = temp_dir / CONFIG_FILE
        config = {"debug": True, "port": 9000}

        result = save_config(config, config_path)

        assert result is True
        assert config_path.exists()

        # Verify contents
        with open(config_path) as f:
            loaded = yaml.safe_load(f)
        assert loaded == config

    def test_creates_new_file_if_none_exists(self, temp_dir):
        """Create new config file when none exists."""
        with patch("aragora.cli.config.find_config", return_value=None):
            with patch("aragora.cli.config.Path.cwd", return_value=temp_dir):
                result = save_config({"test": True})

        assert result is True
        assert (temp_dir / CONFIG_FILE).exists()

    def test_returns_false_on_error(self, temp_dir):
        """Return False when save fails."""
        # Try to save to a non-existent directory
        bad_path = temp_dir / "nonexistent" / "deep" / CONFIG_FILE

        result = save_config({"test": True}, bad_path)
        assert result is False


class TestGetNested:
    """Tests for get_nested function."""

    def test_gets_top_level_value(self):
        """Get a top-level config value."""
        config = {"debug": True}
        assert get_nested(config, "debug") is True

    def test_gets_nested_value(self):
        """Get a nested config value."""
        config = {"debate": {"rounds": 5}}
        assert get_nested(config, "debate.rounds") == 5

    def test_gets_deeply_nested_value(self):
        """Get a deeply nested config value."""
        config = {"a": {"b": {"c": {"d": "deep"}}}}
        assert get_nested(config, "a.b.c.d") == "deep"

    def test_returns_none_for_missing_key(self):
        """Return None for missing key."""
        config = {"existing": "value"}
        assert get_nested(config, "missing") is None

    def test_returns_none_for_missing_nested_key(self):
        """Return None for missing nested key."""
        config = {"debate": {"rounds": 5}}
        assert get_nested(config, "debate.missing") is None
        assert get_nested(config, "nonexistent.key") is None


class TestSetNested:
    """Tests for set_nested function."""

    def test_sets_top_level_value(self):
        """Set a top-level config value."""
        config = {}
        set_nested(config, "debug", True)
        assert config["debug"] is True

    def test_sets_nested_value(self):
        """Set a nested config value."""
        config = {}
        set_nested(config, "debate.rounds", 5)
        assert config["debate"]["rounds"] == 5

    def test_sets_deeply_nested_value(self):
        """Set a deeply nested config value."""
        config = {}
        set_nested(config, "a.b.c.d", "deep")
        assert config["a"]["b"]["c"]["d"] == "deep"

    def test_overwrites_existing_value(self):
        """Overwrite an existing value."""
        config = {"debate": {"rounds": 3}}
        set_nested(config, "debate.rounds", 10)
        assert config["debate"]["rounds"] == 10

    def test_preserves_sibling_values(self):
        """Preserve sibling values when setting."""
        config = {"debate": {"rounds": 3, "agents": ["a", "b"]}}
        set_nested(config, "debate.rounds", 10)
        assert config["debate"]["agents"] == ["a", "b"]


class TestCmdConfig:
    """Tests for cmd_config command handler."""

    def test_show_action(self, capsys, config_file, temp_dir):
        """Show config action."""
        with patch("aragora.cli.config.find_config", return_value=config_file):
            args = MagicMock()
            args.action = "show"
            cmd_config(args)

        captured = capsys.readouterr()
        assert "Configuration from:" in captured.out
        assert "debate:" in captured.out
        assert "rounds: 3" in captured.out

    def test_show_no_config(self, capsys, temp_dir):
        """Show when no config exists."""
        with patch("aragora.cli.config.find_config", return_value=None):
            args = MagicMock()
            args.action = "show"
            cmd_config(args)

        captured = capsys.readouterr()
        assert "No .aragora.yaml found" in captured.out

    def test_get_action(self, capsys, config_file):
        """Get a specific config value."""
        with patch("aragora.cli.config.find_config", return_value=config_file):
            args = MagicMock()
            args.action = "get"
            args.key = "debate.rounds"
            cmd_config(args)

        captured = capsys.readouterr()
        assert "debate.rounds: 3" in captured.out

    def test_get_missing_key(self, capsys, config_file):
        """Get a missing config key."""
        with patch("aragora.cli.config.find_config", return_value=config_file):
            args = MagicMock()
            args.action = "get"
            args.key = "missing.key"
            cmd_config(args)

        captured = capsys.readouterr()
        assert "(not set)" in captured.out

    def test_set_action(self, capsys, config_file, temp_dir):
        """Set a config value."""
        with patch("aragora.cli.config.find_config", return_value=config_file):
            args = MagicMock()
            args.action = "set"
            args.key = "debug"
            args.value = "true"
            cmd_config(args)

        captured = capsys.readouterr()
        assert "Set debug = True" in captured.out

        # Verify the value was actually saved
        config = load_config(config_file)
        assert config["debug"] is True

    def test_set_integer_value(self, config_file, temp_dir):
        """Set an integer config value."""
        with patch("aragora.cli.config.find_config", return_value=config_file):
            args = MagicMock()
            args.action = "set"
            args.key = "server.port"
            args.value = "9000"
            cmd_config(args)

        config = load_config(config_file)
        assert config["server"]["port"] == 9000

    def test_set_float_value(self, config_file, temp_dir):
        """Set a float config value."""
        with patch("aragora.cli.config.find_config", return_value=config_file):
            args = MagicMock()
            args.action = "set"
            args.key = "threshold"
            args.value = "0.75"
            cmd_config(args)

        config = load_config(config_file)
        assert config["threshold"] == 0.75

    def test_env_action(self, capsys):
        """Show environment variables."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test123456789"}):
            args = MagicMock()
            args.action = "env"
            cmd_config(args)

        captured = capsys.readouterr()
        assert "API Key Environment Variables:" in captured.out
        assert "ANTHROPIC_API_KEY:" in captured.out
        # Key should be masked
        assert "sk-a" in captured.out
        assert "..." in captured.out

    def test_env_shows_not_set(self, capsys):
        """Show (not set) for missing env vars."""
        # Clear relevant env vars
        env_copy = {k: v for k, v in os.environ.items() if k not in ENV_KEYS}
        with patch.dict(os.environ, env_copy, clear=True):
            args = MagicMock()
            args.action = "env"
            cmd_config(args)

        captured = capsys.readouterr()
        assert "(not set)" in captured.out

    def test_path_action(self, capsys, config_file):
        """Show config path."""
        with patch("aragora.cli.config.find_config", return_value=config_file):
            args = MagicMock()
            args.action = "path"
            cmd_config(args)

        captured = capsys.readouterr()
        assert "Config file:" in captured.out
        assert str(config_file) in captured.out

    def test_path_no_config(self, capsys, temp_dir):
        """Show expected path when no config exists."""
        with patch("aragora.cli.config.find_config", return_value=None):
            with patch("aragora.cli.config.Path.cwd", return_value=temp_dir):
                args = MagicMock()
                args.action = "path"
                cmd_config(args)

        captured = capsys.readouterr()
        assert "No config file found" in captured.out
        assert CONFIG_FILE in captured.out

    def test_default_action_is_show(self, capsys, config_file):
        """Default action is show."""
        with patch("aragora.cli.config.find_config", return_value=config_file):
            args = MagicMock(spec=[])  # No action attribute
            cmd_config(args)

        captured = capsys.readouterr()
        assert "Configuration from:" in captured.out


class TestEnvKeys:
    """Tests for ENV_KEYS constant."""

    def test_contains_expected_keys(self):
        """ENV_KEYS contains expected API key names."""
        expected = [
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "OPENROUTER_API_KEY",
        ]
        for key in expected:
            assert key in ENV_KEYS
