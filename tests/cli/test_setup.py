"""
Tests for aragora.cli.setup module.

Tests setup wizard CLI commands.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.cli.setup import (
    _confirm,
    _generate_env_content,
    _print_header,
    _print_step,
    _prompt,
    _test_api_key,
    cmd_setup,
    run_setup,
)


# ===========================================================================
# Tests: _prompt
# ===========================================================================


class TestPrompt:
    """Tests for _prompt function."""

    def test_prompt_returns_input(self, monkeypatch):
        """Test prompt returns user input."""
        monkeypatch.setattr("builtins.input", lambda _: "user input")
        result = _prompt("Enter value")
        assert result == "user input"

    def test_prompt_strips_whitespace(self, monkeypatch):
        """Test prompt strips whitespace."""
        monkeypatch.setattr("builtins.input", lambda _: "  value  ")
        result = _prompt("Enter value")
        assert result == "value"

    def test_prompt_with_default(self, monkeypatch):
        """Test prompt returns default when empty input."""
        monkeypatch.setattr("builtins.input", lambda _: "")
        result = _prompt("Enter value", default="default")
        assert result == "default"

    def test_prompt_overrides_default(self, monkeypatch):
        """Test prompt returns input when default is available."""
        monkeypatch.setattr("builtins.input", lambda _: "custom")
        result = _prompt("Enter value", default="default")
        assert result == "custom"

    def test_prompt_secret_mode(self, monkeypatch):
        """Test prompt in secret mode uses getpass."""
        mock_getpass = MagicMock(return_value="secret_value")
        monkeypatch.setattr("getpass.getpass", mock_getpass)
        result = _prompt("Password", secret=True)
        assert result == "secret_value"
        mock_getpass.assert_called_once()

    def test_prompt_eof_exits(self, monkeypatch):
        """Test prompt exits on EOF."""

        def raise_eof(_):
            raise EOFError()

        monkeypatch.setattr("builtins.input", raise_eof)
        with pytest.raises(SystemExit):
            _prompt("Enter value")

    def test_prompt_keyboard_interrupt_exits(self, monkeypatch):
        """Test prompt exits on Ctrl+C."""

        def raise_interrupt(_):
            raise KeyboardInterrupt()

        monkeypatch.setattr("builtins.input", raise_interrupt)
        with pytest.raises(SystemExit):
            _prompt("Enter value")


# ===========================================================================
# Tests: _confirm
# ===========================================================================


class TestConfirm:
    """Tests for _confirm function."""

    def test_confirm_yes(self, monkeypatch):
        """Test confirm returns True for 'y'."""
        monkeypatch.setattr("builtins.input", lambda _: "y")
        assert _confirm("Continue?") is True

    def test_confirm_yes_full(self, monkeypatch):
        """Test confirm returns True for 'yes'."""
        monkeypatch.setattr("builtins.input", lambda _: "yes")
        assert _confirm("Continue?") is True

    def test_confirm_no(self, monkeypatch):
        """Test confirm returns False for 'n'."""
        monkeypatch.setattr("builtins.input", lambda _: "n")
        assert _confirm("Continue?") is False

    def test_confirm_empty_default_true(self, monkeypatch):
        """Test confirm returns default True on empty."""
        monkeypatch.setattr("builtins.input", lambda _: "")
        assert _confirm("Continue?", default=True) is True

    def test_confirm_empty_default_false(self, monkeypatch):
        """Test confirm returns default False on empty."""
        monkeypatch.setattr("builtins.input", lambda _: "")
        assert _confirm("Continue?", default=False) is False

    def test_confirm_eof_exits(self, monkeypatch):
        """Test confirm exits on EOF."""

        def raise_eof(_):
            raise EOFError()

        monkeypatch.setattr("builtins.input", raise_eof)
        with pytest.raises(SystemExit):
            _confirm("Continue?")


# ===========================================================================
# Tests: _print_header
# ===========================================================================


class TestPrintHeader:
    """Tests for _print_header function."""

    def test_prints_header(self, capsys):
        """Test header is printed."""
        _print_header("Test Header")
        captured = capsys.readouterr()
        assert "Test Header" in captured.out
        assert "=" * 60 in captured.out


# ===========================================================================
# Tests: _print_step
# ===========================================================================


class TestPrintStep:
    """Tests for _print_step function."""

    def test_prints_step(self, capsys):
        """Test step is printed."""
        _print_step(1, 5, "First Step")
        captured = capsys.readouterr()
        assert "[1/5]" in captured.out
        assert "First Step" in captured.out
        assert "-" * 40 in captured.out


# ===========================================================================
# Tests: _test_api_key
# ===========================================================================


class TestTestApiKey:
    """Tests for _test_api_key function."""

    def test_empty_key(self):
        """Test empty key returns failure."""
        success, msg = _test_api_key("anthropic", "")
        assert success is False
        assert "No key provided" in msg

    def test_anthropic_valid_key(self):
        """Test valid Anthropic key."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.post", return_value=mock_response):
            success, msg = _test_api_key("anthropic", "valid-key")

        assert success is True
        assert "Valid" in msg

    def test_anthropic_invalid_key(self):
        """Test invalid Anthropic key."""
        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch("httpx.post", return_value=mock_response):
            success, msg = _test_api_key("anthropic", "invalid-key")

        assert success is False
        assert "Invalid" in msg

    def test_anthropic_error_status(self):
        """Test Anthropic error status."""
        mock_response = MagicMock()
        mock_response.status_code = 500

        with patch("httpx.post", return_value=mock_response):
            success, msg = _test_api_key("anthropic", "key")

        assert success is False
        assert "Error: 500" in msg

    def test_openai_valid_key(self):
        """Test valid OpenAI key."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.get", return_value=mock_response):
            success, msg = _test_api_key("openai", "valid-key")

        assert success is True
        assert "Valid" in msg

    def test_openai_invalid_key(self):
        """Test invalid OpenAI key."""
        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch("httpx.get", return_value=mock_response):
            success, msg = _test_api_key("openai", "invalid-key")

        assert success is False
        assert "Invalid" in msg

    def test_openrouter_valid_key(self):
        """Test valid OpenRouter key."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.get", return_value=mock_response):
            success, msg = _test_api_key("openrouter", "valid-key")

        assert success is True
        assert "Valid" in msg

    def test_unknown_provider(self):
        """Test unknown provider returns stored."""
        success, msg = _test_api_key("unknown", "any-key")
        assert success is True
        assert "not validated" in msg

    def test_import_error(self):
        """Test when httpx is not installed."""
        with patch.dict("sys.modules", {"httpx": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                success, msg = _test_api_key("anthropic", "key")
                assert success is True
                assert "not installed" in msg

    def test_connection_error(self):
        """Test connection error."""
        with patch("httpx.post", side_effect=Exception("Connection refused")):
            success, msg = _test_api_key("anthropic", "key")

        assert success is False
        assert "Connection error" in msg


# ===========================================================================
# Tests: _generate_env_content
# ===========================================================================


class TestGenerateEnvContent:
    """Tests for _generate_env_content function."""

    def test_generates_basic_content(self):
        """Test generates basic .env content."""
        config = {}
        content = _generate_env_content(config)

        assert "# Aragora Configuration" in content
        assert "ANTHROPIC_API_KEY" in content
        assert "OPENAI_API_KEY" in content
        assert "ARAGORA_HTTP_PORT=8080" in content
        assert "ARAGORA_WS_PORT=8765" in content

    def test_includes_anthropic_key(self):
        """Test includes Anthropic key when provided."""
        config = {"anthropic_key": "sk-ant-test123"}
        content = _generate_env_content(config)

        assert "ANTHROPIC_API_KEY=sk-ant-test123" in content

    def test_includes_openai_key(self):
        """Test includes OpenAI key when provided."""
        config = {"openai_key": "sk-test123"}
        content = _generate_env_content(config)

        assert "OPENAI_API_KEY=sk-test123" in content

    def test_includes_openrouter_key(self):
        """Test includes OpenRouter key when provided."""
        config = {"openrouter_key": "or-test123"}
        content = _generate_env_content(config)

        assert "OPENROUTER_API_KEY=or-test123" in content

    def test_includes_custom_ports(self):
        """Test includes custom ports."""
        config = {"http_port": 9090, "ws_port": 9999}
        content = _generate_env_content(config)

        assert "ARAGORA_HTTP_PORT=9090" in content
        assert "ARAGORA_WS_PORT=9999" in content

    def test_postgres_database(self):
        """Test PostgreSQL database configuration."""
        config = {"database_mode": "postgres", "database_url": "postgresql://db:5432/test"}
        content = _generate_env_content(config)

        assert "DATABASE_URL=postgresql://db:5432/test" in content

    def test_sqlite_database(self):
        """Test SQLite database configuration."""
        config = {"database_mode": "sqlite"}
        content = _generate_env_content(config)

        assert "# DATABASE_URL=sqlite:" in content

    def test_redis_config(self):
        """Test Redis configuration."""
        config = {"redis_url": "redis://localhost:6379"}
        content = _generate_env_content(config)

        assert "REDIS_URL=redis://localhost:6379" in content

    def test_encryption_key(self):
        """Test encryption key configuration."""
        config = {"encryption_key": "abc123"}
        content = _generate_env_content(config)

        assert "ARAGORA_ENCRYPTION_KEY=abc123" in content

    def test_integrations(self):
        """Test integration tokens."""
        config = {
            "slack_token": "xoxb-test",
            "github_token": "ghp_test",
            "telegram_token": "123456:ABC",
        }
        content = _generate_env_content(config)

        assert "SLACK_BOT_TOKEN=xoxb-test" in content
        assert "GITHUB_TOKEN=ghp_test" in content
        assert "TELEGRAM_BOT_TOKEN=123456:ABC" in content


# ===========================================================================
# Tests: run_setup
# ===========================================================================


class TestRunSetup:
    """Tests for run_setup function."""

    def test_returns_config_dict(self, tmp_path, monkeypatch):
        """Test run_setup returns a dictionary."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        # Use confirm True to accept existing keys
        with patch("aragora.cli.setup._confirm", return_value=True):
            with patch("aragora.cli.setup._prompt", return_value="8080"):
                config = run_setup(
                    output_path=str(tmp_path),
                    minimal=True,
                    skip_test=True,
                    non_interactive=False,
                )

        assert isinstance(config, dict)
        # Either returns config with keys or early exit dict
        assert config is not None

    def test_prints_wizard_header(self, tmp_path, capsys, monkeypatch):
        """Test run_setup prints wizard header."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with patch("aragora.cli.setup._confirm", return_value=True):
            with patch("aragora.cli.setup._prompt", return_value="8080"):
                run_setup(
                    output_path=str(tmp_path),
                    minimal=True,
                    skip_test=True,
                    non_interactive=False,
                )

        captured = capsys.readouterr()
        assert "Aragora Setup Wizard" in captured.out

    def test_detects_existing_anthropic_key(self, tmp_path, capsys, monkeypatch):
        """Test run_setup detects existing Anthropic key."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-existing")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with patch("aragora.cli.setup._confirm", return_value=True):
            with patch("aragora.cli.setup._prompt", return_value="8080"):
                run_setup(
                    output_path=str(tmp_path),
                    minimal=True,
                    skip_test=True,
                    non_interactive=False,
                )

        captured = capsys.readouterr()
        assert "Found existing ANTHROPIC_API_KEY" in captured.out

    def test_detects_existing_openai_key(self, tmp_path, capsys, monkeypatch):
        """Test run_setup detects existing OpenAI key."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-existing")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        with patch("aragora.cli.setup._confirm", return_value=True):
            with patch("aragora.cli.setup._prompt", return_value="8080"):
                run_setup(
                    output_path=str(tmp_path),
                    minimal=True,
                    skip_test=True,
                    non_interactive=False,
                )

        captured = capsys.readouterr()
        assert "Found existing OPENAI_API_KEY" in captured.out

    def test_creates_env_file(self, tmp_path, monkeypatch):
        """Test creates .env file."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with patch("aragora.cli.setup._confirm", return_value=True):
            with patch("aragora.cli.setup._prompt", return_value="8080"):
                run_setup(
                    output_path=str(tmp_path),
                    minimal=True,
                    skip_test=True,
                    non_interactive=False,
                )

        env_file = tmp_path / ".env"
        assert env_file.exists()
        content = env_file.read_text()
        assert "ARAGORA_HTTP_PORT" in content


# ===========================================================================
# Tests: cmd_setup
# ===========================================================================


class TestCmdSetup:
    """Tests for cmd_setup function."""

    def test_cmd_setup_with_defaults(self, tmp_path, monkeypatch):
        """Test cmd_setup with default arguments."""
        args = argparse.Namespace()
        args.output = str(tmp_path)
        args.minimal = True
        args.skip_test = True
        args.yes = True  # non-interactive

        # Mock run_setup to avoid interactive behavior
        with patch("aragora.cli.setup.run_setup") as mock_run:
            cmd_setup(args)

        mock_run.assert_called_once_with(
            output_path=str(tmp_path),
            minimal=True,
            skip_test=True,
            non_interactive=True,
        )

    def test_cmd_setup_without_attributes(self, tmp_path, monkeypatch):
        """Test cmd_setup handles missing attributes."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        args = argparse.Namespace()
        # No attributes set - should use defaults via getattr

        # Mock run_setup to avoid actual execution
        with patch("aragora.cli.setup.run_setup") as mock_run:
            cmd_setup(args)

        mock_run.assert_called_once_with(
            output_path=None,
            minimal=False,
            skip_test=False,
            non_interactive=False,
        )
