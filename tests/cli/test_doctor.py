"""
Tests for aragora.cli.doctor module.

Tests health check CLI commands.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.cli.doctor import (
    check_api_keys,
    check_environment,
    check_icon,
    check_packages,
    check_server,
    check_storage,
    main,
    print_section,
)


# ===========================================================================
# Tests: check_icon
# ===========================================================================


class TestCheckIcon:
    """Tests for check_icon function."""

    def test_true_returns_green(self):
        """Test True returns green checkmark."""
        result = check_icon(True)
        assert "✓" in result
        assert "92m" in result  # Green ANSI code

    def test_false_returns_red(self):
        """Test False returns red X."""
        result = check_icon(False)
        assert "✗" in result
        assert "91m" in result  # Red ANSI code

    def test_none_returns_yellow(self):
        """Test None returns yellow circle."""
        result = check_icon(None)
        assert "○" in result
        assert "93m" in result  # Yellow ANSI code


# ===========================================================================
# Tests: print_section
# ===========================================================================


class TestPrintSection:
    """Tests for print_section function."""

    def test_prints_section(self, capsys):
        """Test printing section header."""
        print_section("Test Section")
        captured = capsys.readouterr()
        assert "Test Section" in captured.out
        assert "-" * 40 in captured.out


# ===========================================================================
# Tests: check_packages
# ===========================================================================


class TestCheckPackages:
    """Tests for check_packages function."""

    def test_returns_list(self):
        """Test returns a list of tuples."""
        result = check_packages()
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) and len(item) == 3 for item in result)

    def test_checks_required_packages(self):
        """Test checks required packages."""
        result = check_packages()
        names = [name for name, _, _ in result]

        # These should be checked
        assert "aiohttp" in names
        assert "pydantic" in names
        assert "sqlite3" in names
        assert "asyncio" in names

    def test_checks_optional_ml_packages(self):
        """Test checks optional ML packages."""
        result = check_packages()
        names = [name for name, _, _ in result]

        # At least one ML package should be checked
        ml_pkgs = [n for n in names if "(ML)" in n]
        assert len(ml_pkgs) >= 1

    def test_checks_optional_integrations(self):
        """Test checks optional integrations."""
        result = check_packages()
        names = [name for name, _, _ in result]

        # At least one integration package should be checked
        int_pkgs = [n for n in names if "(integration)" in n]
        assert len(int_pkgs) >= 1


# ===========================================================================
# Tests: check_api_keys
# ===========================================================================


class TestCheckApiKeys:
    """Tests for check_api_keys function."""

    def test_returns_list(self, monkeypatch):
        """Test returns a list of tuples."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        result = check_api_keys()
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) and len(item) == 3 for item in result)

    def test_no_llm_keys_shows_warning(self, monkeypatch):
        """Test warning when no LLM keys are set."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        result = check_api_keys()
        names = [name for name, _, _ in result]
        statuses = [status for _, status, _ in result]

        # Should have a warning about no LLM provider
        assert "LLM Provider" in names or any("NO API KEY" in s for s in statuses)

    def test_anthropic_key_configured(self, monkeypatch):
        """Test Anthropic key detection."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        result = check_api_keys()
        anthropic = [item for item in result if item[0] == "ANTHROPIC_API_KEY"]

        assert len(anthropic) == 1
        assert anthropic[0][1] == "configured"
        assert anthropic[0][2] is True

    def test_openai_key_configured(self, monkeypatch):
        """Test OpenAI key detection."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        result = check_api_keys()
        openai = [item for item in result if item[0] == "OPENAI_API_KEY"]

        assert len(openai) == 1
        assert openai[0][1] == "configured"
        assert openai[0][2] is True

    def test_optional_keys_detected(self, monkeypatch):
        """Test optional API keys are detected."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        result = check_api_keys()
        openrouter = [item for item in result if item[0] == "OPENROUTER_API_KEY"]

        assert len(openrouter) == 1
        assert openrouter[0][1] == "configured"
        assert openrouter[0][2] is True


# ===========================================================================
# Tests: check_storage
# ===========================================================================


class TestCheckStorage:
    """Tests for check_storage function."""

    def test_returns_list(self):
        """Test returns a list of tuples."""
        result = check_storage()
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) and len(item) == 3 for item in result)

    def test_checks_sqlite(self):
        """Test checks SQLite."""
        result = check_storage()
        sqlite = [item for item in result if "SQLite" in item[0]]

        assert len(sqlite) >= 1
        # SQLite should work
        assert sqlite[0][2] is True

    def test_checks_data_directory(self):
        """Test checks data directory."""
        result = check_storage()
        data_dir = [item for item in result if "Data directory" in item[0]]

        assert len(data_dir) == 1

    def test_database_url_configured(self, monkeypatch):
        """Test DATABASE_URL detection when asyncpg available."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/test")

        result = check_storage()
        db_url = [item for item in result if item[0] == "DATABASE_URL"]

        # Only checked if asyncpg is available
        if db_url:
            assert db_url[0][1] == "configured"

    def test_redis_url_configured(self, monkeypatch):
        """Test ARAGORA_REDIS_URL detection when redis available."""
        monkeypatch.setenv("ARAGORA_REDIS_URL", "redis://localhost:6379")

        result = check_storage()
        redis_url = [item for item in result if item[0] == "ARAGORA_REDIS_URL"]

        # Only checked if redis is available
        if redis_url:
            assert redis_url[0][1] == "configured"


# ===========================================================================
# Tests: check_server
# ===========================================================================


class TestCheckServer:
    """Tests for check_server function."""

    @pytest.mark.asyncio
    async def test_server_not_running(self):
        """Test server not running detection."""
        mock_session = MagicMock()
        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(side_effect=Exception("Connection refused"))
        mock_context.__aexit__ = AsyncMock()
        mock_session.get.return_value = mock_context

        mock_client_session = MagicMock()
        mock_client_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_client_session.__aexit__ = AsyncMock()

        with patch("aiohttp.ClientSession", return_value=mock_client_session):
            result = await check_server()

        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_server_running_healthy(self):
        """Test healthy server detection."""
        mock_response = MagicMock()
        mock_response.status = 200

        mock_response_ctx = MagicMock()
        mock_response_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response_ctx.__aexit__ = AsyncMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response_ctx

        mock_client_session = MagicMock()
        mock_client_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_client_session.__aexit__ = AsyncMock()

        with patch("aiohttp.ClientSession", return_value=mock_client_session):
            result = await check_server()

        assert isinstance(result, list)
        # Check for running status
        server_check = [item for item in result if "Server" in item[0]]
        if server_check:
            assert server_check[0][2] is True
            assert "running" in server_check[0][1]


# ===========================================================================
# Tests: check_environment
# ===========================================================================


class TestCheckEnvironment:
    """Tests for check_environment function."""

    def test_returns_list(self):
        """Test returns a list of tuples."""
        result = check_environment()
        assert isinstance(result, list)
        assert all(isinstance(item, tuple) and len(item) == 3 for item in result)

    def test_checks_python_version(self):
        """Test checks Python version."""
        result = check_environment()
        python = [item for item in result if item[0] == "Python"]

        assert len(python) == 1
        # Should contain version string
        assert "." in python[0][1]

    def test_checks_environment_name(self, monkeypatch):
        """Test checks ARAGORA_ENV."""
        monkeypatch.setenv("ARAGORA_ENV", "production")

        result = check_environment()
        env = [item for item in result if item[0] == "Environment"]

        assert len(env) == 1
        assert env[0][1] == "production"

    def test_checks_debug_mode_enabled(self, monkeypatch):
        """Test checks debug mode when enabled."""
        monkeypatch.setenv("ARAGORA_DEBUG", "true")

        result = check_environment()
        debug = [item for item in result if item[0] == "Debug mode"]

        assert len(debug) == 1
        assert debug[0][1] == "enabled"

    def test_checks_debug_mode_disabled(self, monkeypatch):
        """Test checks debug mode when disabled."""
        monkeypatch.setenv("ARAGORA_DEBUG", "false")

        result = check_environment()
        debug = [item for item in result if item[0] == "Debug mode"]

        assert len(debug) == 1
        assert debug[0][1] == "disabled"


# ===========================================================================
# Tests: main
# ===========================================================================


class TestMain:
    """Tests for main function."""

    def test_runs_all_checks(self, capsys, monkeypatch):
        """Test main runs all check sections."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        # Mock server check to avoid network calls
        with patch("aragora.cli.doctor.check_server", new=AsyncMock(return_value=[])):
            result = main()

        captured = capsys.readouterr()

        # Check all sections are printed
        assert "ARAGORA HEALTH CHECK" in captured.out
        assert "Environment" in captured.out
        assert "Packages" in captured.out
        assert "API Keys" in captured.out
        assert "Storage" in captured.out
        assert "Server" in captured.out
        assert "Summary" in captured.out

    def test_returns_0_when_all_ok(self, monkeypatch):
        """Test returns 0 when all checks pass."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        with patch("aragora.cli.doctor.check_server", new=AsyncMock(return_value=[])):
            result = main()

        assert result == 0

    def test_returns_1_when_checks_fail(self, monkeypatch):
        """Test returns 1 when checks fail."""
        # Remove all LLM keys to cause failure
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with patch("aragora.cli.doctor.check_server", new=AsyncMock(return_value=[])):
            result = main()

        assert result == 1

    def test_shows_summary(self, capsys, monkeypatch):
        """Test shows summary at end."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        with patch("aragora.cli.doctor.check_server", new=AsyncMock(return_value=[])):
            main()

        captured = capsys.readouterr()
        assert "passed" in captured.out
        assert "failed" in captured.out
        assert "optional" in captured.out

    def test_handles_server_check_exception(self, capsys, monkeypatch):
        """Test handles server check exception gracefully."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        # Make asyncio.run raise an exception
        with patch("asyncio.run", side_effect=Exception("Test error")):
            result = main()

        captured = capsys.readouterr()
        assert "skipped" in captured.out
