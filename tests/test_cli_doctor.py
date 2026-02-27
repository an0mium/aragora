"""Tests for CLI doctor command - health checks."""

import sys
from unittest.mock import patch

import pytest

from aragora.cli.doctor import main


class TestDoctorCommand:
    """Tests for the doctor health check command."""

    def test_main_returns_zero_on_success(self, capsys):
        """Doctor returns 0 when all required checks pass."""
        result = main()

        # Should succeed if Python version is compatible
        if sys.version_info >= (3, 10):
            assert result == 0
        captured = capsys.readouterr()
        assert "ARAGORA HEALTH CHECK" in captured.out

    def test_displays_python_version(self, capsys):
        """Doctor displays Python version."""
        main()

        captured = capsys.readouterr()
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
        assert f"Python: {py_ver}" in captured.out

    def test_checks_required_packages(self, capsys):
        """Doctor checks required packages."""
        main()

        captured = capsys.readouterr()
        assert "aiohttp" in captured.out
        assert "pydantic" in captured.out

    def test_checks_optional_packages(self, capsys):
        """Doctor checks optional packages."""
        main()

        captured = capsys.readouterr()
        # Should mention optional packages in summary or by name
        assert "optional" in captured.out or "tiktoken" in captured.out

    def test_checks_api_keys(self, capsys):
        """Doctor checks API keys."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            main()

        captured = capsys.readouterr()
        assert "ANTHROPIC_API_KEY" in captured.out

    def test_shows_api_key_configured(self, capsys):
        """Doctor shows 'configured' for set API keys."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"}):
            main()

        captured = capsys.readouterr()
        assert "ANTHROPIC_API_KEY: configured" in captured.out

    def test_shows_api_key_not_set(self, capsys):
        """Doctor shows 'not set' for missing API keys."""
        import os

        # Ensure key is not set
        env_copy = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        with (
            patch.dict("os.environ", env_copy, clear=True),
            patch("aragora.config.secrets.get_secret", return_value=None),
        ):
            main()

        captured = capsys.readouterr()
        assert "OPENAI_API_KEY: not set" in captured.out

    def test_handles_missing_required_package(self, capsys):
        """Doctor handles missing required packages gracefully."""
        # Just verify the function runs without crashing
        main()

        captured = capsys.readouterr()
        assert "ARAGORA HEALTH CHECK" in captured.out

    def test_displays_header(self, capsys):
        """Doctor displays proper header."""
        main()

        captured = capsys.readouterr()
        assert "ARAGORA HEALTH CHECK" in captured.out
        assert "=" * 50 in captured.out

    def test_displays_check_icons(self, capsys):
        """Doctor displays proper check icons."""
        main()

        captured = capsys.readouterr()
        # Should have at least one positive check (✓ unicode or [+]/[o] legacy)
        assert (
            "✓" in captured.out
            or "[+]" in captured.out
            or "[o]" in captured.out
            or "passed" in captured.out
        )


class TestDoctorMain:
    """Tests for main entry point."""

    def test_main_callable(self):
        """main function is callable."""
        assert callable(main)

    def test_main_returns_int(self):
        """main returns an integer."""
        result = main()
        assert isinstance(result, int)
        assert result in (0, 1)
