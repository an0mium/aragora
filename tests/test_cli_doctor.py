"""Tests for CLI doctor command - health checks."""

import sys
from unittest.mock import patch

import pytest

from aragora.cli.doctor import main


class TestDoctorCommand:
    """Tests for the doctor health check command."""

    def test_main_returns_zero_on_success(self, capsys):
        """Doctor returns 0 when all required checks pass."""
        # Mock required packages as installed
        with patch.dict("sys.modules", {"aiohttp": object(), "pydantic": object()}):
            result = main()

        # Should succeed if Python version is compatible
        if sys.version_info >= (3, 10):
            assert result == 0
        captured = capsys.readouterr()
        assert "ARAGORA HEALTH CHECK" in captured.out

    def test_displays_python_version(self, capsys):
        """Doctor displays Python version."""
        with patch.dict("sys.modules", {"aiohttp": object(), "pydantic": object()}):
            main()

        captured = capsys.readouterr()
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
        assert f"Python: {py_ver}" in captured.out

    def test_checks_required_packages(self, capsys):
        """Doctor checks required packages."""
        with patch.dict("sys.modules", {"aiohttp": object(), "pydantic": object()}):
            main()

        captured = capsys.readouterr()
        assert "aiohttp" in captured.out
        assert "pydantic" in captured.out

    def test_checks_optional_packages(self, capsys):
        """Doctor checks optional packages."""
        with patch.dict("sys.modules", {"aiohttp": object(), "pydantic": object()}):
            main()

        captured = capsys.readouterr()
        # Should show optional packages even if not installed
        assert "tiktoken" in captured.out or "not installed" in captured.out

    def test_checks_api_keys(self, capsys):
        """Doctor checks API keys."""
        with patch.dict("sys.modules", {"aiohttp": object(), "pydantic": object()}):
            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
                main()

        captured = capsys.readouterr()
        assert "ANTHROPIC_API_KEY" in captured.out

    def test_shows_api_key_set(self, capsys):
        """Doctor shows 'set' for configured API keys."""
        with patch.dict("sys.modules", {"aiohttp": object(), "pydantic": object()}):
            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"}):
                main()

        captured = capsys.readouterr()
        assert "ANTHROPIC_API_KEY: set" in captured.out

    def test_shows_api_key_not_set(self, capsys):
        """Doctor shows 'not set' for missing API keys."""
        import os

        # Ensure key is not set
        env_copy = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        with patch.dict("sys.modules", {"aiohttp": object(), "pydantic": object()}):
            with patch.dict("os.environ", env_copy, clear=True):
                main()

        captured = capsys.readouterr()
        assert "OPENAI_API_KEY: not set" in captured.out

    def test_handles_missing_required_package(self, capsys):
        """Doctor handles missing required packages gracefully."""
        # The doctor command uses try/except for imports
        # so it won't crash, but will show the package as missing
        with patch.dict("sys.modules", {"pydantic": object()}):
            # Remove aiohttp from modules to simulate missing
            import sys as _sys

            aiohttp_backup = _sys.modules.get("aiohttp")
            if "aiohttp" in _sys.modules:
                del _sys.modules["aiohttp"]
            try:
                # Can't easily test missing packages since they're
                # already imported. Just verify the function runs.
                main()
            finally:
                if aiohttp_backup:
                    _sys.modules["aiohttp"] = aiohttp_backup

        captured = capsys.readouterr()
        assert "ARAGORA HEALTH CHECK" in captured.out

    def test_displays_header(self, capsys):
        """Doctor displays proper header."""
        with patch.dict("sys.modules", {"aiohttp": object(), "pydantic": object()}):
            main()

        captured = capsys.readouterr()
        assert "ARAGORA HEALTH CHECK" in captured.out
        assert "=" * 50 in captured.out

    def test_displays_check_icons(self, capsys):
        """Doctor displays proper check icons."""
        with patch.dict("sys.modules", {"aiohttp": object(), "pydantic": object()}):
            main()

        captured = capsys.readouterr()
        # Should have at least one positive check
        assert "[+]" in captured.out or "[o]" in captured.out


class TestDoctorMain:
    """Tests for main entry point."""

    def test_main_callable(self):
        """main function is callable."""
        assert callable(main)

    def test_main_returns_int(self):
        """main returns an integer."""
        with patch.dict("sys.modules", {"aiohttp": object(), "pydantic": object()}):
            result = main()
        assert isinstance(result, int)
        assert result in (0, 1)
