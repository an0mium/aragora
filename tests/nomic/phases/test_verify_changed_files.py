"""Tests for VerifyPhase._check_syntax using changed files."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.nomic.phases.verify import VerifyPhase


@pytest.fixture
def verify_phase(tmp_path):
    """Create a VerifyPhase with mocked dependencies."""
    phase = VerifyPhase.__new__(VerifyPhase)
    phase.aragora_path = tmp_path
    phase._log = MagicMock()
    phase._stream_emit = MagicMock()
    phase.nomic_integration = MagicMock()
    phase.cycle_count = 1
    return phase


class TestSyntaxCheckChangedFiles:
    """Tests that syntax check uses changed files from git diff."""

    @pytest.mark.asyncio
    async def test_syntax_check_uses_changed_py_files(self, verify_phase, tmp_path):
        """Syntax check should check changed .py files instead of just __init__.py."""
        # Create test files
        (tmp_path / "foo.py").write_text("x = 1\n")
        (tmp_path / "bar.py").write_text("y = 2\n")

        verify_phase._get_changed_files = AsyncMock(return_value=["foo.py", "bar.py"])

        result = await verify_phase._check_syntax()

        assert result["passed"] is True
        assert result["files_checked"] == 2

    @pytest.mark.asyncio
    async def test_syntax_check_fallback_when_no_changes(self, verify_phase, tmp_path):
        """Falls back to __init__.py when no changed files detected."""
        (tmp_path / "aragora").mkdir(exist_ok=True)
        (tmp_path / "aragora" / "__init__.py").write_text("# init\n")

        verify_phase._get_changed_files = AsyncMock(return_value=[])

        result = await verify_phase._check_syntax()

        assert result["files_checked"] == 1

    @pytest.mark.asyncio
    async def test_non_python_files_filtered_out(self, verify_phase, tmp_path):
        """Non-.py files should be filtered out of syntax checking."""
        (tmp_path / "readme.md").write_text("# Hi\n")
        (tmp_path / "valid.py").write_text("x = 1\n")
        (tmp_path / "aragora").mkdir(exist_ok=True)
        (tmp_path / "aragora" / "__init__.py").write_text("# init\n")

        verify_phase._get_changed_files = AsyncMock(
            return_value=["readme.md", "valid.py", "config.json"]
        )

        result = await verify_phase._check_syntax()

        assert result["passed"] is True
        assert result["files_checked"] == 1  # Only valid.py

    @pytest.mark.asyncio
    async def test_syntax_error_detected_in_changed_file(self, verify_phase, tmp_path):
        """Syntax errors in changed files should be caught."""
        (tmp_path / "broken.py").write_text("def foo(\n")  # SyntaxError

        verify_phase._get_changed_files = AsyncMock(return_value=["broken.py"])

        result = await verify_phase._check_syntax()

        assert result["passed"] is False
        assert "broken.py" in result["output"]

    @pytest.mark.asyncio
    async def test_mixed_valid_invalid_files(self, verify_phase, tmp_path):
        """One bad file among good ones should fail overall."""
        (tmp_path / "good.py").write_text("x = 1\n")
        (tmp_path / "bad.py").write_text("def (\n")  # SyntaxError

        verify_phase._get_changed_files = AsyncMock(return_value=["good.py", "bad.py"])

        result = await verify_phase._check_syntax()

        assert result["passed"] is False
        assert result["files_checked"] == 2
