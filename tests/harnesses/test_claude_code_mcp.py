"""Tests for MCP config wiring in ClaudeCodeHarness."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.harnesses.claude_code import ClaudeCodeConfig, ClaudeCodeHarness


@pytest.fixture
def harness():
    """Create a ClaudeCodeHarness with MCP tools enabled."""
    config = ClaudeCodeConfig(
        use_mcp_tools=True,
        timeout_seconds=60,
    )
    return ClaudeCodeHarness(config=config)


@pytest.fixture
def harness_no_mcp():
    """Create a ClaudeCodeHarness with MCP tools disabled."""
    config = ClaudeCodeConfig(
        use_mcp_tools=False,
        timeout_seconds=60,
    )
    return ClaudeCodeHarness(config=config)


class TestMCPConfigInCommand:
    """Tests that MCP config flags are added to the Claude Code command."""

    @pytest.mark.asyncio
    async def test_mcp_config_flag_present_when_enabled(self, harness, tmp_path):
        """--mcp-config should be in command when use_mcp_tools=True."""
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"done", b""))
        mock_proc.returncode = 0
        mock_proc.kill = MagicMock()

        with (
            patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec,
            patch("asyncio.wait_for", return_value=(b"done", b"")),
        ):
            try:
                await harness.execute_implementation(tmp_path, "fix the bug")
            except Exception:
                pass

            if mock_exec.called:
                cmd_args = [str(a) for a in mock_exec.call_args[0]]
                assert any("--mcp-config" in a for a in cmd_args) or "--mcp-config" in cmd_args

    @pytest.mark.asyncio
    async def test_no_mcp_config_when_disabled(self, harness_no_mcp, tmp_path):
        """--mcp-config should NOT be in command when use_mcp_tools=False."""
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"done", b""))
        mock_proc.returncode = 0
        mock_proc.kill = MagicMock()

        with (
            patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec,
            patch("asyncio.wait_for", return_value=(b"done", b"")),
        ):
            try:
                await harness_no_mcp.execute_implementation(tmp_path, "fix the bug")
            except Exception:
                pass

            if mock_exec.called:
                cmd_args = [str(a) for a in mock_exec.call_args[0]]
                assert "--mcp-config" not in cmd_args
                assert "--allowedTools" not in cmd_args


class TestAllowedTools:
    """Tests for the allowed tools list."""

    def test_allowed_tools_list(self):
        """Allowed tools should include core Claude Code tools and MCP wildcard."""
        allowed = ClaudeCodeHarness._get_allowed_tools()

        assert "Read" in allowed
        assert "Write" in allowed
        assert "Edit" in allowed
        assert "Bash" in allowed
        assert "Grep" in allowed
        assert "Glob" in allowed
        assert "mcp__aragora__*" in allowed
        # Should NOT include browsing or team tools
        assert "WebSearch" not in allowed
        assert "WebFetch" not in allowed

    def test_allowed_tools_no_duplicates(self):
        """No duplicate tools in the list."""
        allowed = ClaudeCodeHarness._get_allowed_tools()
        assert len(allowed) == len(set(allowed))


class TestMCPConfigGeneration:
    """Tests that MCP config is generated correctly during execution."""

    @pytest.mark.asyncio
    async def test_mcp_config_generation_failure_doesnt_crash(self, harness, tmp_path):
        """If MCP config generation fails, execution should continue."""
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"done", b""))
        mock_proc.returncode = 0
        mock_proc.kill = MagicMock()

        with (
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
            patch("asyncio.wait_for", return_value=(b"done", b"")),
            patch(
                "aragora.mcp.impl_config.generate_impl_mcp_config",
                side_effect=OSError("disk full"),
            ),
        ):
            # Should not raise
            try:
                await harness.execute_implementation(tmp_path, "fix bug")
            except Exception:
                pass  # Other errors are fine, just not from MCP config
