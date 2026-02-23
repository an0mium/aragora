"""Tests for CLAUDE.md + MEMORY.md system prompt injection in ClaudeCodeHarness.

Covers Gap 2: _build_system_prompt_injection reads CLAUDE.md from the repo root
(truncated to 2000 chars) and MEMORY.md from ~/.claude/projects/*/memory/MEMORY.md
(truncated to 1000 chars), then execute_implementation passes the result via
--append-system-prompt to the Claude Code CLI.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.harnesses.claude_code import ClaudeCodeConfig, ClaudeCodeHarness


@pytest.fixture
def harness() -> ClaudeCodeHarness:
    """Create a default ClaudeCodeHarness with both injections enabled."""
    config = ClaudeCodeConfig(
        inject_claude_md=True,
        inject_memory_md=True,
        use_mcp_tools=False,
        timeout_seconds=30,
    )
    return ClaudeCodeHarness(config=config)


class TestBuildSystemPromptInjection:
    """Tests for _build_system_prompt_injection."""

    def test_system_prompt_includes_claude_md(self, harness: ClaudeCodeHarness, tmp_path: Path):
        """CLAUDE.md content from repo root should appear in the system prompt."""
        claude_md = tmp_path / "CLAUDE.md"
        claude_md.write_text(
            "# Project Rules\nAlways run tests before committing.", encoding="utf-8"
        )

        result = harness._build_system_prompt_injection(tmp_path)

        assert result is not None
        assert "## Project Conventions (CLAUDE.md)" in result
        assert "# Project Rules" in result
        assert "Always run tests before committing." in result

    def test_system_prompt_includes_memory_md(self, harness: ClaudeCodeHarness, tmp_path: Path):
        """MEMORY.md from the project memory directory should appear in the prompt."""
        # Build the expected directory structure under a fake home
        fake_home = tmp_path / "fakehome"
        project_memory_dir = fake_home / ".claude" / "projects" / "my-project" / "memory"
        project_memory_dir.mkdir(parents=True)
        memory_file = project_memory_dir / "MEMORY.md"
        memory_file.write_text(
            "# Memory\nKey pattern: use static error messages.", encoding="utf-8"
        )

        with patch.object(Path, "home", return_value=fake_home):
            result = harness._build_system_prompt_injection(tmp_path)

        assert result is not None
        assert "## Project Memory (MEMORY.md)" in result
        assert "# Memory" in result
        assert "Key pattern: use static error messages." in result

    def test_system_prompt_truncation(self, harness: ClaudeCodeHarness, tmp_path: Path):
        """Large CLAUDE.md should be truncated to 2000 characters."""
        large_content = "A" * 5000
        claude_md = tmp_path / "CLAUDE.md"
        claude_md.write_text(large_content, encoding="utf-8")

        result = harness._build_system_prompt_injection(tmp_path)

        assert result is not None
        # The header is "## Project Conventions (CLAUDE.md)\n" followed by 2000 A's + truncation marker
        assert "A" * 2000 in result
        assert "A" * 2001 not in result
        assert "... (truncated)" in result

    def test_system_prompt_disabled(self, tmp_path: Path):
        """When inject_claude_md=False, CLAUDE.md should not be included."""
        config = ClaudeCodeConfig(
            inject_claude_md=False,
            inject_memory_md=False,
            use_mcp_tools=False,
        )
        harness = ClaudeCodeHarness(config=config)

        claude_md = tmp_path / "CLAUDE.md"
        claude_md.write_text("# Should Not Appear", encoding="utf-8")

        result = harness._build_system_prompt_injection(tmp_path)

        # With both disabled and no custom append_system_prompt, result should be None
        assert result is None

    def test_system_prompt_missing_files_graceful(self, harness: ClaudeCodeHarness, tmp_path: Path):
        """When no CLAUDE.md or MEMORY.md exist, should return None gracefully."""
        # tmp_path exists but contains no CLAUDE.md
        # Also ensure memory dir lookup finds nothing
        fake_home = tmp_path / "emptyhome"
        fake_home.mkdir()

        with patch.object(Path, "home", return_value=fake_home):
            result = harness._build_system_prompt_injection(tmp_path)

        assert result is None


class TestExecuteImplementationSystemPrompt:
    """Tests that execute_implementation passes --append-system-prompt to CLI."""

    @pytest.mark.asyncio
    async def test_execute_implementation_passes_system_prompt(self, tmp_path: Path):
        """execute_implementation should add --append-system-prompt when prompt injection returns content."""
        config = ClaudeCodeConfig(
            inject_claude_md=True,
            inject_memory_md=False,
            use_mcp_tools=False,
            timeout_seconds=30,
        )
        harness = ClaudeCodeHarness(config=config)

        # Create a CLAUDE.md so the injection is non-None
        claude_md = tmp_path / "CLAUDE.md"
        claude_md.write_text("# Convention\nUse type hints everywhere.", encoding="utf-8")

        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"implementation done", b""))
        mock_proc.returncode = 0
        mock_proc.kill = MagicMock()

        with (
            patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec,
            patch("asyncio.wait_for", return_value=(b"implementation done", b"")),
        ):
            stdout, stderr = await harness.execute_implementation(tmp_path, "Add logging")

            assert mock_exec.called
            cmd_args = [str(a) for a in mock_exec.call_args[0]]

            # Verify --append-system-prompt is in the command
            assert "--append-system-prompt" in cmd_args

            # The argument after --append-system-prompt should contain the CLAUDE.md content
            flag_idx = cmd_args.index("--append-system-prompt")
            prompt_value = cmd_args[flag_idx + 1]
            assert "## Project Conventions (CLAUDE.md)" in prompt_value
            assert "Use type hints everywhere." in prompt_value
