"""Tests for memory gateway integration in HybridExecutor."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.implement.executor import HybridExecutor, TASK_PROMPT_TEMPLATE


@pytest.fixture
def executor(tmp_path):
    """Create a HybridExecutor with defaults."""
    with (
        patch("aragora.implement.executor.ClaudeAgent"),
        patch("aragora.implement.executor.CodexAgent"),
    ):
        return HybridExecutor(
            repo_path=tmp_path,
            use_harness=False,
            sandbox_mode=False,
        )


@pytest.fixture
def mock_gateway():
    """Create a mock MemoryGateway."""
    gw = AsyncMock()
    response = MagicMock()
    result1 = MagicMock()
    result1.source = "km"
    result1.confidence = 0.85
    result1.content = "Rate limiting uses token bucket algorithm"
    result2 = MagicMock()
    result2.source = "continuum"
    result2.confidence = 0.72
    result2.content = "Previous fix applied retry logic"
    response.results = [result1, result2]
    gw.query.return_value = response
    return gw


class TestMemoryContextInPrompt:
    """Tests that memory context flows into implementation prompts."""

    def test_prompt_template_has_memory_placeholder(self):
        """TASK_PROMPT_TEMPLATE should contain {memory_context}."""
        assert "{memory_context}" in TASK_PROMPT_TEMPLATE

    def test_build_prompt_with_memory_context(self, executor):
        """Memory context should appear in built prompt."""
        task = MagicMock()
        task.description = "Add rate limiter"
        task.files = ["api.py"]
        task.complexity = "simple"

        prompt = executor._build_prompt(
            task, memory_context="[km, 85%] Rate limiting uses token bucket"
        )

        assert "Rate limiting uses token bucket" in prompt
        assert "## Historical Context" in prompt

    def test_build_prompt_without_memory_context(self, executor):
        """Default placeholder when no memory context provided."""
        task = MagicMock()
        task.description = "Add rate limiter"
        task.files = ["api.py"]
        task.complexity = "simple"

        prompt = executor._build_prompt(task)

        assert "(No historical context available)" in prompt

    @pytest.mark.asyncio
    async def test_fetch_memory_context_with_gateway(self, executor, mock_gateway):
        """Gateway results are formatted correctly."""
        executor._memory_gateway = mock_gateway

        result = await executor._fetch_memory_context("rate limiting")

        assert "[km, 85%]" in result
        assert "Rate limiting uses token bucket" in result
        assert "[continuum, 72%]" in result

    @pytest.mark.asyncio
    async def test_fetch_memory_context_no_gateway(self, executor):
        """Returns placeholder when no gateway configured."""
        executor._memory_gateway = None

        result = await executor._fetch_memory_context("anything")

        assert "No historical context available" in result

    @pytest.mark.asyncio
    async def test_fetch_memory_context_empty_results_falls_through(self, executor, tmp_path):
        """Empty gateway results fall through to CLAUDE.md fallback."""
        gw = AsyncMock()
        response = MagicMock()
        response.results = []
        gw.query.return_value = response
        executor._memory_gateway = gw

        # No CLAUDE.md in tmp_path → falls to final placeholder
        result = await executor._fetch_memory_context("obscure query")

        assert "No historical context available" in result

    @pytest.mark.asyncio
    async def test_fetch_memory_context_timeout(self, executor):
        """Gateway timeout doesn't crash the executor, falls through to fallback."""

        async def slow_query(*args, **kwargs):
            await asyncio.sleep(100)

        gw = MagicMock()
        gw.query = slow_query
        executor._memory_gateway = gw

        result = await executor._fetch_memory_context("query")

        # Falls through to CLAUDE.md fallback or placeholder
        assert result is not None
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_fetch_memory_context_exception_falls_back_to_claude_md(self, executor, tmp_path):
        """Gateway exceptions fall back to CLAUDE.md patterns."""
        gw = AsyncMock()
        gw.query.side_effect = RuntimeError("Connection refused")
        executor._memory_gateway = gw

        # Create a CLAUDE.md with patterns
        claude_md = tmp_path / "CLAUDE.md"
        claude_md.write_text(
            "# Guide\n## Common Patterns\n### Running a Debate\nUse Arena class\n## Other\n"
        )

        result = await executor._fetch_memory_context("query")

        assert "CLAUDE.md patterns" in result
        assert "Running a Debate" in result

    @pytest.mark.asyncio
    async def test_fetch_memory_no_gateway_no_claude_md(self, executor, tmp_path):
        """No gateway and no CLAUDE.md returns placeholder."""
        executor._memory_gateway = None
        # tmp_path has no CLAUDE.md — repo_path is tmp_path

        result = await executor._fetch_memory_context("query")

        assert "No historical context available" in result

    @pytest.mark.asyncio
    async def test_claude_md_fallback_extracts_patterns(self, executor, tmp_path):
        """CLAUDE.md fallback extracts Common Patterns section."""
        claude_md = tmp_path / "CLAUDE.md"
        claude_md.write_text(
            "# Project\n## Quick Ref\nSome table\n## Common Patterns\n"
            "### Memory Tiers\n| Fast | 1 min |\n## Commands\nbash stuff\n"
        )

        result = executor._read_claude_md_fallback()

        assert "Common Patterns" in result
        assert "Memory Tiers" in result
        assert "Commands" not in result
