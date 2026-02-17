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
    with patch("aragora.implement.executor.ClaudeAgent"), \
         patch("aragora.implement.executor.CodexAgent"):
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
    async def test_fetch_memory_context_empty_results(self, executor):
        """Returns message when gateway returns no results."""
        gw = AsyncMock()
        response = MagicMock()
        response.results = []
        gw.query.return_value = response
        executor._memory_gateway = gw

        result = await executor._fetch_memory_context("obscure query")

        assert "No relevant historical context found" in result

    @pytest.mark.asyncio
    async def test_fetch_memory_context_timeout(self, executor):
        """Gateway timeout doesn't crash the executor."""

        async def slow_query(*args, **kwargs):
            await asyncio.sleep(100)

        gw = MagicMock()
        gw.query = slow_query
        executor._memory_gateway = gw

        result = await executor._fetch_memory_context("query")

        assert "No historical context available" in result

    @pytest.mark.asyncio
    async def test_fetch_memory_context_exception(self, executor):
        """Gateway exceptions are caught gracefully."""
        gw = AsyncMock()
        gw.query.side_effect = RuntimeError("Connection refused")
        executor._memory_gateway = gw

        result = await executor._fetch_memory_context("query")

        assert "No historical context available" in result
