"""Tests for CodebaseContextProvider - thin wrapper for codebase context in debates."""

from __future__ import annotations

import time

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.debate.codebase_context import (
    CodebaseContextConfig,
    CodebaseContextProvider,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestCodebaseContextConfig:
    def test_defaults(self):
        config = CodebaseContextConfig()
        assert config.codebase_path is None
        assert config.max_context_tokens == 500
        assert config.persist_to_km is False
        assert config.include_tests is False

    def test_custom_values(self):
        config = CodebaseContextConfig(
            codebase_path="/repo",
            max_context_tokens=1000,
            persist_to_km=True,
            include_tests=True,
        )
        assert config.codebase_path == "/repo"
        assert config.max_context_tokens == 1000
        assert config.persist_to_km is True


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class TestCodebaseContextProvider:
    def test_init_default_config(self):
        provider = CodebaseContextProvider()
        assert provider.config.codebase_path is None

    def test_init_custom_config(self):
        config = CodebaseContextConfig(codebase_path=".")
        provider = CodebaseContextProvider(config=config)
        assert provider.config.codebase_path == "."

    @pytest.mark.asyncio
    async def test_build_context_no_path(self):
        provider = CodebaseContextProvider()
        result = await provider.build_context("some task")
        assert result == ""

    @pytest.mark.asyncio
    async def test_build_context_with_mock_builder(self, tmp_path):
        config = CodebaseContextConfig(codebase_path=str(tmp_path))
        provider = CodebaseContextProvider(config=config)

        mock_builder = AsyncMock()
        mock_builder.build_debate_context = AsyncMock(
            return_value="# Codebase (100 files)\naragora/debate/ - 50 files"
        )

        with patch(
            "aragora.nomic.context_builder.NomicContextBuilder",
            return_value=mock_builder,
        ):
            result = await provider.build_context("refactor debate module")
            assert "Codebase" in result or "aragora" in result

    @pytest.mark.asyncio
    async def test_build_context_caching(self, tmp_path):
        config = CodebaseContextConfig(codebase_path=str(tmp_path), cache_ttl_seconds=300)
        provider = CodebaseContextProvider(config=config)

        mock_builder = AsyncMock()
        mock_builder.build_debate_context = AsyncMock(return_value="cached context")

        with patch(
            "aragora.nomic.context_builder.NomicContextBuilder",
            return_value=mock_builder,
        ):
            result1 = await provider.build_context("task 1")
            result2 = await provider.build_context("task 2")

            # Should only be called once (second call uses cache)
            assert mock_builder.build_debate_context.call_count == 1
            assert result1 == result2

    @pytest.mark.asyncio
    async def test_cache_invalidation(self, tmp_path):
        config = CodebaseContextConfig(codebase_path=str(tmp_path))
        provider = CodebaseContextProvider(config=config)

        mock_builder = AsyncMock()
        mock_builder.build_debate_context = AsyncMock(return_value="fresh context")

        with patch(
            "aragora.nomic.context_builder.NomicContextBuilder",
            return_value=mock_builder,
        ):
            await provider.build_context("task")
            provider.invalidate_cache()
            await provider.build_context("task")

            assert mock_builder.build_debate_context.call_count == 2

    @pytest.mark.asyncio
    async def test_build_context_error_fallback(self, tmp_path):
        config = CodebaseContextConfig(codebase_path=str(tmp_path))
        provider = CodebaseContextProvider(config=config)

        with patch(
            "aragora.nomic.context_builder.NomicContextBuilder",
            side_effect=RuntimeError("build failed"),
        ):
            result = await provider.build_context("task")
            assert result == ""

    @pytest.mark.asyncio
    async def test_persist_to_km(self, tmp_path):
        config = CodebaseContextConfig(codebase_path=str(tmp_path), persist_to_km=True)
        provider = CodebaseContextProvider(config=config)

        mock_builder = AsyncMock()
        mock_builder.build_debate_context = AsyncMock(return_value="context")
        mock_adapter = AsyncMock()
        mock_adapter.crawl_and_sync = AsyncMock(return_value=5)

        with (
            patch(
                "aragora.nomic.context_builder.NomicContextBuilder",
                return_value=mock_builder,
            ),
            patch(
                "aragora.knowledge.mound.adapters.codebase_adapter.CodebaseAdapter",
                return_value=mock_adapter,
            ),
        ):
            await provider.build_context("task")
            mock_adapter.crawl_and_sync.assert_called_once()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


class TestGetSummary:
    def test_summary_empty_cache(self):
        provider = CodebaseContextProvider()
        assert provider.get_summary() == ""

    @pytest.mark.asyncio
    async def test_summary_truncation(self, tmp_path):
        config = CodebaseContextConfig(codebase_path=str(tmp_path), max_context_tokens=10)
        provider = CodebaseContextProvider(config=config)

        # Manually set cache
        long_content = "x" * 10000
        from aragora.debate.codebase_context import _CacheEntry

        provider._cache = _CacheEntry(context=long_content)

        summary = provider.get_summary()
        # 10 tokens * 4 chars = 40 chars max + truncation marker
        assert len(summary) < 100
        assert "truncated" in summary

    @pytest.mark.asyncio
    async def test_summary_short_content(self, tmp_path):
        config = CodebaseContextConfig(codebase_path=str(tmp_path), max_context_tokens=500)
        provider = CodebaseContextProvider(config=config)

        from aragora.debate.codebase_context import _CacheEntry

        provider._cache = _CacheEntry(context="short context")

        summary = provider.get_summary()
        assert summary == "short context"

    @pytest.mark.asyncio
    async def test_summary_custom_max_tokens(self, tmp_path):
        config = CodebaseContextConfig(codebase_path=str(tmp_path))
        provider = CodebaseContextProvider(config=config)

        from aragora.debate.codebase_context import _CacheEntry

        provider._cache = _CacheEntry(context="x" * 5000)

        summary = provider.get_summary(max_tokens=50)
        assert len(summary) <= 250  # 50 * 4 + truncation
