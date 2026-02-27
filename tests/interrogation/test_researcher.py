"""Tests for the Unified Researcher Component."""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest
from aragora.interrogation.researcher import ResearchContext, SourceResult, UnifiedResearcher


@pytest.fixture
def mock_km():
    km = MagicMock()
    km.query.return_value = [
        {"content": "Prior debate on rate limiting", "relevance": 0.8, "type": "debate"},
        {"content": "ELO history for claude", "relevance": 0.6, "type": "elo"},
    ]
    return km


@pytest.fixture
def mock_obsidian():
    obs = MagicMock()
    obs.search.return_value = [
        {
            "content": "Notes on API design",
            "relevance": 0.7,
            "title": "api-design",
            "path": "notes/api.md",
        },
    ]
    return obs


class TestSourceResult:
    def test_basic(self):
        sr = SourceResult(source="km", content="test", relevance=0.8)
        assert sr.source == "km"
        assert sr.relevance == 0.8


class TestResearchContext:
    def test_empty(self):
        ctx = ResearchContext(query="test")
        assert not ctx.has_results
        assert ctx.summary() == "No research findings."

    def test_with_results(self):
        ctx = ResearchContext(
            query="test",
            results=[
                SourceResult(source="km", content="knowledge data", relevance=0.9),
                SourceResult(source="web", content="web data", relevance=0.5),
            ],
        )
        assert ctx.has_results
        assert len(ctx.top_results) == 2
        assert ctx.top_results[0].relevance == 0.9

    def test_by_source(self):
        ctx = ResearchContext(
            query="test",
            results=[
                SourceResult(source="km", content="a", relevance=0.8),
                SourceResult(source="web", content="b", relevance=0.5),
                SourceResult(source="km", content="c", relevance=0.6),
            ],
        )
        assert len(ctx.by_source("km")) == 2
        assert len(ctx.by_source("web")) == 1

    def test_summary_truncation(self):
        ctx = ResearchContext(
            query="test",
            results=[SourceResult(source="km", content="x" * 500, relevance=0.5)],
        )
        summary = ctx.summary(max_chars=100)
        assert len(summary) > 0


class TestUnifiedResearcher:
    @pytest.mark.asyncio
    async def test_km_only(self, mock_km):
        researcher = UnifiedResearcher(knowledge_mound=mock_km)
        ctx = await researcher.research("rate limiting", sources=["km"])
        assert ctx.has_results
        assert "km" in ctx.sources_queried
        assert len(ctx.results) == 2

    @pytest.mark.asyncio
    async def test_obsidian_only(self, mock_obsidian):
        researcher = UnifiedResearcher(obsidian_adapter=mock_obsidian)
        ctx = await researcher.research("API design", sources=["obsidian"])
        assert ctx.has_results
        assert len(ctx.by_source("obsidian")) == 1

    @pytest.mark.asyncio
    async def test_multiple_sources(self, mock_km, mock_obsidian):
        researcher = UnifiedResearcher(knowledge_mound=mock_km, obsidian_adapter=mock_obsidian)
        ctx = await researcher.research("design patterns", sources=["km", "obsidian"])
        assert len(ctx.sources_queried) == 2
        assert len(ctx.results) == 3

    @pytest.mark.asyncio
    async def test_missing_source_skipped(self):
        researcher = UnifiedResearcher()
        ctx = await researcher.research("anything", sources=["km"])
        assert not ctx.has_results
        assert "km" in ctx.sources_queried

    @pytest.mark.asyncio
    async def test_source_failure_graceful(self):
        km = MagicMock()
        km.query.side_effect = RuntimeError("connection failed")
        researcher = UnifiedResearcher(knowledge_mound=km)
        ctx = await researcher.research("test", sources=["km"])
        assert "km" in ctx.sources_failed

    @pytest.mark.asyncio
    async def test_max_results(self, mock_km):
        mock_km.query.return_value = [
            {"content": f"result {i}", "relevance": 0.5} for i in range(20)
        ]
        researcher = UnifiedResearcher(knowledge_mound=mock_km)
        ctx = await researcher.research("test", sources=["km"], max_results_per_source=3)
        assert len(ctx.by_source("km")) == 3

    @pytest.mark.asyncio
    async def test_web_source(self):
        web = MagicMock()
        web.search.return_value = [
            {
                "snippet": "Web result",
                "url": "https://example.com",
                "title": "Example",
                "relevance": 0.7,
            }
        ]
        researcher = UnifiedResearcher(web_searcher=web)
        ctx = await researcher.research("test", sources=["web"])
        assert ctx.has_results
        assert ctx.results[0].source == "web"

    @pytest.mark.asyncio
    async def test_codebase_source(self):
        analyzer = MagicMock()
        analyzer.analyze.return_value = [
            {"content": "class Arena", "file": "orchestrator.py", "type": "class", "relevance": 0.9}
        ]
        researcher = UnifiedResearcher(codebase_analyzer=analyzer)
        ctx = await researcher.research("Arena class", sources=["codebase"])
        assert ctx.has_results
        assert ctx.results[0].metadata["file"] == "orchestrator.py"
