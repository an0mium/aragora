"""Tests for InterrogationResearcher - multi-source context gathering."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from aragora.interrogation.researcher import (
    InterrogationResearcher,
    ResearchResult,
    ResearchSource,
)
from aragora.interrogation.decomposer import Dimension


class TestInterrogationResearcher:
    @pytest.fixture
    def researcher(self):
        return InterrogationResearcher()

    @pytest.fixture
    def sample_dimensions(self):
        return [
            Dimension(
                name="performance",
                description="Speed improvements",
                vagueness_score=0.6,
                keywords=["fast"],
            ),
            Dimension(
                name="quality",
                description="Test coverage",
                vagueness_score=0.4,
                keywords=["test"],
            ),
        ]

    @pytest.mark.asyncio
    async def test_research_returns_results_per_dimension(
        self, researcher, sample_dimensions
    ):
        result = await researcher.research(sample_dimensions, sources=[])
        assert isinstance(result, ResearchResult)
        assert len(result.findings) == len(sample_dimensions)

    @pytest.mark.asyncio
    async def test_research_with_knowledge_mound(self, sample_dimensions):
        mock_km = AsyncMock()
        mock_km.query.return_value = MagicMock(
            items=[
                MagicMock(
                    content="Prior debate about performance",
                    metadata={"debate_id": "d1"},
                )
            ]
        )
        researcher = InterrogationResearcher(knowledge_mound=mock_km)
        result = await researcher.research(
            sample_dimensions, sources=["knowledge_mound"]
        )
        assert any(
            f.source == ResearchSource.KNOWLEDGE_MOUND
            for findings in result.findings.values()
            for f in findings
        )

    @pytest.mark.asyncio
    async def test_research_with_obsidian(self, sample_dimensions):
        mock_obsidian = AsyncMock()
        mock_obsidian.search.return_value = [
            MagicMock(content="My notes on testing", metadata={})
        ]
        researcher = InterrogationResearcher(obsidian=mock_obsidian)
        result = await researcher.research(sample_dimensions, sources=["obsidian"])
        assert any(
            f.source == ResearchSource.OBSIDIAN
            for findings in result.findings.values()
            for f in findings
        )

    @pytest.mark.asyncio
    async def test_research_empty_dimensions(self, researcher):
        result = await researcher.research([], sources=[])
        assert len(result.findings) == 0

    @pytest.mark.asyncio
    async def test_research_graceful_on_source_failure(self, sample_dimensions):
        mock_km = AsyncMock()
        mock_km.query.side_effect = RuntimeError("Connection failed")
        researcher = InterrogationResearcher(knowledge_mound=mock_km)
        result = await researcher.research(
            sample_dimensions, sources=["knowledge_mound"]
        )
        assert isinstance(result, ResearchResult)

    @pytest.mark.asyncio
    async def test_total_findings_property(self, sample_dimensions):
        mock_km = AsyncMock()
        mock_km.query.return_value = MagicMock(
            items=[
                MagicMock(content="Finding 1", metadata={}),
                MagicMock(content="Finding 2", metadata={}),
            ]
        )
        researcher = InterrogationResearcher(knowledge_mound=mock_km)
        result = await researcher.research(
            sample_dimensions, sources=["knowledge_mound"]
        )
        assert result.total_findings > 0

    @pytest.mark.asyncio
    async def test_for_dimension_accessor(self, sample_dimensions):
        mock_km = AsyncMock()
        mock_km.query.return_value = MagicMock(
            items=[MagicMock(content="Perf data", metadata={})]
        )
        researcher = InterrogationResearcher(knowledge_mound=mock_km)
        result = await researcher.research(
            sample_dimensions, sources=["knowledge_mound"]
        )
        perf_findings = result.for_dimension("performance")
        assert len(perf_findings) >= 1
        missing = result.for_dimension("nonexistent")
        assert missing == []
