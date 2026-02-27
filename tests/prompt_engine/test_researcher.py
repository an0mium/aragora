"""Tests for MultiSourceResearcher fan-out and graceful degradation."""

import pytest
from unittest.mock import AsyncMock, patch
from aragora.prompt_engine.types import (
    RefinedIntent,
    PromptIntent,
    ResearchSource,
)


@pytest.fixture
def sample_refined_intent():
    intent = PromptIntent(
        raw_prompt="add rate limiting",
        intent_type="feature",
        domains=["api", "security"],
        ambiguities=[],
        assumptions=["python backend"],
        scope_estimate="medium",
    )
    return RefinedIntent(intent=intent, answers={"q1": "token-bucket"}, confidence=0.85)


class TestMultiSourceResearcher:
    async def test_fans_out_to_enabled_sources(self, sample_refined_intent):
        from aragora.prompt_engine.researcher import MultiSourceResearcher

        researcher = MultiSourceResearcher()

        with (
            patch.object(researcher, "_research_km", new_callable=AsyncMock) as mock_km,
            patch.object(researcher, "_research_codebase", new_callable=AsyncMock) as mock_code,
        ):
            mock_km.return_value = [{"source": "km", "content": "prior rate limiter"}]
            mock_code.return_value = [{"source": "codebase", "file": "resilience/retry.py"}]

            report = await researcher.research(
                sample_refined_intent,
                sources=[ResearchSource.KNOWLEDGE_MOUND, ResearchSource.CODEBASE],
            )

        assert len(report.km_precedents) == 1
        assert len(report.codebase_context) == 1
        assert ResearchSource.KNOWLEDGE_MOUND in report.sources_used
        assert ResearchSource.CODEBASE in report.sources_used

    async def test_graceful_degradation_on_source_failure(self, sample_refined_intent):
        from aragora.prompt_engine.researcher import MultiSourceResearcher

        researcher = MultiSourceResearcher()

        with (
            patch.object(researcher, "_research_km", new_callable=AsyncMock) as mock_km,
            patch.object(researcher, "_research_codebase", new_callable=AsyncMock) as mock_code,
        ):
            mock_km.side_effect = ConnectionError("KM unavailable")
            mock_code.return_value = [{"source": "codebase", "file": "test.py"}]

            report = await researcher.research(
                sample_refined_intent,
                sources=[ResearchSource.KNOWLEDGE_MOUND, ResearchSource.CODEBASE],
            )

        assert len(report.km_precedents) == 0
        assert len(report.codebase_context) == 1

    async def test_empty_sources_returns_empty_report(self, sample_refined_intent):
        from aragora.prompt_engine.researcher import MultiSourceResearcher

        researcher = MultiSourceResearcher()
        report = await researcher.research(sample_refined_intent, sources=[])
        assert report.km_precedents == []
        assert report.sources_used == []

    async def test_all_sources_fail_returns_empty_report(self, sample_refined_intent):
        from aragora.prompt_engine.researcher import MultiSourceResearcher

        researcher = MultiSourceResearcher()

        with patch.object(researcher, "_research_km", new_callable=AsyncMock) as mock_km:
            mock_km.side_effect = Exception("fail")
            report = await researcher.research(
                sample_refined_intent,
                sources=[ResearchSource.KNOWLEDGE_MOUND],
            )

        assert report.km_precedents == []
        assert report.sources_used == []
