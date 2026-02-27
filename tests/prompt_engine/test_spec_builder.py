"""Tests for SpecBuilder: RefinedIntent + ResearchReport -> SwarmSpec."""

import pytest
from unittest.mock import AsyncMock, patch
from aragora.prompt_engine.types import (
    RefinedIntent,
    PromptIntent,
    ResearchReport,
    ResearchSource,
    UserProfile,
)


@pytest.fixture
def sample_refined_intent():
    intent = PromptIntent(
        raw_prompt="add rate limiting to API",
        intent_type="feature",
        domains=["api", "security"],
        ambiguities=[],
        assumptions=["python backend"],
        scope_estimate="medium",
    )
    return RefinedIntent(intent=intent, answers={"q1": "token-bucket"}, confidence=0.85)


@pytest.fixture
def sample_research():
    return ResearchReport(
        km_precedents=[{"source": "km", "content": "existing retry logic"}],
        codebase_context=[{"file": "aragora/resilience/retry.py", "summary": "Retry with backoff"}],
        obsidian_notes=[],
        web_results=[],
        sources_used=[ResearchSource.KNOWLEDGE_MOUND, ResearchSource.CODEBASE],
    )


class TestSpecBuilder:
    async def test_builds_swarm_spec(self, sample_refined_intent, sample_research):
        from aragora.prompt_engine.spec_builder import SpecBuilder

        builder = SpecBuilder()

        with patch.object(builder, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = {
                "refined_goal": "Implement token-bucket rate limiting for all API endpoints",
                "acceptance_criteria": [
                    "Rate limits enforced per-user",
                    "429 responses on exceed",
                ],
                "constraints": ["No external dependencies"],
                "track_hints": ["developer", "security"],
                "estimated_complexity": "medium",
            }
            spec = await builder.build(
                sample_refined_intent,
                sample_research,
                UserProfile.CTO,
            )

        from aragora.swarm.spec import SwarmSpec

        assert isinstance(spec, SwarmSpec)
        assert "rate limiting" in spec.refined_goal.lower()
        assert len(spec.acceptance_criteria) == 2
        assert spec.estimated_complexity == "medium"

    async def test_sets_raw_goal_from_intent(self, sample_refined_intent, sample_research):
        from aragora.prompt_engine.spec_builder import SpecBuilder

        builder = SpecBuilder()

        with patch.object(builder, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = {
                "refined_goal": "Add rate limiting",
                "acceptance_criteria": [],
                "constraints": [],
                "track_hints": [],
                "estimated_complexity": "small",
            }
            spec = await builder.build(sample_refined_intent, sample_research, UserProfile.FOUNDER)

        assert spec.raw_goal == "add rate limiting to API"

    async def test_applies_profile_constraints(self, sample_refined_intent, sample_research):
        from aragora.prompt_engine.spec_builder import SpecBuilder

        builder = SpecBuilder()

        with patch.object(builder, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = {
                "refined_goal": "Add rate limiting",
                "acceptance_criteria": [],
                "constraints": [],
                "track_hints": [],
                "estimated_complexity": "small",
            }
            spec = await builder.build(sample_refined_intent, sample_research, UserProfile.TEAM)

        assert spec.requires_approval is True
