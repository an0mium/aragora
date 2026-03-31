"""Tests for EssayRefinementPipeline."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.essay.pipeline import EssayRefinementPipeline
from aragora.essay.rubric import EssayScore


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_pipeline_config_defaults() -> None:
    """Verify sensible defaults on a bare pipeline instance."""
    pipe = EssayRefinementPipeline()
    assert pipe.target_words == 1200
    assert pipe.max_rounds == 3
    assert pipe.quality_threshold == 0.8
    assert pipe.models == ["anthropic-api", "openai-api", "gemini"]
    assert pipe.voice_notes == ""
    assert pipe.rubric_path is None


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_dry_run_returns_thesis_and_outline() -> None:
    """dry_run=True should return extraction result without drafting."""
    pipe = EssayRefinementPipeline()

    extraction = {
        "thesis": "AI will reshape education.",
        "outline": "1. Intro\n2. Body\n3. Conclusion",
        "raw_extraction": "full text...",
    }

    with patch.object(pipe, "_extract_ideas", new_callable=AsyncMock, return_value=extraction):
        result = await pipe.run("some raw notes", dry_run=True)

    assert result["thesis"] == "AI will reshape education."
    assert result["outline"] == "1. Intro\n2. Body\n3. Conclusion"
    # dry_run must NOT produce a final essay
    assert "final_essay" not in result


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_full_run_produces_essay_and_score() -> None:
    """Full run should yield final_essay, final_score, and metadata."""
    pipe = EssayRefinementPipeline()

    extraction = {
        "thesis": "Remote work boosts productivity.",
        "outline": "1. Stats\n2. Culture\n3. Conclusion",
        "raw_extraction": "...",
    }

    drafts = ["Draft A text", "Draft B text", "Draft C text"]

    score_good = EssayScore(
        thesis_clarity=0.9,
        argument_coherence=0.85,
        evidence_grounding=0.8,
        rhetorical_force=0.8,
        concision=0.9,
        factual_accuracy=0.85,
        originality=0.8,
    )
    scores = [score_good, score_good, score_good]
    critiques = ["Critique A", "Critique B", "Critique C"]

    final_score = EssayScore(
        thesis_clarity=0.95,
        argument_coherence=0.9,
        evidence_grounding=0.85,
        rhetorical_force=0.85,
        concision=0.9,
        factual_accuracy=0.9,
        originality=0.85,
    )

    with (
        patch.object(pipe, "_extract_ideas", new_callable=AsyncMock, return_value=extraction),
        patch.object(pipe, "_parallel_draft", new_callable=AsyncMock, return_value=drafts),
        patch.object(
            pipe, "_evaluate_drafts", new_callable=AsyncMock, return_value=(scores, critiques)
        ),
        patch.object(
            pipe, "_synthesize", new_callable=AsyncMock, return_value="Synthesized essay text"
        ),
        patch.object(pipe, "_polish", new_callable=AsyncMock, return_value="Polished essay text"),
        patch.object(pipe, "_final_score", new_callable=AsyncMock, return_value=final_score),
    ):
        result = await pipe.run("raw ideas")

    assert result["final_essay"] == "Polished essay text"
    assert result["final_score"] is final_score
    assert result["thesis"] == "Remote work boosts productivity."
    assert result["outline"] == "1. Stats\n2. Culture\n3. Conclusion"
    assert result["rounds_used"] >= 1
    assert isinstance(result["critique_history"], list)


# ---------------------------------------------------------------------------
# End-to-end integration with mocked agents
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_end_to_end_with_mocked_agents():
    with patch("aragora.essay.pipeline.create_agent") as mock_create:
        mock_agent = AsyncMock()
        mock_agent.generate.side_effect = [
            # Extraction
            MagicMock(text="**Thesis:** AI needs testing\n**Outline:**\n1. Problem\n2. Solution"),
            # Draft 1, 2, 3
            MagicMock(text="Draft about AI testing. " * 80),
            MagicMock(text="Alternative AI draft. " * 80),
            MagicMock(text="Third perspective. " * 80),
            # Evaluation scores (3 JSON responses)
            MagicMock(
                text='{"thesis_clarity":0.8,"argument_coherence":0.7,"evidence_grounding":0.6,"rhetorical_force":0.7,"concision":0.8,"factual_accuracy":0.9,"originality":0.5,"severity_notes":["weak"],"suggestions":["add data"]}'
            ),
            MagicMock(
                text='{"thesis_clarity":0.7,"argument_coherence":0.8,"evidence_grounding":0.7,"rhetorical_force":0.6,"concision":0.7,"factual_accuracy":0.8,"originality":0.6}'
            ),
            MagicMock(
                text='{"thesis_clarity":0.6,"argument_coherence":0.6,"evidence_grounding":0.8,"rhetorical_force":0.5,"concision":0.6,"factual_accuracy":0.7,"originality":0.7}'
            ),
            # Synthesis
            MagicMock(text="Synthesized essay. " * 80),
            # Polish
            MagicMock(text="Polished final. " * 80),
            # Final score
            MagicMock(
                text='{"thesis_clarity":0.85,"argument_coherence":0.82,"evidence_grounding":0.78,"rhetorical_force":0.80,"concision":0.85,"factual_accuracy":0.90,"originality":0.70}'
            ),
        ]
        mock_create.return_value = mock_agent

        # Rewrap side_effect so each call returns the .text string value directly.
        # The pipeline passes the generate() return value to re.search and JSON
        # parsers, which require plain strings rather than MagicMock objects.
        _raw_responses = mock_agent.generate.side_effect
        mock_agent.generate.side_effect = [r.text for r in _raw_responses]

        pipeline = EssayRefinementPipeline(
            models=["anthropic-api", "openai-api", "gemini"],
            target_words=1000,
            max_rounds=1,
        )
        result = await pipeline.run("Raw ideas about AI testing")

        assert "final_essay" in result
        assert "final_score" in result
        assert result["final_score"].overall > 0
        assert result["thesis"]
        assert result["rounds_used"] >= 1
