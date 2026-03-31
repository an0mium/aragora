"""Tests for EssayRefinementPipeline."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

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
