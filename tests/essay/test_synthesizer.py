"""Tests for EssaySynthesizer — written FIRST (TDD)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.essay.rubric import EssayScore
from aragora.essay.synthesizer import EssaySynthesizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_score(overall: float) -> EssayScore:
    """Return an EssayScore with a preset overall value."""
    score = EssayScore()
    score.overall = overall
    return score


# ---------------------------------------------------------------------------
# 1. test_synthesizer_calls_agent_with_ranked_drafts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesizer_calls_agent_with_ranked_drafts():
    """Agent.generate is called once and the prompt contains draft text."""
    agent = MagicMock()
    agent.generate = AsyncMock(return_value="The synthesised essay text.")

    synthesizer = EssaySynthesizer(agent)

    drafts = ["Draft alpha about climate.", "Draft beta about climate."]
    scores = [_make_score(0.75), _make_score(0.85)]
    critiques = ["Weak transition", "Good flow"]

    result = await synthesizer.synthesize(drafts, scores, critiques)

    # generate must have been called exactly once
    agent.generate.assert_called_once()

    # The prompt passed to generate must contain both draft texts
    prompt_arg = agent.generate.call_args[0][0]
    assert "Draft alpha about climate." in prompt_arg
    assert "Draft beta about climate." in prompt_arg

    # The return value is the agent response text
    assert result == "The synthesised essay text."


# ---------------------------------------------------------------------------
# 2. test_synthesizer_ranks_by_overall_score
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesizer_ranks_by_overall_score():
    """The higher-scoring draft appears before the lower-scoring one in the prompt."""
    agent = MagicMock()
    agent.generate = AsyncMock(return_value="Synthesised.")

    synthesizer = EssaySynthesizer(agent)

    # Intentionally pass lower-score draft first to confirm ranking is applied
    drafts = ["Lower quality draft.", "Higher quality draft."]
    scores = [_make_score(0.60), _make_score(0.92)]
    critiques = []

    await synthesizer.synthesize(drafts, scores, critiques)

    prompt_arg = agent.generate.call_args[0][0]

    pos_higher = prompt_arg.index("Higher quality draft.")
    pos_lower = prompt_arg.index("Lower quality draft.")

    # Higher-scoring draft must appear earlier in the prompt
    assert pos_higher < pos_lower
