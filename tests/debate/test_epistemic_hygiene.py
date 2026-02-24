"""Tests for epistemic hygiene enforcement in debates.

Tests the four dimensions:
1. Alternatives considered
2. Falsifiability statements
3. Confidence intervals
4. Explicit unknowns

Plus scoring, penalty calculation, tracker, prompt injection, and
consensus integration.
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeProtocol:
    """Minimal protocol stand-in for tests."""

    enable_epistemic_hygiene: bool = True
    epistemic_hygiene_penalty: float = 0.15
    epistemic_min_alternatives: int = 1
    epistemic_require_falsifiers: bool = True
    epistemic_require_confidence: bool = True
    epistemic_require_unknowns: bool = True


COMPLIANT_RESPONSE = """
## My Proposal
I recommend using Redis for the caching layer.

### ALTERNATIVES CONSIDERED
- **Alternative:** Memcached | **Rejected because:** No persistence support

### FALSIFIABILITY
- **Claim:** Redis handles 100k ops/sec | **Falsified if:** Benchmark shows < 50k ops/sec under our workload

### CONFIDENCE LEVELS
- **Claim:** Redis handles 100k ops/sec | **Confidence:** 0.85
- **Claim:** TTL-based eviction sufficient | **Confidence:** 0.7

### EXPLICIT UNKNOWNS
- We do not know the peak concurrent connection count in production
- Uncertain about memory fragmentation under sustained load
"""

PARTIAL_RESPONSE = """
## My Proposal
I recommend using Redis for the caching layer.

### ALTERNATIVES CONSIDERED
- **Alternative:** Memcached | **Rejected because:** No persistence support

We are confident this will work.
"""

BARE_RESPONSE = """
I think we should use Redis for caching. It's fast and reliable.
It supports persistence and pub/sub. Let's go with it.
"""


# ---------------------------------------------------------------------------
# score_response tests
# ---------------------------------------------------------------------------


class TestScoreResponse:
    """Tests for scoring agent responses on epistemic dimensions."""

    def test_fully_compliant_response(self):
        from aragora.debate.epistemic_hygiene import score_response

        score = score_response(COMPLIANT_RESPONSE, agent="claude", round_number=1)
        assert score.has_alternatives is True
        assert score.has_falsifiers is True
        assert score.has_confidence is True
        assert score.has_unknowns is True
        assert score.score == 1.0
        assert score.missing == []

    def test_partial_response(self):
        from aragora.debate.epistemic_hygiene import score_response

        score = score_response(PARTIAL_RESPONSE, agent="gpt4", round_number=2)
        assert score.has_alternatives is True
        # "confident" should not match falsifier patterns but may match confidence
        # The key thing: falsifiers and unknowns are missing
        assert score.has_falsifiers is False
        assert score.has_unknowns is False
        assert score.score < 1.0

    def test_bare_response_scores_low(self):
        from aragora.debate.epistemic_hygiene import score_response

        score = score_response(BARE_RESPONSE, agent="gemini", round_number=1)
        assert score.has_alternatives is False
        assert score.has_falsifiers is False
        assert score.has_unknowns is False
        assert score.score < 0.5

    def test_empty_response(self):
        from aragora.debate.epistemic_hygiene import score_response

        score = score_response("", agent="test")
        assert score.score == 0.0
        assert len(score.missing) == 4

    def test_agent_and_round_tracking(self):
        from aragora.debate.epistemic_hygiene import score_response

        score = score_response("Some text", agent="claude", round_number=3)
        assert score.agent == "claude"
        assert score.round_number == 3

    def test_to_dict(self):
        from aragora.debate.epistemic_hygiene import score_response

        score = score_response(COMPLIANT_RESPONSE, agent="claude", round_number=1)
        d = score.to_dict()
        assert d["agent"] == "claude"
        assert d["round_number"] == 1
        assert d["score"] == 1.0
        assert isinstance(d["missing"], list)

    def test_natural_language_alternatives(self):
        """Detect alternatives expressed naturally, not in strict format."""
        from aragora.debate.epistemic_hygiene import score_response

        text = "We could also use PostgreSQL, but another option is to combine both."
        score = score_response(text)
        assert score.has_alternatives is True

    def test_natural_language_falsifiers(self):
        """Detect falsifiability expressed naturally."""
        from aragora.debate.epistemic_hygiene import score_response

        text = "This claim would be wrong if latency exceeds 500ms."
        score = score_response(text)
        assert score.has_falsifiers is True

    def test_natural_language_confidence(self):
        """Detect confidence expressed as high/medium/low."""
        from aragora.debate.epistemic_hygiene import score_response

        text = "I have high confidence that this approach will work."
        score = score_response(text)
        assert score.has_confidence is True

    def test_natural_language_unknowns(self):
        """Detect unknowns expressed naturally."""
        from aragora.debate.epistemic_hygiene import score_response

        text = "It remains unclear how the system behaves under extreme load. There are limitations of this approach."
        score = score_response(text)
        assert score.has_unknowns is True

    def test_numeric_confidence_detection(self):
        """Detect bare numeric confidence values."""
        from aragora.debate.epistemic_hygiene import score_response

        text = "Confidence: 0.82"
        score = score_response(text)
        assert score.has_confidence is True

    def test_refuted_if_pattern(self):
        """Detect 'refuted if' pattern for falsifiers."""
        from aragora.debate.epistemic_hygiene import score_response

        text = "This would be refuted if we observe more than 10% error rate."
        score = score_response(text)
        assert score.has_falsifiers is True


# ---------------------------------------------------------------------------
# compute_epistemic_penalty tests
# ---------------------------------------------------------------------------


class TestComputeEpistemicPenalty:
    """Tests for consensus penalty calculation."""

    def test_no_penalty_when_disabled(self):
        from aragora.debate.epistemic_hygiene import compute_epistemic_penalty, EpistemicScore

        protocol = _FakeProtocol(enable_epistemic_hygiene=False)
        score = EpistemicScore(has_alternatives=False, has_falsifiers=False)
        assert compute_epistemic_penalty(score, protocol) == 0.0

    def test_no_penalty_for_compliant(self):
        from aragora.debate.epistemic_hygiene import compute_epistemic_penalty, EpistemicScore

        protocol = _FakeProtocol()
        score = EpistemicScore(
            has_alternatives=True,
            has_falsifiers=True,
            has_confidence=True,
            has_unknowns=True,
        )
        assert compute_epistemic_penalty(score, protocol) == 0.0

    def test_full_penalty_for_noncompliant(self):
        from aragora.debate.epistemic_hygiene import compute_epistemic_penalty, EpistemicScore

        protocol = _FakeProtocol(epistemic_hygiene_penalty=0.20)
        score = EpistemicScore(
            has_alternatives=False,
            has_falsifiers=False,
            has_confidence=False,
            has_unknowns=False,
        )
        penalty = compute_epistemic_penalty(score, protocol)
        assert penalty == pytest.approx(0.20)

    def test_partial_penalty(self):
        from aragora.debate.epistemic_hygiene import compute_epistemic_penalty, EpistemicScore

        protocol = _FakeProtocol(epistemic_hygiene_penalty=0.20)
        # Missing 2 of 4 required elements
        score = EpistemicScore(
            has_alternatives=True,
            has_falsifiers=False,
            has_confidence=True,
            has_unknowns=False,
        )
        penalty = compute_epistemic_penalty(score, protocol)
        assert penalty == pytest.approx(0.10)  # 0.20 * (2/4)

    def test_selective_requirements(self):
        """Only penalize dimensions that are required."""
        from aragora.debate.epistemic_hygiene import compute_epistemic_penalty, EpistemicScore

        protocol = _FakeProtocol(
            epistemic_require_falsifiers=False,
            epistemic_require_unknowns=False,
        )
        # Only alternatives + confidence required; both missing
        score = EpistemicScore(
            has_alternatives=False,
            has_falsifiers=False,  # Not required
            has_confidence=False,
            has_unknowns=False,  # Not required
        )
        penalty = compute_epistemic_penalty(score, protocol)
        # 2 required, 2 missing => full penalty
        assert penalty == pytest.approx(0.15)

    def test_no_requirements_no_penalty(self):
        """If nothing is required, no penalty even with low score."""
        from aragora.debate.epistemic_hygiene import compute_epistemic_penalty, EpistemicScore

        protocol = _FakeProtocol(
            epistemic_min_alternatives=0,
            epistemic_require_falsifiers=False,
            epistemic_require_confidence=False,
            epistemic_require_unknowns=False,
        )
        score = EpistemicScore()
        assert compute_epistemic_penalty(score, protocol) == 0.0


# ---------------------------------------------------------------------------
# EpistemicHygieneTracker tests
# ---------------------------------------------------------------------------


class TestEpistemicHygieneTracker:
    """Tests for the per-debate score tracker."""

    def test_record_and_retrieve(self):
        from aragora.debate.epistemic_hygiene import EpistemicHygieneTracker, EpistemicScore

        tracker = EpistemicHygieneTracker(debate_id="test-1")

        s1 = EpistemicScore(
            has_alternatives=True,
            has_falsifiers=True,
            has_confidence=True,
            has_unknowns=True,
            agent="claude",
            round_number=1,
        )
        s2 = EpistemicScore(
            has_alternatives=True,
            has_falsifiers=False,
            has_confidence=True,
            has_unknowns=False,
            agent="gpt4",
            round_number=1,
        )
        tracker.record(s1)
        tracker.record(s2)

        assert len(tracker.scores) == 2
        assert tracker.get_agent_average("claude") == 1.0
        assert tracker.get_agent_average("gpt4") == 0.5
        assert tracker.get_debate_average() == 0.75

    def test_get_round_scores(self):
        from aragora.debate.epistemic_hygiene import EpistemicHygieneTracker, EpistemicScore

        tracker = EpistemicHygieneTracker(debate_id="test-2")

        for r in [1, 1, 2, 2]:
            tracker.record(EpistemicScore(agent="a", round_number=r, has_alternatives=True))

        assert len(tracker.get_round_scores(1)) == 2
        assert len(tracker.get_round_scores(2)) == 2

    def test_summary(self):
        from aragora.debate.epistemic_hygiene import EpistemicHygieneTracker, EpistemicScore

        tracker = EpistemicHygieneTracker(debate_id="test-3")
        tracker.record(
            EpistemicScore(
                has_alternatives=True,
                has_falsifiers=True,
                has_confidence=True,
                has_unknowns=True,
                agent="claude",
                round_number=1,
            )
        )
        summary = tracker.summary()
        assert summary["debate_id"] == "test-3"
        assert summary["total_scores"] == 1
        assert summary["debate_average"] == 1.0
        assert "claude" in summary["agents"]
        assert summary["agents"]["claude"]["fully_compliant"] == 1

    def test_empty_tracker(self):
        from aragora.debate.epistemic_hygiene import EpistemicHygieneTracker

        tracker = EpistemicHygieneTracker()
        assert tracker.get_debate_average() == 0.0
        assert tracker.get_agent_average("nobody") == 0.0
        summary = tracker.summary()
        assert summary["total_scores"] == 0


# ---------------------------------------------------------------------------
# Prompt injection tests
# ---------------------------------------------------------------------------


class TestPromptInjection:
    """Tests for prompt text generation."""

    def test_proposal_prompt_enabled(self):
        from aragora.debate.epistemic_hygiene import get_epistemic_proposal_prompt

        protocol = _FakeProtocol()
        prompt = get_epistemic_proposal_prompt(protocol)
        assert "EPISTEMIC HYGIENE" in prompt
        assert "ALTERNATIVES CONSIDERED" in prompt
        assert "FALSIFIABILITY" in prompt
        assert "CONFIDENCE LEVELS" in prompt
        assert "EXPLICIT UNKNOWNS" in prompt

    def test_proposal_prompt_disabled(self):
        from aragora.debate.epistemic_hygiene import get_epistemic_proposal_prompt

        protocol = _FakeProtocol(enable_epistemic_hygiene=False)
        prompt = get_epistemic_proposal_prompt(protocol)
        assert prompt == ""

    def test_revision_prompt_enabled(self):
        from aragora.debate.epistemic_hygiene import get_epistemic_revision_prompt

        protocol = _FakeProtocol()
        prompt = get_epistemic_revision_prompt(protocol)
        assert "EPISTEMIC HYGIENE" in prompt
        assert "Revision" in prompt

    def test_revision_prompt_disabled(self):
        from aragora.debate.epistemic_hygiene import get_epistemic_revision_prompt

        protocol = _FakeProtocol(enable_epistemic_hygiene=False)
        prompt = get_epistemic_revision_prompt(protocol)
        assert prompt == ""

    def test_min_alternatives_in_prompt(self):
        from aragora.debate.epistemic_hygiene import get_epistemic_proposal_prompt

        protocol = _FakeProtocol(epistemic_min_alternatives=3)
        prompt = get_epistemic_proposal_prompt(protocol)
        assert "3 alternative" in prompt


# ---------------------------------------------------------------------------
# Protocol flag tests
# ---------------------------------------------------------------------------


class TestProtocolFlag:
    """Tests for DebateProtocol integration."""

    def test_default_disabled(self):
        from aragora.debate.protocol import DebateProtocol

        p = DebateProtocol()
        assert p.enable_epistemic_hygiene is False
        assert p.epistemic_hygiene_penalty == 0.15

    def test_enable_via_constructor(self):
        from aragora.debate.protocol import DebateProtocol

        p = DebateProtocol(enable_epistemic_hygiene=True)
        assert p.enable_epistemic_hygiene is True
        assert p.epistemic_min_alternatives == 1
        assert p.epistemic_require_falsifiers is True
        assert p.epistemic_require_confidence is True
        assert p.epistemic_require_unknowns is True

    def test_with_epistemic_hygiene_classmethod(self):
        from aragora.debate.protocol import DebateProtocol

        p = DebateProtocol.with_epistemic_hygiene(penalty=0.25, min_alternatives=2, rounds=5)
        assert p.enable_epistemic_hygiene is True
        assert p.epistemic_hygiene_penalty == 0.25
        assert p.epistemic_min_alternatives == 2
        assert p.rounds == 5

    def test_with_epistemic_hygiene_defaults(self):
        from aragora.debate.protocol import DebateProtocol

        p = DebateProtocol.with_epistemic_hygiene()
        assert p.enable_epistemic_hygiene is True
        assert p.epistemic_hygiene_penalty == 0.15
        assert p.epistemic_require_falsifiers is True
        assert p.epistemic_require_confidence is True
        assert p.epistemic_require_unknowns is True


# ---------------------------------------------------------------------------
# Preset tests
# ---------------------------------------------------------------------------


class TestPreset:
    """Tests for the 'epistemic' preset."""

    def test_epistemic_preset_exists(self):
        from aragora.debate.presets import get_preset, list_presets

        presets = list_presets()
        assert "epistemic" in [p["name"] for p in presets]

    def test_epistemic_preset_contents(self):
        from aragora.debate.presets import get_preset

        preset = get_preset("epistemic")
        assert preset.get("enable_receipt_generation") is True
        assert preset.get("enable_knowledge_injection") is True

    def test_epistemic_preset_protocol_overrides(self):
        from aragora.debate.presets import get_preset

        preset = get_preset("epistemic")
        # Protocol overrides are stored as _protocol_overrides
        overrides = preset.get("_protocol_overrides", {})
        assert overrides.get("enable_epistemic_hygiene") is True


# ---------------------------------------------------------------------------
# VoteBonusCalculator integration tests
# ---------------------------------------------------------------------------


class TestVoteBonusCalculatorIntegration:
    """Tests for epistemic penalties in vote counting."""

    def test_penalties_applied_when_enabled(self):
        from aragora.debate.phases.vote_bonus_calculator import VoteBonusCalculator

        protocol = _FakeProtocol(epistemic_hygiene_penalty=0.10)
        calculator = VoteBonusCalculator(protocol=protocol)

        ctx = MagicMock()
        ctx.proposals = {
            "claude": COMPLIANT_RESPONSE,
            "gpt4": BARE_RESPONSE,
        }

        vote_counts = {"claude": 3.0, "gpt4": 3.0}
        choice_mapping = {"claude": "claude", "gpt4": "gpt4"}

        result = calculator.apply_epistemic_hygiene_penalties(
            ctx, vote_counts, choice_mapping
        )

        # Claude's compliant response should not be penalized
        assert result["claude"] == 3.0
        # GPT4's bare response should be penalized
        assert result["gpt4"] < 3.0

    def test_no_penalties_when_disabled(self):
        from aragora.debate.phases.vote_bonus_calculator import VoteBonusCalculator

        protocol = _FakeProtocol(enable_epistemic_hygiene=False)
        calculator = VoteBonusCalculator(protocol=protocol)

        ctx = MagicMock()
        ctx.proposals = {"claude": BARE_RESPONSE}

        vote_counts = {"claude": 3.0}
        result = calculator.apply_epistemic_hygiene_penalties(
            ctx, vote_counts, {"claude": "claude"}
        )
        assert result["claude"] == 3.0

    def test_no_penalties_without_proposals(self):
        from aragora.debate.phases.vote_bonus_calculator import VoteBonusCalculator

        protocol = _FakeProtocol()
        calculator = VoteBonusCalculator(protocol=protocol)

        ctx = MagicMock()
        ctx.proposals = {}

        vote_counts = {"claude": 3.0}
        result = calculator.apply_epistemic_hygiene_penalties(
            ctx, vote_counts, {"claude": "claude"}
        )
        assert result["claude"] == 3.0

    def test_penalty_does_not_go_below_zero(self):
        from aragora.debate.phases.vote_bonus_calculator import VoteBonusCalculator

        protocol = _FakeProtocol(epistemic_hygiene_penalty=10.0)
        calculator = VoteBonusCalculator(protocol=protocol)

        ctx = MagicMock()
        ctx.proposals = {"claude": BARE_RESPONSE}

        vote_counts = {"claude": 0.5}
        result = calculator.apply_epistemic_hygiene_penalties(
            ctx, vote_counts, {"claude": "claude"}
        )
        assert result["claude"] >= 0.0


# ---------------------------------------------------------------------------
# EpistemicScore edge cases
# ---------------------------------------------------------------------------


class TestEpistemicScoreEdgeCases:
    """Edge cases for the EpistemicScore dataclass."""

    def test_missing_list_completeness(self):
        from aragora.debate.epistemic_hygiene import EpistemicScore

        score = EpistemicScore()
        assert set(score.missing) == {
            "alternatives",
            "falsifiers",
            "confidence_levels",
            "explicit_unknowns",
        }

    def test_score_property_all_true(self):
        from aragora.debate.epistemic_hygiene import EpistemicScore

        score = EpistemicScore(
            has_alternatives=True,
            has_falsifiers=True,
            has_confidence=True,
            has_unknowns=True,
        )
        assert score.score == 1.0
        assert score.missing == []

    def test_score_property_mixed(self):
        from aragora.debate.epistemic_hygiene import EpistemicScore

        score = EpistemicScore(
            has_alternatives=True,
            has_falsifiers=False,
            has_confidence=True,
            has_unknowns=False,
        )
        assert score.score == 0.5
        assert "falsifiers" in score.missing
        assert "explicit_unknowns" in score.missing
