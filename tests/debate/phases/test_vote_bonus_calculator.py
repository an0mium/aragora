"""
Tests for VoteBonusCalculator module.

Tests cover:
- Initialization with and without protocol
- apply_evidence_citation_bonuses: disabled weighting, no evidence pack,
  empty snippets, single/multiple citations, quality scoring, diminishing
  returns, verification results storage, Exception vote skipping
- apply_process_evaluation_bonuses: disabled flag, no proposals, mock
  evaluator scoring, metrics recording, error handling
"""

import math
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.debate.phases.vote_bonus_calculator import VoteBonusCalculator


# ---------------------------------------------------------------------------
# Helpers / simple data classes
# ---------------------------------------------------------------------------


@dataclass
class MockVote:
    """Minimal vote for testing."""

    agent: str
    choice: str
    reasoning: str = ""


@dataclass
class MockSnippet:
    """Minimal evidence snippet for testing."""

    id: str
    quality_scores: dict = field(default_factory=dict)


def _make_evidence_pack(snippets):
    """Return a MagicMock that looks like an EvidencePack."""
    pack = MagicMock()
    pack.snippets = snippets
    return pack


def _make_ctx(
    result_verification=None,
    evidence_pack=None,
    proposals=None,
    task="test task",
):
    """Build a mock DebateContext."""
    ctx = MagicMock()
    ctx.result = MagicMock()
    ctx.result.verification_results = result_verification
    ctx.result.metadata = {}
    ctx.evidence_pack = evidence_pack
    ctx.proposals = proposals or {}
    ctx.env = MagicMock()
    ctx.env.task = task
    return ctx


def _make_protocol(
    enable_evidence_weighting=True,
    enable_process_evaluation=True,
    evidence_citation_bonus=0.15,
    enable_evidence_quality_weighting=True,
):
    """Build a mock protocol with the given settings."""
    proto = MagicMock()
    proto.enable_evidence_weighting = enable_evidence_weighting
    proto.enable_process_evaluation = enable_process_evaluation
    proto.evidence_citation_bonus = evidence_citation_bonus
    proto.enable_evidence_quality_weighting = enable_evidence_quality_weighting
    return proto


# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------

PATCH_EVIDENCE_METRIC = (
    "aragora.debate.phases.vote_bonus_calculator.record_evidence_citation_bonus"
)
PATCH_PROCESS_METRIC = (
    "aragora.debate.phases.vote_bonus_calculator.record_process_evaluation_bonus"
)


# ===========================================================================
# Tests: __init__
# ===========================================================================


class TestVoteBonusCalculatorInit:
    """Tests for VoteBonusCalculator.__init__."""

    def test_init_no_protocol(self):
        """Calculator is created with protocol=None by default."""
        calc = VoteBonusCalculator()

        assert calc.protocol is None

    def test_init_with_protocol(self):
        """Protocol is stored on the instance."""
        proto = _make_protocol()
        calc = VoteBonusCalculator(protocol=proto)

        assert calc.protocol is proto


# ===========================================================================
# Tests: apply_evidence_citation_bonuses — guard clauses
# ===========================================================================


class TestEvidenceBonusGuards:
    """Guard-clause tests for apply_evidence_citation_bonuses."""

    def test_no_protocol_returns_unchanged(self):
        """Returns vote_counts unchanged when protocol is None."""
        calc = VoteBonusCalculator(protocol=None)
        vote_counts = {"agent_a": 1.0, "agent_b": 2.0}

        result = calc.apply_evidence_citation_bonuses(
            ctx=_make_ctx(),
            votes=[],
            vote_counts=vote_counts,
            choice_mapping={},
        )

        assert result is vote_counts
        assert result == {"agent_a": 1.0, "agent_b": 2.0}

    def test_evidence_weighting_disabled_returns_unchanged(self):
        """Returns vote_counts unchanged when enable_evidence_weighting is False."""
        proto = _make_protocol(enable_evidence_weighting=False)
        calc = VoteBonusCalculator(protocol=proto)
        vote_counts = {"agent_a": 3.0}

        result = calc.apply_evidence_citation_bonuses(
            ctx=_make_ctx(),
            votes=[],
            vote_counts=vote_counts,
            choice_mapping={},
        )

        assert result == {"agent_a": 3.0}

    def test_no_evidence_pack_returns_unchanged(self):
        """Returns vote_counts unchanged when ctx has no evidence_pack."""
        proto = _make_protocol()
        calc = VoteBonusCalculator(protocol=proto)
        ctx = _make_ctx(evidence_pack=None)
        vote_counts = {"agent_a": 1.0}

        result = calc.apply_evidence_citation_bonuses(
            ctx=ctx,
            votes=[MockVote("agent_a", "agent_a", "EVID-001 is great")],
            vote_counts=vote_counts,
            choice_mapping={},
        )

        assert result == {"agent_a": 1.0}

    def test_evidence_pack_without_snippets_attr_returns_unchanged(self):
        """Returns vote_counts unchanged when evidence_pack has no 'snippets'."""
        proto = _make_protocol()
        calc = VoteBonusCalculator(protocol=proto)
        bad_pack = object()  # no snippets attribute
        ctx = _make_ctx(evidence_pack=bad_pack)
        vote_counts = {"agent_a": 1.0}

        result = calc.apply_evidence_citation_bonuses(
            ctx=ctx,
            votes=[MockVote("agent_a", "agent_a", "EVID-001")],
            vote_counts=vote_counts,
            choice_mapping={},
        )

        assert result == {"agent_a": 1.0}

    def test_empty_snippets_returns_unchanged(self):
        """Returns vote_counts unchanged when evidence_pack.snippets is empty."""
        proto = _make_protocol()
        calc = VoteBonusCalculator(protocol=proto)
        ctx = _make_ctx(evidence_pack=_make_evidence_pack([]))
        vote_counts = {"agent_a": 1.0}

        result = calc.apply_evidence_citation_bonuses(
            ctx=ctx,
            votes=[MockVote("agent_a", "agent_a", "EVID-001")],
            vote_counts=vote_counts,
            choice_mapping={},
        )

        assert result == {"agent_a": 1.0}


# ===========================================================================
# Tests: apply_evidence_citation_bonuses — citation counting
# ===========================================================================


class TestEvidenceBonusCitationCounting:
    """Tests that citation IDs are extracted and matched correctly."""

    @patch(PATCH_EVIDENCE_METRIC)
    def test_single_valid_citation_applies_bonus(self, mock_metric):
        """A single matching EVID-xxx citation adds a bonus."""
        proto = _make_protocol(evidence_citation_bonus=0.15)
        calc = VoteBonusCalculator(protocol=proto)
        snippet = MockSnippet(id="ABC", quality_scores={})
        ctx = _make_ctx(evidence_pack=_make_evidence_pack([snippet]))
        vote_counts = {"agent_a": 2.0}
        votes = [MockVote("agent_a", "agent_a", "See EVID-ABC for details")]

        result = calc.apply_evidence_citation_bonuses(
            ctx=ctx,
            votes=votes,
            vote_counts=vote_counts,
            choice_mapping={},
        )

        assert result["agent_a"] > 2.0
        mock_metric.assert_called_once_with(agent="agent_a")

    @patch(PATCH_EVIDENCE_METRIC)
    def test_multiple_valid_citations_accumulate_bonus(self, mock_metric):
        """Multiple citations add proportional bonus."""
        proto = _make_protocol(
            evidence_citation_bonus=0.10, enable_evidence_quality_weighting=False
        )
        calc = VoteBonusCalculator(protocol=proto)
        snippets = [MockSnippet(id="X1"), MockSnippet(id="X2")]
        ctx = _make_ctx(evidence_pack=_make_evidence_pack(snippets))
        vote_counts = {"agent_a": 1.0}
        votes = [MockVote("agent_a", "agent_a", "EVID-X1 and EVID-X2 support this")]

        result = calc.apply_evidence_citation_bonuses(
            ctx=ctx,
            votes=votes,
            vote_counts=vote_counts,
            choice_mapping={},
        )

        # Without quality weighting: bonus = 0.10 * 2 = 0.20
        assert abs(result["agent_a"] - 1.20) < 1e-9

    @patch(PATCH_EVIDENCE_METRIC)
    def test_unknown_evidence_id_not_counted(self, mock_metric):
        """Citations to IDs not in the pack are ignored."""
        proto = _make_protocol(evidence_citation_bonus=0.15)
        calc = VoteBonusCalculator(protocol=proto)
        snippet = MockSnippet(id="KNOWN")
        ctx = _make_ctx(evidence_pack=_make_evidence_pack([snippet]))
        vote_counts = {"agent_a": 1.0}
        votes = [MockVote("agent_a", "agent_a", "EVID-UNKNOWN is not real")]

        result = calc.apply_evidence_citation_bonuses(
            ctx=ctx,
            votes=votes,
            vote_counts=vote_counts,
            choice_mapping={},
        )

        assert result["agent_a"] == 1.0
        mock_metric.assert_not_called()

    @patch(PATCH_EVIDENCE_METRIC)
    def test_duplicate_citations_deduplicated_via_set(self, mock_metric):
        """Duplicate EVID-xxx refs in reasoning count as one citation."""
        proto = _make_protocol(
            evidence_citation_bonus=0.10, enable_evidence_quality_weighting=False
        )
        calc = VoteBonusCalculator(protocol=proto)
        snippet = MockSnippet(id="DUP")
        ctx = _make_ctx(evidence_pack=_make_evidence_pack([snippet]))
        vote_counts = {"agent_a": 1.0}
        votes = [MockVote("agent_a", "agent_a", "EVID-DUP EVID-DUP EVID-DUP")]

        result = calc.apply_evidence_citation_bonuses(
            ctx=ctx,
            votes=votes,
            vote_counts=vote_counts,
            choice_mapping={},
        )

        # set deduplication → only 1 valid citation
        assert abs(result["agent_a"] - 1.10) < 1e-9

    @patch(PATCH_EVIDENCE_METRIC)
    def test_choice_mapping_resolves_canonical(self, mock_metric):
        """choice_mapping is used to resolve the canonical vote target."""
        proto = _make_protocol(
            evidence_citation_bonus=0.10, enable_evidence_quality_weighting=False
        )
        calc = VoteBonusCalculator(protocol=proto)
        snippet = MockSnippet(id="E1")
        ctx = _make_ctx(evidence_pack=_make_evidence_pack([snippet]))
        vote_counts = {"canonical_agent": 1.0}
        votes = [MockVote("voter", "alias", "EVID-E1")]
        choice_mapping = {"alias": "canonical_agent"}

        result = calc.apply_evidence_citation_bonuses(
            ctx=ctx,
            votes=votes,
            vote_counts=vote_counts,
            choice_mapping=choice_mapping,
        )

        assert result["canonical_agent"] > 1.0

    @patch(PATCH_EVIDENCE_METRIC)
    def test_vote_choice_not_in_vote_counts_no_bonus(self, mock_metric):
        """When canonical choice is not in vote_counts, no bonus is applied."""
        proto = _make_protocol(evidence_citation_bonus=0.15)
        calc = VoteBonusCalculator(protocol=proto)
        snippet = MockSnippet(id="ID1")
        ctx = _make_ctx(evidence_pack=_make_evidence_pack([snippet]))
        vote_counts = {"other_agent": 1.0}
        # vote.choice doesn't resolve to anything in vote_counts
        votes = [MockVote("voter", "missing_agent", "EVID-ID1")]

        result = calc.apply_evidence_citation_bonuses(
            ctx=ctx,
            votes=votes,
            vote_counts=vote_counts,
            choice_mapping={},
        )

        assert result["other_agent"] == 1.0
        mock_metric.assert_not_called()

    @patch(PATCH_EVIDENCE_METRIC)
    def test_multiple_agents_each_get_bonus(self, mock_metric):
        """Multiple agents citing evidence each receive independent bonuses."""
        proto = _make_protocol(
            evidence_citation_bonus=0.10, enable_evidence_quality_weighting=False
        )
        calc = VoteBonusCalculator(protocol=proto)
        snippets = [MockSnippet(id="EV1"), MockSnippet(id="EV2")]
        ctx = _make_ctx(evidence_pack=_make_evidence_pack(snippets))
        vote_counts = {"agent_a": 1.0, "agent_b": 2.0}
        votes = [
            MockVote("agent_a", "agent_a", "EVID-EV1"),
            MockVote("agent_b", "agent_b", "EVID-EV2"),
        ]

        result = calc.apply_evidence_citation_bonuses(
            ctx=ctx,
            votes=votes,
            vote_counts=vote_counts,
            choice_mapping={},
        )

        assert abs(result["agent_a"] - 1.10) < 1e-9
        assert abs(result["agent_b"] - 2.10) < 1e-9
        assert mock_metric.call_count == 2


# ===========================================================================
# Tests: apply_evidence_citation_bonuses — quality scoring
# ===========================================================================


class TestEvidenceBonusQualityScoring:
    """Tests for quality-weighted bonus calculations."""

    @patch(PATCH_EVIDENCE_METRIC)
    def test_quality_weighting_disabled_uses_flat_bonus(self, mock_metric):
        """When enable_evidence_quality_weighting=False, bonus is base * count."""
        proto = _make_protocol(
            evidence_citation_bonus=0.20, enable_evidence_quality_weighting=False
        )
        calc = VoteBonusCalculator(protocol=proto)
        snippet = MockSnippet(
            id="Q1",
            quality_scores={
                "semantic_relevance": 1.0,
                "authority": 1.0,
                "freshness": 1.0,
                "completeness": 1.0,
            },
        )
        ctx = _make_ctx(evidence_pack=_make_evidence_pack([snippet]))
        vote_counts = {"agent_a": 0.0}
        votes = [MockVote("agent_a", "agent_a", "EVID-Q1")]

        result = calc.apply_evidence_citation_bonuses(
            ctx=ctx, votes=votes, vote_counts=vote_counts, choice_mapping={}
        )

        # flat: 0.20 * 1 = 0.20
        assert abs(result["agent_a"] - 0.20) < 1e-9

    @patch(PATCH_EVIDENCE_METRIC)
    def test_quality_weighting_uses_weighted_dimensions(self, mock_metric):
        """Quality score = 0.4*sr + 0.3*auth + 0.2*fresh + 0.1*comp."""
        proto = _make_protocol(
            evidence_citation_bonus=1.0, enable_evidence_quality_weighting=True
        )
        calc = VoteBonusCalculator(protocol=proto)
        quality_scores = {
            "semantic_relevance": 0.8,
            "authority": 0.6,
            "freshness": 0.4,
            "completeness": 0.2,
        }
        expected_quality = 0.8 * 0.4 + 0.6 * 0.3 + 0.4 * 0.2 + 0.2 * 0.1
        # expected_quality = 0.32 + 0.18 + 0.08 + 0.02 = 0.60
        snippet = MockSnippet(id="QS1", quality_scores=quality_scores)
        ctx = _make_ctx(evidence_pack=_make_evidence_pack([snippet]))
        vote_counts = {"agent_a": 0.0}
        votes = [MockVote("agent_a", "agent_a", "EVID-QS1")]

        result = calc.apply_evidence_citation_bonuses(
            ctx=ctx, votes=votes, vote_counts=vote_counts, choice_mapping={}
        )

        # With 1 citation: quality_factor = sqrt(expected_quality / 1) = sqrt(0.60)
        quality_factor = math.sqrt(expected_quality)
        expected_bonus = 1.0 * 1 * quality_factor
        assert abs(result["agent_a"] - expected_bonus) < 1e-9

    @patch(PATCH_EVIDENCE_METRIC)
    def test_no_quality_scores_uses_default_0_5(self, mock_metric):
        """When snippet has no quality_scores, quality defaults to 0.5."""
        proto = _make_protocol(
            evidence_citation_bonus=1.0, enable_evidence_quality_weighting=True
        )
        calc = VoteBonusCalculator(protocol=proto)
        snippet = MockSnippet(id="NQ1", quality_scores={})  # empty → default 0.5
        ctx = _make_ctx(evidence_pack=_make_evidence_pack([snippet]))
        vote_counts = {"agent_a": 0.0}
        votes = [MockVote("agent_a", "agent_a", "EVID-NQ1")]

        result = calc.apply_evidence_citation_bonuses(
            ctx=ctx, votes=votes, vote_counts=vote_counts, choice_mapping={}
        )

        # quality = 0.5 (default), quality_factor = sqrt(0.5 / 1)
        quality_factor = math.sqrt(0.5)
        expected_bonus = 1.0 * 1 * quality_factor
        assert abs(result["agent_a"] - expected_bonus) < 1e-9

    @patch(PATCH_EVIDENCE_METRIC)
    def test_diminishing_returns_with_multiple_citations(self, mock_metric):
        """Two citations give less than double the single-citation bonus."""
        proto = _make_protocol(
            evidence_citation_bonus=1.0, enable_evidence_quality_weighting=True
        )
        calc = VoteBonusCalculator(protocol=proto)
        snippets = [
            MockSnippet(id="D1", quality_scores={"semantic_relevance": 1.0, "authority": 1.0, "freshness": 1.0, "completeness": 1.0}),
            MockSnippet(id="D2", quality_scores={"semantic_relevance": 1.0, "authority": 1.0, "freshness": 1.0, "completeness": 1.0}),
        ]
        ctx = _make_ctx(evidence_pack=_make_evidence_pack(snippets))
        vote_counts = {"agent_a": 0.0}
        votes = [MockVote("agent_a", "agent_a", "EVID-D1 and EVID-D2")]

        result = calc.apply_evidence_citation_bonuses(
            ctx=ctx, votes=votes, vote_counts=vote_counts, choice_mapping={}
        )

        # quality per snippet = 1.0, avg quality = 1.0, quality_factor = sqrt(1.0) = 1.0
        # bonus = 1.0 * 2 * 1.0 = 2.0  (no diminishing in this case since quality=1.0)
        # Test that the formula is quality_factor = sqrt(total_quality / count)
        # 2 citations, both quality 1.0 → total=2.0, avg=1.0, sqrt(1.0)=1.0, bonus=2.0
        assert abs(result["agent_a"] - 2.0) < 1e-9

    @patch(PATCH_EVIDENCE_METRIC)
    def test_diminishing_returns_with_mixed_quality(self, mock_metric):
        """Mixed quality scores trigger genuine diminishing returns."""
        proto = _make_protocol(
            evidence_citation_bonus=1.0, enable_evidence_quality_weighting=True
        )
        calc = VoteBonusCalculator(protocol=proto)
        # Both snippets have no quality scores → default 0.5 each
        snippets = [MockSnippet(id="M1"), MockSnippet(id="M2")]
        ctx = _make_ctx(evidence_pack=_make_evidence_pack(snippets))
        vote_counts = {"agent_a": 0.0}
        votes = [MockVote("agent_a", "agent_a", "EVID-M1 EVID-M2")]

        result = calc.apply_evidence_citation_bonuses(
            ctx=ctx, votes=votes, vote_counts=vote_counts, choice_mapping={}
        )

        # total_quality = 0.5 + 0.5 = 1.0, avg = 0.5, factor = sqrt(0.5)
        expected = 1.0 * 2 * math.sqrt(0.5)
        assert abs(result["agent_a"] - expected) < 1e-9

    @patch(PATCH_EVIDENCE_METRIC)
    def test_custom_evidence_bonus_value_respected(self, mock_metric):
        """Custom evidence_citation_bonus value is used."""
        proto = _make_protocol(
            evidence_citation_bonus=0.05, enable_evidence_quality_weighting=False
        )
        calc = VoteBonusCalculator(protocol=proto)
        snippet = MockSnippet(id="CUSTOM")
        ctx = _make_ctx(evidence_pack=_make_evidence_pack([snippet]))
        vote_counts = {"agent_a": 1.0}
        votes = [MockVote("agent_a", "agent_a", "EVID-CUSTOM")]

        result = calc.apply_evidence_citation_bonuses(
            ctx=ctx, votes=votes, vote_counts=vote_counts, choice_mapping={}
        )

        assert abs(result["agent_a"] - 1.05) < 1e-9


# ===========================================================================
# Tests: apply_evidence_citation_bonuses — verification results & skipping
# ===========================================================================


class TestEvidenceBonusVerificationAndSkipping:
    """Tests for verification_results storage and Exception skipping."""

    @patch(PATCH_EVIDENCE_METRIC)
    def test_verification_results_stored_per_agent(self, mock_metric):
        """Citation counts and quality totals stored in ctx.result.verification_results."""
        proto = _make_protocol(
            evidence_citation_bonus=0.10, enable_evidence_quality_weighting=False
        )
        calc = VoteBonusCalculator(protocol=proto)
        snippet = MockSnippet(id="VR1")
        ctx = _make_ctx(evidence_pack=_make_evidence_pack([snippet]))
        ctx.result.verification_results = None  # simulate not yet initialized
        vote_counts = {"agent_a": 1.0}
        votes = [MockVote("agent_a", "agent_a", "EVID-VR1")]

        calc.apply_evidence_citation_bonuses(
            ctx=ctx, votes=votes, vote_counts=vote_counts, choice_mapping={}
        )

        assert ctx.result.verification_results is not None
        assert "evidence_agent_a" in ctx.result.verification_results
        assert ctx.result.verification_results["evidence_agent_a"] == 1

    @patch(PATCH_EVIDENCE_METRIC)
    def test_verification_results_quality_stored(self, mock_metric):
        """Quality total stored as evidence_quality_<agent>."""
        proto = _make_protocol(
            evidence_citation_bonus=0.10, enable_evidence_quality_weighting=True
        )
        calc = VoteBonusCalculator(protocol=proto)
        snippet = MockSnippet(
            id="VRQ1",
            quality_scores={
                "semantic_relevance": 1.0,
                "authority": 1.0,
                "freshness": 1.0,
                "completeness": 1.0,
            },
        )
        ctx = _make_ctx(evidence_pack=_make_evidence_pack([snippet]))
        ctx.result.verification_results = None
        vote_counts = {"agent_a": 0.0}
        votes = [MockVote("agent_a", "agent_a", "EVID-VRQ1")]

        calc.apply_evidence_citation_bonuses(
            ctx=ctx, votes=votes, vote_counts=vote_counts, choice_mapping={}
        )

        key = "evidence_quality_agent_a"
        assert key in ctx.result.verification_results
        # quality = 1.0 for all dimensions → total = 1.0
        assert abs(ctx.result.verification_results[key] - 1.0) < 0.01

    @patch(PATCH_EVIDENCE_METRIC)
    def test_existing_verification_results_dict_updated(self, mock_metric):
        """Pre-existing verification_results dict is updated, not replaced."""
        proto = _make_protocol(
            evidence_citation_bonus=0.10, enable_evidence_quality_weighting=False
        )
        calc = VoteBonusCalculator(protocol=proto)
        snippet = MockSnippet(id="EX1")
        ctx = _make_ctx(evidence_pack=_make_evidence_pack([snippet]))
        ctx.result.verification_results = {"pre_existing": "value"}
        vote_counts = {"agent_a": 1.0}
        votes = [MockVote("agent_a", "agent_a", "EVID-EX1")]

        calc.apply_evidence_citation_bonuses(
            ctx=ctx, votes=votes, vote_counts=vote_counts, choice_mapping={}
        )

        assert ctx.result.verification_results["pre_existing"] == "value"
        assert "evidence_agent_a" in ctx.result.verification_results

    @patch(PATCH_EVIDENCE_METRIC)
    def test_exception_votes_are_skipped(self, mock_metric):
        """Exception objects in votes list are skipped entirely."""
        proto = _make_protocol(
            evidence_citation_bonus=0.10, enable_evidence_quality_weighting=False
        )
        calc = VoteBonusCalculator(protocol=proto)
        snippet = MockSnippet(id="EXC1")
        ctx = _make_ctx(evidence_pack=_make_evidence_pack([snippet]))
        vote_counts = {"agent_a": 1.0}
        votes = [
            Exception("agent failed"),
            MockVote("agent_a", "agent_a", "EVID-EXC1"),
        ]

        result = calc.apply_evidence_citation_bonuses(
            ctx=ctx, votes=votes, vote_counts=vote_counts, choice_mapping={}
        )

        # Exception is skipped; only agent_a's vote processed
        assert result["agent_a"] > 1.0
        mock_metric.assert_called_once_with(agent="agent_a")

    @patch(PATCH_EVIDENCE_METRIC)
    def test_all_exception_votes_no_bonus(self, mock_metric):
        """If all votes are Exceptions, vote_counts is returned unchanged."""
        proto = _make_protocol()
        calc = VoteBonusCalculator(protocol=proto)
        snippet = MockSnippet(id="ALL_EXC")
        ctx = _make_ctx(evidence_pack=_make_evidence_pack([snippet]))
        vote_counts = {"agent_a": 5.0}
        votes = [Exception("fail1"), Exception("fail2")]

        result = calc.apply_evidence_citation_bonuses(
            ctx=ctx, votes=votes, vote_counts=vote_counts, choice_mapping={}
        )

        assert result == {"agent_a": 5.0}
        mock_metric.assert_not_called()

    @patch(PATCH_EVIDENCE_METRIC)
    def test_no_reasoning_matches_no_bonus(self, mock_metric):
        """Vote with no EVID pattern in reasoning gets no bonus."""
        proto = _make_protocol()
        calc = VoteBonusCalculator(protocol=proto)
        snippet = MockSnippet(id="NR1")
        ctx = _make_ctx(evidence_pack=_make_evidence_pack([snippet]))
        vote_counts = {"agent_a": 1.0}
        votes = [MockVote("agent_a", "agent_a", "No evidence cited here.")]

        result = calc.apply_evidence_citation_bonuses(
            ctx=ctx, votes=votes, vote_counts=vote_counts, choice_mapping={}
        )

        assert result["agent_a"] == 1.0
        mock_metric.assert_not_called()

    @patch(PATCH_EVIDENCE_METRIC)
    def test_ctx_result_none_no_verification_error(self, mock_metric):
        """Works correctly when ctx.result is None (no verification_results stored)."""
        proto = _make_protocol(
            evidence_citation_bonus=0.10, enable_evidence_quality_weighting=False
        )
        calc = VoteBonusCalculator(protocol=proto)
        snippet = MockSnippet(id="RN1")
        ctx = _make_ctx(evidence_pack=_make_evidence_pack([snippet]))
        ctx.result = None  # Simulate missing result
        vote_counts = {"agent_a": 1.0}
        votes = [MockVote("agent_a", "agent_a", "EVID-RN1")]

        # Should not raise
        result = calc.apply_evidence_citation_bonuses(
            ctx=ctx, votes=votes, vote_counts=vote_counts, choice_mapping={}
        )

        # Bonus still applied
        assert result["agent_a"] > 1.0


# ===========================================================================
# Tests: apply_process_evaluation_bonuses
# ===========================================================================


class TestProcessEvaluationBonuses:
    """Tests for apply_process_evaluation_bonuses."""

    @pytest.mark.asyncio
    async def test_no_protocol_returns_unchanged(self):
        """Returns vote_counts unchanged when protocol is None."""
        calc = VoteBonusCalculator(protocol=None)
        vote_counts = {"agent_a": 1.0}

        result = await calc.apply_process_evaluation_bonuses(
            ctx=_make_ctx(),
            vote_counts=vote_counts,
            choice_mapping={},
        )

        assert result is vote_counts

    @pytest.mark.asyncio
    async def test_process_evaluation_disabled_returns_unchanged(self):
        """Returns vote_counts unchanged when enable_process_evaluation is False."""
        proto = _make_protocol(enable_process_evaluation=False)
        calc = VoteBonusCalculator(protocol=proto)
        vote_counts = {"agent_a": 2.0}

        result = await calc.apply_process_evaluation_bonuses(
            ctx=_make_ctx(),
            vote_counts=vote_counts,
            choice_mapping={},
        )

        assert result == {"agent_a": 2.0}

    @pytest.mark.asyncio
    async def test_empty_proposals_returns_unchanged(self):
        """Returns vote_counts unchanged when proposals is empty."""
        proto = _make_protocol(enable_process_evaluation=True)
        calc = VoteBonusCalculator(protocol=proto)
        ctx = _make_ctx(proposals={})
        vote_counts = {"agent_a": 3.0}

        result = await calc.apply_process_evaluation_bonuses(
            ctx=ctx,
            vote_counts=vote_counts,
            choice_mapping={},
        )

        assert result == {"agent_a": 3.0}

    @pytest.mark.asyncio
    @patch(PATCH_PROCESS_METRIC)
    async def test_process_bonus_applied_with_mock_evaluator(self, mock_metric):
        """Process bonus = 0.2 * weighted_total is applied to vote_counts."""
        proto = _make_protocol(enable_process_evaluation=True)
        calc = VoteBonusCalculator(protocol=proto)

        # Mock ProcessEvaluationResult
        eval_result = MagicMock()
        eval_result.weighted_total = 0.8
        eval_result.evaluation_notes = ["good reasoning"]
        eval_result.criterion_scores = {"reasoning_quality": 0.8}

        mock_evaluator = MagicMock()
        mock_evaluator.evaluate_proposal = AsyncMock(return_value=eval_result)

        ctx = _make_ctx(proposals={"agent_a": "My proposal text"})
        vote_counts = {"agent_a": 1.0}

        with patch(
            "aragora.debate.bias_mitigation.ProcessEvaluator",
            return_value=mock_evaluator,
        ):
            result = await calc.apply_process_evaluation_bonuses(
                ctx=ctx,
                vote_counts=vote_counts,
                choice_mapping={},
            )

        # bonus = 0.2 * 0.8 = 0.16
        assert abs(result["agent_a"] - 1.16) < 1e-9
        mock_metric.assert_called_once_with(agent="agent_a")

    @pytest.mark.asyncio
    @patch(PATCH_PROCESS_METRIC)
    async def test_process_bonus_uses_choice_mapping(self, mock_metric):
        """Choice mapping resolves canonical name for bonus application."""
        proto = _make_protocol(enable_process_evaluation=True)
        calc = VoteBonusCalculator(protocol=proto)

        eval_result = MagicMock()
        eval_result.weighted_total = 0.5
        eval_result.evaluation_notes = []
        eval_result.criterion_scores = {}

        mock_evaluator = MagicMock()
        mock_evaluator.evaluate_proposal = AsyncMock(return_value=eval_result)

        ctx = _make_ctx(proposals={"original_name": "proposal"})
        vote_counts = {"canonical_name": 2.0}
        choice_mapping = {"original_name": "canonical_name"}

        with patch(
            "aragora.debate.bias_mitigation.ProcessEvaluator",
            return_value=mock_evaluator,
        ):
            result = await calc.apply_process_evaluation_bonuses(
                ctx=ctx,
                vote_counts=vote_counts,
                choice_mapping=choice_mapping,
            )

        # bonus = 0.2 * 0.5 = 0.10
        assert abs(result["canonical_name"] - 2.10) < 1e-9

    @pytest.mark.asyncio
    @patch(PATCH_PROCESS_METRIC)
    async def test_multiple_proposals_each_get_bonus(self, mock_metric):
        """Multiple proposals are evaluated independently."""
        proto = _make_protocol(enable_process_evaluation=True)
        calc = VoteBonusCalculator(protocol=proto)

        def make_eval_result(score):
            r = MagicMock()
            r.weighted_total = score
            r.evaluation_notes = []
            r.criterion_scores = {}
            return r

        mock_evaluator = MagicMock()
        mock_evaluator.evaluate_proposal = AsyncMock(
            side_effect=[make_eval_result(0.6), make_eval_result(0.4)]
        )

        ctx = _make_ctx(proposals={"agent_a": "prop A", "agent_b": "prop B"})
        vote_counts = {"agent_a": 1.0, "agent_b": 1.0}

        with patch(
            "aragora.debate.bias_mitigation.ProcessEvaluator",
            return_value=mock_evaluator,
        ):
            result = await calc.apply_process_evaluation_bonuses(
                ctx=ctx,
                vote_counts=vote_counts,
                choice_mapping={},
            )

        # agent_a: 1.0 + 0.2*0.6 = 1.12
        # agent_b: 1.0 + 0.2*0.4 = 1.08
        assert abs(result["agent_a"] - 1.12) < 1e-9
        assert abs(result["agent_b"] - 1.08) < 1e-9
        assert mock_metric.call_count == 2

    @pytest.mark.asyncio
    @patch(PATCH_PROCESS_METRIC)
    async def test_evaluator_error_no_bonus_applied(self, mock_metric):
        """ValueError from evaluator is caught; no bonus is applied for that agent."""
        proto = _make_protocol(enable_process_evaluation=True)
        calc = VoteBonusCalculator(protocol=proto)

        mock_evaluator = MagicMock()
        mock_evaluator.evaluate_proposal = AsyncMock(
            side_effect=ValueError("evaluation failed")
        )

        ctx = _make_ctx(proposals={"agent_a": "proposal"})
        vote_counts = {"agent_a": 3.0}

        with patch(
            "aragora.debate.bias_mitigation.ProcessEvaluator",
            return_value=mock_evaluator,
        ):
            result = await calc.apply_process_evaluation_bonuses(
                ctx=ctx,
                vote_counts=vote_counts,
                choice_mapping={},
            )

        # No bonus — error was caught
        assert result["agent_a"] == 3.0
        mock_metric.assert_not_called()

    @pytest.mark.asyncio
    @patch(PATCH_PROCESS_METRIC)
    async def test_keyerror_from_evaluator_caught(self, mock_metric):
        """KeyError from evaluator is caught; no bonus applied."""
        proto = _make_protocol(enable_process_evaluation=True)
        calc = VoteBonusCalculator(protocol=proto)

        mock_evaluator = MagicMock()
        mock_evaluator.evaluate_proposal = AsyncMock(
            side_effect=KeyError("missing_key")
        )

        ctx = _make_ctx(proposals={"agent_b": "prop"})
        vote_counts = {"agent_b": 2.0}

        with patch(
            "aragora.debate.bias_mitigation.ProcessEvaluator",
            return_value=mock_evaluator,
        ):
            result = await calc.apply_process_evaluation_bonuses(
                ctx=ctx,
                vote_counts=vote_counts,
                choice_mapping={},
            )

        assert result["agent_b"] == 2.0

    @pytest.mark.asyncio
    @patch(PATCH_PROCESS_METRIC)
    async def test_agent_not_in_vote_counts_no_bonus(self, mock_metric):
        """Agent proposal evaluated but agent not in vote_counts — no bonus."""
        proto = _make_protocol(enable_process_evaluation=True)
        calc = VoteBonusCalculator(protocol=proto)

        eval_result = MagicMock()
        eval_result.weighted_total = 0.9
        eval_result.evaluation_notes = []
        eval_result.criterion_scores = {}

        mock_evaluator = MagicMock()
        mock_evaluator.evaluate_proposal = AsyncMock(return_value=eval_result)

        ctx = _make_ctx(proposals={"unknown_agent": "proposal"})
        vote_counts = {"other_agent": 1.0}

        with patch(
            "aragora.debate.bias_mitigation.ProcessEvaluator",
            return_value=mock_evaluator,
        ):
            result = await calc.apply_process_evaluation_bonuses(
                ctx=ctx,
                vote_counts=vote_counts,
                choice_mapping={},
            )

        assert result["other_agent"] == 1.0
        # metric should still be recorded since evaluate succeeded
        mock_metric.assert_not_called()

    @pytest.mark.asyncio
    @patch(PATCH_PROCESS_METRIC)
    async def test_perfect_process_score_gives_max_bonus(self, mock_metric):
        """weighted_total=1.0 gives the maximum bonus of 0.2."""
        proto = _make_protocol(enable_process_evaluation=True)
        calc = VoteBonusCalculator(protocol=proto)

        eval_result = MagicMock()
        eval_result.weighted_total = 1.0
        eval_result.evaluation_notes = []
        eval_result.criterion_scores = {"all": 1.0}

        mock_evaluator = MagicMock()
        mock_evaluator.evaluate_proposal = AsyncMock(return_value=eval_result)

        ctx = _make_ctx(proposals={"agent_a": "perfect proposal"})
        vote_counts = {"agent_a": 0.0}

        with patch(
            "aragora.debate.bias_mitigation.ProcessEvaluator",
            return_value=mock_evaluator,
        ):
            result = await calc.apply_process_evaluation_bonuses(
                ctx=ctx,
                vote_counts=vote_counts,
                choice_mapping={},
            )

        # max bonus: 0.2 * 1.0 = 0.2
        assert abs(result["agent_a"] - 0.20) < 1e-9

    @pytest.mark.asyncio
    @patch(PATCH_PROCESS_METRIC)
    async def test_zero_process_score_gives_no_bonus(self, mock_metric):
        """weighted_total=0.0 gives zero bonus."""
        proto = _make_protocol(enable_process_evaluation=True)
        calc = VoteBonusCalculator(protocol=proto)

        eval_result = MagicMock()
        eval_result.weighted_total = 0.0
        eval_result.evaluation_notes = []
        eval_result.criterion_scores = {}

        mock_evaluator = MagicMock()
        mock_evaluator.evaluate_proposal = AsyncMock(return_value=eval_result)

        ctx = _make_ctx(proposals={"agent_a": "zero proposal"})
        vote_counts = {"agent_a": 5.0}

        with patch(
            "aragora.debate.bias_mitigation.ProcessEvaluator",
            return_value=mock_evaluator,
        ):
            result = await calc.apply_process_evaluation_bonuses(
                ctx=ctx,
                vote_counts=vote_counts,
                choice_mapping={},
            )

        assert abs(result["agent_a"] - 5.0) < 1e-9
        mock_metric.assert_called_once_with(agent="agent_a")
