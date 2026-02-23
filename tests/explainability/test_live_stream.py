"""Tests for LiveExplainabilityStream — real-time factor decomposition."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aragora.explainability.live_stream import (
    ExplanationSnapshot,
    LiveBeliefShift,
    LiveEvidence,
    LiveExplainabilityStream,
    LiveFactor,
    LiveVote,
)


@pytest.fixture
def stream():
    return LiveExplainabilityStream(debate_id="test-debate-1")


@pytest.fixture
def emitter():
    return MagicMock()


@pytest.fixture
def stream_with_emitter(emitter):
    return LiveExplainabilityStream(debate_id="test-2", event_emitter=emitter)


class TestLiveEvidence:
    """Tests for LiveEvidence dataclass."""

    def test_creation(self):
        ev = LiveEvidence(
            content="Rate limiting is effective",
            source="claude",
            evidence_type="proposal",
            round_num=1,
        )
        assert ev.content == "Rate limiting is effective"
        assert ev.source == "claude"
        assert ev.evidence_type == "proposal"
        assert ev.round_num == 1
        assert ev.relevance == 1.0
        assert ev.cited_by == []

    def test_timestamp_auto(self):
        ev = LiveEvidence(content="x", source="a", evidence_type="p", round_num=0)
        assert ev.timestamp > 0


class TestLiveVote:
    """Tests for LiveVote dataclass."""

    def test_creation(self):
        v = LiveVote(agent="claude", choice="option_a", confidence=0.9, round_num=2)
        assert v.agent == "claude"
        assert v.confidence == 0.9
        assert v.flipped is False
        assert v.weight == 1.0

    def test_flipped_vote(self):
        v = LiveVote(agent="gpt4", choice="option_b", confidence=0.7, round_num=3, flipped=True)
        assert v.flipped is True


class TestLiveBeliefShift:
    """Tests for LiveBeliefShift dataclass."""

    def test_creation(self):
        bs = LiveBeliefShift(
            agent="claude",
            round_num=2,
            topic="position",
            prior_confidence=0.8,
            posterior_confidence=0.6,
            trigger="critique_response",
        )
        assert bs.agent == "claude"
        assert bs.prior_confidence == 0.8
        assert bs.posterior_confidence == 0.6


class TestOnProposal:
    """Tests for on_proposal method."""

    def test_basic_proposal(self, stream):
        stream.on_proposal("claude", "We should use caching", round_num=1)
        assert len(stream.evidence) == 1
        assert stream.evidence[0].source == "claude"
        assert stream.evidence[0].evidence_type == "proposal"

    def test_proposal_tracks_position(self, stream):
        stream.on_proposal("claude", "Use rate limiting", round_num=1)
        assert stream._agent_positions["claude"] == "Use rate limiting"

    def test_proposal_with_confidence(self, stream):
        stream.on_proposal("claude", "Use caching", round_num=1, confidence=0.85)
        assert stream._agent_confidences["claude"] == 0.85

    def test_proposal_emits_event(self, stream_with_emitter, emitter):
        stream_with_emitter.on_proposal("claude", "Test", round_num=1)
        emitter.emit.assert_called_once()
        call_args = emitter.emit.call_args
        assert "live_explain_evidence_added" in call_args[0][0]


class TestOnCritique:
    """Tests for on_critique method."""

    def test_basic_critique(self, stream):
        stream.on_critique("gpt4", "This approach is insufficient", round_num=1)
        assert len(stream.evidence) == 1
        assert stream.evidence[0].evidence_type == "critique"
        assert stream.evidence[0].source == "gpt4"

    def test_critique_with_target(self, stream_with_emitter, emitter):
        stream_with_emitter.on_critique("gpt4", "Disagree", target_agent="claude", round_num=1)
        call_data = emitter.emit.call_args[0][1]
        assert call_data["target"] == "claude"


class TestOnRefinement:
    """Tests for on_refinement method."""

    def test_refinement_recorded(self, stream):
        stream.on_refinement("claude", "Updated proposal with caching", round_num=2)
        assert len(stream.evidence) == 1
        assert stream.evidence[0].evidence_type == "refinement"

    def test_refinement_triggers_belief_shift(self, stream):
        stream.on_proposal("claude", "Original position", round_num=1)
        stream.on_refinement("claude", "Revised position after critique", round_num=2)
        assert len(stream.belief_shifts) == 1
        assert stream.belief_shifts[0].trigger == "critique_response"

    def test_refinement_no_shift_if_no_prior(self, stream):
        stream.on_refinement("claude", "First thing said", round_num=1)
        assert len(stream.belief_shifts) == 0


class TestOnVote:
    """Tests for on_vote method."""

    def test_basic_vote(self, stream):
        stream.on_vote("claude", "option_a", confidence=0.9, round_num=2)
        assert len(stream.votes) == 1
        assert stream.votes[0].choice == "option_a"
        assert stream.votes[0].confidence == 0.9
        assert stream.votes[0].flipped is False

    def test_vote_flip_detection(self, stream):
        stream.on_vote("claude", "option_a", confidence=0.8, round_num=1)
        stream.on_vote("claude", "option_b", confidence=0.7, round_num=2)
        assert stream.votes[1].flipped is True
        # Flip should also generate a belief shift
        assert len(stream.belief_shifts) == 1
        assert stream.belief_shifts[0].trigger == "vote_flip"

    def test_same_vote_no_flip(self, stream):
        stream.on_vote("claude", "option_a", confidence=0.8, round_num=1)
        stream.on_vote("claude", "option_a", confidence=0.9, round_num=2)
        assert stream.votes[1].flipped is False

    def test_vote_emits_event(self, stream_with_emitter, emitter):
        stream_with_emitter.on_vote("claude", "opt_a", confidence=0.8, round_num=1)
        emitter.emit.assert_called_once()
        call_data = emitter.emit.call_args[0][1]
        assert call_data["confidence"] == 0.8


class TestOnConsensus:
    """Tests for on_consensus method."""

    def test_consensus_emits(self, stream_with_emitter, emitter):
        stream_with_emitter.on_proposal("claude", "Test", round_num=1)
        stream_with_emitter.on_consensus("Use caching", confidence=0.85)
        calls = emitter.emit.call_args_list
        consensus_call = [c for c in calls if "consensus" in c[0][0]]
        assert len(consensus_call) == 1


class TestOnBeliefChange:
    """Tests for on_belief_change method."""

    def test_explicit_belief_change(self, stream):
        stream.on_belief_change(
            agent="claude",
            topic="rate_limiting",
            prior=0.9,
            posterior=0.6,
            trigger="new_evidence",
            round_num=2,
        )
        assert len(stream.belief_shifts) == 1
        assert stream.belief_shifts[0].prior_confidence == 0.9
        assert stream.belief_shifts[0].posterior_confidence == 0.6
        assert stream._agent_confidences["claude"] == 0.6


class TestComputeFactors:
    """Tests for compute_factors method."""

    def test_no_data_returns_empty(self, stream):
        factors = stream.compute_factors()
        assert factors == []

    def test_evidence_quality_factor(self, stream):
        stream.on_proposal("claude", "Proposal 1", round_num=1)
        stream.on_critique("gpt4", "Critique 1", round_num=1)
        stream.on_refinement("claude", "Refined 1", round_num=2)
        factors = stream.compute_factors()
        eq = [f for f in factors if f.name == "evidence_quality"]
        assert len(eq) == 1
        assert eq[0].contribution > 0
        assert "3 pieces of evidence" in eq[0].explanation

    def test_agent_agreement_factor(self, stream):
        stream.on_vote("claude", "option_a", confidence=0.9, round_num=2)
        stream.on_vote("gpt4", "option_a", confidence=0.8, round_num=2)
        stream.on_vote("gemini", "option_b", confidence=0.7, round_num=2)
        factors = stream.compute_factors()
        aa = [f for f in factors if f.name == "agent_agreement"]
        assert len(aa) == 1
        # 2 out of 3 agree → ~66%
        assert 0.5 < aa[0].raw_value < 0.8

    def test_confidence_weighted_consensus(self, stream):
        stream.on_vote("claude", "opt", confidence=0.95, round_num=1, weight=2.0)
        stream.on_vote("gpt4", "opt", confidence=0.5, round_num=1, weight=1.0)
        factors = stream.compute_factors()
        cwc = [f for f in factors if f.name == "confidence_weighted_consensus"]
        assert len(cwc) == 1
        assert cwc[0].raw_value > 0.7  # Weighted toward claude's higher confidence

    def test_belief_stability_factor(self, stream):
        stream.on_belief_change("claude", "t", 0.9, 0.5, "crit", round_num=1)
        stream.on_belief_change("claude", "t", 0.5, 0.3, "crit2", round_num=3)
        factors = stream.compute_factors()
        bs = [f for f in factors if f.name == "belief_stability"]
        assert len(bs) == 1
        assert bs[0].raw_value < 1.0  # Late shifts reduce stability

    def test_agreement_trend_tracking(self, stream):
        # First snapshot with low agreement
        stream.on_vote("claude", "a", confidence=0.9, round_num=1)
        stream.on_vote("gpt4", "b", confidence=0.8, round_num=1)
        stream.get_snapshot()  # Creates first snapshot

        # Second snapshot with high agreement
        stream.on_vote("claude", "a", confidence=0.9, round_num=2)
        stream.on_vote("gpt4", "a", confidence=0.8, round_num=2)
        factors = stream.compute_factors()
        aa = [f for f in factors if f.name == "agent_agreement"]
        assert len(aa) == 1
        assert aa[0].trend == "rising"


class TestGetLeadingPosition:
    """Tests for get_leading_position method."""

    def test_no_data(self, stream):
        pos, conf = stream.get_leading_position()
        assert pos is None
        assert conf == 0.0

    def test_from_votes(self, stream):
        stream.on_vote("claude", "Use caching", confidence=0.9, round_num=1)
        stream.on_vote("gpt4", "Use caching", confidence=0.8, round_num=1)
        stream.on_vote("gemini", "Use queues", confidence=0.7, round_num=1)
        pos, conf = stream.get_leading_position()
        assert pos == "Use caching"
        assert conf > 0.5

    def test_fallback_to_proposal(self, stream):
        stream.on_proposal("claude", "My proposal", round_num=1)
        pos, conf = stream.get_leading_position()
        assert pos == "My proposal"
        assert conf == 0.5


class TestGenerateNarrative:
    """Tests for generate_narrative method."""

    def test_empty_narrative(self, stream):
        narrative = stream.generate_narrative()
        assert narrative == "Debate is in progress."

    def test_narrative_with_data(self, stream):
        stream.on_proposal("claude", "Use caching for performance", round_num=1)
        stream.on_critique("gpt4", "Caching has invalidation issues", round_num=1)
        stream.on_vote("claude", "Use caching", confidence=0.9, round_num=2)
        stream.on_vote("gpt4", "Use caching", confidence=0.7, round_num=2)
        stream._round_num = 2
        narrative = stream.generate_narrative()
        assert "evidence" in narrative.lower()
        assert "2 votes" in narrative

    def test_narrative_mentions_flips(self, stream):
        stream.on_vote("gpt4", "option_a", confidence=0.8, round_num=1)
        stream.on_vote("gpt4", "option_b", confidence=0.7, round_num=2)
        narrative = stream.generate_narrative()
        assert "gpt4" in narrative
        assert "changed" in narrative.lower()


class TestGetSnapshot:
    """Tests for get_snapshot method."""

    def test_empty_snapshot(self, stream):
        snapshot = stream.get_snapshot()
        assert isinstance(snapshot, ExplanationSnapshot)
        assert snapshot.evidence_count == 0
        assert snapshot.vote_count == 0

    def test_snapshot_with_data(self, stream):
        stream.on_proposal("claude", "Proposal A", round_num=1)
        stream.on_critique("gpt4", "Counter argument", round_num=1)
        stream.on_vote("claude", "opt_a", confidence=0.9, round_num=2)
        stream._round_num = 2
        snapshot = stream.get_snapshot()
        assert snapshot.evidence_count == 2
        assert snapshot.vote_count == 1
        assert snapshot.round_num == 2
        assert len(snapshot.top_factors) > 0
        assert len(snapshot.narrative) > 0

    def test_snapshot_stored(self, stream):
        stream.get_snapshot()
        stream.get_snapshot()
        assert len(stream.snapshots) == 2

    def test_snapshot_emits_event(self, stream_with_emitter, emitter):
        stream_with_emitter.get_snapshot()
        calls = [c for c in emitter.emit.call_args_list if "explanation_snapshot" in c[0][0]]
        assert len(calls) == 1


class TestReset:
    """Tests for reset method."""

    def test_reset_clears_all(self, stream):
        stream.on_proposal("claude", "Test", round_num=1)
        stream.on_vote("claude", "opt", confidence=0.8, round_num=2)
        stream.on_belief_change("claude", "t", 0.8, 0.5, "crit", round_num=1)
        stream.get_snapshot()
        stream.reset()
        assert len(stream.evidence) == 0
        assert len(stream.votes) == 0
        assert len(stream.belief_shifts) == 0
        assert len(stream.snapshots) == 0
        assert stream._round_num == 0


class TestEventEmission:
    """Tests for event emission behavior."""

    def test_no_emitter_no_error(self, stream):
        # Should not raise even without emitter
        stream.on_proposal("claude", "Test", round_num=1)
        stream.on_vote("claude", "opt", confidence=0.8, round_num=1)
        stream.get_snapshot()

    def test_emitter_failure_non_fatal(self):
        bad_emitter = MagicMock()
        bad_emitter.emit.side_effect = ValueError("emit failed")
        s = LiveExplainabilityStream(event_emitter=bad_emitter)
        # Should not raise
        s.on_proposal("claude", "Test", round_num=1)

    def test_debate_id_in_events(self, stream_with_emitter, emitter):
        stream_with_emitter.on_proposal("claude", "Test", round_num=1)
        call_data = emitter.emit.call_args[0][1]
        assert call_data["debate_id"] == "test-2"


class TestEndToEnd:
    """End-to-end debate simulation."""

    def test_multi_round_debate(self):
        stream = LiveExplainabilityStream(debate_id="e2e-test")

        # Round 1: Proposals
        stream.on_round_start(1)
        stream.on_proposal(
            "claude", "Implement rate limiting with sliding window", round_num=1, confidence=0.85
        )
        stream.on_proposal(
            "gpt4", "Use token bucket algorithm instead", round_num=1, confidence=0.80
        )
        stream.on_proposal(
            "gemini", "Combine both approaches for hybrid solution", round_num=1, confidence=0.75
        )

        snap1 = stream.get_snapshot()
        assert snap1.evidence_count == 3

        # Round 1: Critiques
        stream.on_critique(
            "gpt4", "Sliding window has memory overhead", target_agent="claude", round_num=1
        )
        stream.on_critique(
            "claude", "Token bucket can lead to bursts", target_agent="gpt4", round_num=1
        )

        # Round 2: Refinements
        stream.on_round_start(2)
        stream.on_refinement(
            "claude", "Rate limiting with sliding window and burst protection", round_num=2
        )

        # Round 2: Votes
        stream.on_vote("claude", "hybrid", confidence=0.90, round_num=2)
        stream.on_vote("gpt4", "hybrid", confidence=0.85, round_num=2)
        stream.on_vote("gemini", "hybrid", confidence=0.95, round_num=2)

        snap2 = stream.get_snapshot()
        assert snap2.evidence_count == 6  # 3 proposals + 2 critiques + 1 refinement
        assert snap2.vote_count == 3
        assert snap2.agent_agreement == 1.0  # All voted hybrid
        assert snap2.position_confidence > 0.8

        # Verify factors
        factors = stream.compute_factors()
        factor_names = {f.name for f in factors}
        assert "evidence_quality" in factor_names
        assert "agent_agreement" in factor_names
        assert "confidence_weighted_consensus" in factor_names

        # Consensus
        stream.on_consensus("Use hybrid rate limiting approach", confidence=0.90)

        # Verify narrative mentions key facts
        narrative = stream.generate_narrative()
        assert "6 pieces of evidence" in narrative

        # Verify belief shift from refinement
        assert len(stream.belief_shifts) >= 1


class TestImports:
    """Tests for module imports."""

    def test_import_from_module(self):
        from aragora.explainability.live_stream import LiveExplainabilityStream

        assert LiveExplainabilityStream is not None

    def test_import_from_package(self):
        from aragora.explainability import LiveExplainabilityStream

        assert LiveExplainabilityStream is not None

    def test_all_types_importable(self):
        from aragora.explainability import (
            ExplanationSnapshot,
            LiveBeliefShift,
            LiveEvidence,
            LiveExplainabilityStream,
            LiveFactor,
            LiveVote,
        )

        assert all(
            [
                LiveExplainabilityStream,
                LiveEvidence,
                LiveVote,
                LiveBeliefShift,
                LiveFactor,
                ExplanationSnapshot,
            ]
        )
