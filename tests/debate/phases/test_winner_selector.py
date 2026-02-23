"""
Tests for winner selection and result finalization.

Tests cover:
- WinnerSelector initialization
- determine_majority_winner: clear winner, threshold met/not met, threshold_override,
  empty votes, variance calculation (strong/medium/weak/unanimous)
- determine_majority_winner: dissenting views, position_tracker, recorder, spectator
- determine_majority_winner: calibration predictions recorded for all votes
- set_unanimous_winner: all fields set correctly, calibration recorded
- set_no_unanimity: final_answer includes all proposals, consensus_reached=False
- analyze_belief_network: with/without belief analyzer, with messages, import error handling
- Error handling: graceful when position_tracker/recorder/calibration_tracker fail
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.phases.winner_selector import WinnerSelector


# ---------------------------------------------------------------------------
# Lightweight fakes
# ---------------------------------------------------------------------------


@dataclass
class FakeVote:
    """Mock vote with agent, choice, confidence."""

    agent: str
    choice: str
    confidence: float = 0.8


@dataclass
class FakeMessage:
    """Mock message with role, agent, content."""

    role: str
    agent: str
    content: str


@dataclass
class FakeResult:
    """Mock debate result."""

    final_answer: str = ""
    consensus_reached: bool = False
    confidence: float = 0.0
    winner: str = ""
    consensus_strength: str = ""
    consensus_variance: float = 0.0
    status: str = ""
    votes: list = field(default_factory=list)
    messages: list = field(default_factory=list)
    dissenting_views: list = field(default_factory=list)
    critiques: list = field(default_factory=list)
    id: str = "debate-001"


@dataclass
class FakeResultNoId:
    """Mock debate result without an 'id' attribute (for fallback path testing)."""

    final_answer: str = ""
    consensus_reached: bool = False
    confidence: float = 0.0
    winner: str = ""
    consensus_strength: str = ""
    consensus_variance: float = 0.0
    status: str = ""
    votes: list = field(default_factory=list)
    messages: list = field(default_factory=list)
    dissenting_views: list = field(default_factory=list)
    critiques: list = field(default_factory=list)


@dataclass
class FakeEnv:
    """Mock environment."""

    task: str = "Design a rate limiter"


@dataclass
class FakeCtx:
    """Mock debate context."""

    result: Any = field(default_factory=FakeResult)
    proposals: dict = field(default_factory=dict)
    agents: list = field(default_factory=list)
    env: Any = field(default_factory=FakeEnv)
    winner_agent: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_selector(**kwargs) -> WinnerSelector:
    """Create a WinnerSelector with sensible defaults; override via kwargs."""
    defaults: dict[str, Any] = {
        "protocol": MagicMock(consensus_threshold=0.5),
        "position_tracker": MagicMock(),
        "calibration_tracker": MagicMock(),
        "recorder": MagicMock(),
        "notify_spectator": MagicMock(),
        "extract_debate_domain": MagicMock(return_value="technology"),
        "get_belief_analyzer": None,
    }
    defaults.update(kwargs)
    return WinnerSelector(**defaults)


def _identity_normalize(choice, agents, proposals):
    """Trivial normalize: return choice as-is."""
    return choice


_SENTINEL = object()


def _make_ctx(proposals=_SENTINEL, agents=_SENTINEL, votes=None, messages=None):
    """Build a FakeCtx pre-populated with proposals, agents, votes, messages."""
    if proposals is _SENTINEL:
        proposals = {"agent_a": "Proposal A", "agent_b": "Proposal B"}
    if agents is _SENTINEL:
        agents = ["agent_a", "agent_b"]
    ctx = FakeCtx(proposals=proposals, agents=agents)
    if votes is not None:
        ctx.result.votes = votes
    if messages is not None:
        ctx.result.messages = messages
    return ctx


# ===========================================================================
# WinnerSelector initialization
# ===========================================================================


class TestWinnerSelectorInit:
    """Tests for WinnerSelector __init__."""

    def test_init_defaults(self):
        selector = WinnerSelector()
        assert selector.protocol is None
        assert selector.position_tracker is None
        assert selector.calibration_tracker is None
        assert selector.recorder is None
        assert selector._notify_spectator is None
        assert selector._extract_debate_domain is None
        assert selector._get_belief_analyzer is None

    def test_init_with_dependencies(self):
        protocol = MagicMock()
        tracker = MagicMock()
        notify = MagicMock()
        selector = WinnerSelector(
            protocol=protocol,
            position_tracker=tracker,
            notify_spectator=notify,
        )
        assert selector.protocol is protocol
        assert selector.position_tracker is tracker
        assert selector._notify_spectator is notify


# ===========================================================================
# determine_majority_winner
# ===========================================================================


class TestDetermineMajorityWinner:
    """Tests for determine_majority_winner."""

    def test_clear_winner_above_threshold(self):
        selector = _make_selector()
        ctx = _make_ctx()
        vote_counts = {"agent_a": 3.0, "agent_b": 1.0}

        selector.determine_majority_winner(ctx, vote_counts, 4.0, {}, _identity_normalize)

        assert ctx.result.consensus_reached is True
        assert ctx.result.confidence == pytest.approx(0.75)
        assert ctx.result.winner == "agent_a"
        assert ctx.result.final_answer == "Proposal A"
        assert ctx.winner_agent == "agent_a"

    def test_winner_below_threshold_not_reached(self):
        selector = _make_selector()
        selector.protocol.consensus_threshold = 0.8
        ctx = _make_ctx()
        vote_counts = {"agent_a": 3.0, "agent_b": 2.0}

        selector.determine_majority_winner(ctx, vote_counts, 5.0, {}, _identity_normalize)

        assert ctx.result.consensus_reached is False
        assert ctx.result.confidence == pytest.approx(0.6)

    def test_threshold_override(self):
        selector = _make_selector()
        selector.protocol.consensus_threshold = 0.9  # high default
        ctx = _make_ctx()
        vote_counts = {"agent_a": 3.0, "agent_b": 2.0}

        selector.determine_majority_winner(
            ctx, vote_counts, 5.0, {}, _identity_normalize, threshold_override=0.5
        )

        # 3/5 = 0.6 >= 0.5 override -> reached
        assert ctx.result.consensus_reached is True

    def test_empty_vote_counts(self):
        selector = _make_selector()
        ctx = _make_ctx()

        selector.determine_majority_winner(ctx, {}, 0.0, {}, _identity_normalize)

        assert ctx.result.consensus_reached is False
        assert ctx.result.confidence == 0.0
        assert ctx.result.status == "insufficient_participation"
        # Falls back to first proposal value
        assert ctx.result.final_answer == "Proposal A"

    def test_empty_vote_counts_no_proposals(self):
        selector = _make_selector()
        ctx = _make_ctx(proposals={})

        selector.determine_majority_winner(ctx, {}, 0.0, {}, _identity_normalize)

        assert ctx.result.final_answer == ""
        assert ctx.result.status == "insufficient_participation"

    def test_total_votes_zero_insufficient(self):
        """Votes exist but total_votes is 0 -> insufficient participation."""
        selector = _make_selector()
        ctx = _make_ctx()
        vote_counts = {"agent_a": 0.0}

        selector.determine_majority_winner(ctx, vote_counts, 0.0, {}, _identity_normalize)

        assert ctx.result.consensus_reached is False
        assert ctx.result.status == "insufficient_participation"

    def test_no_protocol_uses_default_threshold(self):
        """When protocol is None, default threshold of 0.5 is used."""
        selector = _make_selector(protocol=None)
        ctx = _make_ctx()
        vote_counts = {"agent_a": 3.0, "agent_b": 2.0}

        selector.determine_majority_winner(ctx, vote_counts, 5.0, {}, _identity_normalize)

        # 3/5 = 0.6 >= 0.5 default -> reached
        assert ctx.result.consensus_reached is True

    def test_winner_proposal_not_found_falls_back(self):
        """If winner isn't in proposals dict, falls back to first proposal."""
        selector = _make_selector()
        ctx = _make_ctx()

        def remap_normalize(choice, agents, proposals):
            return "unknown_agent"

        vote_counts = {"agent_a": 3.0}

        selector.determine_majority_winner(ctx, vote_counts, 3.0, {}, remap_normalize)

        # Falls back to first proposal value
        assert ctx.result.final_answer == "Proposal A"

    # ---- variance / strength buckets ----

    def test_variance_strong(self):
        """Variance < 1 -> 'strong'."""
        selector = _make_selector()
        ctx = _make_ctx()
        # counts=[4, 3.5] mean=3.75 var=0.0625
        vote_counts = {"agent_a": 4.0, "agent_b": 3.5}

        selector.determine_majority_winner(ctx, vote_counts, 7.5, {}, _identity_normalize)

        assert ctx.result.consensus_strength == "strong"
        assert ctx.result.consensus_variance == pytest.approx(0.0625)

    def test_variance_medium(self):
        """1 <= variance < 2 -> 'medium'."""
        selector = _make_selector()
        ctx = _make_ctx()
        # counts=[4, 2] mean=3 var=1.0
        vote_counts = {"agent_a": 4.0, "agent_b": 2.0}

        selector.determine_majority_winner(ctx, vote_counts, 6.0, {}, _identity_normalize)

        assert ctx.result.consensus_strength == "medium"
        assert ctx.result.consensus_variance == pytest.approx(1.0)

    def test_variance_weak(self):
        """Variance >= 2 -> 'weak'."""
        selector = _make_selector()
        ctx = _make_ctx()
        # counts=[5, 1] mean=3 var=4.0
        vote_counts = {"agent_a": 5.0, "agent_b": 1.0}

        selector.determine_majority_winner(ctx, vote_counts, 6.0, {}, _identity_normalize)

        assert ctx.result.consensus_strength == "weak"

    def test_single_choice_unanimous(self):
        """Only one key in vote_counts -> 'unanimous'."""
        selector = _make_selector()
        ctx = _make_ctx()
        vote_counts = {"agent_a": 5.0}

        selector.determine_majority_winner(ctx, vote_counts, 5.0, {}, _identity_normalize)

        assert ctx.result.consensus_strength == "unanimous"
        assert ctx.result.consensus_variance == 0.0

    # ---- dissenting views ----

    def test_dissenting_views_tracked(self):
        selector = _make_selector()
        proposals = {"agent_a": "Prop A", "agent_b": "Prop B", "agent_c": "Prop C"}
        ctx = _make_ctx(proposals=proposals, agents=["agent_a", "agent_b", "agent_c"])
        vote_counts = {"agent_a": 3.0, "agent_b": 1.0, "agent_c": 1.0}

        selector.determine_majority_winner(ctx, vote_counts, 5.0, {}, _identity_normalize)

        assert len(ctx.result.dissenting_views) == 2
        assert "[agent_b]: Prop B" in ctx.result.dissenting_views
        assert "[agent_c]: Prop C" in ctx.result.dissenting_views

    # ---- side-effect calls ----

    def test_spectator_notified(self):
        spectator = MagicMock()
        selector = _make_selector(notify_spectator=spectator)
        ctx = _make_ctx()
        vote_counts = {"agent_a": 3.0, "agent_b": 1.0}

        selector.determine_majority_winner(ctx, vote_counts, 4.0, {}, _identity_normalize)

        spectator.assert_called_once_with(
            "consensus",
            details="Majority vote: agent_a",
            metric=pytest.approx(0.75),
        )

    def test_no_spectator_is_fine(self):
        selector = _make_selector(notify_spectator=None)
        ctx = _make_ctx()

        selector.determine_majority_winner(ctx, {"agent_a": 1.0}, 1.0, {}, _identity_normalize)
        assert ctx.result.winner == "agent_a"

    def test_recorder_called(self):
        recorder = MagicMock()
        selector = _make_selector(recorder=recorder)
        ctx = _make_ctx()
        vote_counts = {"agent_a": 3.0}

        selector.determine_majority_winner(ctx, vote_counts, 3.0, {}, _identity_normalize)

        recorder.record_phase_change.assert_called_once_with("consensus_reached: agent_a")

    def test_position_tracker_called(self):
        tracker = MagicMock()
        selector = _make_selector(position_tracker=tracker)
        ctx = _make_ctx()
        vote_counts = {"agent_a": 4.0}

        selector.determine_majority_winner(ctx, vote_counts, 4.0, {}, _identity_normalize)

        tracker.finalize_debate.assert_called_once()
        call_kwargs = tracker.finalize_debate.call_args.kwargs
        assert call_kwargs["winning_agent"] == "agent_a"
        assert call_kwargs["debate_id"] == "debate-001"

    def test_position_tracker_uses_env_task_when_no_id(self):
        tracker = MagicMock()
        selector = _make_selector(position_tracker=tracker)
        ctx = _make_ctx()
        # Replace result with a FakeResultNoId that has no 'id' attribute
        ctx.result = FakeResultNoId()
        vote_counts = {"agent_a": 4.0}

        selector.determine_majority_winner(ctx, vote_counts, 4.0, {}, _identity_normalize)

        call_kwargs = tracker.finalize_debate.call_args.kwargs
        assert call_kwargs["debate_id"] == "Design a rate limiter"

    # ---- calibration ----

    def test_calibration_predictions_recorded(self):
        cal = MagicMock()
        selector = _make_selector(calibration_tracker=cal)
        votes = [
            FakeVote("agent_a", "agent_a", 0.9),
            FakeVote("agent_b", "agent_a", 0.7),
            FakeVote("agent_c", "agent_b", 0.6),
        ]
        choice_mapping = {"agent_a": "agent_a", "agent_b": "agent_b"}
        ctx = _make_ctx(votes=votes)
        vote_counts = {"agent_a": 3.0}

        selector.determine_majority_winner(
            ctx, vote_counts, 3.0, choice_mapping, _identity_normalize
        )

        assert cal.record_prediction.call_count == 3
        # First vote: agent_a voted for agent_a (winner) -> correct=True
        first_call = cal.record_prediction.call_args_list[0]
        assert first_call.kwargs["agent"] == "agent_a"
        assert first_call.kwargs["correct"] is True
        # Third vote: agent_c voted for agent_b (not winner) -> correct=False
        third_call = cal.record_prediction.call_args_list[2]
        assert third_call.kwargs["agent"] == "agent_c"
        assert third_call.kwargs["correct"] is False

    def test_calibration_uses_domain_from_extractor(self):
        cal = MagicMock()
        domain_fn = MagicMock(return_value="healthcare")
        selector = _make_selector(calibration_tracker=cal, extract_debate_domain=domain_fn)
        votes = [FakeVote("agent_a", "agent_a", 0.9)]
        ctx = _make_ctx(votes=votes)

        selector.determine_majority_winner(
            ctx, {"agent_a": 1.0}, 1.0, {"agent_a": "agent_a"}, _identity_normalize
        )

        call_kwargs = cal.record_prediction.call_args.kwargs
        assert call_kwargs["domain"] == "healthcare"

    def test_calibration_defaults_to_general_domain(self):
        cal = MagicMock()
        selector = _make_selector(calibration_tracker=cal, extract_debate_domain=None)
        votes = [FakeVote("agent_a", "agent_a", 0.9)]
        ctx = _make_ctx(votes=votes)

        selector.determine_majority_winner(
            ctx, {"agent_a": 1.0}, 1.0, {"agent_a": "agent_a"}, _identity_normalize
        )

        call_kwargs = cal.record_prediction.call_args.kwargs
        assert call_kwargs["domain"] == "general"

    def test_calibration_skips_exception_votes(self):
        cal = MagicMock()
        selector = _make_selector(calibration_tracker=cal)
        votes = [
            FakeVote("agent_a", "agent_a", 0.9),
            RuntimeError("agent failed"),  # Exception object in votes list
        ]
        ctx = _make_ctx(votes=votes)

        selector.determine_majority_winner(
            ctx, {"agent_a": 1.0}, 1.0, {"agent_a": "agent_a"}, _identity_normalize
        )

        assert cal.record_prediction.call_count == 1


# ===========================================================================
# Error resilience in determine_majority_winner
# ===========================================================================


class TestMajorityWinnerErrorHandling:
    """Tests that determine_majority_winner is resilient to dependency failures."""

    def test_recorder_failure_is_swallowed(self):
        recorder = MagicMock()
        recorder.record_phase_change.side_effect = RuntimeError("boom")
        selector = _make_selector(recorder=recorder)
        ctx = _make_ctx()

        # Should not raise
        selector.determine_majority_winner(ctx, {"agent_a": 3.0}, 3.0, {}, _identity_normalize)
        assert ctx.result.winner == "agent_a"

    def test_position_tracker_failure_is_swallowed(self):
        tracker = MagicMock()
        tracker.finalize_debate.side_effect = TypeError("bad arg")
        selector = _make_selector(position_tracker=tracker)
        ctx = _make_ctx()

        selector.determine_majority_winner(ctx, {"agent_a": 3.0}, 3.0, {}, _identity_normalize)
        assert ctx.result.winner == "agent_a"

    @patch(
        "aragora.debate.phases.winner_selector._build_error_action",
        return_value=("cal", "msg", False),
    )
    def test_calibration_tracker_failure_is_swallowed(self, _mock_bea):
        cal = MagicMock()
        cal.record_prediction.side_effect = ValueError("bad value")
        selector = _make_selector(calibration_tracker=cal)
        votes = [FakeVote("agent_a", "agent_a", 0.9)]
        ctx = _make_ctx(votes=votes)

        selector.determine_majority_winner(
            ctx, {"agent_a": 1.0}, 1.0, {"agent_a": "agent_a"}, _identity_normalize
        )
        assert ctx.result.winner == "agent_a"


# ===========================================================================
# set_unanimous_winner
# ===========================================================================


class TestSetUnanimousWinner:
    """Tests for set_unanimous_winner."""

    def test_all_fields_set(self):
        selector = _make_selector()
        ctx = _make_ctx()

        selector.set_unanimous_winner(ctx, "agent_a", 1.0, 5, 5)

        assert ctx.result.final_answer == "Proposal A"
        assert ctx.result.consensus_reached is True
        assert ctx.result.confidence == 1.0
        assert ctx.result.consensus_strength == "unanimous"
        assert ctx.result.consensus_variance == 0.0
        assert ctx.result.winner == "agent_a"
        assert ctx.winner_agent == "agent_a"

    def test_spectator_notified(self):
        spectator = MagicMock()
        selector = _make_selector(notify_spectator=spectator)
        ctx = _make_ctx()

        selector.set_unanimous_winner(ctx, "agent_a", 1.0, 4, 4)

        spectator.assert_called_once_with(
            "consensus",
            details="Unanimous: agent_a",
            metric=1.0,
        )

    def test_recorder_called(self):
        recorder = MagicMock()
        selector = _make_selector(recorder=recorder)
        ctx = _make_ctx()

        selector.set_unanimous_winner(ctx, "agent_a", 1.0, 3, 3)

        recorder.record_phase_change.assert_called_once_with("consensus_reached: agent_a")

    def test_recorder_error_swallowed(self):
        recorder = MagicMock()
        recorder.record_phase_change.side_effect = AttributeError("gone")
        selector = _make_selector(recorder=recorder)
        ctx = _make_ctx()

        # Should not raise
        selector.set_unanimous_winner(ctx, "agent_a", 1.0, 3, 3)
        assert ctx.result.winner == "agent_a"

    def test_calibration_recorded(self):
        cal = MagicMock()
        selector = _make_selector(calibration_tracker=cal)
        votes = [
            FakeVote("agent_a", "agent_a", 0.95),
            FakeVote("agent_b", "agent_a", 0.80),
        ]
        ctx = _make_ctx(votes=votes)

        selector.set_unanimous_winner(ctx, "agent_a", 1.0, 2, 2)

        assert cal.record_prediction.call_count == 2
        # Both voted for agent_a (the winner) -> correct=True
        for call in cal.record_prediction.call_args_list:
            assert call.kwargs["correct"] is True

    @patch(
        "aragora.debate.phases.winner_selector._build_error_action",
        return_value=("cal", "msg", False),
    )
    def test_calibration_error_swallowed(self, _mock_bea):
        cal = MagicMock()
        cal.record_prediction.side_effect = TypeError("oops")
        selector = _make_selector(calibration_tracker=cal)
        votes = [FakeVote("agent_a", "agent_a", 0.9)]
        ctx = _make_ctx(votes=votes)

        # Should not raise
        selector.set_unanimous_winner(ctx, "agent_a", 1.0, 1, 1)
        assert ctx.result.winner == "agent_a"

    def test_winner_not_in_proposals_falls_back(self):
        selector = _make_selector()
        ctx = _make_ctx(proposals={"agent_a": "Prop A"})

        selector.set_unanimous_winner(ctx, "unknown", 1.0, 3, 3)

        # Falls back to first proposal value
        assert ctx.result.final_answer == "Prop A"

    def test_no_calibration_tracker_skips(self):
        selector = _make_selector(calibration_tracker=None)
        votes = [FakeVote("agent_a", "agent_a", 0.9)]
        ctx = _make_ctx(votes=votes)

        selector.set_unanimous_winner(ctx, "agent_a", 1.0, 1, 1)
        assert ctx.result.winner == "agent_a"


# ===========================================================================
# set_no_unanimity
# ===========================================================================


class TestSetNoUnanimity:
    """Tests for set_no_unanimity."""

    def test_final_answer_includes_all_proposals(self):
        selector = _make_selector()
        votes = [
            FakeVote("agent_a", "agent_a", 0.9),
            FakeVote("agent_b", "agent_b", 0.7),
        ]
        proposals = {"agent_a": "Proposal A", "agent_b": "Proposal B"}
        choice_mapping = {"agent_a": "agent_a", "agent_b": "agent_b"}
        ctx = _make_ctx(proposals=proposals, votes=votes)

        selector.set_no_unanimity(ctx, "agent_a", 0.6, 3, 2, choice_mapping)

        assert "[No unanimous consensus reached]" in ctx.result.final_answer
        assert "agent_a" in ctx.result.final_answer
        assert "Proposal A" in ctx.result.final_answer
        assert "agent_b" in ctx.result.final_answer
        assert "Proposal B" in ctx.result.final_answer

    def test_consensus_not_reached(self):
        selector = _make_selector()
        ctx = _make_ctx(votes=[])

        selector.set_no_unanimity(ctx, "agent_a", 0.5, 4, 2, {})

        assert ctx.result.consensus_reached is False
        assert ctx.result.consensus_strength == "none"
        assert ctx.result.confidence == pytest.approx(0.5)

    def test_dissenting_views_all_proposals(self):
        selector = _make_selector()
        proposals = {"agent_a": "PA", "agent_b": "PB", "agent_c": "PC"}
        ctx = _make_ctx(proposals=proposals, votes=[])

        selector.set_no_unanimity(ctx, "agent_a", 0.4, 3, 1, {})

        assert len(ctx.result.dissenting_views) == 3

    def test_spectator_notified(self):
        spectator = MagicMock()
        selector = _make_selector(notify_spectator=spectator)
        ctx = _make_ctx(votes=[])

        selector.set_no_unanimity(ctx, "agent_a", 0.5, 4, 2, {})

        spectator.assert_called_once()
        call_kwargs = spectator.call_args.kwargs
        assert "No unanimity" in call_kwargs["details"]

    def test_vote_counts_in_final_answer(self):
        selector = _make_selector()
        votes = [
            FakeVote("v1", "agent_a", 0.9),
            FakeVote("v2", "agent_a", 0.8),
            FakeVote("v3", "agent_b", 0.7),
        ]
        proposals = {"agent_a": "PA", "agent_b": "PB"}
        choice_mapping = {"agent_a": "agent_a", "agent_b": "agent_b"}
        ctx = _make_ctx(proposals=proposals, votes=votes)

        selector.set_no_unanimity(ctx, "agent_a", 0.66, 3, 2, choice_mapping)

        assert "(2 votes)" in ctx.result.final_answer
        assert "(1 votes)" in ctx.result.final_answer


# ===========================================================================
# analyze_belief_network
# ===========================================================================


class TestAnalyzeBeliefNetwork:
    """Tests for analyze_belief_network."""

    def test_no_analyzer_returns_early(self):
        selector = _make_selector(get_belief_analyzer=None)
        ctx = _make_ctx()

        # Should not raise
        selector.analyze_belief_network(ctx)

    def test_no_messages_returns_early(self):
        mock_analyzer = MagicMock(return_value=(MagicMock, MagicMock))
        selector = _make_selector(get_belief_analyzer=mock_analyzer)
        ctx = _make_ctx(messages=[])

        selector.analyze_belief_network(ctx)
        # Analyzer should not be called since no messages
        mock_analyzer.assert_not_called()

    def test_bn_class_is_none_returns_early(self):
        mock_analyzer = MagicMock(return_value=(None, MagicMock()))
        selector = _make_selector(get_belief_analyzer=mock_analyzer)
        ctx = _make_ctx(messages=[FakeMessage("proposer", "agent_a", "claim")])

        # Should not raise; BN is None so returns early
        selector.analyze_belief_network(ctx)

    def test_with_messages_and_cruxes(self):
        """Full path: messages exist, BN works, cruxes found."""
        mock_network = MagicMock()
        mock_bn_cls = MagicMock(return_value=mock_network)

        mock_crux = MagicMock()
        mock_crux.statement = "key disagreement"
        mock_crux.crux_score = 0.85
        mock_crux.contesting_agents = ["agent_a", "agent_b"]

        mock_analysis = MagicMock()
        mock_analysis.cruxes = [mock_crux]

        mock_detector = MagicMock()
        mock_detector.detect_cruxes.return_value = mock_analysis
        mock_crux_detector_cls = MagicMock(return_value=mock_detector)

        mock_crux_module = MagicMock(CruxDetector=mock_crux_detector_cls)

        with patch.dict("sys.modules", {"aragora.reasoning.crux_detector": mock_crux_module}):
            selector = _make_selector(
                get_belief_analyzer=MagicMock(return_value=(mock_bn_cls, MagicMock()))
            )
            messages = [
                FakeMessage("proposer", "agent_a", "We should use token bucket"),
                FakeMessage("critic", "agent_b", "Sliding window is better"),
                FakeMessage("moderator", "mod", "Summarizing"),  # skipped
            ]
            ctx = _make_ctx(messages=messages)

            selector.analyze_belief_network(ctx)

            # BN was created with max_iterations=3
            mock_bn_cls.assert_called_once_with(max_iterations=3)
            # Only proposer/critic messages added (2 of 3)
            assert mock_network.add_claim.call_count == 2
            # Result has cruxes
            assert hasattr(ctx.result, "cruxes")
            assert len(ctx.result.cruxes) == 1
            assert ctx.result.cruxes[0]["claim"] == "key disagreement"
            assert ctx.result.cruxes[0]["score"] == 0.85

    def test_import_error_swallowed(self):
        """If CruxDetector import fails, error is swallowed."""
        mock_bn_cls = MagicMock(return_value=MagicMock())
        selector = _make_selector(
            get_belief_analyzer=MagicMock(return_value=(mock_bn_cls, MagicMock()))
        )
        messages = [FakeMessage("proposer", "agent_a", "claim")]
        ctx = _make_ctx(messages=messages)

        with patch.dict("sys.modules", {"aragora.reasoning.crux_detector": None}):
            # Should not raise; ImportError is caught
            selector.analyze_belief_network(ctx)

    def test_runtime_error_in_analysis_swallowed(self):
        """RuntimeError during BN construction is caught."""
        mock_bn_cls = MagicMock(side_effect=RuntimeError("analysis failed"))
        selector = _make_selector(
            get_belief_analyzer=MagicMock(return_value=(mock_bn_cls, MagicMock()))
        )
        messages = [FakeMessage("proposer", "agent_a", "claim")]
        ctx = _make_ctx(messages=messages)

        mock_crux_module = MagicMock()
        with patch.dict("sys.modules", {"aragora.reasoning.crux_detector": mock_crux_module}):
            # Should not raise
            selector.analyze_belief_network(ctx)
