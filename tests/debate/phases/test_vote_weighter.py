"""
Tests for vote_weighter module.

Tests cover:
- VoteWeighterConfig dataclass defaults
- VoteWeighterDeps dataclass defaults
- VoteWeighter.apply_calibration_to_votes (no tracker, adjusted, unchanged, exceptions)
- VoteWeighter.count_weighted_votes (basic, mapping, exceptions, empty, cache miss)
- VoteWeighter.add_user_votes (drain, base weight from protocol, default weight, multiplier)
- VoteWeighter.compute_vote_results (with/without user votes)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from unittest.mock import MagicMock, call, patch

import pytest

from aragora.debate.phases.vote_weighter import (
    VoteWeighter,
    VoteWeighterConfig,
    VoteWeighterDeps,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeVote:
    """Minimal stand-in for aragora.core_types.Vote."""

    agent: str
    choice: str
    reasoning: str = "test reasoning"
    confidence: float = 0.8
    continue_debate: bool = False


def make_vote(agent="agent1", choice="proposal_a", confidence=0.8) -> FakeVote:
    return FakeVote(agent=agent, choice=choice, confidence=confidence)


def make_weighter(
    calibration_tracker=None,
    protocol=None,
    user_votes=None,
    drain_user_events=None,
    user_vote_multiplier=None,
    default_user_vote_weight=0.5,
) -> VoteWeighter:
    config = VoteWeighterConfig(default_user_vote_weight=default_user_vote_weight)
    deps = VoteWeighterDeps(
        calibration_tracker=calibration_tracker,
        protocol=protocol,
        user_votes=user_votes if user_votes is not None else [],
        drain_user_events=drain_user_events,
        user_vote_multiplier=user_vote_multiplier,
    )
    return VoteWeighter(config=config, deps=deps)


# ---------------------------------------------------------------------------
# VoteWeighterConfig tests
# ---------------------------------------------------------------------------


class TestVoteWeighterConfig:
    def test_default_user_vote_weight(self):
        """Default user vote weight is 0.5."""
        config = VoteWeighterConfig()
        assert config.default_user_vote_weight == 0.5

    def test_custom_user_vote_weight(self):
        """Custom user vote weight is stored correctly."""
        config = VoteWeighterConfig(default_user_vote_weight=0.75)
        assert config.default_user_vote_weight == 0.75

    def test_zero_user_vote_weight(self):
        """Zero user vote weight is valid."""
        config = VoteWeighterConfig(default_user_vote_weight=0.0)
        assert config.default_user_vote_weight == 0.0


# ---------------------------------------------------------------------------
# VoteWeighterDeps tests
# ---------------------------------------------------------------------------


class TestVoteWeighterDeps:
    def test_all_defaults_are_none_or_empty(self):
        """All deps default to None/empty."""
        deps = VoteWeighterDeps()
        assert deps.calibration_tracker is None
        assert deps.protocol is None
        assert deps.user_votes == []
        assert deps.drain_user_events is None
        assert deps.user_vote_multiplier is None

    def test_user_votes_default_is_independent(self):
        """Each VoteWeighterDeps instance has its own user_votes list."""
        deps1 = VoteWeighterDeps()
        deps2 = VoteWeighterDeps()
        deps1.user_votes.append({"choice": "a"})
        assert deps2.user_votes == []

    def test_custom_deps(self):
        """Custom deps are stored correctly."""
        tracker = MagicMock()
        protocol = MagicMock()
        drain = MagicMock()
        multiplier = MagicMock()
        deps = VoteWeighterDeps(
            calibration_tracker=tracker,
            protocol=protocol,
            user_votes=[{"choice": "a"}],
            drain_user_events=drain,
            user_vote_multiplier=multiplier,
        )
        assert deps.calibration_tracker is tracker
        assert deps.protocol is protocol
        assert deps.user_votes == [{"choice": "a"}]
        assert deps.drain_user_events is drain
        assert deps.user_vote_multiplier is multiplier


# ---------------------------------------------------------------------------
# VoteWeighter initialisation tests
# ---------------------------------------------------------------------------


class TestVoteWeighterInit:
    def test_defaults_when_no_args(self):
        """VoteWeighter uses default config and deps when none provided."""
        weighter = VoteWeighter()
        assert isinstance(weighter.config, VoteWeighterConfig)
        assert isinstance(weighter.deps, VoteWeighterDeps)
        assert weighter.config.default_user_vote_weight == 0.5

    def test_custom_config_and_deps(self):
        """VoteWeighter stores provided config and deps."""
        config = VoteWeighterConfig(default_user_vote_weight=0.3)
        deps = VoteWeighterDeps(user_votes=[{"choice": "x"}])
        weighter = VoteWeighter(config=config, deps=deps)
        assert weighter.config.default_user_vote_weight == 0.3
        assert weighter.deps.user_votes == [{"choice": "x"}]


# ---------------------------------------------------------------------------
# apply_calibration_to_votes tests
# ---------------------------------------------------------------------------


class TestApplyCalibrationToVotes:
    def _ctx(self):
        return MagicMock()

    def test_no_calibration_tracker_returns_original_list(self):
        """Without a calibration tracker, votes are returned unchanged."""
        weighter = make_weighter()
        votes = [make_vote("a"), make_vote("b")]
        result = weighter.apply_calibration_to_votes(votes, self._ctx())
        assert result is votes

    def test_exception_items_pass_through(self):
        """Exception objects in votes list are preserved as-is."""
        tracker = MagicMock()
        # Tracker should never be asked about an Exception item
        tracker.get_calibration_summary.side_effect = AssertionError(
            "should not be called for Exception items"
        )
        weighter = make_weighter(calibration_tracker=tracker)
        exc = ValueError("bad vote")
        votes = [exc]

        # adjust_agent_confidence is lazy-imported, so it's not a module attribute
        # at patch time; instead verify the tracker is never invoked
        result = weighter.apply_calibration_to_votes(votes, self._ctx())

        assert result == [exc]
        tracker.get_calibration_summary.assert_not_called()

    def test_confidence_changed_creates_new_vote(self):
        """When calibration changes confidence a new Vote is created."""
        tracker = MagicMock()
        summary = MagicMock()
        summary.bias_direction = "overconfident"
        tracker.get_calibration_summary.return_value = summary

        weighter = make_weighter(calibration_tracker=tracker)
        original = make_vote("agent1", "proposal_a", confidence=0.9)

        # The function is imported lazily inside apply_calibration_to_votes,
        # so we patch it at its defining module.
        with patch(
            "aragora.agents.calibration.adjust_agent_confidence",
            return_value=0.7,
        ):
            result = weighter.apply_calibration_to_votes([original], self._ctx())

        assert len(result) == 1
        adjusted = result[0]
        # Should be a Vote-like object, not the original
        assert adjusted is not original
        assert adjusted.agent == "agent1"
        assert adjusted.choice == "proposal_a"
        assert adjusted.confidence == 0.7

    def test_confidence_unchanged_returns_original_vote(self):
        """When calibration returns the same confidence the original vote is kept."""
        tracker = MagicMock()
        summary = MagicMock()
        tracker.get_calibration_summary.return_value = summary

        original = make_vote("agent1", confidence=0.8)
        weighter = make_weighter(calibration_tracker=tracker)

        with patch(
            "aragora.agents.calibration.adjust_agent_confidence",
            return_value=0.8,  # same as original
        ):
            result = weighter.apply_calibration_to_votes([original], self._ctx())

        assert len(result) == 1
        assert result[0] is original

    def test_calibration_exception_keeps_original_vote(self):
        """ValueError/KeyError/TypeError during calibration keeps original vote."""
        tracker = MagicMock()
        tracker.get_calibration_summary.side_effect = KeyError("missing agent")

        original = make_vote("agent_x")
        weighter = make_weighter(calibration_tracker=tracker)

        result = weighter.apply_calibration_to_votes([original], self._ctx())

        assert len(result) == 1
        assert result[0] is original

    def test_mixed_votes_and_exceptions(self):
        """Mix of Exceptions and valid votes handled correctly."""
        tracker = MagicMock()
        summary = MagicMock()
        summary.bias_direction = "well_calibrated"
        tracker.get_calibration_summary.return_value = summary

        exc = RuntimeError("oops")
        good_vote = make_vote("agent2", confidence=0.6)
        weighter = make_weighter(calibration_tracker=tracker)

        with patch(
            "aragora.agents.calibration.adjust_agent_confidence",
            return_value=0.6,  # unchanged → original vote kept
        ):
            result = weighter.apply_calibration_to_votes([exc, good_vote], self._ctx())

        assert result[0] is exc
        assert result[1] is good_vote

    def test_multiple_votes_adjusted(self):
        """Multiple votes are all processed in order."""
        tracker = MagicMock()
        summary = MagicMock()
        summary.bias_direction = "overconfident"
        tracker.get_calibration_summary.return_value = summary

        votes = [make_vote(f"agent{i}", confidence=0.9) for i in range(3)]
        weighter = make_weighter(calibration_tracker=tracker)

        with patch(
            "aragora.agents.calibration.adjust_agent_confidence",
            return_value=0.5,
        ):
            result = weighter.apply_calibration_to_votes(votes, self._ctx())

        assert len(result) == 3
        for v in result:
            assert v.confidence == 0.5


# ---------------------------------------------------------------------------
# count_weighted_votes tests
# ---------------------------------------------------------------------------


class TestCountWeightedVotes:
    def test_empty_votes_returns_zeros(self):
        """Empty vote list gives empty counts and zero total."""
        weighter = make_weighter()
        counts, total = weighter.count_weighted_votes([], {}, {})
        assert counts == {}
        assert total == 0.0

    def test_single_vote_default_weight(self):
        """Single vote with no cache entry uses weight 1.0."""
        weighter = make_weighter()
        vote = make_vote("agent1", "proposal_a")
        counts, total = weighter.count_weighted_votes([vote], {}, {})
        assert counts["proposal_a"] == 1.0
        assert total == 1.0

    def test_choice_mapping_applied(self):
        """Votes are mapped to canonical choices."""
        weighter = make_weighter()
        vote = make_vote("agent1", "PROPOSAL_A")
        mapping = {"PROPOSAL_A": "proposal_a"}
        counts, total = weighter.count_weighted_votes([vote], mapping, {})
        assert "proposal_a" in counts
        assert counts["proposal_a"] == 1.0

    def test_unmapped_choice_kept_as_is(self):
        """Choices not in the mapping are kept verbatim."""
        weighter = make_weighter()
        vote = make_vote("agent1", "unknown_choice")
        counts, total = weighter.count_weighted_votes([vote], {}, {})
        assert "unknown_choice" in counts

    def test_weight_cache_applied(self):
        """Per-agent weights from cache are used."""
        weighter = make_weighter()
        vote = make_vote("agent1", "proposal_a")
        cache = {"agent1": 2.5}
        counts, total = weighter.count_weighted_votes([vote], {}, cache)
        assert counts["proposal_a"] == pytest.approx(2.5)
        assert total == pytest.approx(2.5)

    def test_exception_items_skipped(self):
        """Exception objects in the votes list do not contribute to counts."""
        weighter = make_weighter()
        exc = ValueError("bad")
        vote = make_vote("agent1", "proposal_b")
        counts, total = weighter.count_weighted_votes([exc, vote], {}, {})
        assert "proposal_b" in counts
        assert total == 1.0

    def test_multiple_votes_accumulate(self):
        """Multiple votes for same choice accumulate."""
        weighter = make_weighter()
        votes = [
            make_vote("agent1", "choice_x"),
            make_vote("agent2", "choice_x"),
            make_vote("agent3", "choice_y"),
        ]
        cache = {"agent1": 1.0, "agent2": 2.0, "agent3": 1.0}
        counts, total = weighter.count_weighted_votes(votes, {}, cache)
        assert counts["choice_x"] == pytest.approx(3.0)
        assert counts["choice_y"] == pytest.approx(1.0)
        assert total == pytest.approx(4.0)

    def test_multiple_choices_mapping(self):
        """Multiple choices are all remapped through choice_mapping."""
        weighter = make_weighter()
        votes = [
            make_vote("agent1", "A"),
            make_vote("agent2", "B"),
        ]
        mapping = {"A": "canonical_a", "B": "canonical_b"}
        counts, total = weighter.count_weighted_votes(votes, mapping, {})
        assert "canonical_a" in counts
        assert "canonical_b" in counts
        assert total == pytest.approx(2.0)

    def test_missing_agent_in_cache_defaults_to_one(self):
        """Agents not in cache use weight 1.0."""
        weighter = make_weighter()
        vote = make_vote("unknown_agent", "proposal_z")
        counts, total = weighter.count_weighted_votes([vote], {}, {"other_agent": 5.0})
        assert counts["proposal_z"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# add_user_votes tests
# ---------------------------------------------------------------------------


class TestAddUserVotes:
    def test_drain_user_events_called(self):
        """drain_user_events callback is invoked before processing user votes."""
        drain = MagicMock()
        weighter = make_weighter(
            drain_user_events=drain,
            user_votes=[{"choice": "proposal_a"}],
        )
        vote_counts: dict[str, float] = defaultdict(float)
        weighter.add_user_votes(vote_counts, 0.0, {})
        drain.assert_called_once()

    def test_no_drain_callback_is_fine(self):
        """When drain_user_events is None no error is raised."""
        weighter = make_weighter(user_votes=[{"choice": "proposal_a"}])
        vote_counts: dict[str, float] = defaultdict(float)
        # Should not raise
        weighter.add_user_votes(vote_counts, 0.0, {})

    def test_user_vote_added_with_default_weight(self):
        """User vote is added with the default 0.5 weight when no protocol or multiplier."""
        weighter = make_weighter(
            user_votes=[{"choice": "proposal_a", "intensity": 5}],
            default_user_vote_weight=0.5,
        )
        vote_counts: dict[str, float] = defaultdict(float)
        updated_counts, updated_total = weighter.add_user_votes(vote_counts, 0.0, {})
        assert updated_counts["proposal_a"] == pytest.approx(0.5)
        assert updated_total == pytest.approx(0.5)

    def test_user_vote_weight_from_protocol(self):
        """user_vote_weight is read from protocol when available."""
        protocol = MagicMock()
        protocol.user_vote_weight = 0.8
        weighter = make_weighter(
            protocol=protocol,
            user_votes=[{"choice": "proposal_b"}],
        )
        vote_counts: dict[str, float] = defaultdict(float)
        updated_counts, updated_total = weighter.add_user_votes(vote_counts, 0.0, {})
        assert updated_counts["proposal_b"] == pytest.approx(0.8)

    def test_protocol_weight_none_falls_back_to_default(self):
        """If protocol.user_vote_weight is None/0, default is used."""
        protocol = MagicMock()
        protocol.user_vote_weight = None
        weighter = make_weighter(
            protocol=protocol,
            user_votes=[{"choice": "proposal_c"}],
            default_user_vote_weight=0.4,
        )
        vote_counts: dict[str, float] = defaultdict(float)
        updated_counts, _ = weighter.add_user_votes(vote_counts, 0.0, {})
        assert updated_counts["proposal_c"] == pytest.approx(0.4)

    def test_intensity_multiplier_applied(self):
        """user_vote_multiplier scales final weight by intensity."""
        multiplier = MagicMock(return_value=2.0)
        weighter = make_weighter(
            user_votes=[{"choice": "proposal_d", "intensity": 7}],
            user_vote_multiplier=multiplier,
            default_user_vote_weight=0.5,
        )
        vote_counts: dict[str, float] = defaultdict(float)
        updated_counts, updated_total = weighter.add_user_votes(vote_counts, 0.0, {})
        # final_weight = 0.5 * 2.0 = 1.0
        assert updated_counts["proposal_d"] == pytest.approx(1.0)
        assert updated_total == pytest.approx(1.0)

    def test_intensity_multiplier_receives_correct_args(self):
        """user_vote_multiplier is called with (intensity, protocol)."""
        protocol = MagicMock()
        protocol.user_vote_weight = None
        multiplier = MagicMock(return_value=1.5)
        weighter = make_weighter(
            protocol=protocol,
            user_votes=[{"choice": "proposal_e", "intensity": 9}],
            user_vote_multiplier=multiplier,
            default_user_vote_weight=0.5,
        )
        vote_counts: dict[str, float] = defaultdict(float)
        weighter.add_user_votes(vote_counts, 0.0, {})
        multiplier.assert_called_once_with(9, protocol)

    def test_no_multiplier_uses_1_0(self):
        """When user_vote_multiplier is None, intensity multiplier defaults to 1.0."""
        weighter = make_weighter(
            user_votes=[{"choice": "proposal_f", "intensity": 10}],
            user_vote_multiplier=None,
            default_user_vote_weight=0.6,
        )
        vote_counts: dict[str, float] = defaultdict(float)
        updated_counts, _ = weighter.add_user_votes(vote_counts, 0.0, {})
        # 0.6 * 1.0 = 0.6
        assert updated_counts["proposal_f"] == pytest.approx(0.6)

    def test_user_vote_without_intensity_defaults_to_5(self):
        """A user vote without 'intensity' key uses intensity=5 (does not error)."""
        weighter = make_weighter(
            user_votes=[{"choice": "proposal_g"}],
            default_user_vote_weight=0.5,
        )
        vote_counts: dict[str, float] = defaultdict(float)
        # Should not raise
        updated_counts, _ = weighter.add_user_votes(vote_counts, 0.0, {})
        assert "proposal_g" in updated_counts

    def test_user_vote_with_empty_choice_skipped(self):
        """User votes with empty choice string are ignored."""
        weighter = make_weighter(
            user_votes=[{"choice": "", "user_id": "u1"}],
        )
        vote_counts: dict[str, float] = defaultdict(float)
        updated_counts, updated_total = weighter.add_user_votes(vote_counts, 0.0, {})
        assert updated_total == 0.0

    def test_user_vote_choice_mapping_applied(self):
        """choice_mapping is applied to user vote choices."""
        weighter = make_weighter(
            user_votes=[{"choice": "ALIAS_A"}],
            default_user_vote_weight=0.5,
        )
        mapping = {"ALIAS_A": "canonical_a"}
        vote_counts: dict[str, float] = defaultdict(float)
        updated_counts, _ = weighter.add_user_votes(vote_counts, 0.0, mapping)
        assert "canonical_a" in updated_counts
        assert "ALIAS_A" not in updated_counts

    def test_total_weighted_accumulates(self):
        """total_weighted grows with each user vote added."""
        weighter = make_weighter(
            user_votes=[
                {"choice": "a"},
                {"choice": "b"},
            ],
            default_user_vote_weight=0.5,
        )
        vote_counts: dict[str, float] = defaultdict(float)
        _, total = weighter.add_user_votes(vote_counts, 1.0, {})  # starts at 1.0
        assert total == pytest.approx(1.0 + 0.5 + 0.5)

    def test_no_user_votes_returns_unchanged(self):
        """With no user votes, counts and total are returned unchanged."""
        weighter = make_weighter(user_votes=[])
        vote_counts: dict[str, float] = {"proposal_a": 3.0}
        updated_counts, updated_total = weighter.add_user_votes(vote_counts, 5.0, {})
        assert updated_counts == {"proposal_a": 3.0}
        assert updated_total == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# compute_vote_results tests
# ---------------------------------------------------------------------------


class TestComputeVoteResults:
    def test_returns_counts_and_total(self):
        """compute_vote_results returns tuple of (counts, total)."""
        weighter = make_weighter()
        votes = [make_vote("a", "choice_x"), make_vote("b", "choice_y")]
        counts, total = weighter.compute_vote_results(votes, {}, {})
        assert "choice_x" in counts
        assert "choice_y" in counts
        assert total == pytest.approx(2.0)

    def test_include_user_votes_true_calls_add_user_votes(self):
        """include_user_votes=True (default) includes user votes."""
        weighter = make_weighter(
            user_votes=[{"choice": "choice_x"}],
            default_user_vote_weight=0.5,
        )
        votes = [make_vote("a", "choice_x")]
        counts, total = weighter.compute_vote_results(votes, {}, {}, include_user_votes=True)
        # agent vote (1.0) + user vote (0.5) = 1.5
        assert counts["choice_x"] == pytest.approx(1.5)
        assert total == pytest.approx(1.5)

    def test_include_user_votes_false_skips_user_votes(self):
        """include_user_votes=False omits user votes."""
        drain = MagicMock()
        weighter = make_weighter(
            user_votes=[{"choice": "choice_x"}],
            drain_user_events=drain,
            default_user_vote_weight=0.5,
        )
        votes = [make_vote("a", "choice_x")]
        counts, total = weighter.compute_vote_results(votes, {}, {}, include_user_votes=False)
        # Only agent vote
        assert counts["choice_x"] == pytest.approx(1.0)
        assert total == pytest.approx(1.0)
        # drain should NOT have been called
        drain.assert_not_called()

    def test_empty_votes_with_user_votes(self):
        """Empty agent votes + user votes yields only user vote contribution."""
        weighter = make_weighter(
            user_votes=[{"choice": "proposal_a"}],
            default_user_vote_weight=0.5,
        )
        counts, total = weighter.compute_vote_results([], {}, {})
        assert counts["proposal_a"] == pytest.approx(0.5)
        assert total == pytest.approx(0.5)

    def test_compute_with_weight_cache_and_user_votes(self):
        """Weight cache and user votes combine correctly."""
        weighter = make_weighter(
            user_votes=[{"choice": "proposal_b"}],
            default_user_vote_weight=0.4,
        )
        votes = [make_vote("expert", "proposal_b")]
        cache = {"expert": 3.0}
        counts, total = weighter.compute_vote_results(votes, {}, cache)
        # expert (3.0) + user (0.4) = 3.4
        assert counts["proposal_b"] == pytest.approx(3.4)
        assert total == pytest.approx(3.4)

    def test_default_include_user_votes_is_true(self):
        """include_user_votes defaults to True."""
        drain = MagicMock()
        weighter = make_weighter(
            user_votes=[{"choice": "proposal_c"}],
            drain_user_events=drain,
            default_user_vote_weight=0.5,
        )
        weighter.compute_vote_results([], {}, {})
        drain.assert_called_once()

    def test_exception_votes_skipped_in_pipeline(self):
        """Exceptions in votes are skipped through the full pipeline."""
        weighter = make_weighter()
        exc = ValueError("failure")
        vote = make_vote("ok_agent", "winning_choice")
        counts, total = weighter.compute_vote_results([exc, vote], {}, {})
        assert "winning_choice" in counts
        assert total == pytest.approx(1.0)
