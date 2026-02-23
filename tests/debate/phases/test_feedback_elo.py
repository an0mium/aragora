"""Tests for aragora.debate.phases.feedback_elo — EloFeedback."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


from aragora.debate.phases.feedback_elo import EloFeedback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(name: str, role: str = "agent") -> MagicMock:
    a = MagicMock()
    a.name = name
    a.role = role
    return a


def _make_ctx(
    debate_id: str = "debate-1",
    domain: str = "general",
    agents: list | None = None,
    winner: str | None = "alice",
    consensus_reached: bool = True,
    votes: list | None = None,
) -> MagicMock:
    ctx = MagicMock()
    ctx.debate_id = debate_id
    ctx.domain = domain
    ctx.agents = agents or [_make_agent("alice"), _make_agent("bob")]
    result = MagicMock()
    result.winner = winner
    result.consensus_reached = consensus_reached
    result.votes = votes
    result.final_answer = "answer"
    ctx.result = result
    return ctx


def _make_vote(agent: str, choice: str) -> MagicMock:
    v = MagicMock()
    v.agent = agent
    v.choice = choice
    return v


# ---------------------------------------------------------------------------
# EloFeedback — init
# ---------------------------------------------------------------------------


class TestEloFeedbackInit:
    def test_defaults(self):
        ef = EloFeedback()
        assert ef.elo_system is None
        assert ef.event_emitter is None
        assert ef.loop_id is None

    def test_custom(self):
        elo = MagicMock()
        emitter = MagicMock()
        ef = EloFeedback(elo_system=elo, event_emitter=emitter, loop_id="loop-1")
        assert ef.elo_system is elo
        assert ef.event_emitter is emitter
        assert ef.loop_id == "loop-1"


# ---------------------------------------------------------------------------
# record_elo_match
# ---------------------------------------------------------------------------


class TestRecordEloMatch:
    def test_no_elo_system_returns_early(self):
        ef = EloFeedback(elo_system=None)
        ctx = _make_ctx()
        ef.record_elo_match(ctx)  # Should not raise

    def test_no_result_returns_early(self):
        ef = EloFeedback(elo_system=MagicMock())
        ctx = _make_ctx()
        ctx.result = None
        ef.record_elo_match(ctx)
        ef.elo_system.record_match.assert_not_called()

    def test_no_winner_returns_early(self):
        ef = EloFeedback(elo_system=MagicMock())
        ctx = _make_ctx(winner=None)
        ef.record_elo_match(ctx)
        ef.elo_system.record_match.assert_not_called()

    def test_records_match_with_scores(self):
        elo = MagicMock()
        ef = EloFeedback(elo_system=elo)
        ctx = _make_ctx(winner="alice", consensus_reached=True)
        ef.record_elo_match(ctx)
        elo.record_match.assert_called_once()
        args = elo.record_match.call_args
        assert args[1]["domain"] == "general"
        scores = args[0][2]  # Third positional arg
        assert scores["alice"] == 1.0
        assert scores["bob"] == 0.5  # consensus draw

    def test_non_consensus_loser_gets_zero(self):
        elo = MagicMock()
        ef = EloFeedback(elo_system=elo)
        ctx = _make_ctx(winner="alice", consensus_reached=False)
        ef.record_elo_match(ctx)
        scores = elo.record_match.call_args[0][2]
        assert scores["alice"] == 1.0
        assert scores["bob"] == 0.0

    def test_exception_handled_gracefully(self):
        elo = MagicMock()
        elo.record_match.side_effect = RuntimeError("elo failure")
        ef = EloFeedback(elo_system=elo)
        ctx = _make_ctx()
        ef.record_elo_match(ctx)  # Should not raise


# ---------------------------------------------------------------------------
# _emit_match_recorded_event
# ---------------------------------------------------------------------------


class TestEmitMatchRecordedEvent:
    def test_no_emitter_returns_early(self):
        ef = EloFeedback(elo_system=MagicMock(), event_emitter=None)
        ctx = _make_ctx()
        ef._emit_match_recorded_event(ctx, ["alice", "bob"])
        # No assertion needed — just shouldn't raise

    def test_no_elo_system_returns_early(self):
        ef = EloFeedback(elo_system=None, event_emitter=MagicMock())
        ctx = _make_ctx()
        ef._emit_match_recorded_event(ctx, ["alice", "bob"])

    def test_emits_match_recorded_and_per_agent(self):
        elo = MagicMock()
        rating_alice = MagicMock()
        rating_alice.elo = 1600
        rating_bob = MagicMock()
        rating_bob.elo = 1400
        elo.get_ratings_batch.return_value = {
            "alice": rating_alice,
            "bob": rating_bob,
        }
        emitter = MagicMock()
        ef = EloFeedback(elo_system=elo, event_emitter=emitter, loop_id="loop-1")
        ctx = _make_ctx()

        with (
            patch("aragora.events.types.StreamEvent") as MockSE,
            patch("aragora.events.types.StreamEventType") as MockSET,
        ):
            ef._emit_match_recorded_event(ctx, ["alice", "bob"])

        # 1 MATCH_RECORDED + 2 AGENT_ELO_UPDATED = 3 emit calls
        assert emitter.emit.call_count == 3

    def test_missing_rating_defaults_to_1500(self):
        elo = MagicMock()
        elo.get_ratings_batch.return_value = {}  # No ratings
        emitter = MagicMock()
        ef = EloFeedback(elo_system=elo, event_emitter=emitter)
        ctx = _make_ctx()

        with (
            patch("aragora.events.types.StreamEvent"),
            patch("aragora.events.types.StreamEventType"),
        ):
            ef._emit_match_recorded_event(ctx, ["alice"])

        # Only 1 MATCH_RECORDED (no per-agent since rating not in batch)
        assert emitter.emit.call_count == 1

    def test_event_emission_error_handled(self):
        elo = MagicMock()
        elo.get_ratings_batch.side_effect = TypeError("bad")
        emitter = MagicMock()
        ef = EloFeedback(elo_system=elo, event_emitter=emitter)
        ctx = _make_ctx()
        ef._emit_match_recorded_event(ctx, ["alice"])  # Should not raise


# ---------------------------------------------------------------------------
# record_voting_accuracy
# ---------------------------------------------------------------------------


class TestRecordVotingAccuracy:
    def test_no_elo_system_returns_early(self):
        ef = EloFeedback(elo_system=None)
        ctx = _make_ctx(votes=[_make_vote("alice", "X")])
        ef.record_voting_accuracy(ctx)

    def test_no_result_returns_early(self):
        ef = EloFeedback(elo_system=MagicMock())
        ctx = _make_ctx()
        ctx.result = None
        ef.record_voting_accuracy(ctx)
        ef.elo_system.update_voting_accuracy.assert_not_called()

    def test_no_votes_returns_early(self):
        ef = EloFeedback(elo_system=MagicMock())
        ctx = _make_ctx(votes=None)
        ef.record_voting_accuracy(ctx)
        ef.elo_system.update_voting_accuracy.assert_not_called()

    def test_no_winner_returns_early(self):
        ef = EloFeedback(elo_system=MagicMock())
        ctx = _make_ctx(winner=None, votes=[_make_vote("a", "X")])
        ef.record_voting_accuracy(ctx)
        ef.elo_system.update_voting_accuracy.assert_not_called()

    def test_records_accuracy_for_each_vote(self):
        elo = MagicMock()
        ef = EloFeedback(elo_system=elo)
        votes = [_make_vote("alice", "alice"), _make_vote("bob", "carol")]
        ctx = _make_ctx(winner="alice", votes=votes)
        ef.record_voting_accuracy(ctx)
        assert elo.update_voting_accuracy.call_count == 2

    def test_matching_vote_is_consensus(self):
        elo = MagicMock()
        ef = EloFeedback(elo_system=elo)
        votes = [_make_vote("alice", "alice")]
        ctx = _make_ctx(winner="alice", votes=votes)
        ef.record_voting_accuracy(ctx)
        call_kwargs = elo.update_voting_accuracy.call_args[1]
        assert call_kwargs["voted_for_consensus"] is True

    def test_non_matching_vote_not_consensus(self):
        elo = MagicMock()
        ef = EloFeedback(elo_system=elo)
        votes = [_make_vote("bob", "carol")]
        ctx = _make_ctx(winner="alice", votes=votes)
        ef.record_voting_accuracy(ctx)
        call_kwargs = elo.update_voting_accuracy.call_args[1]
        assert call_kwargs["voted_for_consensus"] is False

    def test_vote_without_agent_attr_skipped(self):
        elo = MagicMock()
        ef = EloFeedback(elo_system=elo)
        bad_vote = MagicMock(spec=[])  # No agent/choice attrs
        ctx = _make_ctx(winner="alice", votes=[bad_vote])
        ef.record_voting_accuracy(ctx)
        elo.update_voting_accuracy.assert_not_called()

    def test_exception_handled_gracefully(self):
        elo = MagicMock()
        elo.update_voting_accuracy.side_effect = RuntimeError("oops")
        ef = EloFeedback(elo_system=elo)
        votes = [_make_vote("alice", "alice")]
        ctx = _make_ctx(winner="alice", votes=votes)
        ef.record_voting_accuracy(ctx)  # Should not raise


# ---------------------------------------------------------------------------
# apply_learning_bonuses
# ---------------------------------------------------------------------------


class TestApplyLearningBonuses:
    def test_no_elo_system_returns_early(self):
        ef = EloFeedback(elo_system=None)
        ctx = _make_ctx()
        ef.apply_learning_bonuses(ctx)

    def test_no_result_returns_early(self):
        ef = EloFeedback(elo_system=MagicMock())
        ctx = _make_ctx()
        ctx.result = None
        ef.apply_learning_bonuses(ctx)
        ef.elo_system.apply_learning_bonus.assert_not_called()

    def test_no_winner_returns_early(self):
        ef = EloFeedback(elo_system=MagicMock())
        ctx = _make_ctx(winner=None)
        ef.apply_learning_bonuses(ctx)
        ef.elo_system.apply_learning_bonus.assert_not_called()

    def test_no_consensus_returns_early(self):
        ef = EloFeedback(elo_system=MagicMock())
        ctx = _make_ctx(consensus_reached=False)
        ef.apply_learning_bonuses(ctx)
        ef.elo_system.apply_learning_bonus.assert_not_called()

    def test_applies_bonus_for_each_agent(self):
        elo = MagicMock()
        elo.apply_learning_bonus.return_value = 5.0
        ef = EloFeedback(elo_system=elo)
        agents = [_make_agent("alice"), _make_agent("bob"), _make_agent("carol")]
        ctx = _make_ctx(agents=agents, consensus_reached=True)
        ef.apply_learning_bonuses(ctx)
        assert elo.apply_learning_bonus.call_count == 3

    def test_bonus_uses_correct_params(self):
        elo = MagicMock()
        elo.apply_learning_bonus.return_value = 3.0
        ef = EloFeedback(elo_system=elo)
        ctx = _make_ctx(domain="programming", consensus_reached=True)
        ef.apply_learning_bonuses(ctx)
        call_kwargs = elo.apply_learning_bonus.call_args[1]
        assert call_kwargs["domain"] == "programming"
        assert call_kwargs["bonus_factor"] == 0.5

    def test_empty_domain_defaults_to_general(self):
        elo = MagicMock()
        elo.apply_learning_bonus.return_value = 1.0
        ef = EloFeedback(elo_system=elo)
        ctx = _make_ctx(domain="", consensus_reached=True)
        ef.apply_learning_bonuses(ctx)
        call_kwargs = elo.apply_learning_bonus.call_args[1]
        assert call_kwargs["domain"] == "general"

    def test_per_agent_exception_handled(self):
        elo = MagicMock()
        elo.apply_learning_bonus.side_effect = RuntimeError("fail")
        ef = EloFeedback(elo_system=elo)
        ctx = _make_ctx(consensus_reached=True)
        ef.apply_learning_bonuses(ctx)  # Should not raise

    def test_outer_exception_handled(self):
        elo = MagicMock()
        ef = EloFeedback(elo_system=elo)
        ctx = _make_ctx(consensus_reached=True)
        ctx.agents = None  # Will cause TypeError when iterating
        ef.apply_learning_bonuses(ctx)  # Should not raise
