"""
Tests for JudgmentPhase in aragora/debate/phases/judgment.py.

Covers:
- Initialization and attribute storage
- select_judge with all selection strategies
- _select_last with and without synthesizer
- _select_elo_ranked with leaderboard, fallback, and exceptions
- _select_calibrated with scores, missing elo, missing composite fn
- _require_agents raises on empty list
- should_terminate with all decision branches
- get_judge_stats with and without elo/calibration
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.phases.judgment import JudgmentPhase


# ---------------------------------------------------------------------------
# Lightweight stubs
# ---------------------------------------------------------------------------


@dataclass
class MockAgent:
    name: str
    role: str = "debater"
    model: str = "mock-model"


@dataclass
class LeaderboardEntry:
    agent_name: str
    elo: float


def make_protocol(
    judge_selection: str = "random",
    judge_termination: bool = False,
    min_rounds_before_judge_check: int = 2,
) -> MagicMock:
    proto = MagicMock()
    proto.judge_selection = judge_selection
    proto.judge_termination = judge_termination
    proto.min_rounds_before_judge_check = min_rounds_before_judge_check
    return proto


def make_elo_system(leaderboard=None, rating_elo=1000.0):
    elo = MagicMock()
    if leaderboard is not None:
        elo.get_leaderboard.return_value = leaderboard
    rating = MagicMock()
    rating.elo = rating_elo
    elo.get_rating.return_value = rating
    return elo


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------


class TestInit:
    """JudgmentPhase stores constructor arguments."""

    def test_stores_protocol(self):
        proto = make_protocol()
        agents = [MockAgent("a1")]
        jp = JudgmentPhase(proto, agents)
        assert jp.protocol is proto

    def test_stores_agents(self):
        proto = make_protocol()
        agents = [MockAgent("a1"), MockAgent("a2")]
        jp = JudgmentPhase(proto, agents)
        assert jp.agents is agents

    def test_stores_elo_system(self):
        proto = make_protocol()
        elo = make_elo_system()
        jp = JudgmentPhase(proto, [], elo_system=elo)
        assert jp.elo_system is elo

    def test_elo_system_defaults_to_none(self):
        jp = JudgmentPhase(make_protocol(), [])
        assert jp.elo_system is None

    def test_stores_calibration_weight_fn(self):
        fn = lambda name: 0.9
        jp = JudgmentPhase(make_protocol(), [], calibration_weight_fn=fn)
        assert jp._get_calibration_weight is fn

    def test_stores_composite_score_fn(self):
        fn = lambda name: 1.5
        jp = JudgmentPhase(make_protocol(), [], composite_score_fn=fn)
        assert jp._compute_composite_score is fn

    def test_optional_fns_default_to_none(self):
        jp = JudgmentPhase(make_protocol(), [])
        assert jp._get_calibration_weight is None
        assert jp._compute_composite_score is None


# ---------------------------------------------------------------------------
# TestRequireAgents
# ---------------------------------------------------------------------------


class TestRequireAgents:
    """_require_agents raises ValueError on empty agent list."""

    def test_raises_on_empty(self):
        jp = JudgmentPhase(make_protocol(), [])
        with pytest.raises(ValueError, match="No agents available"):
            jp._require_agents()

    def test_returns_agents_when_non_empty(self):
        agents = [MockAgent("a1"), MockAgent("a2")]
        jp = JudgmentPhase(make_protocol(), agents)
        result = jp._require_agents()
        assert result is agents


# ---------------------------------------------------------------------------
# TestSelectJudge — selection strategies
# ---------------------------------------------------------------------------


class TestSelectJudgeLast:
    """select_judge with 'last' selection."""

    def test_returns_synthesizer_when_present(self):
        proto = make_protocol(judge_selection="last")
        agents = [MockAgent("a1"), MockAgent("synth", role="synthesizer"), MockAgent("a3")]
        jp = JudgmentPhase(proto, agents)
        judge = jp.select_judge({}, [])
        assert judge.name == "synth"

    def test_returns_last_agent_when_no_synthesizer(self):
        proto = make_protocol(judge_selection="last")
        agents = [MockAgent("a1"), MockAgent("a2"), MockAgent("a3")]
        jp = JudgmentPhase(proto, agents)
        judge = jp.select_judge({}, [])
        assert judge.name == "a3"

    def test_single_agent_without_synthesizer(self):
        proto = make_protocol(judge_selection="last")
        agents = [MockAgent("solo")]
        jp = JudgmentPhase(proto, agents)
        judge = jp.select_judge({}, [])
        assert judge.name == "solo"

    def test_first_synthesizer_chosen_when_multiple(self):
        proto = make_protocol(judge_selection="last")
        agents = [
            MockAgent("a1"),
            MockAgent("synth1", role="synthesizer"),
            MockAgent("synth2", role="synthesizer"),
        ]
        jp = JudgmentPhase(proto, agents)
        judge = jp.select_judge({}, [])
        assert judge.name == "synth1"


class TestSelectJudgeRandom:
    """select_judge with 'random' selection."""

    def test_returns_agent_from_list(self):
        random.seed(42)
        proto = make_protocol(judge_selection="random")
        agents = [MockAgent("a1"), MockAgent("a2"), MockAgent("a3")]
        jp = JudgmentPhase(proto, agents)
        judge = jp.select_judge({}, [])
        assert judge in agents

    def test_raises_on_empty_agents(self):
        proto = make_protocol(judge_selection="random")
        jp = JudgmentPhase(proto, [])
        with pytest.raises(ValueError):
            jp.select_judge({}, [])

    def test_always_returns_only_agent(self):
        proto = make_protocol(judge_selection="random")
        agents = [MockAgent("only")]
        jp = JudgmentPhase(proto, agents)
        for _ in range(5):
            assert jp.select_judge({}, []).name == "only"


class TestSelectJudgeVoted:
    """select_judge with 'voted' selection falls back to random with a warning."""

    def test_falls_back_to_random_without_vote_fn(self):
        random.seed(0)
        proto = make_protocol(judge_selection="voted")
        agents = [MockAgent("a1"), MockAgent("a2")]
        jp = JudgmentPhase(proto, agents)
        judge = jp.select_judge({}, [])
        assert judge in agents

    def test_logs_warning_when_vote_fn_provided(self):
        proto = make_protocol(judge_selection="voted")
        agents = [MockAgent("a1"), MockAgent("a2")]
        jp = JudgmentPhase(proto, agents)
        vote_fn = MagicMock()
        with patch("aragora.debate.phases.judgment.logger") as mock_logger:
            jp.select_judge({}, [], vote_for_judge_fn=vote_fn)
            mock_logger.warning.assert_called_once()
            assert (
                "voted" in mock_logger.warning.call_args[0][0].lower()
                or "async" in mock_logger.warning.call_args[0][0].lower()
            )

    def test_vote_fn_not_called(self):
        proto = make_protocol(judge_selection="voted")
        agents = [MockAgent("a1")]
        jp = JudgmentPhase(proto, agents)
        vote_fn = MagicMock()
        jp.select_judge({}, [], vote_for_judge_fn=vote_fn)
        vote_fn.assert_not_called()


class TestSelectJudgeEloRanked:
    """select_judge with 'elo_ranked' selection."""

    def test_returns_top_elo_agent(self):
        proto = make_protocol(judge_selection="elo_ranked")
        agents = [MockAgent("a1"), MockAgent("a2"), MockAgent("a3")]
        leaderboard = [
            LeaderboardEntry("a2", 1500),
            LeaderboardEntry("a1", 1400),
            LeaderboardEntry("a3", 1300),
        ]
        elo = make_elo_system(leaderboard=leaderboard)
        jp = JudgmentPhase(proto, agents, elo_system=elo)
        judge = jp.select_judge({}, [])
        assert judge.name == "a2"

    def test_skips_non_participant_in_leaderboard(self):
        proto = make_protocol(judge_selection="elo_ranked")
        agents = [MockAgent("a1"), MockAgent("a2")]
        leaderboard = [
            LeaderboardEntry("stranger", 9999),
            LeaderboardEntry("a1", 1400),
        ]
        elo = make_elo_system(leaderboard=leaderboard)
        jp = JudgmentPhase(proto, agents, elo_system=elo)
        judge = jp.select_judge({}, [])
        assert judge.name == "a1"

    def test_falls_back_to_random_without_elo_system(self):
        random.seed(7)
        proto = make_protocol(judge_selection="elo_ranked")
        agents = [MockAgent("a1"), MockAgent("a2")]
        jp = JudgmentPhase(proto, agents, elo_system=None)
        judge = jp.select_judge({}, [])
        assert judge in agents

    def test_falls_back_on_elo_exception(self):
        random.seed(7)
        proto = make_protocol(judge_selection="elo_ranked")
        agents = [MockAgent("a1"), MockAgent("a2")]
        elo = MagicMock()
        elo.get_leaderboard.side_effect = RuntimeError("db down")
        jp = JudgmentPhase(proto, agents, elo_system=elo)
        judge = jp.select_judge({}, [])
        assert judge in agents

    def test_falls_back_when_leaderboard_has_no_participants(self):
        random.seed(3)
        proto = make_protocol(judge_selection="elo_ranked")
        agents = [MockAgent("a1")]
        leaderboard = [LeaderboardEntry("outsider", 2000)]
        elo = make_elo_system(leaderboard=leaderboard)
        jp = JudgmentPhase(proto, agents, elo_system=elo)
        judge = jp.select_judge({}, [])
        assert judge.name == "a1"


class TestSelectJudgeCalibrated:
    """select_judge with 'calibrated' selection."""

    def test_selects_highest_composite_score(self):
        proto = make_protocol(judge_selection="calibrated")
        agents = [MockAgent("low"), MockAgent("high"), MockAgent("mid")]
        scores = {"low": 0.5, "high": 0.9, "mid": 0.7}
        elo = make_elo_system(leaderboard=[])
        jp = JudgmentPhase(
            proto,
            agents,
            elo_system=elo,
            composite_score_fn=lambda name: scores[name],
        )
        judge = jp.select_judge({}, [])
        assert judge.name == "high"

    def test_falls_back_to_random_without_elo_system(self):
        random.seed(1)
        proto = make_protocol(judge_selection="calibrated")
        agents = [MockAgent("a1"), MockAgent("a2")]
        jp = JudgmentPhase(proto, agents, elo_system=None, composite_score_fn=lambda n: 1.0)
        judge = jp.select_judge({}, [])
        assert judge in agents

    def test_falls_back_to_elo_ranked_without_composite_fn(self):
        proto = make_protocol(judge_selection="calibrated")
        agents = [MockAgent("a1"), MockAgent("a2")]
        leaderboard = [
            LeaderboardEntry("a2", 1600),
            LeaderboardEntry("a1", 1400),
        ]
        elo = make_elo_system(leaderboard=leaderboard)
        jp = JudgmentPhase(proto, agents, elo_system=elo, composite_score_fn=None)
        judge = jp.select_judge({}, [])
        assert judge.name == "a2"

    def test_falls_back_to_random_when_all_scores_fail(self):
        random.seed(5)
        proto = make_protocol(judge_selection="calibrated")
        agents = [MockAgent("a1"), MockAgent("a2")]
        elo = make_elo_system(leaderboard=[])
        jp = JudgmentPhase(
            proto,
            agents,
            elo_system=elo,
            composite_score_fn=lambda n: (_ for _ in ()).throw(ValueError("no score")),
        )
        judge = jp.select_judge({}, [])
        assert judge in agents

    def test_skips_failed_agent_scores(self):
        proto = make_protocol(judge_selection="calibrated")
        agents = [MockAgent("good"), MockAgent("bad")]
        elo = make_elo_system(leaderboard=[])

        def score_fn(name):
            if name == "bad":
                raise KeyError("missing")
            return 0.8

        jp = JudgmentPhase(proto, agents, elo_system=elo, composite_score_fn=score_fn)
        judge = jp.select_judge({}, [])
        assert judge.name == "good"


class TestSelectJudgeDefault:
    """select_judge with unknown selection falls back to random."""

    def test_unknown_strategy_uses_random(self):
        random.seed(42)
        proto = make_protocol(judge_selection="unknown_strategy")
        agents = [MockAgent("a1"), MockAgent("a2"), MockAgent("a3")]
        jp = JudgmentPhase(proto, agents)
        judge = jp.select_judge({}, [])
        assert judge in agents


# ---------------------------------------------------------------------------
# TestShouldTerminate
# ---------------------------------------------------------------------------


class TestShouldTerminate:
    """should_terminate returns (continue, reason) tuples."""

    def test_returns_continue_when_judge_termination_disabled(self):
        proto = make_protocol(judge_termination=False)
        jp = JudgmentPhase(proto, [MockAgent("a1")])
        result = jp.should_terminate(5, {}, "Conclusive: Yes\nReason: Clear winner")
        assert result == (True, "")

    def test_returns_continue_before_min_rounds(self):
        proto = make_protocol(judge_termination=True, min_rounds_before_judge_check=3)
        jp = JudgmentPhase(proto, [MockAgent("a1")])
        result = jp.should_terminate(2, {}, "Conclusive: Yes\nReason: done")
        assert result == (True, "")

    def test_returns_continue_at_exactly_min_rounds_minus_one(self):
        proto = make_protocol(judge_termination=True, min_rounds_before_judge_check=5)
        jp = JudgmentPhase(proto, [MockAgent("a1")])
        result = jp.should_terminate(4, {}, "Conclusive: Yes\nReason: done")
        assert result == (True, "")

    def test_returns_continue_when_no_judge_response(self):
        proto = make_protocol(judge_termination=True, min_rounds_before_judge_check=1)
        jp = JudgmentPhase(proto, [MockAgent("a1")])
        result = jp.should_terminate(3, {}, None)
        assert result == (True, "")

    def test_returns_continue_when_judge_response_empty_string(self):
        proto = make_protocol(judge_termination=True, min_rounds_before_judge_check=1)
        jp = JudgmentPhase(proto, [MockAgent("a1")])
        result = jp.should_terminate(3, {}, "")
        assert result == (True, "")

    def test_stops_debate_when_conclusive_yes(self):
        proto = make_protocol(judge_termination=True, min_rounds_before_judge_check=1)
        jp = JudgmentPhase(proto, [MockAgent("a1")])
        response = "Conclusive: Yes\nReason: Consensus reached"
        should_continue, reason = jp.should_terminate(3, {}, response)
        assert should_continue is False
        assert "Consensus reached" in reason

    def test_continues_when_conclusive_no(self):
        proto = make_protocol(judge_termination=True, min_rounds_before_judge_check=1)
        jp = JudgmentPhase(proto, [MockAgent("a1")])
        response = "Conclusive: No\nReason: Still debating"
        result = jp.should_terminate(3, {}, response)
        assert result == (True, "")

    def test_continues_when_conclusive_not_found(self):
        proto = make_protocol(judge_termination=True, min_rounds_before_judge_check=1)
        jp = JudgmentPhase(proto, [MockAgent("a1")])
        response = "This is some response without the keyword"
        result = jp.should_terminate(3, {}, response)
        assert result == (True, "")

    def test_case_insensitive_conclusive_parsing(self):
        proto = make_protocol(judge_termination=True, min_rounds_before_judge_check=1)
        jp = JudgmentPhase(proto, [MockAgent("a1")])
        response = "CONCLUSIVE: YES\nREASON: Done"
        should_continue, reason = jp.should_terminate(3, {}, response)
        assert should_continue is False

    def test_extracts_reason_text(self):
        proto = make_protocol(judge_termination=True, min_rounds_before_judge_check=1)
        jp = JudgmentPhase(proto, [MockAgent("a1")])
        response = "Conclusive: Yes\nReason: The debate has reached a clear verdict"
        _, reason = jp.should_terminate(3, {}, response)
        assert reason == "The debate has reached a clear verdict"

    def test_reason_empty_when_conclusive_no(self):
        proto = make_protocol(judge_termination=True, min_rounds_before_judge_check=1)
        jp = JudgmentPhase(proto, [MockAgent("a1")])
        response = "Conclusive: No"
        should_continue, reason = jp.should_terminate(3, {}, response)
        assert should_continue is True
        assert reason == ""

    def test_at_min_rounds_boundary_checks_response(self):
        proto = make_protocol(judge_termination=True, min_rounds_before_judge_check=3)
        jp = JudgmentPhase(proto, [MockAgent("a1")])
        response = "Conclusive: Yes\nReason: boundary test"
        should_continue, reason = jp.should_terminate(3, {}, response)
        assert should_continue is False
        assert "boundary test" in reason


# ---------------------------------------------------------------------------
# TestGetJudgeStats
# ---------------------------------------------------------------------------


class TestGetJudgeStats:
    """get_judge_stats returns correct dict."""

    def test_returns_basic_stats(self):
        proto = make_protocol(judge_selection="random")
        agent = MockAgent("judge_agent", role="synthesizer")
        jp = JudgmentPhase(proto, [agent])
        stats = jp.get_judge_stats(agent)
        assert stats["name"] == "judge_agent"
        assert stats["role"] == "synthesizer"
        assert stats["selection_method"] == "random"

    def test_includes_elo_when_elo_system_present(self):
        proto = make_protocol(judge_selection="elo_ranked")
        agent = MockAgent("a1")
        elo = make_elo_system(rating_elo=1750.0)
        jp = JudgmentPhase(proto, [agent], elo_system=elo)
        stats = jp.get_judge_stats(agent)
        assert "elo" in stats
        assert stats["elo"] == 1750.0

    def test_excludes_elo_when_no_elo_system(self):
        proto = make_protocol()
        agent = MockAgent("a1")
        jp = JudgmentPhase(proto, [agent])
        stats = jp.get_judge_stats(agent)
        assert "elo" not in stats

    def test_elo_missing_when_get_rating_raises_key_error(self):
        proto = make_protocol(judge_selection="elo_ranked")
        agent = MockAgent("a1")
        elo = MagicMock()
        elo.get_rating.side_effect = KeyError("a1")
        jp = JudgmentPhase(proto, [agent], elo_system=elo)
        stats = jp.get_judge_stats(agent)
        assert "elo" not in stats

    def test_includes_calibration_weight_when_fn_provided(self):
        proto = make_protocol(judge_selection="calibrated")
        agent = MockAgent("a1")
        jp = JudgmentPhase(proto, [agent], calibration_weight_fn=lambda name: 0.85)
        stats = jp.get_judge_stats(agent)
        assert "calibration_weight" in stats
        assert stats["calibration_weight"] == pytest.approx(0.85)

    def test_excludes_calibration_weight_when_no_fn(self):
        proto = make_protocol()
        agent = MockAgent("a1")
        jp = JudgmentPhase(proto, [agent])
        stats = jp.get_judge_stats(agent)
        assert "calibration_weight" not in stats

    def test_calibration_weight_missing_when_fn_raises(self):
        proto = make_protocol()
        agent = MockAgent("a1")

        def bad_weight(name):
            raise AttributeError("not found")

        jp = JudgmentPhase(proto, [agent], calibration_weight_fn=bad_weight)
        stats = jp.get_judge_stats(agent)
        assert "calibration_weight" not in stats

    def test_both_elo_and_calibration_weight_included(self):
        proto = make_protocol(judge_selection="calibrated")
        agent = MockAgent("a1")
        elo = make_elo_system(rating_elo=1200.0)
        jp = JudgmentPhase(
            proto,
            [agent],
            elo_system=elo,
            calibration_weight_fn=lambda name: 0.75,
        )
        stats = jp.get_judge_stats(agent)
        assert stats["elo"] == 1200.0
        assert stats["calibration_weight"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# TestSelectLast (unit-level)
# ---------------------------------------------------------------------------


class TestSelectLast:
    """_select_last selects synthesizer or last agent."""

    def test_prefers_synthesizer_over_last(self):
        proto = make_protocol()
        agents = [MockAgent("first"), MockAgent("synth", role="synthesizer"), MockAgent("last")]
        jp = JudgmentPhase(proto, agents)
        result = jp._select_last()
        assert result.name == "synth"

    def test_returns_last_when_no_synthesizer(self):
        proto = make_protocol()
        agents = [MockAgent("a"), MockAgent("b"), MockAgent("c")]
        jp = JudgmentPhase(proto, agents)
        result = jp._select_last()
        assert result.name == "c"


# ---------------------------------------------------------------------------
# TestSelectEloRanked (unit-level)
# ---------------------------------------------------------------------------


class TestSelectEloRanked:
    """_select_elo_ranked returns highest-ELO agent."""

    def test_logs_warning_without_elo_system(self):
        proto = make_protocol()
        agents = [MockAgent("a1")]
        jp = JudgmentPhase(proto, agents, elo_system=None)
        with patch("aragora.debate.phases.judgment.logger") as mock_logger:
            jp._select_elo_ranked()
            mock_logger.warning.assert_called()

    def test_returns_top_ranked_participant(self):
        proto = make_protocol()
        agents = [MockAgent("a1"), MockAgent("a2")]
        leaderboard = [
            LeaderboardEntry("a2", 1800),
            LeaderboardEntry("a1", 1600),
        ]
        elo = make_elo_system(leaderboard=leaderboard)
        jp = JudgmentPhase(proto, agents, elo_system=elo)
        result = jp._select_elo_ranked()
        assert result.name == "a2"


# ---------------------------------------------------------------------------
# TestSelectCalibrated (unit-level)
# ---------------------------------------------------------------------------


class TestSelectCalibrated:
    """_select_calibrated returns highest composite-score agent."""

    def test_returns_highest_scoring_agent(self):
        proto = make_protocol()
        agents = [MockAgent("low"), MockAgent("high")]
        elo = make_elo_system(leaderboard=[])
        scores = {"low": 0.3, "high": 0.8}
        jp = JudgmentPhase(proto, agents, elo_system=elo, composite_score_fn=lambda n: scores[n])
        result = jp._select_calibrated()
        assert result.name == "high"

    def test_falls_back_to_elo_ranked_when_no_composite_fn(self):
        proto = make_protocol()
        agents = [MockAgent("a1"), MockAgent("a2")]
        leaderboard = [LeaderboardEntry("a1", 1500), LeaderboardEntry("a2", 1000)]
        elo = make_elo_system(leaderboard=leaderboard)
        jp = JudgmentPhase(proto, agents, elo_system=elo, composite_score_fn=None)
        result = jp._select_calibrated()
        assert result.name == "a1"
