"""Tests for the relationship-to-bias mitigation bridge.

Covers EchoChamberRisk, RelationshipBiasBridgeConfig, RelationshipBiasBridge
(echo risk, vote weight adjustments, diverse teams, echo chamber pairs,
diversity score, cache), and create_relationship_bias_bridge factory.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aragora.debate.relationship_bias_bridge import (
    EchoChamberRisk,
    RelationshipBiasBridge,
    RelationshipBiasBridgeConfig,
    create_relationship_bias_bridge,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_metrics(
    alliance_score=0.5,
    agreement_rate=0.5,
    debate_count=10,
    rivalry_score=0.3,
):
    m = MagicMock()
    m.alliance_score = alliance_score
    m.agreement_rate = agreement_rate
    m.debate_count = debate_count
    m.rivalry_score = rivalry_score
    return m


def make_tracker(metrics_map=None):
    """Create mock tracker. metrics_map: {(a,b): metrics}"""
    tracker = MagicMock()

    def compute_metrics(a, b):
        # Normalize key order
        key = (a, b) if a <= b else (b, a)
        if metrics_map and key in metrics_map:
            return metrics_map[key]
        return None

    tracker.compute_metrics = compute_metrics
    return tracker


def make_vote(agent, choice):
    v = MagicMock()
    v.agent = agent
    v.choice = choice
    return v


# ---------------------------------------------------------------------------
# EchoChamberRisk
# ---------------------------------------------------------------------------


class TestEchoChamberRisk:
    def test_fields(self):
        risk = EchoChamberRisk(
            team=["claude", "gpt"],
            overall_risk=0.5,
            high_alliance_pairs=[("claude", "gpt")],
            agreement_stats={"claude:gpt": 0.8},
            recommendation="caution",
        )
        assert risk.team == ["claude", "gpt"]
        assert risk.overall_risk == 0.5
        assert len(risk.high_alliance_pairs) == 1
        assert risk.recommendation == "caution"


# ---------------------------------------------------------------------------
# RelationshipBiasBridgeConfig
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self):
        cfg = RelationshipBiasBridgeConfig()
        assert cfg.echo_chamber_alliance_threshold == 0.7
        assert cfg.echo_chamber_agreement_threshold == 0.8
        assert cfg.alliance_vote_penalty == 0.3
        assert cfg.min_debates_for_relationship == 5
        assert cfg.high_risk_threshold == 0.6
        assert cfg.caution_threshold == 0.3
        assert cfg.apply_vote_adjustments is True
        assert cfg.diversity_bonus == 0.1


# ---------------------------------------------------------------------------
# compute_team_echo_risk
# ---------------------------------------------------------------------------


class TestTeamEchoRisk:
    def test_single_agent_safe(self):
        bridge = RelationshipBiasBridge()
        risk = bridge.compute_team_echo_risk(["claude"])
        assert risk.overall_risk == 0.0
        assert risk.recommendation == "safe"

    def test_empty_team_safe(self):
        bridge = RelationshipBiasBridge()
        risk = bridge.compute_team_echo_risk([])
        assert risk.overall_risk == 0.0
        assert risk.recommendation == "safe"

    def test_no_tracker_safe(self):
        bridge = RelationshipBiasBridge()
        risk = bridge.compute_team_echo_risk(["claude", "gpt"])
        assert risk.overall_risk == 0.0
        assert risk.recommendation == "safe"

    def test_high_alliance_detected(self):
        tracker = make_tracker(
            {
                ("claude", "gpt"): make_metrics(alliance_score=0.9, agreement_rate=0.9),
            }
        )
        bridge = RelationshipBiasBridge(relationship_tracker=tracker)
        risk = bridge.compute_team_echo_risk(["claude", "gpt"])
        assert risk.overall_risk > 0.0
        assert len(risk.high_alliance_pairs) > 0

    def test_high_risk_recommendation(self):
        tracker = make_tracker(
            {
                ("claude", "gpt"): make_metrics(alliance_score=0.95, agreement_rate=0.95),
            }
        )
        bridge = RelationshipBiasBridge(relationship_tracker=tracker)
        risk = bridge.compute_team_echo_risk(["claude", "gpt"])
        assert risk.recommendation in ("high_risk", "caution")

    def test_low_alliance_safe(self):
        tracker = make_tracker(
            {
                ("claude", "gpt"): make_metrics(alliance_score=0.1, agreement_rate=0.2),
            }
        )
        bridge = RelationshipBiasBridge(relationship_tracker=tracker)
        risk = bridge.compute_team_echo_risk(["claude", "gpt"])
        assert risk.recommendation == "safe"

    def test_below_min_debates_ignored(self):
        tracker = make_tracker(
            {
                ("claude", "gpt"): make_metrics(
                    alliance_score=0.95, agreement_rate=0.95, debate_count=2
                ),
            }
        )
        bridge = RelationshipBiasBridge(relationship_tracker=tracker)
        risk = bridge.compute_team_echo_risk(["claude", "gpt"])
        assert risk.overall_risk == 0.0  # Not enough data

    def test_three_agent_team(self):
        tracker = make_tracker(
            {
                ("claude", "gemini"): make_metrics(alliance_score=0.8, agreement_rate=0.85),
                ("claude", "gpt"): make_metrics(alliance_score=0.2, agreement_rate=0.3),
                ("gemini", "gpt"): make_metrics(alliance_score=0.15, agreement_rate=0.25),
            }
        )
        bridge = RelationshipBiasBridge(relationship_tracker=tracker)
        risk = bridge.compute_team_echo_risk(["claude", "gpt", "gemini"])
        # Only claude-gemini pair is high risk
        assert risk.overall_risk > 0.0
        assert len(risk.agreement_stats) == 3


# ---------------------------------------------------------------------------
# compute_vote_weight_adjustments
# ---------------------------------------------------------------------------


class TestVoteWeightAdjustments:
    def test_adjustments_disabled(self):
        bridge = RelationshipBiasBridge(
            config=RelationshipBiasBridgeConfig(apply_vote_adjustments=False)
        )
        votes = [make_vote("claude", "gpt")]
        adj = bridge.compute_vote_weight_adjustments(votes, {"gpt": "proposal"})
        assert adj["claude"] == 1.0

    def test_no_tracker_all_neutral(self):
        bridge = RelationshipBiasBridge()
        votes = [make_vote("claude", "gpt")]
        adj = bridge.compute_vote_weight_adjustments(votes, {"gpt": "proposal"})
        assert adj["claude"] == 1.0

    def test_vote_for_non_agent_neutral(self):
        tracker = make_tracker({})
        bridge = RelationshipBiasBridge(relationship_tracker=tracker)
        votes = [make_vote("claude", "option_a")]
        adj = bridge.compute_vote_weight_adjustments(votes, {"gpt": "proposal"})
        assert adj["claude"] == 1.0

    def test_allied_vote_penalized(self):
        tracker = make_tracker(
            {
                ("claude", "gpt"): make_metrics(alliance_score=0.9, agreement_rate=0.9),
            }
        )
        bridge = RelationshipBiasBridge(relationship_tracker=tracker)
        votes = [make_vote("claude", "gpt")]
        adj = bridge.compute_vote_weight_adjustments(votes, {"gpt": "proposal"})
        assert adj["claude"] < 1.0

    def test_non_allied_vote_neutral(self):
        tracker = make_tracker(
            {
                ("claude", "gpt"): make_metrics(alliance_score=0.2, agreement_rate=0.3),
            }
        )
        bridge = RelationshipBiasBridge(relationship_tracker=tracker)
        votes = [make_vote("claude", "gpt")]
        adj = bridge.compute_vote_weight_adjustments(votes, {"gpt": "proposal"})
        assert adj["claude"] == 1.0

    def test_below_min_debates_neutral(self):
        tracker = make_tracker(
            {
                ("claude", "gpt"): make_metrics(alliance_score=0.95, debate_count=2),
            }
        )
        bridge = RelationshipBiasBridge(relationship_tracker=tracker)
        votes = [make_vote("claude", "gpt")]
        adj = bridge.compute_vote_weight_adjustments(votes, {"gpt": "proposal"})
        assert adj["claude"] == 1.0


# ---------------------------------------------------------------------------
# get_diverse_team_candidates
# ---------------------------------------------------------------------------


class TestDiverseTeamCandidates:
    def test_required_agents_fill_team(self):
        bridge = RelationshipBiasBridge()
        result = bridge.get_diverse_team_candidates(
            ["a", "b", "c"], team_size=2, required_agents=["a", "b", "c"]
        )
        assert len(result) == 1
        assert len(result[0]) == 2

    def test_not_enough_agents(self):
        bridge = RelationshipBiasBridge()
        result = bridge.get_diverse_team_candidates(["a"], team_size=3)
        assert len(result) == 1
        assert "a" in result[0]

    def test_sorted_by_diversity(self):
        tracker = make_tracker(
            {
                ("a", "b"): make_metrics(alliance_score=0.9, agreement_rate=0.9),
                ("a", "c"): make_metrics(alliance_score=0.1, agreement_rate=0.2),
                ("b", "c"): make_metrics(alliance_score=0.1, agreement_rate=0.2),
            }
        )
        bridge = RelationshipBiasBridge(relationship_tracker=tracker)
        result = bridge.get_diverse_team_candidates(["a", "b", "c"], team_size=2)
        # Most diverse pair (a,c or b,c) should come first
        assert len(result) >= 1
        first_team = result[0]
        # a,b should not be first (highest risk pair)
        assert not (set(first_team) == {"a", "b"})


# ---------------------------------------------------------------------------
# get_echo_chamber_pairs
# ---------------------------------------------------------------------------


class TestEchoChamberPairs:
    def test_no_pairs_no_tracker(self):
        bridge = RelationshipBiasBridge()
        pairs = bridge.get_echo_chamber_pairs(agents=["a", "b"])
        assert pairs == []

    def test_high_risk_pair_detected(self):
        tracker = make_tracker(
            {
                ("a", "b"): make_metrics(alliance_score=0.8, agreement_rate=0.85),
            }
        )
        bridge = RelationshipBiasBridge(relationship_tracker=tracker)
        pairs = bridge.get_echo_chamber_pairs(agents=["a", "b"])
        assert len(pairs) > 0
        assert pairs[0][2] > 0.3  # combined_risk > caution_threshold

    def test_uses_cached_metrics(self):
        bridge = RelationshipBiasBridge()
        # Manually populate cache
        metrics = make_metrics(alliance_score=0.8, agreement_rate=0.85, debate_count=10)
        bridge._metrics_cache[("a", "b")] = metrics
        pairs = bridge.get_echo_chamber_pairs()
        assert len(pairs) > 0

    def test_sorted_by_risk_descending(self):
        bridge = RelationshipBiasBridge()
        bridge._metrics_cache[("a", "b")] = make_metrics(
            alliance_score=0.5, agreement_rate=0.5, debate_count=10
        )
        bridge._metrics_cache[("c", "d")] = make_metrics(
            alliance_score=0.9, agreement_rate=0.9, debate_count=10
        )
        pairs = bridge.get_echo_chamber_pairs()
        if len(pairs) >= 2:
            assert pairs[0][2] >= pairs[1][2]


# ---------------------------------------------------------------------------
# compute_diversity_score
# ---------------------------------------------------------------------------


class TestDiversityScore:
    def test_single_agent_max_diverse(self):
        bridge = RelationshipBiasBridge()
        assert bridge.compute_diversity_score(["claude"]) == 1.0

    def test_unknown_relationships_moderate(self):
        bridge = RelationshipBiasBridge()
        score = bridge.compute_diversity_score(["a", "b", "c"])
        assert score == 0.5  # No tracker → unknown → 0.5

    def test_high_agreement_low_diversity(self):
        tracker = make_tracker(
            {
                ("a", "b"): make_metrics(alliance_score=0.3, agreement_rate=0.9, rivalry_score=0.1),
            }
        )
        bridge = RelationshipBiasBridge(relationship_tracker=tracker)
        score = bridge.compute_diversity_score(["a", "b"])
        assert score < 0.5

    def test_rival_pair_high_diversity(self):
        tracker = make_tracker(
            {
                ("a", "b"): make_metrics(alliance_score=0.1, agreement_rate=0.2, rivalry_score=0.8),
            }
        )
        bridge = RelationshipBiasBridge(relationship_tracker=tracker)
        score = bridge.compute_diversity_score(["a", "b"])
        assert score > 0.7


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------


class TestCacheManagement:
    def test_refresh_cache(self):
        tracker = make_tracker(
            {
                ("a", "b"): make_metrics(),
                ("a", "c"): make_metrics(),
                ("b", "c"): make_metrics(),
            }
        )
        bridge = RelationshipBiasBridge(relationship_tracker=tracker)
        cached = bridge.refresh_cache(["a", "b", "c"])
        assert cached == 3

    def test_clear_cache(self):
        bridge = RelationshipBiasBridge()
        bridge._metrics_cache[("a", "b")] = make_metrics()
        bridge.clear_cache()
        assert len(bridge._metrics_cache) == 0


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_empty_stats(self):
        bridge = RelationshipBiasBridge()
        stats = bridge.get_stats()
        assert stats["relationships_cached"] == 0
        assert stats["echo_chamber_pairs"] == 0
        assert stats["apply_vote_adjustments"] is True

    def test_stats_after_caching(self):
        bridge = RelationshipBiasBridge()
        bridge._metrics_cache[("a", "b")] = make_metrics(
            alliance_score=0.8, agreement_rate=0.85, debate_count=10
        )
        stats = bridge.get_stats()
        assert stats["relationships_cached"] == 1
        assert stats["echo_chamber_pairs"] >= 1


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestFactory:
    def test_create_default(self):
        bridge = create_relationship_bias_bridge()
        assert isinstance(bridge, RelationshipBiasBridge)
        assert bridge.relationship_tracker is None

    def test_create_with_config(self):
        bridge = create_relationship_bias_bridge(
            echo_chamber_alliance_threshold=0.9,
            alliance_vote_penalty=0.5,
        )
        assert bridge.config.echo_chamber_alliance_threshold == 0.9
        assert bridge.config.alliance_vote_penalty == 0.5

    def test_create_with_tracker(self):
        tracker = MagicMock()
        bridge = create_relationship_bias_bridge(relationship_tracker=tracker)
        assert bridge.relationship_tracker is tracker
