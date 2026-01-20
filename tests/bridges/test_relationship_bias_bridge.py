"""Tests for RelationshipBiasBridge."""

from __future__ import annotations

import pytest
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from aragora.debate.relationship_bias_bridge import (
    RelationshipBiasBridge,
    RelationshipBiasBridgeConfig,
    EchoChamberRisk,
    create_relationship_bias_bridge,
)


@dataclass
class MockRelationshipMetrics:
    """Mock relationship metrics for testing."""

    agent_a: str = ""
    agent_b: str = ""
    debate_count: int = 10
    agreement_rate: float = 0.5
    alliance_score: float = 0.3
    rivalry_score: float = 0.2


@dataclass
class MockVote:
    """Mock vote for testing."""

    agent: str = "voter"
    choice: str = "choice"


class MockRelationshipTracker:
    """Mock relationship tracker."""

    def __init__(self):
        self._metrics: Dict[tuple, MockRelationshipMetrics] = {}

    def compute_metrics(self, agent_a: str, agent_b: str) -> MockRelationshipMetrics:
        """Compute relationship metrics."""
        # Normalize key order
        if agent_a > agent_b:
            agent_a, agent_b = agent_b, agent_a
        key = (agent_a, agent_b)
        return self._metrics.get(
            key,
            MockRelationshipMetrics(agent_a=agent_a, agent_b=agent_b),
        )

    def set_metrics(
        self, agent_a: str, agent_b: str, metrics: MockRelationshipMetrics
    ) -> None:
        """Set metrics for a pair."""
        if agent_a > agent_b:
            agent_a, agent_b = agent_b, agent_a
        self._metrics[(agent_a, agent_b)] = metrics


class TestEchoChamberRisk:
    """Tests for EchoChamberRisk dataclass."""

    def test_defaults(self):
        """Test default values."""
        risk = EchoChamberRisk(
            team=["a", "b"],
            overall_risk=0.5,
            high_alliance_pairs=[],
            agreement_stats={},
            recommendation="caution",
        )
        assert risk.overall_risk == 0.5
        assert risk.recommendation == "caution"


class TestRelationshipBiasBridge:
    """Tests for RelationshipBiasBridge."""

    def test_create_bridge(self):
        """Test bridge creation."""
        bridge = RelationshipBiasBridge()
        assert bridge.relationship_tracker is None
        assert bridge.config is not None

    def test_create_with_config(self):
        """Test bridge creation with custom config."""
        config = RelationshipBiasBridgeConfig(
            echo_chamber_alliance_threshold=0.8,
            alliance_vote_penalty=0.4,
        )
        bridge = RelationshipBiasBridge(config=config)
        assert bridge.config.echo_chamber_alliance_threshold == 0.8
        assert bridge.config.alliance_vote_penalty == 0.4

    def test_compute_team_echo_risk_single_agent(self):
        """Test echo risk for single agent team."""
        bridge = RelationshipBiasBridge()
        risk = bridge.compute_team_echo_risk(["claude"])

        assert risk.overall_risk == 0.0
        assert risk.recommendation == "safe"

    def test_compute_team_echo_risk_no_data(self):
        """Test echo risk with no relationship data."""
        tracker = MockRelationshipTracker()
        # Set low debate count
        tracker.set_metrics(
            "claude",
            "gpt-4",
            MockRelationshipMetrics(
                agent_a="claude", agent_b="gpt-4", debate_count=1
            ),
        )

        bridge = RelationshipBiasBridge(
            relationship_tracker=tracker,
            config=RelationshipBiasBridgeConfig(min_debates_for_relationship=5),
        )

        risk = bridge.compute_team_echo_risk(["claude", "gpt-4"])
        assert risk.overall_risk == 0.0

    def test_compute_team_echo_risk_high_alliance(self):
        """Test echo risk with high alliance pairs."""
        tracker = MockRelationshipTracker()
        tracker.set_metrics(
            "claude",
            "gpt-4",
            MockRelationshipMetrics(
                agent_a="claude",
                agent_b="gpt-4",
                debate_count=20,
                alliance_score=0.85,
                agreement_rate=0.9,
            ),
        )

        bridge = RelationshipBiasBridge(
            relationship_tracker=tracker,
            config=RelationshipBiasBridgeConfig(
                echo_chamber_alliance_threshold=0.7,
                high_risk_threshold=0.6,
            ),
        )

        risk = bridge.compute_team_echo_risk(["claude", "gpt-4"])

        assert risk.overall_risk > 0.5
        assert len(risk.high_alliance_pairs) > 0
        assert risk.recommendation == "high_risk"

    def test_compute_team_echo_risk_safe_team(self):
        """Test echo risk for diverse team."""
        tracker = MockRelationshipTracker()
        tracker.set_metrics(
            "claude",
            "gpt-4",
            MockRelationshipMetrics(
                agent_a="claude",
                agent_b="gpt-4",
                debate_count=20,
                alliance_score=0.3,
                agreement_rate=0.4,
            ),
        )

        bridge = RelationshipBiasBridge(
            relationship_tracker=tracker,
            config=RelationshipBiasBridgeConfig(caution_threshold=0.5),
        )

        risk = bridge.compute_team_echo_risk(["claude", "gpt-4"])

        assert risk.overall_risk < 0.5
        assert risk.recommendation == "safe"

    def test_compute_vote_weight_adjustments(self):
        """Test vote weight adjustments."""
        tracker = MockRelationshipTracker()
        tracker.set_metrics(
            "ally1",
            "ally2",
            MockRelationshipMetrics(
                agent_a="ally1",
                agent_b="ally2",
                debate_count=20,
                alliance_score=0.9,
            ),
        )

        bridge = RelationshipBiasBridge(
            relationship_tracker=tracker,
            config=RelationshipBiasBridgeConfig(
                echo_chamber_alliance_threshold=0.7,
                alliance_vote_penalty=0.3,
            ),
        )

        votes = [MockVote(agent="ally1", choice="ally2")]
        proposals = {"ally2": "Some proposal"}

        weights = bridge.compute_vote_weight_adjustments(votes, proposals)

        assert "ally1" in weights
        assert weights["ally1"] < 1.0  # Penalized

    def test_compute_vote_weight_no_adjustment(self):
        """Test vote weights with no relationship."""
        bridge = RelationshipBiasBridge()

        votes = [MockVote(agent="voter", choice="option1")]
        proposals = {"agent1": "Proposal 1"}

        weights = bridge.compute_vote_weight_adjustments(votes, proposals)

        assert weights["voter"] == 1.0  # No adjustment

    def test_compute_vote_weight_disabled(self):
        """Test vote weights when adjustments disabled."""
        bridge = RelationshipBiasBridge(
            config=RelationshipBiasBridgeConfig(apply_vote_adjustments=False)
        )

        votes = [MockVote(agent="voter", choice="choice")]
        proposals = {}

        weights = bridge.compute_vote_weight_adjustments(votes, proposals)

        assert weights["voter"] == 1.0

    def test_get_diverse_team_candidates(self):
        """Test getting diverse team candidates."""
        tracker = MockRelationshipTracker()
        # Set high alliance between a and b
        tracker.set_metrics(
            "a",
            "b",
            MockRelationshipMetrics(
                agent_a="a",
                agent_b="b",
                debate_count=20,
                alliance_score=0.9,
                agreement_rate=0.9,
            ),
        )
        # Set low alliance between a and c
        tracker.set_metrics(
            "a",
            "c",
            MockRelationshipMetrics(
                agent_a="a",
                agent_b="c",
                debate_count=20,
                alliance_score=0.2,
                agreement_rate=0.3,
            ),
        )

        bridge = RelationshipBiasBridge(relationship_tracker=tracker)

        candidates = bridge.get_diverse_team_candidates(
            available_agents=["a", "b", "c"],
            team_size=2,
        )

        # Should prefer a+c over a+b due to lower alliance
        assert len(candidates) > 0
        # First candidate should be most diverse
        if len(candidates) > 1:
            first_team = set(candidates[0])
            # a+c should be preferred over a+b
            if "a" in first_team:
                assert "c" in first_team or "b" not in first_team

    def test_get_echo_chamber_pairs(self):
        """Test getting echo chamber pairs."""
        bridge = RelationshipBiasBridge(
            config=RelationshipBiasBridgeConfig(
                min_debates_for_relationship=5,
                caution_threshold=0.4,
            )
        )

        # Add high-risk pair to cache
        bridge._metrics_cache[("a", "b")] = MockRelationshipMetrics(
            agent_a="a",
            agent_b="b",
            debate_count=20,
            alliance_score=0.8,
            agreement_rate=0.9,
        )
        # Add low-risk pair
        bridge._metrics_cache[("c", "d")] = MockRelationshipMetrics(
            agent_a="c",
            agent_b="d",
            debate_count=20,
            alliance_score=0.2,
            agreement_rate=0.2,
        )

        pairs = bridge.get_echo_chamber_pairs()

        assert len(pairs) >= 1
        # High-risk pair should be first
        assert pairs[0][0:2] == ("a", "b")

    def test_compute_diversity_score(self):
        """Test computing diversity score."""
        tracker = MockRelationshipTracker()
        # Diverse pair
        tracker.set_metrics(
            "diverse1",
            "diverse2",
            MockRelationshipMetrics(
                agent_a="diverse1",
                agent_b="diverse2",
                debate_count=20,
                agreement_rate=0.3,
                rivalry_score=0.5,
                alliance_score=0.1,
            ),
        )
        # Homogeneous pair
        tracker.set_metrics(
            "echo1",
            "echo2",
            MockRelationshipMetrics(
                agent_a="echo1",
                agent_b="echo2",
                debate_count=20,
                agreement_rate=0.9,
                alliance_score=0.8,
            ),
        )

        bridge = RelationshipBiasBridge(relationship_tracker=tracker)

        diverse_score = bridge.compute_diversity_score(["diverse1", "diverse2"])
        echo_score = bridge.compute_diversity_score(["echo1", "echo2"])

        assert diverse_score > echo_score

    def test_compute_diversity_score_single_agent(self):
        """Test diversity score for single agent."""
        bridge = RelationshipBiasBridge()
        score = bridge.compute_diversity_score(["single"])
        assert score == 1.0

    def test_refresh_cache(self):
        """Test refreshing cache."""
        tracker = MockRelationshipTracker()
        bridge = RelationshipBiasBridge(relationship_tracker=tracker)

        cached = bridge.refresh_cache(["a", "b", "c"])
        assert cached >= 0

    def test_clear_cache(self):
        """Test clearing cache."""
        bridge = RelationshipBiasBridge()
        bridge._metrics_cache[("a", "b")] = MockRelationshipMetrics()

        bridge.clear_cache()
        assert len(bridge._metrics_cache) == 0

    def test_get_stats(self):
        """Test getting bridge stats."""
        bridge = RelationshipBiasBridge()
        bridge._metrics_cache[("a", "b")] = MockRelationshipMetrics()

        stats = bridge.get_stats()
        assert "relationships_cached" in stats
        assert stats["relationships_cached"] == 1

    def test_factory_function(self):
        """Test factory function."""
        tracker = MockRelationshipTracker()
        bridge = create_relationship_bias_bridge(
            relationship_tracker=tracker,
            echo_chamber_alliance_threshold=0.8,
        )
        assert bridge.relationship_tracker is tracker
        assert bridge.config.echo_chamber_alliance_threshold == 0.8
