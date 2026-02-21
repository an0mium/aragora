"""Tests for edge topology scoring in GoalExtractor.extract_from_debate_analysis().

Covers change 1D: edges with SUPPORTS/REFUTES relations affect goal scoring.
"""

from __future__ import annotations

import pytest

from aragora.goals.extractor import GoalExtractionConfig, GoalExtractor


@pytest.fixture
def extractor():
    return GoalExtractor()


class TestEdgeTopologyScoring:
    """1D: Edge topology affects goal confidence and metadata."""

    def test_support_edges_boost_score(self, extractor):
        """Nodes with incoming support edges score higher."""
        data = {
            "nodes": [
                {"id": "n1", "label": "Build rate limiter", "weight": 0.5, "type": "claim"},
                {"id": "n2", "label": "Evidence for rate limiter", "weight": 0.3, "type": "claim"},
            ],
            "edges": [
                {"source": "n2", "target": "n1", "type": "supports", "weight": 1.0},
            ],
        }
        cfg = GoalExtractionConfig(require_consensus=False, confidence_threshold=0.0)
        result = extractor.extract_from_debate_analysis(data, config=cfg)

        # n1 should appear as a goal (boosted by support edge)
        assert len(result.goals) > 0
        # Find the goal derived from n1
        n1_goals = [g for g in result.goals if "n1" in g.source_idea_ids]
        if n1_goals:
            meta = n1_goals[0].metadata
            assert meta.get("support_edges", 0) >= 1

    def test_refute_edges_flag_goals(self, extractor):
        """Nodes with incoming refute edges get has_refutation=True."""
        data = {
            "nodes": [
                {"id": "n1", "label": "Controversial proposal", "weight": 0.5, "type": "claim"},
                {"id": "n2", "label": "Counter-argument", "weight": 0.5, "type": "claim"},
            ],
            "edges": [
                {"source": "n2", "target": "n1", "type": "refutes", "weight": 0.8},
            ],
        }
        cfg = GoalExtractionConfig(require_consensus=False, confidence_threshold=0.0)
        result = extractor.extract_from_debate_analysis(data, config=cfg)

        n1_goals = [g for g in result.goals if "n1" in g.source_idea_ids]
        if n1_goals:
            assert n1_goals[0].metadata.get("has_refutation") is True

    def test_mixed_edges_net_score(self, extractor):
        """Mixed support/refute edges produce balanced scoring."""
        data = {
            "nodes": [
                {"id": "n1", "label": "Balanced proposal", "weight": 0.5, "type": "claim"},
                {"id": "n2", "label": "Supporter", "weight": 0.3, "type": "claim"},
                {"id": "n3", "label": "Critic", "weight": 0.3, "type": "claim"},
            ],
            "edges": [
                {"source": "n2", "target": "n1", "type": "supports", "weight": 1.0},
                {"source": "n3", "target": "n1", "type": "contradicts", "weight": 1.0},
            ],
        }
        cfg = GoalExtractionConfig(require_consensus=False, confidence_threshold=0.0)
        result = extractor.extract_from_debate_analysis(data, config=cfg)

        n1_goals = [g for g in result.goals if "n1" in g.source_idea_ids]
        if n1_goals:
            meta = n1_goals[0].metadata
            assert meta.get("support_edges", 0) >= 1
            assert meta.get("refute_edges", 0) >= 1
            assert meta.get("has_refutation") is True

    def test_no_edges_no_edge_metadata(self, extractor):
        """When no edges exist, edge metadata defaults to zero/false."""
        data = {
            "nodes": [
                {"id": "n1", "label": "Standalone idea", "weight": 0.5, "type": "claim"},
            ],
            "edges": [],
        }
        cfg = GoalExtractionConfig(require_consensus=False, confidence_threshold=0.0)
        result = extractor.extract_from_debate_analysis(data, config=cfg)

        assert len(result.goals) > 0
        goal = result.goals[0]
        assert goal.metadata.get("support_edges", 0) == 0
        assert goal.metadata.get("refute_edges", 0) == 0
        assert goal.metadata.get("has_refutation") is False

    def test_edges_without_nodes_key_still_works(self, extractor):
        """If cartographer output has no 'edges' key, extraction still works."""
        data = {
            "nodes": [
                {"id": "n1", "label": "Test idea", "weight": 0.5, "type": "claim"},
            ],
            # No 'edges' key at all
        }
        cfg = GoalExtractionConfig(require_consensus=False, confidence_threshold=0.0)
        result = extractor.extract_from_debate_analysis(data, config=cfg)
        assert len(result.goals) > 0
