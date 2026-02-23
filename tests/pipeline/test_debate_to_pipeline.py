"""Tests for debate result to pipeline creation."""

from __future__ import annotations

import pytest

from aragora.canvas.stages import PipelineStage
from aragora.pipeline.stage_transitions import ideas_to_goals
from aragora.pipeline.universal_node import UniversalGraph, UniversalNode


def _debate_proposals_to_idea_nodes(proposals: list[dict]) -> list[UniversalNode]:
    """Convert debate proposals to idea nodes (mirrors backend handler logic)."""
    nodes = []
    for i, prop in enumerate(proposals):
        node = UniversalNode(
            id=prop.get("id", f"debate-{i}"),
            stage=PipelineStage.IDEAS,
            node_subtype="concept",
            label=prop.get("summary", prop.get("content", "")),
            description=prop.get("content", prop.get("summary", "")),
            data={"source": "debate", "type": prop.get("type", "proposal")},
        )
        nodes.append(node)
    return nodes


class TestDebateToPipeline:
    def test_convert_proposals_to_ideas(self):
        proposals = [
            {
                "id": "p1",
                "type": "proposal",
                "summary": "Use Redis for caching",
                "content": "Implement Redis caching layer",
            },
            {
                "id": "p2",
                "type": "critique",
                "summary": "Security concern",
                "content": "Need to validate cache keys",
            },
        ]
        nodes = _debate_proposals_to_idea_nodes(proposals)
        assert len(nodes) == 2
        assert nodes[0].label == "Use Redis for caching"
        assert nodes[1].data["type"] == "critique"

    def test_debate_ideas_feed_into_goals(self):
        proposals = [
            {"id": "p1", "summary": "Build rate limiter", "content": "Token bucket algorithm"},
            {"id": "p2", "summary": "Add monitoring", "content": "Prometheus metrics"},
        ]
        nodes = _debate_proposals_to_idea_nodes(proposals)

        graph = UniversalGraph(id="debate-pipeline", name="From Debate")
        for n in nodes:
            graph.add_node(n)

        idea_ids = [n.id for n in nodes]
        goals = ideas_to_goals(graph, idea_ids)
        assert len(goals) >= 1
        # Each goal should trace back to debate proposals
        for g in goals:
            assert g.parent_ids

    def test_provenance_preserves_debate_source(self):
        proposals = [
            {"id": "debate-node-1", "summary": "Improve error handling"},
        ]
        nodes = _debate_proposals_to_idea_nodes(proposals)
        graph = UniversalGraph(id="debate-pipe", name="From Debate")
        for n in nodes:
            graph.add_node(n)

        goals = ideas_to_goals(graph, [n.id for n in nodes])
        # The goal should reference the debate node in parent_ids
        assert any("debate-node-1" in g.parent_ids for g in goals)

    def test_empty_proposals_produce_no_ideas(self):
        nodes = _debate_proposals_to_idea_nodes([])
        assert len(nodes) == 0
