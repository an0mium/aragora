"""
Tests for ArgumentCartographer WebSocket integration.

Covers:
- ArgumentCartographer.to_dict() serialization
- EventEmitterBridge emits GRAPH_UPDATE after cartographer updates
- GRAPH_UPDATE event type registration
- No emission when cartographer or emitter is absent
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aragora.visualization.mapper import (
    ArgumentCartographer,
    ArgumentEdge,
    ArgumentNode,
    EdgeRelation,
    NodeType,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def cartographer():
    """A cartographer with some pre-populated data."""
    c = ArgumentCartographer()
    c.set_debate_context(debate_id="d-001", topic="Rate limiter design")
    c.update_from_message(agent="claude", content="I propose a token bucket", role="proposer", round_num=1)
    c.update_from_message(agent="gpt4", content="I disagree, sliding window is better", role="critic", round_num=1)
    return c


@pytest.fixture()
def bridge():
    """An EventEmitterBridge with mock emitter and cartographer."""
    from aragora.debate.event_bridge import EventEmitterBridge

    emitter = MagicMock()
    carto = ArgumentCartographer()
    carto.set_debate_context(debate_id="d-002", topic="Architecture")
    return EventEmitterBridge(
        spectator=None,
        event_emitter=emitter,
        cartographer=carto,
        loop_id="loop-1",
    )


# ---------------------------------------------------------------------------
# ArgumentCartographer.to_dict
# ---------------------------------------------------------------------------


class TestCartographerToDict:
    """Test to_dict() serialization."""

    def test_empty_graph(self):
        c = ArgumentCartographer()
        d = c.to_dict()

        assert d["debate_id"] is None
        assert d["topic"] is None
        assert d["nodes"] == []
        assert d["edges"] == []
        assert "statistics" in d
        assert d["statistics"]["node_count"] == 0

    def test_with_context(self, cartographer):
        d = cartographer.to_dict()

        assert d["debate_id"] == "d-001"
        assert d["topic"] == "Rate limiter design"
        assert len(d["nodes"]) == 2
        assert isinstance(d["edges"], list)

    def test_nodes_are_dicts(self, cartographer):
        d = cartographer.to_dict()

        for node in d["nodes"]:
            assert isinstance(node, dict)
            assert "id" in node
            assert "agent" in node
            assert "node_type" in node
            assert "summary" in node

    def test_edges_are_dicts(self, cartographer):
        d = cartographer.to_dict()

        for edge in d["edges"]:
            assert isinstance(edge, dict)
            assert "source_id" in edge
            assert "target_id" in edge
            assert "relation" in edge

    def test_statistics_included(self, cartographer):
        d = cartographer.to_dict()
        stats = d["statistics"]

        assert stats["node_count"] == 2
        assert "edge_count" in stats
        assert "max_depth" in stats
        assert "complexity_score" in stats


# ---------------------------------------------------------------------------
# GRAPH_UPDATE event type registration
# ---------------------------------------------------------------------------


class TestGraphUpdateEventType:
    """Test GRAPH_UPDATE is registered in StreamEventType."""

    def test_graph_update_exists(self):
        from aragora.events.types import StreamEventType

        assert hasattr(StreamEventType, "GRAPH_UPDATE")
        assert StreamEventType.GRAPH_UPDATE.value == "graph_update"

    def test_event_type_mapping_includes_graph_update(self):
        from aragora.debate.event_bridge import EventEmitterBridge

        assert "graph_update" in EventEmitterBridge.EVENT_TYPE_MAPPING
        assert EventEmitterBridge.EVENT_TYPE_MAPPING["graph_update"] == "GRAPH_UPDATE"


# ---------------------------------------------------------------------------
# EventEmitterBridge graph_update emission
# ---------------------------------------------------------------------------


class TestBridgeGraphUpdateEmission:
    """Test that EventEmitterBridge emits graph_update after cartographer updates."""

    def test_proposal_emits_graph_update(self, bridge):
        bridge.notify("proposal", agent="claude", details="I propose X", round_number=1)

        # The emitter should have been called (at least once for the proposal event,
        # plus once for the graph_update)
        calls = bridge.event_emitter.emit.call_args_list
        graph_events = [
            c for c in calls
            if hasattr(c[0][0], "type") and c[0][0].type.value == "graph_update"
        ]
        assert len(graph_events) >= 1

    def test_critique_emits_graph_update(self, bridge):
        # First add a proposal so there's a node for the critique target
        bridge.notify("proposal", agent="claude", details="I propose X", round_number=1)
        bridge.event_emitter.emit.reset_mock()

        bridge.notify("critique", agent="gpt4", details="Critiqued claude: weak", round_number=1, metric=0.8)

        calls = bridge.event_emitter.emit.call_args_list
        graph_events = [
            c for c in calls
            if hasattr(c[0][0], "type") and c[0][0].type.value == "graph_update"
        ]
        assert len(graph_events) >= 1

    def test_vote_emits_graph_update(self, bridge):
        bridge.notify("proposal", agent="claude", details="I propose X", round_number=1)
        bridge.event_emitter.emit.reset_mock()

        bridge.notify("vote", agent="gpt4", details="approve: yes", round_number=1)

        calls = bridge.event_emitter.emit.call_args_list
        graph_events = [
            c for c in calls
            if hasattr(c[0][0], "type") and c[0][0].type.value == "graph_update"
        ]
        assert len(graph_events) >= 1

    def test_consensus_emits_graph_update(self, bridge):
        bridge.notify("proposal", agent="claude", details="I propose X", round_number=1)
        bridge.event_emitter.emit.reset_mock()

        bridge.notify("consensus", agent="system", details="approved", round_number=1)

        calls = bridge.event_emitter.emit.call_args_list
        graph_events = [
            c for c in calls
            if hasattr(c[0][0], "type") and c[0][0].type.value == "graph_update"
        ]
        assert len(graph_events) >= 1

    def test_graph_update_contains_nodes_and_edges(self, bridge):
        bridge.notify("proposal", agent="claude", details="I propose X", round_number=1)

        calls = bridge.event_emitter.emit.call_args_list
        graph_events = [
            c for c in calls
            if hasattr(c[0][0], "type") and c[0][0].type.value == "graph_update"
        ]
        assert len(graph_events) >= 1

        event = graph_events[-1][0][0]
        assert "nodes" in event.data
        assert "edges" in event.data
        assert "statistics" in event.data

    def test_no_emission_without_emitter(self):
        from aragora.debate.event_bridge import EventEmitterBridge

        carto = ArgumentCartographer()
        bridge = EventEmitterBridge(
            spectator=None,
            event_emitter=None,
            cartographer=carto,
            loop_id="loop-2",
        )

        # Should not raise
        bridge._emit_graph_update()

    def test_no_emission_without_cartographer(self):
        from aragora.debate.event_bridge import EventEmitterBridge

        emitter = MagicMock()
        bridge = EventEmitterBridge(
            spectator=None,
            event_emitter=emitter,
            cartographer=None,
            loop_id="loop-3",
        )

        bridge._emit_graph_update()
        emitter.emit.assert_not_called()

    def test_graph_update_has_loop_id(self, bridge):
        bridge.notify("proposal", agent="claude", details="I propose X", round_number=1)

        calls = bridge.event_emitter.emit.call_args_list
        graph_events = [
            c for c in calls
            if hasattr(c[0][0], "type") and c[0][0].type.value == "graph_update"
        ]
        assert len(graph_events) >= 1

        event = graph_events[-1][0][0]
        assert event.loop_id == "loop-1"

    def test_unmapped_event_no_graph_update(self, bridge):
        """Events not handled by _update_cartographer should not emit graph_update."""
        bridge.event_emitter.emit.reset_mock()
        bridge._update_cartographer("unknown_event", agent="claude", details="something")

        # No graph_update should be emitted for unknown events
        calls = bridge.event_emitter.emit.call_args_list
        graph_events = [
            c for c in calls
            if hasattr(c[0][0], "type") and c[0][0].type.value == "graph_update"
        ]
        assert len(graph_events) == 0
