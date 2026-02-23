"""Tests for GoalCanvasAdapter KM persistence."""

from __future__ import annotations

import pytest

from aragora.knowledge.mound.adapters.goal_canvas_adapter import GoalCanvasAdapter


@pytest.fixture
def adapter():
    return GoalCanvasAdapter(event_callback=None, enable_resilience=False)


class TestSyncNodeToKM:
    """sync_node_to_km() tests."""

    def test_returns_km_node_id(self, adapter):
        node = {
            "id": "node-1",
            "label": "Achieve 99% uptime",
            "data": {"goal_type": "goal", "priority": "high"},
            "position": {"x": 100, "y": 200},
            "size": {"width": 220, "height": 80},
            "style": {},
        }
        km_id = adapter.sync_node_to_km(node, "canvas-1", "user-1")
        assert km_id.startswith("kn_goal_")

    def test_deterministic_id(self, adapter):
        node = {"id": "node-1", "label": "Test", "data": {"goal_type": "goal"}}
        id1 = adapter.sync_node_to_km(node, "c1", "u1")
        id2 = adapter.sync_node_to_km(node, "c1", "u1")
        assert id1 == id2

    def test_maps_goal_type(self, adapter):
        for goal_type, expected_km_type in [
            ("goal", "goal_goal"),
            ("principle", "goal_principle"),
            ("strategy", "goal_strategy"),
            ("milestone", "goal_milestone"),
            ("metric", "goal_metric"),
            ("risk", "goal_risk"),
        ]:
            node = {"id": f"n-{goal_type}", "label": "X", "data": {"goal_type": goal_type}}
            adapter.sync_node_to_km(node, "c1", "u1")

    def test_unknown_goal_type_defaults(self, adapter):
        node = {"id": "n1", "label": "X", "data": {"goal_type": "unknown"}}
        km_id = adapter.sync_node_to_km(node, "c1", "u1")
        assert km_id.startswith("kn_goal_")

    def test_node_map_populated(self, adapter):
        node = {"id": "n1", "label": "X", "data": {}}
        adapter.sync_node_to_km(node, "c1", "u1")
        assert "n1" in adapter._node_map

    def test_description_included_in_content(self, adapter):
        node = {"id": "n1", "label": "Title", "data": {"description": "Detailed desc"}}
        adapter.sync_node_to_km(node, "c1", "u1")

    def test_priority_stored_in_metadata(self, adapter):
        node = {"id": "n1", "label": "X", "data": {"goal_type": "goal", "priority": "critical"}}
        adapter.sync_node_to_km(node, "c1", "u1")
        # Node map should have the entry
        assert "n1" in adapter._node_map


class TestSyncEdgeToKM:
    """sync_edge_to_km() tests."""

    def test_returns_kr_id(self, adapter):
        adapter.sync_node_to_km({"id": "n1", "label": "A", "data": {}}, "c1", "u1")
        adapter.sync_node_to_km({"id": "n2", "label": "B", "data": {}}, "c1", "u1")

        edge = {
            "id": "e1",
            "source_id": "n1",
            "target_id": "n2",
            "edge_type": "requires",
            "label": "requires",
        }
        kr_id = adapter.sync_edge_to_km(edge, "c1")
        assert kr_id.startswith("kr_goal_")

    def test_edge_map_populated(self, adapter):
        edge = {"id": "e1", "source_id": "n1", "target_id": "n2", "edge_type": "requires"}
        adapter.sync_edge_to_km(edge, "c1")
        assert "e1" in adapter._edge_map

    def test_edge_type_mapping(self, adapter):
        for edge_type in [
            "requires",
            "blocks",
            "follows",
            "derived_from",
            "supports",
            "conflicts",
            "decomposes_into",
        ]:
            edge = {
                "id": f"e-{edge_type}",
                "source_id": "a",
                "target_id": "b",
                "edge_type": edge_type,
            }
            kr_id = adapter.sync_edge_to_km(edge, "c1")
            assert kr_id.startswith("kr_goal_")


class TestSyncCanvasToKM:
    """sync_canvas_to_km() batch sync tests."""

    def test_syncs_all_nodes(self, adapter):
        canvas = {
            "id": "c1",
            "nodes": [
                {"id": "n1", "label": "Goal A", "data": {"goal_type": "goal"}},
                {"id": "n2", "label": "Strategy B", "data": {"goal_type": "strategy"}},
            ],
            "edges": [
                {"id": "e1", "source_id": "n1", "target_id": "n2", "edge_type": "decomposes_into"},
            ],
        }
        result = adapter.sync_canvas_to_km(canvas, "user-1")
        assert len(result) == 2
        assert "n1" in result
        assert "n2" in result

    def test_empty_canvas(self, adapter):
        canvas = {"id": "c1", "nodes": [], "edges": []}
        result = adapter.sync_canvas_to_km(canvas, "user-1")
        assert len(result) == 0


class TestLoadCanvasFromKM:
    """load_canvas_from_km() reconstruction tests."""

    def test_basic_reconstruction(self, adapter):
        km_nodes = [
            {
                "id": "kn_goal_abc",
                "content": "Achieve 99% Uptime\nEnsure all services meet SLA",
                "confidence": 0.8,
                "topics": ["reliability"],
                "metadata": {
                    "canvas_id": "c1",
                    "canvas_node_id": "n1",
                    "goal_type": "goal",
                    "priority": "high",
                    "measurable": "99% uptime over 30 days",
                    "canvas": {
                        "position": {"x": 100, "y": 200},
                        "size": {"width": 220, "height": 80},
                        "style": {},
                    },
                },
            }
        ]
        result = adapter.load_canvas_from_km("c1", km_nodes)
        assert result is not None
        assert len(result["nodes"]) == 1
        assert result["nodes"][0]["label"] == "Achieve 99% Uptime"
        assert result["nodes"][0]["data"]["goal_type"] == "goal"
        assert result["nodes"][0]["data"]["priority"] == "high"

    def test_no_matching_nodes(self, adapter):
        km_nodes = [
            {
                "id": "kn1",
                "content": "X",
                "metadata": {"canvas_id": "other"},
            }
        ]
        result = adapter.load_canvas_from_km("c1", km_nodes)
        assert result is None

    def test_empty_km_nodes(self, adapter):
        result = adapter.load_canvas_from_km("c1", [])
        assert result is None

    def test_preserves_position(self, adapter):
        km_nodes = [
            {
                "id": "kn1",
                "content": "Test",
                "metadata": {
                    "canvas_id": "c1",
                    "canvas_node_id": "n1",
                    "goal_type": "milestone",
                    "canvas": {"position": {"x": 42, "y": 99}},
                },
            }
        ]
        result = adapter.load_canvas_from_km("c1", km_nodes)
        assert result["nodes"][0]["position"] == {"x": 42, "y": 99}

    def test_stage_metadata(self, adapter):
        km_nodes = [
            {
                "id": "kn1",
                "content": "Test",
                "metadata": {
                    "canvas_id": "c1",
                    "canvas_node_id": "n1",
                    "goal_type": "goal",
                    "canvas": {},
                },
            }
        ]
        result = adapter.load_canvas_from_km("c1", km_nodes)
        assert result["metadata"]["stage"] == "goals"
        assert result["nodes"][0]["data"]["stage"] == "goals"
        assert result["nodes"][0]["data"]["rf_type"] == "goalNode"


class TestSyncToKM:
    """sync_to_km() interface method."""

    @pytest.mark.asyncio
    async def test_returns_counts(self, adapter):
        adapter.sync_node_to_km({"id": "n1", "label": "A", "data": {}}, "c1", "u1")
        result = await adapter.sync_to_km()
        assert result["synced_nodes"] == 1
        assert result["synced_edges"] == 0


class TestEdgeTypeMapping:
    """_map_edge_type static method."""

    def test_requires(self):
        assert GoalCanvasAdapter._map_edge_type("requires") == "related_to"

    def test_supports(self):
        assert GoalCanvasAdapter._map_edge_type("supports") == "supports"

    def test_conflicts(self):
        assert GoalCanvasAdapter._map_edge_type("conflicts") == "contradicts"

    def test_derived_from(self):
        assert GoalCanvasAdapter._map_edge_type("derived_from") == "derived_from"

    def test_unknown_defaults(self):
        assert GoalCanvasAdapter._map_edge_type("unknown") == "related_to"
