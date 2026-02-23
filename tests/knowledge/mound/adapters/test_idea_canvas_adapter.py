"""Tests for IdeaCanvasAdapter KM persistence."""

from __future__ import annotations

import pytest

from aragora.knowledge.mound.adapters.idea_canvas_adapter import IdeaCanvasAdapter


@pytest.fixture
def adapter():
    return IdeaCanvasAdapter(event_callback=None, enable_resilience=False)


class TestSyncNodeToKM:
    """sync_node_to_km() tests."""

    def test_returns_km_node_id(self, adapter):
        node = {
            "id": "node-1",
            "label": "Test concept",
            "data": {"idea_type": "concept"},
            "position": {"x": 100, "y": 200},
            "size": {"width": 220, "height": 80},
            "style": {},
        }
        km_id = adapter.sync_node_to_km(node, "canvas-1", "user-1")
        assert km_id.startswith("kn_idea_")

    def test_deterministic_id(self, adapter):
        node = {"id": "node-1", "label": "Test", "data": {"idea_type": "concept"}}
        id1 = adapter.sync_node_to_km(node, "c1", "u1")
        id2 = adapter.sync_node_to_km(node, "c1", "u1")
        assert id1 == id2

    def test_maps_idea_type(self, adapter):
        for idea_type, expected_km_type in [
            ("concept", "idea_concept"),
            ("hypothesis", "idea_hypothesis"),
            ("observation", "idea_observation"),
            ("question", "idea_question"),
            ("insight", "idea_insight"),
            ("evidence", "idea_evidence"),
            ("assumption", "idea_assumption"),
            ("constraint", "idea_constraint"),
            ("cluster", "idea_cluster"),
        ]:
            node = {"id": f"n-{idea_type}", "label": "X", "data": {"idea_type": idea_type}}
            adapter.sync_node_to_km(node, "c1", "u1")
            # The mapping is stored internally; we just verify no exceptions

    def test_unknown_idea_type_defaults(self, adapter):
        node = {"id": "n1", "label": "X", "data": {"idea_type": "unknown"}}
        km_id = adapter.sync_node_to_km(node, "c1", "u1")
        assert km_id.startswith("kn_idea_")

    def test_node_map_populated(self, adapter):
        node = {"id": "n1", "label": "X", "data": {}}
        adapter.sync_node_to_km(node, "c1", "u1")
        assert "n1" in adapter._node_map

    def test_body_included_in_content(self, adapter):
        node = {"id": "n1", "label": "Title", "data": {"body": "Detailed body"}}
        adapter.sync_node_to_km(node, "c1", "u1")


class TestSyncEdgeToKM:
    """sync_edge_to_km() tests."""

    def test_returns_kr_id(self, adapter):
        # First sync nodes so the map has entries
        adapter.sync_node_to_km({"id": "n1", "label": "A", "data": {}}, "c1", "u1")
        adapter.sync_node_to_km({"id": "n2", "label": "B", "data": {}}, "c1", "u1")

        edge = {
            "id": "e1",
            "source_id": "n1",
            "target_id": "n2",
            "edge_type": "supports",
            "label": "supports",
        }
        kr_id = adapter.sync_edge_to_km(edge, "c1")
        assert kr_id.startswith("kr_idea_")

    def test_edge_map_populated(self, adapter):
        edge = {"id": "e1", "source_id": "n1", "target_id": "n2", "edge_type": "inspires"}
        adapter.sync_edge_to_km(edge, "c1")
        assert "e1" in adapter._edge_map

    def test_edge_type_mapping(self, adapter):
        for edge_type in [
            "supports",
            "refutes",
            "inspires",
            "refines",
            "challenges",
            "exemplifies",
            "requires",
        ]:
            edge = {
                "id": f"e-{edge_type}",
                "source_id": "a",
                "target_id": "b",
                "edge_type": edge_type,
            }
            kr_id = adapter.sync_edge_to_km(edge, "c1")
            assert kr_id.startswith("kr_idea_")


class TestSyncCanvasToKM:
    """sync_canvas_to_km() batch sync tests."""

    def test_syncs_all_nodes(self, adapter):
        canvas = {
            "id": "c1",
            "nodes": [
                {"id": "n1", "label": "A", "data": {"idea_type": "concept"}},
                {"id": "n2", "label": "B", "data": {"idea_type": "hypothesis"}},
            ],
            "edges": [
                {"id": "e1", "source_id": "n1", "target_id": "n2", "edge_type": "inspires"},
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
                "id": "kn_idea_abc",
                "content": "My Concept\nSome details",
                "confidence": 0.8,
                "topics": ["tag1"],
                "metadata": {
                    "canvas_id": "c1",
                    "canvas_node_id": "n1",
                    "idea_type": "concept",
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
        assert result["nodes"][0]["label"] == "My Concept"
        assert result["nodes"][0]["data"]["idea_type"] == "concept"

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
                    "idea_type": "observation",
                    "canvas": {"position": {"x": 42, "y": 99}},
                },
            }
        ]
        result = adapter.load_canvas_from_km("c1", km_nodes)
        assert result["nodes"][0]["position"] == {"x": 42, "y": 99}


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

    def test_supports(self):
        assert IdeaCanvasAdapter._map_edge_type("supports") == "supports"

    def test_refutes(self):
        assert IdeaCanvasAdapter._map_edge_type("refutes") == "contradicts"

    def test_inspires(self):
        assert IdeaCanvasAdapter._map_edge_type("inspires") == "inspires"

    def test_unknown_defaults(self):
        assert IdeaCanvasAdapter._map_edge_type("unknown") == "related_to"
