"""Tests for the PipelineGraphHandler REST endpoints.

Covers all 11 endpoints:
- POST   /api/v1/pipeline/graph (create)
- GET    /api/v1/pipeline/graph (list)
- GET    /api/v1/pipeline/graph/{id} (get)
- DELETE /api/v1/pipeline/graph/{id} (delete)
- POST   /api/v1/pipeline/graph/{id}/node (add node)
- DELETE /api/v1/pipeline/graph/{id}/node/{nid} (remove node)
- GET    /api/v1/pipeline/graph/{id}/nodes (query)
- POST   /api/v1/pipeline/graph/{id}/promote (stage transition)
- GET    /api/v1/pipeline/graph/{id}/provenance/{nid}
- GET    /api/v1/pipeline/graph/{id}/react-flow
- GET    /api/v1/pipeline/graph/{id}/integrity
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.pipeline_graph import PipelineGraphHandler


def _body(result: object) -> dict:
    """Extract JSON body from a HandlerResult."""
    return json.loads(result.body)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_node(
    node_id: str = "n-1",
    stage: str = "ideas",
    subtype: str = "concept",
    label: str = "Test Node",
) -> MagicMock:
    node = MagicMock()
    node.id = node_id
    node.stage.value = stage
    node.node_subtype = subtype
    node.label = label
    node.to_dict.return_value = {
        "id": node_id,
        "stage": stage,
        "node_subtype": subtype,
        "label": label,
    }
    node.content_hash = "abc123"
    node.parent_ids = []
    return node


def _make_graph(
    graph_id: str = "g-1",
    name: str = "Test Graph",
    nodes: dict | None = None,
    edges: dict | None = None,
    transitions: list | None = None,
) -> MagicMock:
    graph = MagicMock()
    graph.id = graph_id
    graph.name = name
    graph.nodes = nodes or {}
    graph.edges = edges or {}
    graph.transitions = transitions or []
    graph.owner_id = None
    graph.workspace_id = None
    graph.metadata = {}
    graph.to_dict.return_value = {
        "id": graph_id,
        "name": name,
        "nodes": [],
        "edges": [],
    }
    graph.to_react_flow.return_value = {"nodes": [], "edges": []}
    graph.integrity_hash.return_value = "deadbeef12345678"
    return graph


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def handler():
    return PipelineGraphHandler()


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.get.return_value = None
    store.list.return_value = []
    store.delete.return_value = False
    store.query_nodes.return_value = []
    store.get_provenance_chain.return_value = []
    with patch(
        "aragora.server.handlers.pipeline_graph._get_store",
        return_value=store,
    ):
        yield store


# ---------------------------------------------------------------------------
# can_handle
# ---------------------------------------------------------------------------

class TestCanHandle:
    def test_v1_pipeline_graph(self, handler):
        assert handler.can_handle("/api/v1/pipeline/graph")

    def test_v1_pipeline_graph_with_id(self, handler):
        assert handler.can_handle("/api/v1/pipeline/graph/g-123")

    def test_v1_pipeline_graph_node(self, handler):
        assert handler.can_handle("/api/v1/pipeline/graph/g-1/node")

    def test_v1_pipeline_graph_promote(self, handler):
        assert handler.can_handle("/api/v1/pipeline/graph/g-1/promote")

    def test_v1_pipeline_graph_react_flow(self, handler):
        assert handler.can_handle("/api/v1/pipeline/graph/g-1/react-flow")

    def test_v1_pipeline_graph_integrity(self, handler):
        assert handler.can_handle("/api/v1/pipeline/graph/g-1/integrity")

    def test_unversioned_path(self, handler):
        assert handler.can_handle("/api/pipeline/graph")

    def test_unversioned_path_with_id(self, handler):
        assert handler.can_handle("/api/pipeline/graph/g-1")

    def test_unrelated_path_debates(self, handler):
        assert not handler.can_handle("/api/v1/debates")

    def test_unrelated_path_canvas(self, handler):
        assert not handler.can_handle("/api/v1/canvas/pipeline")

    def test_unrelated_path_empty(self, handler):
        assert not handler.can_handle("/")

    def test_unrelated_path_pipeline_no_graph(self, handler):
        assert not handler.can_handle("/api/v1/pipeline/run")


# ---------------------------------------------------------------------------
# Dispatch routing
# ---------------------------------------------------------------------------

class TestDispatch:
    def test_handle_dispatches_get_graph(self, handler, mock_store):
        mock_store.get.return_value = _make_graph()
        result = handler.handle("/api/v1/pipeline/graph/g-1", {}, None)
        assert result is not None

    def test_handle_dispatches_list_graphs(self, handler, mock_store):
        result = handler.handle("/api/v1/pipeline/graph", {}, None)
        assert result is not None

    def test_handle_dispatches_react_flow(self, handler, mock_store):
        mock_store.get.return_value = _make_graph()
        result = handler.handle("/api/v1/pipeline/graph/g-1/react-flow", {}, None)
        assert result is not None

    def test_handle_dispatches_integrity(self, handler, mock_store):
        mock_store.get.return_value = _make_graph()
        result = handler.handle("/api/v1/pipeline/graph/g-1/integrity", {}, None)
        assert result is not None

    def test_handle_dispatches_provenance(self, handler, mock_store):
        result = handler.handle("/api/v1/pipeline/graph/g-1/provenance/n-1", {}, None)
        assert result is not None

    def test_handle_dispatches_query_nodes(self, handler, mock_store):
        result = handler.handle("/api/v1/pipeline/graph/g-1/nodes", {}, None)
        assert result is not None

    def test_handle_returns_none_for_unknown(self, handler, mock_store):
        result = handler.handle("/api/v1/pipeline/unknown", {}, None)
        assert result is None

    def test_handle_post_dispatches_create(self, handler, mock_store):
        mock_handler = MagicMock()
        mock_handler.request.body = b'{}'
        result = handler.handle_post("/api/v1/pipeline/graph", {}, mock_handler)
        assert result is not None

    def test_handle_post_dispatches_add_node(self, handler, mock_store):
        mock_handler = MagicMock()
        mock_handler.request.body = b'{}'
        result = handler.handle_post("/api/v1/pipeline/graph/g-1/node", {}, mock_handler)
        assert result is not None

    def test_handle_post_dispatches_promote(self, handler, mock_store):
        mock_handler = MagicMock()
        mock_handler.request.body = b'{}'
        result = handler.handle_post("/api/v1/pipeline/graph/g-1/promote", {}, mock_handler)
        assert result is not None

    def test_handle_post_returns_none_for_unknown(self, handler, mock_store):
        mock_handler = MagicMock()
        mock_handler.request.body = b'{}'
        result = handler.handle_post("/api/v1/pipeline/unknown", {}, mock_handler)
        assert result is None

    def test_handle_delete_dispatches_graph(self, handler, mock_store):
        result = handler.handle_delete("/api/v1/pipeline/graph/g-1", {}, None)
        assert result is not None

    def test_handle_delete_dispatches_node(self, handler, mock_store):
        result = handler.handle_delete("/api/v1/pipeline/graph/g-1/node/n-1", {}, None)
        assert result is not None

    def test_handle_delete_returns_none_for_unknown(self, handler, mock_store):
        result = handler.handle_delete("/api/v1/pipeline/unknown", {}, None)
        assert result is None


# ---------------------------------------------------------------------------
# _get_request_body
# ---------------------------------------------------------------------------

class TestGetRequestBody:
    def test_parses_json_body(self):
        handler = MagicMock()
        handler.request.body = b'{"key": "value"}'
        result = PipelineGraphHandler._get_request_body(handler)
        assert result == {"key": "value"}

    def test_string_body(self):
        handler = MagicMock()
        handler.request.body = '{"key": "value"}'
        result = PipelineGraphHandler._get_request_body(handler)
        assert result == {"key": "value"}

    def test_invalid_json_returns_empty_dict(self):
        handler = MagicMock()
        handler.request.body = b'not json'
        result = PipelineGraphHandler._get_request_body(handler)
        assert result == {}

    def test_empty_body_returns_empty_dict(self):
        handler = MagicMock()
        handler.request.body = b''
        result = PipelineGraphHandler._get_request_body(handler)
        assert result == {}

    def test_missing_request_attribute(self):
        handler = object()
        result = PipelineGraphHandler._get_request_body(handler)
        assert result == {}


# ---------------------------------------------------------------------------
# POST create graph
# ---------------------------------------------------------------------------

class TestCreateGraph:
    @pytest.mark.asyncio
    async def test_success_with_name(self, handler, mock_store):
        result = await handler.handle_create_graph({"name": "My Graph"})
        assert _body(result)["name"] == "My Graph"
        assert _body(result)["created"] is True
        mock_store.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_success_with_explicit_id(self, handler, mock_store):
        result = await handler.handle_create_graph({"id": "custom-id"})
        assert _body(result)["id"] == "custom-id"

    @pytest.mark.asyncio
    async def test_default_name(self, handler, mock_store):
        result = await handler.handle_create_graph({})
        assert _body(result)["name"] == "Untitled Pipeline"

    @pytest.mark.asyncio
    async def test_auto_generated_id(self, handler, mock_store):
        result = await handler.handle_create_graph({})
        assert _body(result)["id"].startswith("graph-")

    @pytest.mark.asyncio
    async def test_with_owner_and_workspace(self, handler, mock_store):
        result = await handler.handle_create_graph({
            "owner_id": "user-1",
            "workspace_id": "ws-1",
        })
        assert _body(result)["created"] is True

    @pytest.mark.asyncio
    async def test_with_metadata(self, handler, mock_store):
        result = await handler.handle_create_graph({
            "metadata": {"purpose": "test"},
        })
        assert _body(result)["created"] is True

    @pytest.mark.asyncio
    async def test_with_nodes(self, handler, mock_store):
        result = await handler.handle_create_graph({
            "nodes": [
                {"stage": "ideas", "node_subtype": "concept", "label": "Idea 1"},
            ],
        })
        assert _body(result)["node_count"] == 1

    @pytest.mark.asyncio
    async def test_import_error_fallback(self, handler, mock_store):
        with patch.dict("sys.modules", {"aragora.pipeline.universal_node": None}):
            result = await handler.handle_create_graph({"name": "X"})
        assert result.status_code >= 400


# ---------------------------------------------------------------------------
# GET get graph
# ---------------------------------------------------------------------------

class TestGetGraph:
    @pytest.mark.asyncio
    async def test_found(self, handler, mock_store):
        graph = _make_graph("g-1")
        mock_store.get.return_value = graph
        result = await handler.handle_get_graph("g-1")
        assert _body(result)["id"] == "g-1"

    @pytest.mark.asyncio
    async def test_not_found(self, handler, mock_store):
        mock_store.get.return_value = None
        result = await handler.handle_get_graph("nonexistent")
        assert result.status_code >= 400

    @pytest.mark.asyncio
    async def test_import_error_fallback(self, handler):
        with patch(
            "aragora.server.handlers.pipeline_graph._get_store",
            side_effect=ImportError("no store"),
        ):
            result = await handler.handle_get_graph("g-1")
        assert result.status_code >= 400


# ---------------------------------------------------------------------------
# GET list graphs
# ---------------------------------------------------------------------------

class TestListGraphs:
    @pytest.mark.asyncio
    async def test_empty_list(self, handler, mock_store):
        result = await handler.handle_list_graphs({})
        assert _body(result)["graphs"] == []
        assert _body(result)["count"] == 0

    @pytest.mark.asyncio
    async def test_with_results(self, handler, mock_store):
        mock_store.list.return_value = [
            {"id": "g-1", "name": "Graph 1", "node_count": 5},
        ]
        result = await handler.handle_list_graphs({})
        assert _body(result)["count"] == 1

    @pytest.mark.asyncio
    async def test_with_owner_filter(self, handler, mock_store):
        await handler.handle_list_graphs({"owner_id": "user-1"})
        mock_store.list.assert_called_once_with(
            owner_id="user-1", workspace_id=None, limit=50,
        )

    @pytest.mark.asyncio
    async def test_with_workspace_filter(self, handler, mock_store):
        await handler.handle_list_graphs({"workspace_id": "ws-1"})
        mock_store.list.assert_called_once_with(
            owner_id=None, workspace_id="ws-1", limit=50,
        )

    @pytest.mark.asyncio
    async def test_combined_filters(self, handler, mock_store):
        await handler.handle_list_graphs({
            "owner_id": "u-1", "workspace_id": "ws-1",
        })
        mock_store.list.assert_called_once_with(
            owner_id="u-1", workspace_id="ws-1", limit=50,
        )

    @pytest.mark.asyncio
    async def test_with_limit(self, handler, mock_store):
        await handler.handle_list_graphs({"limit": "10"})
        mock_store.list.assert_called_once_with(
            owner_id=None, workspace_id=None, limit=10,
        )

    @pytest.mark.asyncio
    async def test_default_limit(self, handler, mock_store):
        await handler.handle_list_graphs({})
        mock_store.list.assert_called_once_with(
            owner_id=None, workspace_id=None, limit=50,
        )

    @pytest.mark.asyncio
    async def test_import_error_fallback(self, handler):
        with patch(
            "aragora.server.handlers.pipeline_graph._get_store",
            side_effect=ImportError("no store"),
        ):
            result = await handler.handle_list_graphs({})
        assert result.status_code >= 400


# ---------------------------------------------------------------------------
# DELETE delete graph
# ---------------------------------------------------------------------------

class TestDeleteGraph:
    @pytest.mark.asyncio
    async def test_found(self, handler, mock_store):
        mock_store.delete.return_value = True
        result = await handler.handle_delete_graph("g-1")
        assert _body(result)["deleted"] is True
        assert _body(result)["id"] == "g-1"

    @pytest.mark.asyncio
    async def test_not_found(self, handler, mock_store):
        mock_store.delete.return_value = False
        result = await handler.handle_delete_graph("nonexistent")
        assert result.status_code >= 400

    @pytest.mark.asyncio
    async def test_import_error_fallback(self, handler):
        with patch(
            "aragora.server.handlers.pipeline_graph._get_store",
            side_effect=ImportError("no store"),
        ):
            result = await handler.handle_delete_graph("g-1")
        assert result.status_code >= 400


# ---------------------------------------------------------------------------
# POST add node
# ---------------------------------------------------------------------------

class TestAddNode:
    @pytest.mark.asyncio
    async def test_success(self, handler, mock_store):
        mock_store.get.return_value = _make_graph()
        result = await handler.handle_add_node("g-1", {
            "stage": "ideas",
            "node_subtype": "concept",
            "label": "New Idea",
        })
        assert _body(result)["added"] is True
        assert _body(result)["graph_id"] == "g-1"

    @pytest.mark.asyncio
    async def test_graph_not_found(self, handler, mock_store):
        mock_store.get.return_value = None
        result = await handler.handle_add_node("g-1", {
            "stage": "ideas", "node_subtype": "concept",
        })
        assert result.status_code >= 400
        assert "not found" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_missing_stage(self, handler, mock_store):
        mock_store.get.return_value = _make_graph()
        result = await handler.handle_add_node("g-1", {"node_subtype": "concept"})
        assert result.status_code >= 400
        assert "stage" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_missing_node_subtype(self, handler, mock_store):
        mock_store.get.return_value = _make_graph()
        result = await handler.handle_add_node("g-1", {"stage": "ideas"})
        assert result.status_code >= 400
        assert "node_subtype" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_missing_both_fields(self, handler, mock_store):
        mock_store.get.return_value = _make_graph()
        result = await handler.handle_add_node("g-1", {})
        assert result.status_code >= 400

    @pytest.mark.asyncio
    async def test_import_error_fallback(self, handler, mock_store):
        with patch.dict("sys.modules", {"aragora.pipeline.universal_node": None}):
            result = await handler.handle_add_node("g-1", {
                "stage": "ideas", "node_subtype": "concept",
            })
        assert result.status_code >= 400


# ---------------------------------------------------------------------------
# DELETE remove node
# ---------------------------------------------------------------------------

class TestRemoveNode:
    @pytest.mark.asyncio
    async def test_success(self, handler, mock_store):
        graph = _make_graph(nodes={"n-1": _make_node()})
        mock_store.get.return_value = graph
        result = await handler.handle_remove_node("g-1", "n-1")
        assert _body(result)["removed"] is True

    @pytest.mark.asyncio
    async def test_graph_not_found(self, handler, mock_store):
        mock_store.get.return_value = None
        result = await handler.handle_remove_node("g-1", "n-1")
        assert result.status_code >= 400

    @pytest.mark.asyncio
    async def test_node_not_found(self, handler, mock_store):
        mock_store.get.return_value = _make_graph(nodes={})
        result = await handler.handle_remove_node("g-1", "n-1")
        assert result.status_code >= 400
        assert "Node" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_import_error_fallback(self, handler):
        with patch(
            "aragora.server.handlers.pipeline_graph._get_store",
            side_effect=ImportError("no store"),
        ):
            result = await handler.handle_remove_node("g-1", "n-1")
        assert result.status_code >= 400


# ---------------------------------------------------------------------------
# GET query nodes
# ---------------------------------------------------------------------------

class TestQueryNodes:
    @pytest.mark.asyncio
    async def test_without_filters(self, handler, mock_store):
        mock_store.query_nodes.return_value = [_make_node()]
        result = await handler.handle_query_nodes("g-1", {})
        assert _body(result)["count"] == 1

    @pytest.mark.asyncio
    async def test_with_stage_filter(self, handler, mock_store):
        mock_store.query_nodes.return_value = []
        result = await handler.handle_query_nodes("g-1", {"stage": "ideas"})
        assert _body(result)["count"] == 0

    @pytest.mark.asyncio
    async def test_with_subtype_filter(self, handler, mock_store):
        mock_store.query_nodes.return_value = []
        result = await handler.handle_query_nodes("g-1", {"subtype": "concept"})
        assert _body(result)["count"] == 0

    @pytest.mark.asyncio
    async def test_with_both_filters(self, handler, mock_store):
        result = await handler.handle_query_nodes(
            "g-1", {"stage": "goals", "subtype": "goal"},
        )
        assert _body(result)["count"] == 0

    @pytest.mark.asyncio
    async def test_invalid_stage_returns_error(self, handler, mock_store):
        result = await handler.handle_query_nodes("g-1", {"stage": "invalid"})
        assert result.status_code >= 400

    @pytest.mark.asyncio
    async def test_import_error_fallback(self, handler):
        with patch(
            "aragora.server.handlers.pipeline_graph._get_store",
            side_effect=ImportError("no store"),
        ):
            result = await handler.handle_query_nodes("g-1", {})
        assert result.status_code >= 400


# ---------------------------------------------------------------------------
# POST promote
# ---------------------------------------------------------------------------

class TestPromote:
    @pytest.mark.asyncio
    async def test_ideas_to_goals(self, handler, mock_store):
        graph = _make_graph("g-1")
        mock_store.get.return_value = graph
        created_node = _make_node("goal-1", "goals", "goal", "New Goal")
        with patch(
            "aragora.pipeline.stage_transitions.ideas_to_goals",
            return_value=[created_node],
        ) as mock_fn:
            result = await handler.handle_promote(
                "g-1", {"node_ids": ["n-1"], "target_stage": "goals"},
            )
        assert _body(result)["graph_id"] == "g-1"
        assert _body(result)["target_stage"] == "goals"
        assert _body(result)["promoted_count"] == 1
        assert "goal-1" in _body(result)["new_node_ids"]
        mock_fn.assert_called_once_with(graph, ["n-1"])
        mock_store.update.assert_called_once_with(graph)

    @pytest.mark.asyncio
    async def test_goals_to_actions(self, handler, mock_store):
        graph = _make_graph("g-1")
        mock_store.get.return_value = graph
        created = _make_node("action-1", "actions", "task", "Do Thing")
        with patch(
            "aragora.pipeline.stage_transitions.goals_to_actions",
            return_value=[created],
        ):
            result = await handler.handle_promote(
                "g-1", {"node_ids": ["goal-1"], "target_stage": "actions"},
            )
        assert _body(result)["promoted_count"] == 1

    @pytest.mark.asyncio
    async def test_actions_to_orchestration(self, handler, mock_store):
        graph = _make_graph("g-1")
        mock_store.get.return_value = graph
        created = _make_node("orch-1", "orchestration", "agent_task", "Run")
        with patch(
            "aragora.pipeline.stage_transitions.actions_to_orchestration",
            return_value=[created],
        ):
            result = await handler.handle_promote(
                "g-1", {"node_ids": ["action-1"], "target_stage": "orchestration"},
            )
        assert _body(result)["promoted_count"] == 1

    @pytest.mark.asyncio
    async def test_cannot_promote_to_ideas(self, handler, mock_store):
        mock_store.get.return_value = _make_graph()
        result = await handler.handle_promote(
            "g-1", {"node_ids": ["n-1"], "target_stage": "ideas"},
        )
        assert result.status_code >= 400

    @pytest.mark.asyncio
    async def test_invalid_target_stage(self, handler, mock_store):
        mock_store.get.return_value = _make_graph()
        result = await handler.handle_promote(
            "g-1", {"node_ids": ["n-1"], "target_stage": "invalid"},
        )
        assert result.status_code >= 400
        assert "Invalid" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_missing_node_ids(self, handler, mock_store):
        mock_store.get.return_value = _make_graph()
        result = await handler.handle_promote(
            "g-1", {"target_stage": "goals"},
        )
        assert result.status_code >= 400
        assert "node_ids" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_missing_target_stage(self, handler, mock_store):
        mock_store.get.return_value = _make_graph()
        result = await handler.handle_promote(
            "g-1", {"node_ids": ["n-1"]},
        )
        assert result.status_code >= 400
        assert "target_stage" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_missing_both_fields(self, handler, mock_store):
        mock_store.get.return_value = _make_graph()
        result = await handler.handle_promote("g-1", {})
        assert result.status_code >= 400

    @pytest.mark.asyncio
    async def test_graph_not_found(self, handler, mock_store):
        mock_store.get.return_value = None
        result = await handler.handle_promote(
            "g-1", {"node_ids": ["n-1"], "target_stage": "goals"},
        )
        assert result.status_code >= 400
        assert "not found" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_transition_count_in_result(self, handler, mock_store):
        graph = _make_graph("g-1")
        graph.transitions = [MagicMock(), MagicMock()]
        mock_store.get.return_value = graph
        created = _make_node("goal-1", "goals", "goal", "Goal")
        with patch(
            "aragora.pipeline.stage_transitions.ideas_to_goals",
            return_value=[created],
        ):
            result = await handler.handle_promote(
                "g-1", {"node_ids": ["n-1"], "target_stage": "goals"},
            )
        assert _body(result)["transition_count"] == 2

    @pytest.mark.asyncio
    async def test_store_add_node_called_for_each_created(self, handler, mock_store):
        graph = _make_graph("g-1")
        mock_store.get.return_value = graph
        nodes = [_make_node(f"goal-{i}") for i in range(3)]
        with patch(
            "aragora.pipeline.stage_transitions.ideas_to_goals",
            return_value=nodes,
        ):
            result = await handler.handle_promote(
                "g-1", {"node_ids": ["n-1"], "target_stage": "goals"},
            )
        assert _body(result)["promoted_count"] == 3
        assert mock_store.add_node.call_count == 3

    @pytest.mark.asyncio
    async def test_import_error_fallback(self, handler, mock_store):
        with patch.dict("sys.modules", {"aragora.pipeline.stage_transitions": None}):
            result = await handler.handle_promote(
                "g-1", {"node_ids": ["n-1"], "target_stage": "goals"},
            )
        assert result.status_code >= 400


# ---------------------------------------------------------------------------
# GET provenance
# ---------------------------------------------------------------------------

class TestProvenance:
    @pytest.mark.asyncio
    async def test_chain_exists(self, handler, mock_store):
        chain = [_make_node("n-1"), _make_node("n-2")]
        mock_store.get_provenance_chain.return_value = chain
        result = await handler.handle_provenance("g-1", "n-1")
        assert _body(result)["depth"] == 2
        assert len(_body(result)["chain"]) == 2

    @pytest.mark.asyncio
    async def test_empty_chain(self, handler, mock_store):
        mock_store.get_provenance_chain.return_value = []
        result = await handler.handle_provenance("g-1", "n-1")
        assert _body(result)["depth"] == 0

    @pytest.mark.asyncio
    async def test_single_node_chain(self, handler, mock_store):
        mock_store.get_provenance_chain.return_value = [_make_node()]
        result = await handler.handle_provenance("g-1", "n-1")
        assert _body(result)["depth"] == 1

    @pytest.mark.asyncio
    async def test_import_error_fallback(self, handler):
        with patch(
            "aragora.server.handlers.pipeline_graph._get_store",
            side_effect=ImportError("no store"),
        ):
            result = await handler.handle_provenance("g-1", "n-1")
        assert result.status_code >= 400


# ---------------------------------------------------------------------------
# GET react-flow
# ---------------------------------------------------------------------------

class TestReactFlow:
    @pytest.mark.asyncio
    async def test_full_graph(self, handler, mock_store):
        graph = _make_graph()
        graph.to_react_flow.return_value = {
            "nodes": [{"id": "n-1"}],
            "edges": [{"id": "e-1"}],
        }
        mock_store.get.return_value = graph
        result = await handler.handle_react_flow("g-1", {})
        assert _body(result)["node_count"] == 1
        assert _body(result)["edge_count"] == 1

    @pytest.mark.asyncio
    async def test_empty_graph(self, handler, mock_store):
        graph = _make_graph()
        mock_store.get.return_value = graph
        result = await handler.handle_react_flow("g-1", {})
        assert _body(result)["node_count"] == 0

    @pytest.mark.asyncio
    async def test_with_stage_filter(self, handler, mock_store):
        graph = _make_graph()
        mock_store.get.return_value = graph
        result = await handler.handle_react_flow("g-1", {"stage": "ideas"})
        assert "graph_id" in _body(result)

    @pytest.mark.asyncio
    async def test_without_stage_filter(self, handler, mock_store):
        graph = _make_graph()
        mock_store.get.return_value = graph
        result = await handler.handle_react_flow("g-1", {})
        assert "graph_name" in _body(result)

    @pytest.mark.asyncio
    async def test_graph_not_found(self, handler, mock_store):
        mock_store.get.return_value = None
        result = await handler.handle_react_flow("g-1", {})
        assert result.status_code >= 400

    @pytest.mark.asyncio
    async def test_invalid_stage_returns_error(self, handler, mock_store):
        mock_store.get.return_value = _make_graph()
        result = await handler.handle_react_flow("g-1", {"stage": "invalid"})
        assert result.status_code >= 400

    @pytest.mark.asyncio
    async def test_import_error_fallback(self, handler):
        with patch(
            "aragora.server.handlers.pipeline_graph._get_store",
            side_effect=ImportError("no store"),
        ):
            result = await handler.handle_react_flow("g-1", {})
        assert result.status_code >= 400


# ---------------------------------------------------------------------------
# GET integrity
# ---------------------------------------------------------------------------

class TestIntegrity:
    @pytest.mark.asyncio
    async def test_success(self, handler, mock_store):
        graph = _make_graph()
        mock_store.get.return_value = graph
        result = await handler.handle_integrity("g-1")
        assert _body(result)["integrity_hash"] == "deadbeef12345678"
        assert _body(result)["graph_id"] == "g-1"

    @pytest.mark.asyncio
    async def test_not_found(self, handler, mock_store):
        mock_store.get.return_value = None
        result = await handler.handle_integrity("nonexistent")
        assert result.status_code >= 400

    @pytest.mark.asyncio
    async def test_empty_graph_integrity(self, handler, mock_store):
        graph = _make_graph(nodes={}, edges={})
        mock_store.get.return_value = graph
        result = await handler.handle_integrity("g-1")
        assert "integrity_hash" in _body(result)
        assert _body(result)["node_count"] == 0

    @pytest.mark.asyncio
    async def test_import_error_fallback(self, handler):
        with patch(
            "aragora.server.handlers.pipeline_graph._get_store",
            side_effect=ImportError("no store"),
        ):
            result = await handler.handle_integrity("g-1")
        assert result.status_code >= 400


# ---------------------------------------------------------------------------
# Import fallback paths
# ---------------------------------------------------------------------------

class TestImportFallbacks:
    @pytest.mark.asyncio
    async def test_create_graph_import_error(self, handler, mock_store):
        with patch.dict("sys.modules", {"aragora.pipeline.universal_node": None}):
            result = await handler.handle_create_graph({"name": "X"})
        assert result.status_code >= 400

    @pytest.mark.asyncio
    async def test_get_graph_os_error(self, handler):
        with patch(
            "aragora.server.handlers.pipeline_graph._get_store",
            side_effect=OSError("disk"),
        ):
            result = await handler.handle_get_graph("g-1")
        assert result.status_code >= 400

    @pytest.mark.asyncio
    async def test_list_graphs_os_error(self, handler):
        with patch(
            "aragora.server.handlers.pipeline_graph._get_store",
            side_effect=OSError("disk"),
        ):
            result = await handler.handle_list_graphs({})
        assert result.status_code >= 400

    @pytest.mark.asyncio
    async def test_delete_graph_os_error(self, handler):
        with patch(
            "aragora.server.handlers.pipeline_graph._get_store",
            side_effect=OSError("disk"),
        ):
            result = await handler.handle_delete_graph("g-1")
        assert result.status_code >= 400

    @pytest.mark.asyncio
    async def test_add_node_value_error(self, handler, mock_store):
        mock_store.get.return_value = _make_graph()
        with patch(
            "aragora.pipeline.universal_node.UniversalNode.from_dict",
            side_effect=ValueError("bad data"),
        ):
            result = await handler.handle_add_node("g-1", {
                "stage": "ideas", "node_subtype": "concept",
            })
        assert result.status_code >= 400

    @pytest.mark.asyncio
    async def test_remove_node_os_error(self, handler):
        with patch(
            "aragora.server.handlers.pipeline_graph._get_store",
            side_effect=OSError("disk"),
        ):
            result = await handler.handle_remove_node("g-1", "n-1")
        assert result.status_code >= 400

    @pytest.mark.asyncio
    async def test_query_nodes_import_error(self, handler):
        with patch(
            "aragora.server.handlers.pipeline_graph._get_store",
            side_effect=ImportError("no store"),
        ):
            result = await handler.handle_query_nodes("g-1", {})
        assert result.status_code >= 400

    @pytest.mark.asyncio
    async def test_promote_import_error(self, handler, mock_store):
        with patch.dict("sys.modules", {"aragora.pipeline.stage_transitions": None}):
            result = await handler.handle_promote(
                "g-1", {"node_ids": ["n-1"], "target_stage": "goals"},
            )
        assert result.status_code >= 400

    @pytest.mark.asyncio
    async def test_provenance_os_error(self, handler):
        with patch(
            "aragora.server.handlers.pipeline_graph._get_store",
            side_effect=OSError("disk"),
        ):
            result = await handler.handle_provenance("g-1", "n-1")
        assert result.status_code >= 400

    @pytest.mark.asyncio
    async def test_react_flow_os_error(self, handler):
        with patch(
            "aragora.server.handlers.pipeline_graph._get_store",
            side_effect=OSError("disk"),
        ):
            result = await handler.handle_react_flow("g-1", {})
        assert result.status_code >= 400

    @pytest.mark.asyncio
    async def test_integrity_os_error(self, handler):
        with patch(
            "aragora.server.handlers.pipeline_graph._get_store",
            side_effect=OSError("disk"),
        ):
            result = await handler.handle_integrity("g-1")
        assert result.status_code >= 400


# ---------------------------------------------------------------------------
# Constructor / routes
# ---------------------------------------------------------------------------

class TestConstructor:
    def test_default_context(self):
        h = PipelineGraphHandler()
        assert h.ctx == {}

    def test_custom_context(self):
        h = PipelineGraphHandler(ctx={"key": "val"})
        assert h.ctx["key"] == "val"


class TestRoutes:
    def test_route_count(self):
        assert len(PipelineGraphHandler.ROUTES) == 11
