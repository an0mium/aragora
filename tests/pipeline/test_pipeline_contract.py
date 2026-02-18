"""E2E contract test: validates pipeline handler output matches frontend expectations.

This test exercises the full pipeline through the handler layer and validates
the JSON shape matches what PipelineResultResponse expects in TypeScript:

    interface PipelineResultResponse {
      pipeline_id: string;
      ideas: ReactFlowData | null;       // {nodes, edges, metadata}
      goals: Record<string, unknown>;     // {goals: [...], provenance: [...]}
      actions: ReactFlowData | null;
      orchestration: ReactFlowData | null;
      transitions: Array<Record>;
      provenance_count: number;
      stage_status: Record<PipelineStageType, string>;
      integrity_hash: string;
    }
"""

from __future__ import annotations

import pytest

from aragora.server.handlers.canvas_pipeline import (
    CanvasPipelineHandler,
    _pipeline_objects,
)


@pytest.fixture(autouse=True)
def _clear_stores():
    _pipeline_objects.clear()
    yield
    _pipeline_objects.clear()


@pytest.fixture
def handler():
    return CanvasPipelineHandler()


# =============================================================================
# Full contract: from-ideas → auto_advance → validate shape
# =============================================================================


class TestFromIdeasContract:
    """Validate the full from-ideas pipeline output matches the frontend contract."""

    @pytest.mark.asyncio
    async def test_result_has_pipeline_id(self, handler):
        resp = await handler.handle_from_ideas({
            "ideas": ["Build rate limiter", "Add caching", "Improve docs"],
            "auto_advance": True,
        })
        result = resp["result"]
        assert isinstance(result["pipeline_id"], str)
        assert result["pipeline_id"].startswith("pipe-")

    @pytest.mark.asyncio
    async def test_result_has_stage_status(self, handler):
        resp = await handler.handle_from_ideas({
            "ideas": ["Build rate limiter", "Add caching"],
            "auto_advance": True,
        })
        result = resp["result"]
        status = result["stage_status"]
        assert set(status.keys()) == {"ideas", "goals", "actions", "orchestration"}
        for stage in ("ideas", "goals", "actions", "orchestration"):
            assert status[stage] in ("pending", "complete"), f"{stage}={status[stage]}"

    @pytest.mark.asyncio
    async def test_result_has_integrity_hash(self, handler):
        resp = await handler.handle_from_ideas({
            "ideas": ["Build rate limiter"],
            "auto_advance": True,
        })
        result = resp["result"]
        assert isinstance(result["integrity_hash"], str)
        assert len(result["integrity_hash"]) == 16

    @pytest.mark.asyncio
    async def test_result_has_provenance_count(self, handler):
        resp = await handler.handle_from_ideas({
            "ideas": ["Build rate limiter", "Add caching"],
            "auto_advance": True,
        })
        result = resp["result"]
        assert isinstance(result["provenance_count"], int)
        assert result["provenance_count"] >= 0

    @pytest.mark.asyncio
    async def test_result_has_transitions_array(self, handler):
        resp = await handler.handle_from_ideas({
            "ideas": ["Build rate limiter", "Add caching"],
            "auto_advance": True,
        })
        result = resp["result"]
        assert isinstance(result["transitions"], list)
        for transition in result["transitions"]:
            assert "from_stage" in transition
            assert "to_stage" in transition
            assert "confidence" in transition
            assert "ai_rationale" in transition

    @pytest.mark.asyncio
    async def test_ideas_is_react_flow_data(self, handler):
        """Ideas stage should be ReactFlowData: {nodes, edges, metadata}"""
        resp = await handler.handle_from_ideas({
            "ideas": ["Build rate limiter", "Add caching"],
            "auto_advance": True,
        })
        ideas = resp["result"]["ideas"]
        assert ideas is not None
        assert "nodes" in ideas
        assert "edges" in ideas
        assert isinstance(ideas["nodes"], list)
        assert isinstance(ideas["edges"], list)

        for node in ideas["nodes"]:
            assert "id" in node
            assert "type" in node
            assert "position" in node
            assert "data" in node
            assert isinstance(node["position"], dict)
            assert "x" in node["position"]
            assert "y" in node["position"]

    @pytest.mark.asyncio
    async def test_goals_has_goals_list(self, handler):
        """Goals stage should have {goals: [...], provenance: [...]}"""
        resp = await handler.handle_from_ideas({
            "ideas": ["Build rate limiter", "Add caching"],
            "auto_advance": True,
        })
        goals = resp["result"]["goals"]
        assert goals is not None
        assert "goals" in goals
        assert isinstance(goals["goals"], list)
        assert len(goals["goals"]) > 0

        for goal in goals["goals"]:
            assert "id" in goal
            assert "title" in goal

    @pytest.mark.asyncio
    async def test_actions_is_react_flow_data(self, handler):
        """Actions stage should be ReactFlowData: {nodes, edges, metadata}"""
        resp = await handler.handle_from_ideas({
            "ideas": ["Build rate limiter", "Add caching"],
            "auto_advance": True,
        })
        actions = resp["result"]["actions"]
        assert actions is not None
        assert "nodes" in actions
        assert "edges" in actions
        assert isinstance(actions["nodes"], list)

    @pytest.mark.asyncio
    async def test_orchestration_is_react_flow_data(self, handler):
        """Orchestration stage should be ReactFlowData: {nodes, edges, metadata}"""
        resp = await handler.handle_from_ideas({
            "ideas": ["Build rate limiter", "Add caching"],
            "auto_advance": True,
        })
        orch = resp["result"]["orchestration"]
        assert orch is not None
        assert "nodes" in orch
        assert "edges" in orch
        assert isinstance(orch["nodes"], list)


# =============================================================================
# Full contract: from-debate → auto_advance → validate shape
# =============================================================================


class TestFromDebateContract:
    """Validate from-debate pipeline output matches the frontend contract."""

    @pytest.mark.asyncio
    async def test_debate_result_shape(self, handler):
        resp = await handler.handle_from_debate({
            "cartographer_data": {
                "nodes": [
                    {"id": "n1", "type": "proposal", "summary": "Rate limiter", "content": "Build it"},
                    {"id": "n2", "type": "evidence", "summary": "Reduces errors", "content": "Proof"},
                ],
                "edges": [
                    {"source_id": "n2", "target_id": "n1", "relation": "supports"},
                ],
            },
            "auto_advance": True,
        })
        result = resp["result"]

        # Top-level required fields
        assert "pipeline_id" in result
        assert "ideas" in result
        assert "goals" in result
        assert "actions" in result
        assert "orchestration" in result
        assert "transitions" in result
        assert "stage_status" in result
        assert "integrity_hash" in result
        assert "provenance_count" in result


# =============================================================================
# Advance stage contract
# =============================================================================


class TestAdvanceContract:
    """Validate advance endpoint returns updated result."""

    @pytest.mark.asyncio
    async def test_advance_returns_result_dict(self, handler):
        create_resp = await handler.handle_from_ideas({
            "ideas": ["Build rate limiter", "Add caching"],
            "auto_advance": False,
        })
        pid = create_resp["pipeline_id"]

        advance_resp = await handler.handle_advance({
            "pipeline_id": pid,
            "target_stage": "actions",
        })
        assert "result" in advance_resp
        assert advance_resp["result"]["stage_status"]["actions"] == "complete"
        # Result should still have the full shape
        result = advance_resp["result"]
        assert "ideas" in result
        assert "goals" in result
        assert "actions" in result


# =============================================================================
# Get pipeline contract
# =============================================================================


class TestGetPipelineContract:
    """Validate GET pipeline returns the full PipelineResultResponse shape."""

    @pytest.mark.asyncio
    async def test_get_returns_full_shape(self, handler):
        create_resp = await handler.handle_from_ideas({
            "ideas": ["Test idea"],
            "auto_advance": True,
        })
        pid = create_resp["pipeline_id"]

        result = await handler.handle_get_pipeline(pid)
        # Must have all required fields for PipelineResultResponse
        assert "pipeline_id" in result
        assert "ideas" in result
        assert "goals" in result
        assert "actions" in result
        assert "orchestration" in result
        assert "transitions" in result
        assert "stage_status" in result
        assert "integrity_hash" in result
        assert "provenance_count" in result


# =============================================================================
# Node type contract
# =============================================================================


class TestNodeTypeContract:
    """Validate node types match what nodeTypes registry expects."""

    VALID_IDEA_TYPES = {"concept", "cluster", "question", "insight", "evidence", "assumption", "constraint"}
    VALID_GOAL_TYPES = {"goal", "principle", "strategy", "milestone", "metric", "risk"}
    VALID_NODE_TYPES = {"ideaNode", "goalNode", "actionNode", "orchestrationNode"}

    @pytest.mark.asyncio
    async def test_idea_nodes_have_valid_types(self, handler):
        resp = await handler.handle_from_ideas({
            "ideas": ["Build rate limiter", "Add caching", "Improve docs"],
            "auto_advance": True,
        })
        ideas = resp["result"]["ideas"]
        for node in ideas["nodes"]:
            assert node["type"] in self.VALID_NODE_TYPES, f"Bad node type: {node['type']}"
            data = node["data"]
            assert "idea_type" in data or "ideaType" in data

    @pytest.mark.asyncio
    async def test_goals_have_required_fields(self, handler):
        resp = await handler.handle_from_ideas({
            "ideas": ["Build rate limiter", "Add caching"],
            "auto_advance": True,
        })
        goals = resp["result"]["goals"]["goals"]
        for goal in goals:
            assert "id" in goal
            assert "title" in goal
            assert "type" in goal
            assert "confidence" in goal
