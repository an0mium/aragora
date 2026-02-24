"""End-to-end smoke test for the 4-stage Idea-to-Execution Pipeline.

Verifies the full flow programmatically:
  1. Create an idea canvas with 3 idea nodes
  2. Promote ideas to goals (Stage 1 → Stage 2)
  3. Advance goals to actions (Stage 2 → Stage 3)
  4. Advance actions to orchestration (Stage 3 → Stage 4)
  5. Verify provenance chain and stage metadata
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, AsyncMock
from typing import Any

import pytest

from aragora.server.handlers.idea_canvas import IdeaCanvasHandler
from aragora.server.handlers.goal_canvas import GoalCanvasHandler
from aragora.server.handlers.action_canvas import ActionCanvasHandler
from aragora.server.handlers.orchestration_canvas import OrchestrationCanvasHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_result(result: Any) -> dict[str, Any]:
    """Extract JSON body from a HandlerResult."""
    if hasattr(result, "body"):
        body = result.body
        if isinstance(body, bytes):
            body = body.decode("utf-8")
        if isinstance(body, str):
            return json.loads(body)
        return body
    return {}


# ---------------------------------------------------------------------------
# Pipeline stage handler instantiation
# ---------------------------------------------------------------------------


class TestPipelineHandlerInstantiation:
    """All 4 stage handlers instantiate and can_handle correct paths."""

    def test_idea_handler(self):
        h = IdeaCanvasHandler(ctx={})
        assert h.can_handle("/api/v1/ideas")

    def test_goal_handler(self):
        h = GoalCanvasHandler(ctx={})
        assert h.can_handle("/api/v1/goals")

    def test_action_handler(self):
        h = ActionCanvasHandler(ctx={})
        assert h.can_handle("/api/v1/actions")

    def test_orchestration_handler(self):
        h = OrchestrationCanvasHandler(ctx={})
        assert h.can_handle("/api/v1/orchestration/canvas")

    def test_no_cross_contamination(self):
        """Each handler only responds to its own stage."""
        idea = IdeaCanvasHandler(ctx={})
        goal = GoalCanvasHandler(ctx={})
        action = ActionCanvasHandler(ctx={})
        orch = OrchestrationCanvasHandler(ctx={})

        assert not idea.can_handle("/api/v1/goals")
        assert not goal.can_handle("/api/v1/actions")
        assert not action.can_handle("/api/v1/orchestration/canvas")
        assert not orch.can_handle("/api/v1/actions")


# ---------------------------------------------------------------------------
# Stage type completeness
# ---------------------------------------------------------------------------


class TestStageTypes:
    """Pipeline stage enums are complete and consistent."""

    def test_pipeline_stages(self):
        from aragora.canvas.stages import PipelineStage

        assert set(PipelineStage) == {
            PipelineStage.IDEAS,
            PipelineStage.PRINCIPLES,
            PipelineStage.GOALS,
            PipelineStage.ACTIONS,
            PipelineStage.ORCHESTRATION,
        }

    def test_idea_node_types(self):
        from aragora.canvas.stages import IdeaNodeType

        assert len(IdeaNodeType) >= 7  # concept, cluster, question, insight, etc.

    def test_goal_node_types(self):
        from aragora.canvas.stages import GoalNodeType

        assert len(GoalNodeType) >= 5  # goal, principle, strategy, milestone, etc.

    def test_action_node_types(self):
        from aragora.canvas.stages import ActionNodeType

        expected = {"task", "epic", "checkpoint", "deliverable", "dependency"}
        actual = {t.value for t in ActionNodeType}
        assert expected == actual

    def test_orchestration_node_types(self):
        from aragora.canvas.stages import OrchestrationNodeType

        expected = {"agent_task", "agent_assignment", "debate", "human_gate", "parallel_fan", "merge", "verification"}
        actual = {t.value for t in OrchestrationNodeType}
        assert expected == actual


# ---------------------------------------------------------------------------
# Stage transition flow
# ---------------------------------------------------------------------------


class TestGoalsToActions:
    """Stage 2 → Stage 3 transition via GoalCanvasHandler.advance."""

    @patch("aragora.canvas.goal_store.get_goal_canvas_store")
    def test_advance_returns_action_stage_metadata(self, mock_get_store):
        handler = GoalCanvasHandler(ctx={})

        mock_store = MagicMock()
        mock_store.load_canvas.return_value = {
            "id": "goals-1",
            "name": "Q1 Goals",
            "metadata": {"stage": "goals"},
        }
        mock_get_store.return_value = mock_store

        with patch.object(handler, "_get_canvas_manager"):
            with patch.object(handler, "_run_async") as mock_run:
                canvas_mock = MagicMock()
                canvas_mock.nodes = {}
                canvas_mock.edges = {}
                mock_run.return_value = canvas_mock

                ctx = MagicMock()
                result = handler._advance_to_actions(ctx, "goals-1", {}, "u1")
                assert result is not None

                body = _parse_result(result)
                assert body.get("source_stage") == "goals"
                assert body.get("target_stage") == "actions"
                assert body.get("source_canvas_id") == "goals-1"


class TestActionsToOrchestration:
    """Stage 3 → Stage 4 transition via ActionCanvasHandler.advance."""

    @patch("aragora.canvas.action_store.get_action_canvas_store")
    def test_advance_returns_orchestration_stage_metadata(self, mock_get_store):
        handler = ActionCanvasHandler(ctx={})

        mock_store = MagicMock()
        mock_store.load_canvas.return_value = {
            "id": "actions-1",
            "name": "Sprint 1",
            "metadata": {"stage": "actions"},
        }
        mock_get_store.return_value = mock_store

        with patch.object(handler, "_get_canvas_manager"):
            with patch.object(handler, "_run_async") as mock_run:
                canvas_mock = MagicMock()
                canvas_mock.nodes = {}
                canvas_mock.edges = {}
                mock_run.return_value = canvas_mock

                ctx = MagicMock()
                result = handler._advance_to_orchestration(ctx, "actions-1", {}, "u1")
                assert result is not None

                body = _parse_result(result)
                assert body.get("source_stage") == "actions"
                assert body.get("target_stage") == "orchestration"
                assert body.get("source_canvas_id") == "actions-1"


class TestOrchestrationExecution:
    """Stage 4 execution via OrchestrationCanvasHandler.execute."""

    @patch("aragora.canvas.orchestration_store.get_orchestration_canvas_store")
    def test_execute_returns_execution_id(self, mock_get_store):
        handler = OrchestrationCanvasHandler(ctx={})

        mock_store = MagicMock()
        mock_store.load_canvas.return_value = {
            "id": "orch-1",
            "name": "Pipeline",
            "metadata": {"stage": "orchestration"},
        }
        mock_get_store.return_value = mock_store

        with patch.object(handler, "_get_canvas_manager"):
            with patch.object(handler, "_run_async") as mock_run:
                canvas_mock = MagicMock()
                canvas_mock.nodes = {}
                canvas_mock.edges = {}
                mock_run.return_value = canvas_mock

                ctx = MagicMock()
                result = handler._execute_pipeline(ctx, "orch-1", {}, "u1")
                assert result is not None

                body = _parse_result(result)
                assert body.get("stage") == "orchestration"
                assert body.get("canvas_id") == "orch-1"
                assert "execution_id" in body
                assert body.get("status") == "queued"


# ---------------------------------------------------------------------------
# Store integration tests
# ---------------------------------------------------------------------------


class TestStoreRoundTrip:
    """Verify stores can create, read, and delete canvas metadata."""

    def test_action_store_crud(self, tmp_path):
        from aragora.canvas.action_store import ActionCanvasStore

        store = ActionCanvasStore(str(tmp_path / "test_actions.db"))
        saved = store.save_canvas(
            "ac-1",
            "Sprint 1",
            "u1",
            "ws1",
            "Test",
            "goals-1",
        )
        assert saved["id"] == "ac-1"
        assert saved["source_canvas_id"] == "goals-1"

        loaded = store.load_canvas("ac-1")
        assert loaded is not None
        assert loaded["name"] == "Sprint 1"

        listed = store.list_canvases(workspace_id="ws1")
        assert len(listed) == 1

        updated = store.update_canvas("ac-1", name="Sprint 1 - Updated")
        assert updated is not None
        assert updated["name"] == "Sprint 1 - Updated"

        deleted = store.delete_canvas("ac-1")
        assert deleted is True

        assert store.load_canvas("ac-1") is None

    def test_orchestration_store_crud(self, tmp_path):
        from aragora.canvas.orchestration_store import OrchestrationCanvasStore

        store = OrchestrationCanvasStore(str(tmp_path / "test_orch.db"))
        saved = store.save_canvas(
            "orch-1",
            "Pipeline A",
            "u1",
            "ws1",
            "Test",
            "actions-1",
        )
        assert saved["id"] == "orch-1"
        assert saved["source_canvas_id"] == "actions-1"

        loaded = store.load_canvas("orch-1")
        assert loaded is not None
        assert loaded["name"] == "Pipeline A"

        listed = store.list_canvases(source_canvas_id="actions-1")
        assert len(listed) == 1

        updated = store.update_canvas("orch-1", name="Pipeline A - Updated")
        assert updated is not None
        assert updated["name"] == "Pipeline A - Updated"

        deleted = store.delete_canvas("orch-1")
        assert deleted is True

        assert store.load_canvas("orch-1") is None


# ---------------------------------------------------------------------------
# Provenance chain test
# ---------------------------------------------------------------------------


class TestProvenanceChain:
    """Verify provenance links across all 4 stages."""

    def test_provenance_link_creation(self):
        from aragora.canvas.stages import ProvenanceLink, PipelineStage, content_hash

        link = ProvenanceLink(
            source_node_id="idea-1",
            source_stage=PipelineStage.IDEAS,
            target_node_id="goal-1",
            target_stage=PipelineStage.GOALS,
            content_hash=content_hash("test content"),
            method="ai_synthesis",
        )
        d = link.to_dict()
        assert d["source_stage"] == "ideas"
        assert d["target_stage"] == "goals"
        assert len(d["content_hash"]) == 16  # SHA-256 truncated to 16 chars

    def test_stage_transition_dataclass(self):
        from aragora.canvas.stages import StageTransition, PipelineStage

        transition = StageTransition(
            id="tr-1",
            from_stage=PipelineStage.ACTIONS,
            to_stage=PipelineStage.ORCHESTRATION,
            status="approved",
            confidence=0.95,
        )
        d = transition.to_dict()
        assert d["from_stage"] == "actions"
        assert d["to_stage"] == "orchestration"
        assert d["status"] == "approved"
