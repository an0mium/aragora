"""Tests for PruningOperationsMixin (aragora/server/handlers/knowledge_base/mound/pruning.py).

Covers all six async methods on the mixin:
- get_prunable_items     (GET  /pruning/items)
- execute_prune          (POST /pruning/execute)
- auto_prune             (POST /pruning/auto)
- get_prune_history      (GET  /pruning/history)
- restore_pruned_item    (POST /pruning/restore)
- apply_confidence_decay (POST /pruning/decay)

Each method is tested for:
- Success with valid inputs
- Mound not available (503)
- Missing required parameters (400)
- Invalid parameter values (400)
- Internal errors from mound operations (500)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.knowledge_base.mound.pruning import (
    PruningOperationsMixin,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    raw = result.body
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return -1
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ---------------------------------------------------------------------------
# Mock domain objects
# ---------------------------------------------------------------------------


class MockPruningAction(str, Enum):
    ARCHIVE = "archive"
    DELETE = "delete"
    DEMOTE = "demote"
    FLAG = "flag"


@dataclass
class MockPruningPolicy:
    policy_id: str = "auto_default"
    workspace_id: str = "default"
    name: str = "Auto Prune Policy"
    staleness_threshold: float = 0.9
    min_age_days: int = 30
    action: MockPruningAction = MockPruningAction.ARCHIVE
    enabled: bool = True


@dataclass
class MockPrunableItem:
    node_id: str = "node-001"
    content_preview: str = "Some stale content..."
    staleness_score: float = 0.95
    confidence: float = 0.3
    retrieval_count: int = 2
    last_retrieved_at: datetime | None = None
    tier: str = "cold"
    created_at: datetime = field(default_factory=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc))
    prune_reason: str = "stale_and_unused"
    recommended_action: MockPruningAction = MockPruningAction.ARCHIVE


@dataclass
class MockPruneResult:
    workspace_id: str = "default"
    executed_at: datetime = field(default_factory=lambda: datetime(2026, 2, 1, tzinfo=timezone.utc))
    items_analyzed: int = 10
    items_pruned: int = 5
    items_archived: int = 3
    items_deleted: int = 1
    items_demoted: int = 1
    items_flagged: int = 0
    pruned_item_ids: list[str] = field(default_factory=lambda: ["n1", "n2", "n3", "n4", "n5"])
    errors: list[str] = field(default_factory=list)
    policy_id: str = "policy-001"


@dataclass
class MockPruneHistory:
    history_id: str = "hist-001"
    executed_at: datetime = field(
        default_factory=lambda: datetime(2026, 1, 15, tzinfo=timezone.utc)
    )
    policy_id: str = "policy-001"
    action: MockPruningAction = MockPruningAction.ARCHIVE
    items_pruned: int = 7
    pruned_item_ids: list[str] = field(default_factory=lambda: ["n10", "n11"])
    reason: str = "auto_prune"
    executed_by: str = "system"


# ---------------------------------------------------------------------------
# Concrete test class combining the mixin with stubs
# ---------------------------------------------------------------------------


class PruningTestHandler(PruningOperationsMixin):
    """Concrete handler for testing the pruning mixin."""

    def __init__(self, mound=None):
        self._mound = mound
        self.ctx: dict[str, Any] = {}

    def _get_mound(self):
        return self._mound


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_mound():
    """Create a mock KnowledgeMound with AsyncMock methods for pruning."""
    mound = MagicMock()
    mound.get_prunable_items = AsyncMock(return_value=[])
    mound.prune_items = AsyncMock(return_value=MockPruneResult())
    mound.auto_prune = AsyncMock(return_value=MockPruneResult())
    mound.get_prune_history = AsyncMock(return_value=[])
    mound.restore_pruned_item = AsyncMock(return_value=True)
    mound.apply_confidence_decay = AsyncMock(return_value=42)
    return mound


@pytest.fixture
def handler(mock_mound):
    """Create a PruningTestHandler with a mock mound."""
    return PruningTestHandler(mound=mock_mound)


@pytest.fixture
def handler_no_mound():
    """Create a PruningTestHandler with no mound (returns None)."""
    return PruningTestHandler(mound=None)


# Patch target for PruningAction used inside execute_prune and auto_prune
_PRUNING_ACTION_PATCH = "aragora.server.handlers.knowledge_base.mound.pruning.PruningAction"
_PRUNING_POLICY_PATCH = "aragora.server.handlers.knowledge_base.mound.pruning.PruningPolicy"


# ============================================================================
# Tests: get_prunable_items
# ============================================================================


class TestGetPrunableItems:
    """Test the get_prunable_items method."""

    @pytest.mark.asyncio
    async def test_success_returns_items(self, handler, mock_mound):
        """Successful call returns items list with metadata."""
        items = [
            MockPrunableItem(
                node_id="n1",
                staleness_score=0.95,
                confidence=0.2,
                retrieval_count=1,
                last_retrieved_at=datetime(2025, 6, 1, tzinfo=timezone.utc),
            ),
            MockPrunableItem(
                node_id="n2",
                staleness_score=0.98,
                confidence=0.1,
                last_retrieved_at=None,
            ),
        ]
        mock_mound.get_prunable_items = AsyncMock(return_value=items)

        result = await handler.get_prunable_items(workspace_id="ws-1")
        body = _body(result)

        assert _status(result) == 200
        assert body["workspace_id"] == "ws-1"
        assert body["items_found"] == 2
        assert len(body["items"]) == 2
        assert body["items"][0]["node_id"] == "n1"
        assert body["items"][0]["staleness_score"] == 0.95
        assert body["items"][0]["confidence"] == 0.2
        assert body["items"][0]["retrieval_count"] == 1
        assert body["items"][0]["last_retrieved_at"] == "2025-06-01T00:00:00+00:00"
        assert body["items"][0]["recommended_action"] == "archive"
        # Second item has None last_retrieved_at
        assert body["items"][1]["last_retrieved_at"] is None

    @pytest.mark.asyncio
    async def test_success_empty_list(self, handler, mock_mound):
        """No prunable items returns empty list with count 0."""
        mock_mound.get_prunable_items = AsyncMock(return_value=[])
        result = await handler.get_prunable_items(workspace_id="ws-1")
        body = _body(result)
        assert _status(result) == 200
        assert body["items_found"] == 0
        assert body["items"] == []

    @pytest.mark.asyncio
    async def test_custom_parameters(self, handler, mock_mound):
        """Custom staleness_threshold, min_age_days, limit are forwarded."""
        mock_mound.get_prunable_items = AsyncMock(return_value=[])
        result = await handler.get_prunable_items(
            workspace_id="ws-2",
            staleness_threshold=0.7,
            min_age_days=60,
            limit=25,
        )
        body = _body(result)
        assert body["staleness_threshold"] == 0.7
        assert body["min_age_days"] == 60

        mock_mound.get_prunable_items.assert_awaited_once_with(
            workspace_id="ws-2",
            staleness_threshold=0.7,
            min_age_days=60,
            limit=25,
        )

    @pytest.mark.asyncio
    async def test_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        result = await handler_no_mound.get_prunable_items(workspace_id="ws-1")
        assert _status(result) == 503
        body = _body(result)
        assert "not available" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_empty_workspace_returns_400(self, handler):
        """Empty workspace_id returns 400."""
        result = await handler.get_prunable_items(workspace_id="")
        assert _status(result) == 400
        assert "workspace_id" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_mound_raises_value_error_returns_500(self, handler, mock_mound):
        """ValueError from mound returns 500."""
        mock_mound.get_prunable_items = AsyncMock(side_effect=ValueError("bad query"))
        result = await handler.get_prunable_items(workspace_id="ws-1")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_key_error_returns_500(self, handler, mock_mound):
        """KeyError from mound returns 500."""
        mock_mound.get_prunable_items = AsyncMock(side_effect=KeyError("missing"))
        result = await handler.get_prunable_items(workspace_id="ws-1")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_os_error_returns_500(self, handler, mock_mound):
        """OSError from mound returns 500."""
        mock_mound.get_prunable_items = AsyncMock(side_effect=OSError("disk fail"))
        result = await handler.get_prunable_items(workspace_id="ws-1")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_type_error_returns_500(self, handler, mock_mound):
        """TypeError from mound returns 500."""
        mock_mound.get_prunable_items = AsyncMock(side_effect=TypeError("type mismatch"))
        result = await handler.get_prunable_items(workspace_id="ws-1")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_raises_attribute_error_returns_500(self, handler, mock_mound):
        """AttributeError from mound returns 500."""
        mock_mound.get_prunable_items = AsyncMock(side_effect=AttributeError("no attr"))
        result = await handler.get_prunable_items(workspace_id="ws-1")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_item_fields_present(self, handler, mock_mound):
        """All expected item fields are present in the response."""
        item = MockPrunableItem(
            node_id="n-check",
            content_preview="preview text",
            staleness_score=0.92,
            confidence=0.4,
            retrieval_count=5,
            last_retrieved_at=datetime(2025, 12, 1, tzinfo=timezone.utc),
            tier="warm",
            created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
            prune_reason="low_confidence",
            recommended_action=MockPruningAction.DEMOTE,
        )
        mock_mound.get_prunable_items = AsyncMock(return_value=[item])
        result = await handler.get_prunable_items(workspace_id="ws-1")
        body = _body(result)
        item_dict = body["items"][0]

        expected_fields = [
            "node_id",
            "content_preview",
            "staleness_score",
            "confidence",
            "retrieval_count",
            "last_retrieved_at",
            "tier",
            "created_at",
            "prune_reason",
            "recommended_action",
        ]
        for f in expected_fields:
            assert f in item_dict, f"Missing field: {f}"

        assert item_dict["tier"] == "warm"
        assert item_dict["prune_reason"] == "low_confidence"
        assert item_dict["recommended_action"] == "demote"

    @pytest.mark.asyncio
    async def test_default_parameters_in_response(self, handler, mock_mound):
        """Default staleness_threshold and min_age_days are reflected in response."""
        mock_mound.get_prunable_items = AsyncMock(return_value=[])
        result = await handler.get_prunable_items(workspace_id="ws-1")
        body = _body(result)
        assert body["staleness_threshold"] == 0.9
        assert body["min_age_days"] == 30


# ============================================================================
# Tests: execute_prune
# ============================================================================


class TestExecutePrune:
    """Test the execute_prune method."""

    @pytest.mark.asyncio
    async def test_success(self, handler, mock_mound):
        """Successful prune returns result counts."""
        result_obj = MockPruneResult(
            workspace_id="ws-1",
            items_analyzed=10,
            items_pruned=5,
            items_archived=3,
            items_deleted=1,
            items_demoted=1,
            items_flagged=0,
            pruned_item_ids=["a", "b", "c", "d", "e"],
        )
        mock_mound.prune_items = AsyncMock(return_value=result_obj)

        with patch(
            "aragora.knowledge.mound.ops.pruning.PruningAction",
            MockPruningAction,
        ):
            result = await handler.execute_prune(
                workspace_id="ws-1",
                item_ids=["a", "b", "c"],
                action="archive",
                reason="manual_cleanup",
            )

        body = _body(result)
        assert _status(result) == 200
        assert body["success"] is True
        assert body["workspace_id"] == "ws-1"
        assert body["items_analyzed"] == 10
        assert body["items_pruned"] == 5
        assert body["items_archived"] == 3
        assert body["items_deleted"] == 1
        assert body["items_demoted"] == 1
        assert body["items_flagged"] == 0
        assert len(body["pruned_item_ids"]) == 5
        assert "executed_at" in body

    @pytest.mark.asyncio
    async def test_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        result = await handler_no_mound.execute_prune(workspace_id="ws-1", item_ids=["a"])
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_empty_workspace_returns_400(self, handler):
        """Empty workspace_id returns 400."""
        result = await handler.execute_prune(workspace_id="", item_ids=["a"])
        assert _status(result) == 400
        assert "required" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_empty_item_ids_returns_400(self, handler):
        """Empty item_ids returns 400."""
        result = await handler.execute_prune(workspace_id="ws-1", item_ids=[])
        assert _status(result) == 400
        assert "required" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_invalid_action_returns_400(self, handler):
        """Invalid action string returns 400 with valid actions listed."""
        with patch(
            "aragora.knowledge.mound.ops.pruning.PruningAction",
            MockPruningAction,
        ):
            result = await handler.execute_prune(
                workspace_id="ws-1",
                item_ids=["a"],
                action="invalid_action",
            )
        assert _status(result) == 400
        body = _body(result)
        assert "invalid" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_all_actions_accepted(self, handler, mock_mound):
        """All valid action values are accepted."""
        for action_val in ["archive", "delete", "demote", "flag"]:
            mock_mound.prune_items = AsyncMock(return_value=MockPruneResult())
            with patch(
                "aragora.knowledge.mound.ops.pruning.PruningAction",
                MockPruningAction,
            ):
                result = await handler.execute_prune(
                    workspace_id="ws-1",
                    item_ids=["a"],
                    action=action_val,
                )
            assert _status(result) == 200, f"Action '{action_val}' should be accepted"

    @pytest.mark.asyncio
    async def test_mound_raises_error_returns_500(self, handler, mock_mound):
        """Error from mound.prune_items returns 500."""
        mock_mound.prune_items = AsyncMock(side_effect=ValueError("prune failed"))
        with patch(
            "aragora.knowledge.mound.ops.pruning.PruningAction",
            MockPruningAction,
        ):
            result = await handler.execute_prune(
                workspace_id="ws-1", item_ids=["a"], action="archive"
            )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_default_action_and_reason(self, handler, mock_mound):
        """Default action is 'archive' and default reason is 'manual_prune'."""
        mock_mound.prune_items = AsyncMock(return_value=MockPruneResult())
        with patch(
            "aragora.knowledge.mound.ops.pruning.PruningAction",
            MockPruningAction,
        ):
            await handler.execute_prune(workspace_id="ws-1", item_ids=["x"])
        call_kwargs = mock_mound.prune_items.call_args.kwargs
        assert call_kwargs["action"] == MockPruningAction.ARCHIVE
        assert call_kwargs["reason"] == "manual_prune"

    @pytest.mark.asyncio
    async def test_result_has_errors_field(self, handler, mock_mound):
        """Result includes errors list even when empty."""
        result_obj = MockPruneResult(errors=["partial failure on node-3"])
        mock_mound.prune_items = AsyncMock(return_value=result_obj)
        with patch(
            "aragora.knowledge.mound.ops.pruning.PruningAction",
            MockPruningAction,
        ):
            result = await handler.execute_prune(workspace_id="ws-1", item_ids=["a"])
        body = _body(result)
        assert body["errors"] == ["partial failure on node-3"]

    @pytest.mark.asyncio
    async def test_result_without_errors_attr(self, handler, mock_mound):
        """Result without errors attr returns empty list."""
        result_obj = MockPruneResult()
        # Remove the errors attribute to trigger hasattr check
        del result_obj.errors
        mock_mound.prune_items = AsyncMock(return_value=result_obj)
        with patch(
            "aragora.knowledge.mound.ops.pruning.PruningAction",
            MockPruningAction,
        ):
            result = await handler.execute_prune(workspace_id="ws-1", item_ids=["a"])
        body = _body(result)
        assert body["errors"] == []


# ============================================================================
# Tests: auto_prune
# ============================================================================


class TestAutoPrune:
    """Test the auto_prune method."""

    @pytest.mark.asyncio
    async def test_success_dry_run(self, handler, mock_mound):
        """Successful dry run returns result with dry_run=True."""
        result_obj = MockPruneResult(
            workspace_id="ws-1",
            items_analyzed=50,
            items_pruned=10,
            items_archived=8,
            items_deleted=0,
            items_demoted=2,
            items_flagged=0,
        )
        mock_mound.auto_prune = AsyncMock(return_value=result_obj)

        with (
            patch(
                "aragora.knowledge.mound.ops.pruning.PruningAction",
                MockPruningAction,
            ),
            patch(
                "aragora.knowledge.mound.ops.pruning.PruningPolicy",
                MockPruningPolicy,
            ),
        ):
            result = await handler.auto_prune(
                workspace_id="ws-1",
                dry_run=True,
            )

        body = _body(result)
        assert _status(result) == 200
        assert body["success"] is True
        assert body["dry_run"] is True
        assert body["workspace_id"] == "ws-1"
        assert body["items_analyzed"] == 50
        assert body["items_pruned"] == 10

    @pytest.mark.asyncio
    async def test_success_actual_run(self, handler, mock_mound):
        """Successful actual prune (dry_run=False)."""
        result_obj = MockPruneResult(workspace_id="ws-1")
        mock_mound.auto_prune = AsyncMock(return_value=result_obj)

        with (
            patch(
                "aragora.knowledge.mound.ops.pruning.PruningAction",
                MockPruningAction,
            ),
            patch(
                "aragora.knowledge.mound.ops.pruning.PruningPolicy",
                MockPruningPolicy,
            ),
        ):
            result = await handler.auto_prune(
                workspace_id="ws-1",
                dry_run=False,
            )

        body = _body(result)
        assert _status(result) == 200
        assert body["dry_run"] is False

    @pytest.mark.asyncio
    async def test_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        result = await handler_no_mound.auto_prune(workspace_id="ws-1")
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_empty_workspace_returns_400(self, handler):
        """Empty workspace_id returns 400."""
        result = await handler.auto_prune(workspace_id="")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_invalid_action_returns_400(self, handler):
        """Invalid action returns 400."""
        with patch(
            "aragora.knowledge.mound.ops.pruning.PruningAction",
            MockPruningAction,
        ):
            result = await handler.auto_prune(
                workspace_id="ws-1",
                action="nonexistent",
            )
        assert _status(result) == 400
        body = _body(result)
        assert "invalid" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_custom_policy_params(self, handler, mock_mound):
        """Custom staleness_threshold, min_age_days, policy_id are forwarded."""
        result_obj = MockPruneResult()
        mock_mound.auto_prune = AsyncMock(return_value=result_obj)

        with (
            patch(
                "aragora.knowledge.mound.ops.pruning.PruningAction",
                MockPruningAction,
            ),
            patch(
                "aragora.knowledge.mound.ops.pruning.PruningPolicy",
                MockPruningPolicy,
            ),
        ):
            result = await handler.auto_prune(
                workspace_id="ws-custom",
                policy_id="custom-policy-123",
                staleness_threshold=0.8,
                min_age_days=60,
                action="delete",
                dry_run=False,
            )

        assert _status(result) == 200
        mock_mound.auto_prune.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_auto_generated_policy_id(self, handler, mock_mound):
        """When policy_id is None, an auto-generated ID is used."""
        result_obj = MockPruneResult()
        # Remove policy_id from result so the fallback in the handler is used
        del result_obj.policy_id
        mock_mound.auto_prune = AsyncMock(return_value=result_obj)

        with (
            patch(
                "aragora.knowledge.mound.ops.pruning.PruningAction",
                MockPruningAction,
            ),
            patch(
                "aragora.knowledge.mound.ops.pruning.PruningPolicy",
                MockPruningPolicy,
            ),
        ):
            result = await handler.auto_prune(
                workspace_id="ws-1",
                policy_id=None,
            )

        body = _body(result)
        assert _status(result) == 200
        assert "policy_id" in body

    @pytest.mark.asyncio
    async def test_result_has_policy_id_from_result(self, handler, mock_mound):
        """When result has policy_id attr, it is used."""
        result_obj = MockPruneResult(policy_id="from-result")
        mock_mound.auto_prune = AsyncMock(return_value=result_obj)

        with (
            patch(
                "aragora.knowledge.mound.ops.pruning.PruningAction",
                MockPruningAction,
            ),
            patch(
                "aragora.knowledge.mound.ops.pruning.PruningPolicy",
                MockPruningPolicy,
            ),
        ):
            result = await handler.auto_prune(workspace_id="ws-1")

        body = _body(result)
        assert body["policy_id"] == "from-result"

    @pytest.mark.asyncio
    async def test_mound_raises_error_returns_500(self, handler, mock_mound):
        """Error from mound.auto_prune returns 500."""
        mock_mound.auto_prune = AsyncMock(side_effect=OSError("storage failure"))

        with (
            patch(
                "aragora.knowledge.mound.ops.pruning.PruningAction",
                MockPruningAction,
            ),
            patch(
                "aragora.knowledge.mound.ops.pruning.PruningPolicy",
                MockPruningPolicy,
            ),
        ):
            result = await handler.auto_prune(workspace_id="ws-1")

        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_result_without_errors_attr(self, handler, mock_mound):
        """Result without errors attr returns empty list."""
        result_obj = MockPruneResult()
        del result_obj.errors
        mock_mound.auto_prune = AsyncMock(return_value=result_obj)

        with (
            patch(
                "aragora.knowledge.mound.ops.pruning.PruningAction",
                MockPruningAction,
            ),
            patch(
                "aragora.knowledge.mound.ops.pruning.PruningPolicy",
                MockPruningPolicy,
            ),
        ):
            result = await handler.auto_prune(workspace_id="ws-1")

        body = _body(result)
        assert body["errors"] == []

    @pytest.mark.asyncio
    async def test_default_dry_run_is_true(self, handler, mock_mound):
        """Default dry_run parameter is True."""
        result_obj = MockPruneResult()
        mock_mound.auto_prune = AsyncMock(return_value=result_obj)

        with (
            patch(
                "aragora.knowledge.mound.ops.pruning.PruningAction",
                MockPruningAction,
            ),
            patch(
                "aragora.knowledge.mound.ops.pruning.PruningPolicy",
                MockPruningPolicy,
            ),
        ):
            result = await handler.auto_prune(workspace_id="ws-1")

        body = _body(result)
        assert body["dry_run"] is True
        # Verify dry_run=True was passed to mound
        call_kwargs = mock_mound.auto_prune.call_args.kwargs
        assert call_kwargs["dry_run"] is True


# ============================================================================
# Tests: get_prune_history
# ============================================================================


class TestGetPruneHistory:
    """Test the get_prune_history method."""

    @pytest.mark.asyncio
    async def test_success_returns_entries(self, handler, mock_mound):
        """Successful call returns history entries."""
        history = [
            MockPruneHistory(
                history_id="h1",
                items_pruned=5,
                reason="auto_prune",
            ),
            MockPruneHistory(
                history_id="h2",
                items_pruned=3,
                action=MockPruningAction.DELETE,
                reason="manual_cleanup",
                executed_by="admin",
            ),
        ]
        mock_mound.get_prune_history = AsyncMock(return_value=history)

        result = await handler.get_prune_history(workspace_id="ws-1")
        body = _body(result)

        assert _status(result) == 200
        assert body["workspace_id"] == "ws-1"
        assert len(body["entries"]) == 2
        e0 = body["entries"][0]
        assert e0["history_id"] == "h1"
        assert e0["items_pruned"] == 5
        assert e0["action"] == "archive"
        assert e0["reason"] == "auto_prune"
        e1 = body["entries"][1]
        assert e1["history_id"] == "h2"
        assert e1["action"] == "delete"
        assert e1["executed_by"] == "admin"

    @pytest.mark.asyncio
    async def test_success_empty_history(self, handler, mock_mound):
        """Empty history returns empty entries list."""
        mock_mound.get_prune_history = AsyncMock(return_value=[])
        result = await handler.get_prune_history(workspace_id="ws-1")
        body = _body(result)
        assert _status(result) == 200
        assert body["entries"] == []

    @pytest.mark.asyncio
    async def test_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        result = await handler_no_mound.get_prune_history(workspace_id="ws-1")
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_empty_workspace_returns_400(self, handler):
        """Empty workspace_id returns 400."""
        result = await handler.get_prune_history(workspace_id="")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_custom_limit(self, handler, mock_mound):
        """Custom limit is forwarded to mound."""
        mock_mound.get_prune_history = AsyncMock(return_value=[])
        await handler.get_prune_history(workspace_id="ws-1", limit=10)
        mock_mound.get_prune_history.assert_awaited_once_with(
            workspace_id="ws-1",
            limit=10,
            since=None,
        )

    @pytest.mark.asyncio
    async def test_since_parameter_valid(self, handler, mock_mound):
        """Valid since datetime string is parsed and forwarded."""
        mock_mound.get_prune_history = AsyncMock(return_value=[])
        result = await handler.get_prune_history(
            workspace_id="ws-1",
            since="2026-01-01T00:00:00",
        )
        assert _status(result) == 200
        call_kwargs = mock_mound.get_prune_history.call_args.kwargs
        assert call_kwargs["since"] == datetime.fromisoformat("2026-01-01T00:00:00")

    @pytest.mark.asyncio
    async def test_since_parameter_invalid_returns_400(self, handler):
        """Invalid since datetime string returns 400."""
        result = await handler.get_prune_history(
            workspace_id="ws-1",
            since="not-a-date",
        )
        assert _status(result) == 400
        body = _body(result)
        assert "since" in body["error"].lower() or "datetime" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_since_none_passes_none(self, handler, mock_mound):
        """since=None passes None to mound."""
        mock_mound.get_prune_history = AsyncMock(return_value=[])
        await handler.get_prune_history(workspace_id="ws-1", since=None)
        call_kwargs = mock_mound.get_prune_history.call_args.kwargs
        assert call_kwargs["since"] is None

    @pytest.mark.asyncio
    async def test_mound_raises_error_returns_500(self, handler, mock_mound):
        """Error from mound.get_prune_history returns 500."""
        mock_mound.get_prune_history = AsyncMock(side_effect=TypeError("corrupt"))
        result = await handler.get_prune_history(workspace_id="ws-1")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_history_entry_fields(self, handler, mock_mound):
        """All expected entry fields are present."""
        entry = MockPruneHistory()
        mock_mound.get_prune_history = AsyncMock(return_value=[entry])
        result = await handler.get_prune_history(workspace_id="ws-1")
        body = _body(result)
        e = body["entries"][0]
        expected = [
            "history_id",
            "executed_at",
            "policy_id",
            "action",
            "items_pruned",
            "pruned_item_ids",
            "reason",
            "executed_by",
        ]
        for f in expected:
            assert f in e, f"Missing field: {f}"

    @pytest.mark.asyncio
    async def test_action_as_string(self, handler, mock_mound):
        """Action without .value attribute (plain string) is stringified."""
        entry = MockPruneHistory()
        # Replace enum action with a plain string
        entry.action = "archive"  # type: ignore[assignment]
        mock_mound.get_prune_history = AsyncMock(return_value=[entry])
        result = await handler.get_prune_history(workspace_id="ws-1")
        body = _body(result)
        assert body["entries"][0]["action"] == "archive"


# ============================================================================
# Tests: restore_pruned_item
# ============================================================================


class TestRestorePrunedItem:
    """Test the restore_pruned_item method."""

    @pytest.mark.asyncio
    async def test_success(self, handler, mock_mound):
        """Successful restore returns success message."""
        mock_mound.restore_pruned_item = AsyncMock(return_value=True)
        result = await handler.restore_pruned_item(workspace_id="ws-1", node_id="n-42")
        body = _body(result)
        assert _status(result) == 200
        assert body["success"] is True
        assert body["workspace_id"] == "ws-1"
        assert body["node_id"] == "n-42"
        assert "restored" in body["message"].lower()

    @pytest.mark.asyncio
    async def test_not_found_returns_404(self, handler, mock_mound):
        """When mound returns False, 404 is returned."""
        mock_mound.restore_pruned_item = AsyncMock(return_value=False)
        result = await handler.restore_pruned_item(workspace_id="ws-1", node_id="n-gone")
        assert _status(result) == 404
        body = _body(result)
        assert "n-gone" in body["error"]

    @pytest.mark.asyncio
    async def test_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        result = await handler_no_mound.restore_pruned_item(workspace_id="ws-1", node_id="n-1")
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_empty_workspace_returns_400(self, handler):
        """Empty workspace_id returns 400."""
        result = await handler.restore_pruned_item(workspace_id="", node_id="n-1")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_empty_node_id_returns_400(self, handler):
        """Empty node_id returns 400."""
        result = await handler.restore_pruned_item(workspace_id="ws-1", node_id="")
        assert _status(result) == 400
        assert "required" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_mound_raises_error_returns_500(self, handler, mock_mound):
        """Error from mound.restore_pruned_item returns 500."""
        mock_mound.restore_pruned_item = AsyncMock(side_effect=KeyError("bad node"))
        result = await handler.restore_pruned_item(workspace_id="ws-1", node_id="n-1")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_mound_called_with_correct_args(self, handler, mock_mound):
        """Mound is called with the correct workspace_id and node_id."""
        mock_mound.restore_pruned_item = AsyncMock(return_value=True)
        await handler.restore_pruned_item(workspace_id="ws-x", node_id="n-y")
        mock_mound.restore_pruned_item.assert_awaited_once_with(
            workspace_id="ws-x",
            node_id="n-y",
        )


# ============================================================================
# Tests: apply_confidence_decay
# ============================================================================


class TestApplyConfidenceDecay:
    """Test the apply_confidence_decay method."""

    @pytest.mark.asyncio
    async def test_success(self, handler, mock_mound):
        """Successful decay returns items_decayed count."""
        mock_mound.apply_confidence_decay = AsyncMock(return_value=42)
        result = await handler.apply_confidence_decay(workspace_id="ws-1")
        body = _body(result)
        assert _status(result) == 200
        assert body["success"] is True
        assert body["workspace_id"] == "ws-1"
        assert body["items_decayed"] == 42
        assert body["decay_rate"] == 0.01
        assert body["min_confidence"] == 0.1

    @pytest.mark.asyncio
    async def test_custom_parameters(self, handler, mock_mound):
        """Custom decay_rate and min_confidence are forwarded."""
        mock_mound.apply_confidence_decay = AsyncMock(return_value=100)
        result = await handler.apply_confidence_decay(
            workspace_id="ws-1",
            decay_rate=0.05,
            min_confidence=0.2,
        )
        body = _body(result)
        assert body["decay_rate"] == 0.05
        assert body["min_confidence"] == 0.2
        assert body["items_decayed"] == 100

        mock_mound.apply_confidence_decay.assert_awaited_once_with(
            workspace_id="ws-1",
            decay_rate=0.05,
            min_confidence=0.2,
        )

    @pytest.mark.asyncio
    async def test_no_mound_returns_503(self, handler_no_mound):
        """Missing mound returns 503."""
        result = await handler_no_mound.apply_confidence_decay(workspace_id="ws-1")
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_empty_workspace_returns_400(self, handler):
        """Empty workspace_id returns 400."""
        result = await handler.apply_confidence_decay(workspace_id="")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_decay_rate_zero_returns_400(self, handler):
        """decay_rate=0 returns 400 (must be between 0 and 1 exclusive)."""
        result = await handler.apply_confidence_decay(workspace_id="ws-1", decay_rate=0.0)
        assert _status(result) == 400
        assert "decay_rate" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_decay_rate_one_returns_400(self, handler):
        """decay_rate=1.0 returns 400."""
        result = await handler.apply_confidence_decay(workspace_id="ws-1", decay_rate=1.0)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_decay_rate_negative_returns_400(self, handler):
        """Negative decay_rate returns 400."""
        result = await handler.apply_confidence_decay(workspace_id="ws-1", decay_rate=-0.1)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_min_confidence_zero_returns_400(self, handler):
        """min_confidence=0 returns 400."""
        result = await handler.apply_confidence_decay(workspace_id="ws-1", min_confidence=0.0)
        assert _status(result) == 400
        assert "min_confidence" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_min_confidence_one_returns_400(self, handler):
        """min_confidence=1.0 returns 400."""
        result = await handler.apply_confidence_decay(workspace_id="ws-1", min_confidence=1.0)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_min_confidence_negative_returns_400(self, handler):
        """Negative min_confidence returns 400."""
        result = await handler.apply_confidence_decay(workspace_id="ws-1", min_confidence=-0.5)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_valid_boundary_decay_rate(self, handler, mock_mound):
        """decay_rate just above 0 is accepted."""
        mock_mound.apply_confidence_decay = AsyncMock(return_value=1)
        result = await handler.apply_confidence_decay(workspace_id="ws-1", decay_rate=0.001)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_valid_boundary_min_confidence(self, handler, mock_mound):
        """min_confidence just above 0 is accepted."""
        mock_mound.apply_confidence_decay = AsyncMock(return_value=1)
        result = await handler.apply_confidence_decay(workspace_id="ws-1", min_confidence=0.001)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_mound_raises_error_returns_500(self, handler, mock_mound):
        """Error from mound.apply_confidence_decay returns 500."""
        mock_mound.apply_confidence_decay = AsyncMock(side_effect=AttributeError("missing method"))
        result = await handler.apply_confidence_decay(workspace_id="ws-1")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_zero_items_decayed(self, handler, mock_mound):
        """Zero items_decayed is valid."""
        mock_mound.apply_confidence_decay = AsyncMock(return_value=0)
        result = await handler.apply_confidence_decay(workspace_id="ws-1")
        body = _body(result)
        assert _status(result) == 200
        assert body["items_decayed"] == 0
