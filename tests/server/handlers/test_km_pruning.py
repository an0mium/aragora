"""
Tests for Knowledge Mound Pruning handler endpoints.

Tests pruning operations:
- GET /api/knowledge/mound/pruning/items - Get prunable items
- POST /api/knowledge/mound/pruning/execute - Prune specified items
- POST /api/knowledge/mound/pruning/auto - Run auto-prune with policy
- GET /api/knowledge/mound/pruning/history - Get pruning history
- POST /api/knowledge/mound/pruning/restore - Restore archived item
- POST /api/knowledge/mound/pruning/decay - Apply confidence decay
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from aragora.server.handlers.knowledge_base.mound.pruning import (
    PruningOperationsMixin,
)


class MockPruningAction(str, Enum):
    """Mock pruning action for testing."""

    ARCHIVE = "archive"
    DELETE = "delete"
    DEMOTE = "demote"
    FLAG = "flag"


@dataclass
class MockPrunableItem:
    """Mock prunable item for testing."""

    node_id: str = "node-123"
    content_preview: str = "Some old content..."
    staleness_score: float = 0.95
    confidence: float = 0.3
    retrieval_count: int = 2
    last_retrieved_at: datetime | None = None
    tier: str = "glacial"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    prune_reason: str = "Low confidence and usage"
    recommended_action: MockPruningAction = MockPruningAction.ARCHIVE


@dataclass
class MockPruneResult:
    """Mock prune result for testing."""

    workspace_id: str = "workspace-123"
    executed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    items_analyzed: int = 100
    items_pruned: int = 10
    items_archived: int = 8
    items_deleted: int = 0
    items_demoted: int = 2
    items_flagged: int = 0
    pruned_item_ids: list = field(default_factory=lambda: ["node-1", "node-2"])
    errors: list = field(default_factory=list)
    policy_id: str = "policy-123"


@dataclass
class MockPruneHistory:
    """Mock prune history entry for testing."""

    history_id: str = "history-123"
    executed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    policy_id: str = "policy-123"
    action: MockPruningAction = MockPruningAction.ARCHIVE
    items_pruned: int = 5
    pruned_item_ids: list = field(default_factory=lambda: ["node-1"])
    reason: str = "Automated pruning"
    executed_by: str = "system"


class MockMound:
    """Mock KnowledgeMound for testing."""

    def __init__(self):
        self.get_prunable_items = AsyncMock(
            return_value=[MockPrunableItem(), MockPrunableItem(node_id="node-456")]
        )
        self.prune_items = AsyncMock(return_value=MockPruneResult())
        self.auto_prune = AsyncMock(return_value=MockPruneResult())
        self.get_prune_history = AsyncMock(
            return_value=[MockPruneHistory(), MockPruneHistory(history_id="h-456")]
        )
        self.restore_pruned_item = AsyncMock(return_value=True)
        self.apply_confidence_decay = AsyncMock(return_value=50)


class MockPruningHandler(PruningOperationsMixin):
    """Handler for testing PruningOperationsMixin."""

    def __init__(self):
        self.mound = MockMound()
        self.ctx = {"user_id": "test-user", "org_id": "test-org"}

    def _get_mound(self):
        return self.mound


class MockPruningHandlerNoMound(PruningOperationsMixin):
    """Handler with no mound available."""

    def __init__(self):
        self.ctx = {}

    def _get_mound(self):
        return None


def parse_json_response(result):
    """Parse JSON response from HandlerResult dataclass."""
    body = result.body
    if isinstance(body, bytes):
        body = body.decode("utf-8")
    return json.loads(body)


@pytest.fixture
def handler():
    """Create test handler with mocked mound."""
    return MockPruningHandler()


@pytest.fixture
def handler_no_mound():
    """Create test handler without mound."""
    return MockPruningHandlerNoMound()


# Mock the decorators to bypass RBAC and rate limiting for tests
@pytest.fixture(autouse=True)
def mock_decorators():
    """Mock RBAC and rate limit decorators."""
    with (
        patch(
            "aragora.server.handlers.knowledge_base.mound.pruning.require_permission",
            lambda perm: lambda fn: fn,
        ),
        patch(
            "aragora.server.handlers.knowledge_base.mound.pruning.rate_limit",
            lambda **kwargs: lambda fn: fn,
        ),
    ):
        yield


class TestGetPrunableItems:
    """Tests for get_prunable_items endpoint."""

    @pytest.mark.asyncio
    async def test_get_items_success(self, handler):
        """Test successful retrieval of prunable items."""
        test_handler = MockPruningHandler()
        result = await test_handler.get_prunable_items(workspace_id="workspace-123")

        data = parse_json_response(result)
        assert data["workspace_id"] == "workspace-123"
        assert data["items_found"] == 2
        assert len(data["items"]) == 2

    @pytest.mark.asyncio
    async def test_get_items_with_threshold(self, handler):
        """Test retrieval with custom staleness threshold."""
        test_handler = MockPruningHandler()
        result = await test_handler.get_prunable_items(
            workspace_id="workspace-123",
            staleness_threshold=0.8,
            min_age_days=60,
        )

        data = parse_json_response(result)
        assert data["staleness_threshold"] == 0.8
        assert data["min_age_days"] == 60

    @pytest.mark.asyncio
    async def test_get_items_with_limit(self, handler):
        """Test retrieval with custom limit."""
        test_handler = MockPruningHandler()
        await test_handler.get_prunable_items(
            workspace_id="workspace-123",
            limit=50,
        )

        test_handler.mound.get_prunable_items.assert_called_once_with(
            workspace_id="workspace-123",
            staleness_threshold=0.9,
            min_age_days=30,
            limit=50,
        )

    @pytest.mark.asyncio
    async def test_get_items_missing_workspace_id(self, handler):
        """Test fails without workspace_id."""
        test_handler = MockPruningHandler()
        result = await test_handler.get_prunable_items(workspace_id="")

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 400
        assert "workspace_id is required" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_get_items_mound_unavailable(self, handler_no_mound):
        """Test when mound is unavailable."""
        result = await handler_no_mound.get_prunable_items(workspace_id="workspace-123")

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 503
        assert "not available" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_get_items_mound_error(self, handler):
        """Test handles mound errors."""
        test_handler = MockPruningHandler()
        test_handler.mound.get_prunable_items = AsyncMock(side_effect=Exception("Items error"))

        result = await test_handler.get_prunable_items(workspace_id="workspace-123")

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 500
        assert "error" in data


class TestExecutePrune:
    """Tests for execute_prune endpoint."""

    @pytest.mark.asyncio
    async def test_execute_success(self, handler):
        """Test successful pruning execution."""
        test_handler = MockPruningHandler()

        with patch(
            "aragora.knowledge.mound.ops.pruning.PruningAction",
            MockPruningAction,
        ):
            result = await test_handler.execute_prune(
                workspace_id="workspace-123",
                item_ids=["node-1", "node-2"],
                action="archive",
            )

        data = parse_json_response(result)
        assert data["success"] is True
        assert data["items_pruned"] == 10

    @pytest.mark.asyncio
    async def test_execute_missing_workspace_id(self, handler):
        """Test fails without workspace_id."""
        test_handler = MockPruningHandler()
        result = await test_handler.execute_prune(
            workspace_id="",
            item_ids=["node-1"],
        )

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 400
        assert "workspace_id and item_ids are required" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_execute_missing_item_ids(self, handler):
        """Test fails without item_ids."""
        test_handler = MockPruningHandler()
        result = await test_handler.execute_prune(
            workspace_id="workspace-123",
            item_ids=[],
        )

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 400
        assert "workspace_id and item_ids are required" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_execute_invalid_action(self, handler):
        """Test fails with invalid action."""
        test_handler = MockPruningHandler()

        with patch(
            "aragora.knowledge.mound.ops.pruning.PruningAction",
            MockPruningAction,
        ):
            result = await test_handler.execute_prune(
                workspace_id="workspace-123",
                item_ids=["node-1"],
                action="invalid_action",
            )

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 400
        assert "Invalid action" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_execute_mound_unavailable(self, handler_no_mound):
        """Test when mound is unavailable."""
        result = await handler_no_mound.execute_prune(
            workspace_id="workspace-123",
            item_ids=["node-1"],
        )

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 503
        assert "not available" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_execute_mound_error(self, handler):
        """Test handles mound errors."""
        test_handler = MockPruningHandler()
        test_handler.mound.prune_items = AsyncMock(side_effect=Exception("Prune error"))

        with patch(
            "aragora.knowledge.mound.ops.pruning.PruningAction",
            MockPruningAction,
        ):
            result = await test_handler.execute_prune(
                workspace_id="workspace-123",
                item_ids=["node-1"],
                action="archive",
            )

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 500
        assert "error" in data


class TestAutoPrune:
    """Tests for auto_prune endpoint."""

    @pytest.mark.asyncio
    async def test_auto_prune_dry_run(self, handler):
        """Test auto-prune in dry run mode."""
        test_handler = MockPruningHandler()

        with (
            patch(
                "aragora.knowledge.mound.ops.pruning.PruningAction",
                MockPruningAction,
            ),
            patch(
                "aragora.knowledge.mound.ops.pruning.PruningPolicy",
                lambda **kwargs: type("MockPolicy", (), kwargs)(),
            ),
        ):
            result = await test_handler.auto_prune(
                workspace_id="workspace-123",
                dry_run=True,
            )

        data = parse_json_response(result)
        assert data["success"] is True
        assert data["dry_run"] is True

    @pytest.mark.asyncio
    async def test_auto_prune_execute(self, handler):
        """Test auto-prune with execution."""
        test_handler = MockPruningHandler()

        with (
            patch(
                "aragora.knowledge.mound.ops.pruning.PruningAction",
                MockPruningAction,
            ),
            patch(
                "aragora.knowledge.mound.ops.pruning.PruningPolicy",
                lambda **kwargs: type("MockPolicy", (), kwargs)(),
            ),
        ):
            result = await test_handler.auto_prune(
                workspace_id="workspace-123",
                dry_run=False,
            )

        data = parse_json_response(result)
        assert data["success"] is True
        assert data["dry_run"] is False

    @pytest.mark.asyncio
    async def test_auto_prune_missing_workspace_id(self, handler):
        """Test fails without workspace_id."""
        test_handler = MockPruningHandler()
        result = await test_handler.auto_prune(workspace_id="")

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 400
        assert "workspace_id is required" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_auto_prune_invalid_action(self, handler):
        """Test fails with invalid action."""
        test_handler = MockPruningHandler()

        with patch(
            "aragora.knowledge.mound.ops.pruning.PruningAction",
            MockPruningAction,
        ):
            result = await test_handler.auto_prune(
                workspace_id="workspace-123",
                action="invalid_action",
            )

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 400
        assert "Invalid action" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_auto_prune_mound_unavailable(self, handler_no_mound):
        """Test when mound is unavailable."""
        result = await handler_no_mound.auto_prune(workspace_id="workspace-123")

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 503
        assert "not available" in data.get("error", "")


class TestGetPruneHistory:
    """Tests for get_prune_history endpoint."""

    @pytest.mark.asyncio
    async def test_get_history_success(self, handler):
        """Test successful history retrieval."""
        test_handler = MockPruningHandler()
        result = await test_handler.get_prune_history(workspace_id="workspace-123")

        data = parse_json_response(result)
        assert data["workspace_id"] == "workspace-123"
        assert len(data["entries"]) == 2

    @pytest.mark.asyncio
    async def test_get_history_with_limit(self, handler):
        """Test history with custom limit."""
        test_handler = MockPruningHandler()
        await test_handler.get_prune_history(
            workspace_id="workspace-123",
            limit=10,
        )

        test_handler.mound.get_prune_history.assert_called_once_with(
            workspace_id="workspace-123",
            limit=10,
            since=None,
        )

    @pytest.mark.asyncio
    async def test_get_history_with_since(self, handler):
        """Test history with since filter."""
        test_handler = MockPruningHandler()
        await test_handler.get_prune_history(
            workspace_id="workspace-123",
            since="2024-01-01T00:00:00",
        )

        call_args = test_handler.mound.get_prune_history.call_args
        assert call_args.kwargs["since"] is not None

    @pytest.mark.asyncio
    async def test_get_history_invalid_since(self, handler):
        """Test fails with invalid since datetime."""
        test_handler = MockPruningHandler()
        result = await test_handler.get_prune_history(
            workspace_id="workspace-123",
            since="not-a-date",
        )

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 400
        assert "Invalid 'since'" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_get_history_missing_workspace_id(self, handler):
        """Test fails without workspace_id."""
        test_handler = MockPruningHandler()
        result = await test_handler.get_prune_history(workspace_id="")

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 400
        assert "workspace_id is required" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_get_history_mound_unavailable(self, handler_no_mound):
        """Test when mound is unavailable."""
        result = await handler_no_mound.get_prune_history(workspace_id="workspace-123")

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 503
        assert "not available" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_get_history_mound_error(self, handler):
        """Test handles mound errors."""
        test_handler = MockPruningHandler()
        test_handler.mound.get_prune_history = AsyncMock(side_effect=Exception("History error"))

        result = await test_handler.get_prune_history(workspace_id="workspace-123")

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 500
        assert "error" in data


class TestRestorePrunedItem:
    """Tests for restore_pruned_item endpoint."""

    @pytest.mark.asyncio
    async def test_restore_success(self, handler):
        """Test successful item restoration."""
        test_handler = MockPruningHandler()
        result = await test_handler.restore_pruned_item(
            workspace_id="workspace-123",
            node_id="node-123",
        )

        data = parse_json_response(result)
        assert data["success"] is True
        assert data["node_id"] == "node-123"
        assert "restored" in data["message"]

    @pytest.mark.asyncio
    async def test_restore_not_found(self, handler):
        """Test restore when item not found."""
        test_handler = MockPruningHandler()
        test_handler.mound.restore_pruned_item = AsyncMock(return_value=False)

        result = await test_handler.restore_pruned_item(
            workspace_id="workspace-123",
            node_id="nonexistent",
        )

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 404
        assert "Could not restore" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_restore_missing_workspace_id(self, handler):
        """Test fails without workspace_id."""
        test_handler = MockPruningHandler()
        result = await test_handler.restore_pruned_item(
            workspace_id="",
            node_id="node-123",
        )

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 400
        assert "workspace_id and node_id are required" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_restore_missing_node_id(self, handler):
        """Test fails without node_id."""
        test_handler = MockPruningHandler()
        result = await test_handler.restore_pruned_item(
            workspace_id="workspace-123",
            node_id="",
        )

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 400
        assert "workspace_id and node_id are required" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_restore_mound_unavailable(self, handler_no_mound):
        """Test when mound is unavailable."""
        result = await handler_no_mound.restore_pruned_item(
            workspace_id="workspace-123",
            node_id="node-123",
        )

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 503
        assert "not available" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_restore_mound_error(self, handler):
        """Test handles mound errors."""
        test_handler = MockPruningHandler()
        test_handler.mound.restore_pruned_item = AsyncMock(side_effect=Exception("Restore error"))

        result = await test_handler.restore_pruned_item(
            workspace_id="workspace-123",
            node_id="node-123",
        )

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 500
        assert "error" in data


class TestApplyConfidenceDecay:
    """Tests for apply_confidence_decay endpoint."""

    @pytest.mark.asyncio
    async def test_decay_success(self, handler):
        """Test successful confidence decay."""
        test_handler = MockPruningHandler()
        result = await test_handler.apply_confidence_decay(
            workspace_id="workspace-123",
            decay_rate=0.05,
            min_confidence=0.2,
        )

        data = parse_json_response(result)
        assert data["success"] is True
        assert data["items_decayed"] == 50
        assert data["decay_rate"] == 0.05
        assert data["min_confidence"] == 0.2

    @pytest.mark.asyncio
    async def test_decay_missing_workspace_id(self, handler):
        """Test fails without workspace_id."""
        test_handler = MockPruningHandler()
        result = await test_handler.apply_confidence_decay(workspace_id="")

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 400
        assert "workspace_id is required" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_decay_invalid_rate_zero(self, handler):
        """Test fails with decay_rate of 0."""
        test_handler = MockPruningHandler()
        result = await test_handler.apply_confidence_decay(
            workspace_id="workspace-123",
            decay_rate=0,
        )

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 400
        assert "decay_rate must be between 0 and 1" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_decay_invalid_rate_one(self, handler):
        """Test fails with decay_rate of 1."""
        test_handler = MockPruningHandler()
        result = await test_handler.apply_confidence_decay(
            workspace_id="workspace-123",
            decay_rate=1.0,
        )

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 400
        assert "decay_rate must be between 0 and 1" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_decay_invalid_min_confidence(self, handler):
        """Test fails with invalid min_confidence."""
        test_handler = MockPruningHandler()
        result = await test_handler.apply_confidence_decay(
            workspace_id="workspace-123",
            decay_rate=0.05,
            min_confidence=0,
        )

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 400
        assert "min_confidence must be between 0 and 1" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_decay_mound_unavailable(self, handler_no_mound):
        """Test when mound is unavailable."""
        result = await handler_no_mound.apply_confidence_decay(workspace_id="workspace-123")

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 503
        assert "not available" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_decay_mound_error(self, handler):
        """Test handles mound errors."""
        test_handler = MockPruningHandler()
        test_handler.mound.apply_confidence_decay = AsyncMock(side_effect=Exception("Decay error"))

        result = await test_handler.apply_confidence_decay(
            workspace_id="workspace-123",
            decay_rate=0.05,
            min_confidence=0.2,
        )

        status_code = result.status_code
        data = parse_json_response(result)
        assert status_code == 500
        assert "error" in data
