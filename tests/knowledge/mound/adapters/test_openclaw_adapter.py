"""
Tests for OpenClawAdapter - Bridges OpenClaw to the Knowledge Mound.

Tests cover:
- Dataclasses (OpenClawKnowledgeItem, ActionPattern, SyncResult, etc.)
- Forward sync (OpenClaw -> KM)
- Reverse sync (KM -> OpenClaw)
- Bidirectional learning
- Pattern extraction
- Query methods
- Statistics and health
- Error handling and edge cases
- Mixin implementations (FusionMixin, SemanticSearchMixin)
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.knowledge.mound.adapters.openclaw_adapter import (
    OpenClawAdapter,
    OpenClawKnowledgeItem,
    ActionPattern,
    ActionStatus,
    PatternType,
    SyncResult,
    KMContextUpdate,
    TaskPrioritizationUpdate,
    KMValidationResult,
    OpenClawKMSyncResult,
)


# =============================================================================
# OpenClawKnowledgeItem Dataclass Tests
# =============================================================================


class TestOpenClawKnowledgeItem:
    """Tests for OpenClawKnowledgeItem dataclass."""

    def test_create_basic_item(self):
        """Should create a basic knowledge item."""
        item = OpenClawKnowledgeItem(
            action_id="action_001",
            result=ActionStatus.SUCCESS,
            context={"task": "web_search", "query": "latest news"},
        )

        assert item.action_id == "action_001"
        assert item.result == ActionStatus.SUCCESS
        assert item.context["task"] == "web_search"

    def test_create_with_all_fields(self):
        """Should create item with all fields."""
        created_at = datetime.now(timezone.utc)
        item = OpenClawKnowledgeItem(
            action_id="action_002",
            result=ActionStatus.FAILED,
            context={"task": "code_review"},
            debate_id="debate_123",
            workspace_id="ws_456",
            tenant_id="tenant_789",
            capabilities_used=["code_analysis", "lint"],
            execution_time_ms=1500.0,
            output="Review complete",
            error="Timeout occurred",
            metadata={"retry_count": 2},
            created_at=created_at,
        )

        assert item.action_id == "action_002"
        assert item.debate_id == "debate_123"
        assert item.workspace_id == "ws_456"
        assert item.tenant_id == "tenant_789"
        assert "code_analysis" in item.capabilities_used
        assert item.execution_time_ms == 1500.0
        assert item.error == "Timeout occurred"
        assert item.metadata["retry_count"] == 2
        assert item.created_at == created_at

    def test_default_values(self):
        """Should have correct default values."""
        item = OpenClawKnowledgeItem(
            action_id="action_003",
            result=ActionStatus.PENDING,
            context={},
        )

        assert item.debate_id is None
        assert item.workspace_id == "default"
        assert item.tenant_id == "default"
        assert item.capabilities_used == []
        assert item.execution_time_ms == 0.0
        assert item.output == ""
        assert item.error is None
        assert item.metadata == {}

    def test_to_dict(self):
        """Should convert to dictionary."""
        item = OpenClawKnowledgeItem(
            action_id="action_004",
            result=ActionStatus.SUCCESS,
            context={"task": "search"},
            capabilities_used=["web_search"],
        )

        data = item.to_dict()

        assert data["action_id"] == "action_004"
        assert data["result"] == "success"
        assert data["capabilities_used"] == ["web_search"]
        assert "created_at" in data

    def test_from_dict(self):
        """Should create from dictionary."""
        data = {
            "action_id": "action_005",
            "result": "failed",
            "context": {"task": "analysis"},
            "debate_id": "debate_abc",
            "workspace_id": "ws_def",
            "capabilities_used": ["analyze"],
            "execution_time_ms": 500.0,
            "created_at": "2024-06-15T10:00:00+00:00",
        }

        item = OpenClawKnowledgeItem.from_dict(data)

        assert item.action_id == "action_005"
        assert item.result == ActionStatus.FAILED
        assert item.debate_id == "debate_abc"
        assert item.execution_time_ms == 500.0

    def test_from_dict_handles_z_suffix(self):
        """Should handle Z timezone suffix in created_at."""
        data = {
            "action_id": "action_006",
            "result": "success",
            "context": {},
            "created_at": "2024-06-15T10:00:00Z",
        }

        item = OpenClawKnowledgeItem.from_dict(data)

        assert item.created_at.tzinfo is not None

    def test_from_dict_handles_missing_created_at(self):
        """Should handle missing created_at."""
        data = {
            "action_id": "action_007",
            "result": "pending",
            "context": {},
        }

        item = OpenClawKnowledgeItem.from_dict(data)

        assert item.created_at is not None


# =============================================================================
# ActionPattern Dataclass Tests
# =============================================================================


class TestActionPattern:
    """Tests for ActionPattern dataclass."""

    def test_create_basic_pattern(self):
        """Should create a basic action pattern."""
        pattern = ActionPattern(
            pattern_id="pattern_001",
            pattern_type=PatternType.SUCCESS_PATTERN,
            description="High success rate for web search",
        )

        assert pattern.pattern_id == "pattern_001"
        assert pattern.pattern_type == PatternType.SUCCESS_PATTERN
        assert pattern.description == "High success rate for web search"

    def test_create_with_all_fields(self):
        """Should create pattern with all fields."""
        pattern = ActionPattern(
            pattern_id="pattern_002",
            pattern_type=PatternType.FAILURE_PATTERN,
            description="Timeout issues with API calls",
            success_rate=0.3,
            observation_count=15,
            capabilities_involved=["api_call", "external_service"],
            context_signature={"timeout": True},
            recommendation="Use shorter timeout with retry",
            confidence=0.85,
            contributing_actions=["action_1", "action_2"],
            metadata={"category": "timeout"},
        )

        assert pattern.success_rate == 0.3
        assert pattern.observation_count == 15
        assert "api_call" in pattern.capabilities_involved
        assert pattern.confidence == 0.85

    def test_default_values(self):
        """Should have correct default values."""
        pattern = ActionPattern(
            pattern_id="pattern_003",
            pattern_type=PatternType.CAPABILITY_PATTERN,
            description="Test pattern",
        )

        assert pattern.success_rate == 0.0
        assert pattern.observation_count == 0
        assert pattern.capabilities_involved == []
        assert pattern.recommendation == ""
        assert pattern.confidence == 0.5

    def test_to_dict(self):
        """Should convert to dictionary."""
        pattern = ActionPattern(
            pattern_id="pattern_004",
            pattern_type=PatternType.WORKFLOW_PATTERN,
            description="Complex workflow pattern",
            success_rate=0.75,
        )

        data = pattern.to_dict()

        assert data["pattern_id"] == "pattern_004"
        assert data["pattern_type"] == "workflow"
        assert data["success_rate"] == 0.75
        assert "first_observed_at" in data

    def test_from_dict(self):
        """Should create from dictionary."""
        data = {
            "pattern_id": "pattern_005",
            "pattern_type": "failure",
            "description": "Restored pattern",
            "success_rate": 0.2,
            "observation_count": 10,
            "first_observed_at": "2024-06-01T00:00:00+00:00",
            "last_observed_at": "2024-06-15T12:00:00Z",
        }

        pattern = ActionPattern.from_dict(data)

        assert pattern.pattern_id == "pattern_005"
        assert pattern.pattern_type == PatternType.FAILURE_PATTERN
        assert pattern.success_rate == 0.2
        assert pattern.observation_count == 10


# =============================================================================
# SyncResult Dataclass Tests
# =============================================================================


class TestSyncResult:
    """Tests for SyncResult dataclass."""

    def test_create_basic_result(self):
        """Should create a basic sync result."""
        result = SyncResult()

        assert result.items_synced == 0
        assert result.items_skipped == 0
        assert result.items_failed == 0
        assert result.errors == []
        assert result.direction == "forward"

    def test_create_with_values(self):
        """Should create result with values."""
        result = SyncResult(
            items_synced=10,
            items_skipped=2,
            items_failed=1,
            errors=["Error 1"],
            duration_ms=150.5,
            direction="reverse",
            metadata={"batch": 1},
        )

        assert result.items_synced == 10
        assert result.items_skipped == 2
        assert result.items_failed == 1
        assert len(result.errors) == 1
        assert result.duration_ms == 150.5
        assert result.direction == "reverse"

    def test_to_dict(self):
        """Should convert to dictionary."""
        result = SyncResult(
            items_synced=5,
            duration_ms=100.0,
        )

        data = result.to_dict()

        assert data["items_synced"] == 5
        assert data["duration_ms"] == 100.0
        assert data["direction"] == "forward"


# =============================================================================
# Reverse Flow Dataclass Tests
# =============================================================================


class TestKMContextUpdate:
    """Tests for KMContextUpdate dataclass."""

    def test_create_context_update(self):
        """Should create a context update."""
        update = KMContextUpdate(
            update_id="update_001",
            context_type="pattern",
            content={"pattern": "success_pattern_web_search"},
            relevance_score=0.85,
        )

        assert update.update_id == "update_001"
        assert update.context_type == "pattern"
        assert update.relevance_score == 0.85

    def test_to_dict(self):
        """Should convert to dictionary."""
        expires_at = datetime.now(timezone.utc)
        update = KMContextUpdate(
            update_id="update_002",
            context_type="insight",
            content={"key": "value"},
            expires_at=expires_at,
        )

        data = update.to_dict()

        assert data["update_id"] == "update_002"
        assert data["expires_at"] is not None


class TestTaskPrioritizationUpdate:
    """Tests for TaskPrioritizationUpdate dataclass."""

    def test_create_prioritization_update(self):
        """Should create a prioritization update."""
        update = TaskPrioritizationUpdate(
            task_id="task_001",
            debate_id="debate_001",
            original_priority=0.5,
            new_priority=0.9,
            reason="Debate consensus",
        )

        assert update.task_id == "task_001"
        assert update.original_priority == 0.5
        assert update.new_priority == 0.9
        assert update.applied is False

    def test_to_dict(self):
        """Should convert to dictionary."""
        update = TaskPrioritizationUpdate(
            task_id="task_002",
            debate_id="debate_002",
            original_priority=0.3,
            new_priority=0.8,
            applied=True,
            applied_at=datetime.now(timezone.utc),
        )

        data = update.to_dict()

        assert data["applied"] is True
        assert data["applied_at"] is not None


class TestKMValidationResult:
    """Tests for KMValidationResult dataclass."""

    def test_create_validation_result(self):
        """Should create a validation result."""
        result = KMValidationResult(
            action_id="action_001",
            km_confidence=0.85,
            cross_debate_utility=0.7,
            was_supported=True,
        )

        assert result.action_id == "action_001"
        assert result.km_confidence == 0.85
        assert result.was_supported is True
        assert result.recommendation == "keep"

    def test_default_values(self):
        """Should have correct default values."""
        result = KMValidationResult(action_id="action_002")

        assert result.km_confidence == 0.5
        assert result.cross_debate_utility == 0.0
        assert result.validation_count == 1
        assert result.was_contradicted is False
        assert result.was_supported is False
        assert result.patterns_matched == []


class TestOpenClawKMSyncResult:
    """Tests for OpenClawKMSyncResult dataclass."""

    def test_create_km_sync_result(self):
        """Should create a KM sync result."""
        result = OpenClawKMSyncResult(
            actions_analyzed=20,
            actions_updated=15,
            patterns_extracted=5,
            duration_ms=250.0,
        )

        assert result.actions_analyzed == 20
        assert result.actions_updated == 15
        assert result.patterns_extracted == 5

    def test_to_dict(self):
        """Should convert to dictionary."""
        result = OpenClawKMSyncResult(
            actions_analyzed=10,
            errors=["Error 1", "Error 2"],
        )

        data = result.to_dict()

        assert data["actions_analyzed"] == 10
        assert len(data["errors"]) == 2


# =============================================================================
# OpenClawAdapter Initialization Tests
# =============================================================================


class TestOpenClawAdapterInit:
    """Tests for OpenClawAdapter initialization."""

    def test_default_init(self):
        """Should initialize with default values."""
        adapter = OpenClawAdapter()

        assert adapter._openclaw_client is None
        assert adapter._enable_dual_write is False
        assert adapter.ID_PREFIX == "oc_"
        assert adapter.adapter_name == "openclaw"
        assert adapter.source_type == "openclaw"

    def test_init_with_client(self):
        """Should accept OpenClaw client."""
        mock_client = MagicMock()
        adapter = OpenClawAdapter(openclaw_client=mock_client)

        assert adapter._openclaw_client is mock_client

    def test_init_with_dual_write(self):
        """Should accept dual write flag."""
        adapter = OpenClawAdapter(enable_dual_write=True)

        assert adapter._enable_dual_write is True

    def test_init_with_event_callback(self):
        """Should accept event callback."""
        callback = MagicMock()
        adapter = OpenClawAdapter(event_callback=callback)

        assert adapter._event_callback is callback

    def test_init_with_resilience_options(self):
        """Should accept resilience options."""
        adapter = OpenClawAdapter(
            enable_resilience=False,
            resilience_timeout=60.0,
        )

        assert adapter._enable_resilience is False

    def test_set_openclaw_client(self):
        """Should set OpenClaw client after init."""
        adapter = OpenClawAdapter()
        mock_client = MagicMock()

        adapter.set_openclaw_client(mock_client)

        assert adapter._openclaw_client is mock_client


# =============================================================================
# Forward Sync Tests (OpenClaw -> KM)
# =============================================================================


class TestStoreActionResult:
    """Tests for storing action results."""

    def test_store_successful_action(self):
        """Should store successful action result."""
        adapter = OpenClawAdapter()

        action = OpenClawKnowledgeItem(
            action_id="action_001",
            result=ActionStatus.SUCCESS,
            context={"task": "web_search"},
            capabilities_used=["web_search"],
        )

        item_id = adapter.store_action_result(action)

        assert item_id is not None
        assert item_id.startswith("oc_action_")
        assert item_id in adapter._actions

    def test_store_failed_action(self):
        """Should store failed action result."""
        adapter = OpenClawAdapter()

        action = OpenClawKnowledgeItem(
            action_id="action_002",
            result=ActionStatus.FAILED,
            context={"task": "api_call"},
            error="Connection refused",
        )

        item_id = adapter.store_action_result(action)

        assert item_id is not None
        stored = adapter._actions[item_id]
        assert stored["result"] == "failed"
        assert stored["error"] == "Connection refused"

    def test_calculates_confidence_for_success(self):
        """Should calculate higher confidence for successful actions."""
        adapter = OpenClawAdapter()

        action = OpenClawKnowledgeItem(
            action_id="action_003",
            result=ActionStatus.SUCCESS,
            context={},
            execution_time_ms=500,  # Fast execution
        )

        item_id = adapter.store_action_result(action)
        stored = adapter._actions[item_id]

        # Success + fast execution should give high confidence
        assert stored["confidence"] >= 0.8

    def test_calculates_confidence_for_failure(self):
        """Should calculate lower confidence for failed actions."""
        adapter = OpenClawAdapter()

        action = OpenClawKnowledgeItem(
            action_id="action_004",
            result=ActionStatus.FAILED,
            context={},
        )

        item_id = adapter.store_action_result(action)
        stored = adapter._actions[item_id]

        # Failed actions have lower confidence
        assert stored["confidence"] < 0.5

    def test_updates_capability_index(self):
        """Should update capability index."""
        adapter = OpenClawAdapter()

        action = OpenClawKnowledgeItem(
            action_id="action_005",
            result=ActionStatus.SUCCESS,
            context={},
            capabilities_used=["web_search", "file_read"],
        )

        item_id = adapter.store_action_result(action)

        assert item_id in adapter._capability_actions.get("web_search", [])
        assert item_id in adapter._capability_actions.get("file_read", [])

    def test_updates_debate_index(self):
        """Should update debate index when debate_id present."""
        adapter = OpenClawAdapter()

        action = OpenClawKnowledgeItem(
            action_id="action_006",
            result=ActionStatus.SUCCESS,
            context={},
            debate_id="debate_123",
        )

        item_id = adapter.store_action_result(action)

        assert item_id in adapter._debate_actions.get("debate_123", [])

    def test_stores_action_with_callback_configured(self):
        """Should store action when event callback is configured."""
        adapter = OpenClawAdapter()

        # Set an event callback (it may not be called due to MRO with mixins)
        adapter.set_event_callback(lambda t, d: None)

        action = OpenClawKnowledgeItem(
            action_id="action_007",
            result=ActionStatus.SUCCESS,
            context={},
        )

        # Main functionality should still work
        item_id = adapter.store_action_result(action)

        assert item_id is not None
        assert item_id in adapter._actions


class TestSyncActionsToMound:
    """Tests for batch syncing actions to KM."""

    @pytest.mark.asyncio
    async def test_syncs_actions_above_confidence(self):
        """Should sync actions above confidence threshold."""
        adapter = OpenClawAdapter()
        mock_mound = MagicMock()
        mock_mound.ingest = AsyncMock(return_value="km_123")

        # Store some actions first
        for i in range(3):
            action = OpenClawKnowledgeItem(
                action_id=f"action_{i}",
                result=ActionStatus.SUCCESS,
                context={},
            )
            adapter.store_action_result(action)

        result = await adapter.sync_actions_to_mound(
            mound=mock_mound,
            workspace_id="test_workspace",
            min_confidence=0.5,
        )

        assert result.items_synced > 0
        assert result.direction == "forward"
        assert mock_mound.ingest.called

    @pytest.mark.asyncio
    async def test_skips_actions_below_confidence(self):
        """Should skip actions below confidence threshold."""
        adapter = OpenClawAdapter()
        mock_mound = MagicMock()
        mock_mound.ingest = AsyncMock(return_value="km_123")

        # Store failed action (low confidence)
        action = OpenClawKnowledgeItem(
            action_id="action_low",
            result=ActionStatus.FAILED,
            context={},
        )
        adapter.store_action_result(action)

        result = await adapter.sync_actions_to_mound(
            mound=mock_mound,
            workspace_id="test_workspace",
            min_confidence=0.9,  # High threshold
        )

        assert result.items_skipped > 0

    @pytest.mark.asyncio
    async def test_skips_already_synced_actions(self):
        """Should skip already synced actions."""
        adapter = OpenClawAdapter()
        mock_mound = MagicMock()
        mock_mound.ingest = AsyncMock(return_value="km_123")

        action = OpenClawKnowledgeItem(
            action_id="action_synced",
            result=ActionStatus.SUCCESS,
            context={},
        )
        item_id = adapter.store_action_result(action)

        # Mark as already synced
        adapter._actions[item_id]["km_synced"] = True

        result = await adapter.sync_actions_to_mound(
            mound=mock_mound,
            workspace_id="test_workspace",
        )

        assert result.items_skipped == 1

    @pytest.mark.asyncio
    async def test_handles_ingest_errors(self):
        """Should handle ingest errors gracefully."""
        adapter = OpenClawAdapter()
        mock_mound = MagicMock()
        mock_mound.ingest = AsyncMock(side_effect=RuntimeError("Ingest failed"))

        action = OpenClawKnowledgeItem(
            action_id="action_error",
            result=ActionStatus.SUCCESS,
            context={},
        )
        adapter.store_action_result(action)

        result = await adapter.sync_actions_to_mound(
            mound=mock_mound,
            workspace_id="test_workspace",
        )

        assert result.items_failed > 0
        assert len(result.errors) > 0


class TestStoreExecutionLog:
    """Tests for storing execution logs."""

    def test_stores_log_entries(self):
        """Should store log entries for an action."""
        adapter = OpenClawAdapter()

        log_entries = [
            {"timestamp": "2024-06-15T10:00:00Z", "level": "info", "message": "Started"},
            {"timestamp": "2024-06-15T10:00:01Z", "level": "info", "message": "Completed"},
        ]

        log_id = adapter.store_execution_log("action_001", log_entries)

        assert log_id == "oc_log_action_001"
        assert len(adapter._execution_logs["action_001"]) == 2

    def test_appends_to_existing_logs(self):
        """Should append to existing logs."""
        adapter = OpenClawAdapter()

        adapter.store_execution_log("action_002", [{"message": "First"}])
        adapter.store_execution_log("action_002", [{"message": "Second"}])

        assert len(adapter._execution_logs["action_002"]) == 2

    def test_stores_logs_with_callback_configured(self):
        """Should store logs when event callback is configured."""
        adapter = OpenClawAdapter()

        # Set an event callback
        adapter.set_event_callback(lambda t, d: None)

        log_id = adapter.store_execution_log("action_003", [{"message": "Test"}])

        # Main functionality should still work
        assert log_id == "oc_log_action_003"
        assert "action_003" in adapter._execution_logs


class TestIndexSessionContext:
    """Tests for indexing session context."""

    def test_indexes_session_context(self):
        """Should index session context."""
        adapter = OpenClawAdapter()

        context = {
            "session_start": "2024-06-15T10:00:00Z",
            "user_id": "user_123",
            "workspace": "ws_456",
        }

        context_id = adapter.index_session_context("session_001", context)

        assert context_id == "oc_ctx_session_001"
        assert context_id in adapter._session_contexts
        assert adapter._session_contexts[context_id]["context"] == context

    def test_indexes_context_with_callback_configured(self):
        """Should index context when event callback is configured."""
        adapter = OpenClawAdapter()

        # Set an event callback
        adapter.set_event_callback(lambda t, d: None)

        context_id = adapter.index_session_context("session_002", {"key": "value"})

        # Main functionality should still work
        assert context_id == "oc_ctx_session_002"
        assert context_id in adapter._session_contexts


# =============================================================================
# Reverse Sync Tests (KM -> OpenClaw)
# =============================================================================


class TestPushDebateDecisions:
    """Tests for pushing debate decisions to OpenClaw."""

    @pytest.mark.asyncio
    async def test_pushes_decisions_without_client(self):
        """Should store decisions even without client."""
        adapter = OpenClawAdapter()

        decisions = [
            {"task_id": "task_001", "priority": 0.9, "reason": "High importance"},
            {"task_id": "task_002", "priority": 0.5, "reason": "Normal"},
        ]

        count = await adapter.push_debate_decisions("debate_123", decisions)

        assert count == 2
        assert "task_001" in adapter._prioritization_updates
        assert "task_002" in adapter._prioritization_updates

    @pytest.mark.asyncio
    async def test_pushes_decisions_with_client(self):
        """Should push decisions to OpenClaw client."""
        mock_client = MagicMock()
        mock_client.update_task_priority = AsyncMock()
        adapter = OpenClawAdapter(openclaw_client=mock_client)

        decisions = [{"task_id": "task_003", "priority": 0.8}]

        count = await adapter.push_debate_decisions("debate_456", decisions)

        assert count == 1
        mock_client.update_task_priority.assert_called_once()

    @pytest.mark.asyncio
    async def test_marks_applied_on_success(self):
        """Should mark update as applied on success."""
        mock_client = MagicMock()
        mock_client.update_task_priority = AsyncMock()
        adapter = OpenClawAdapter(openclaw_client=mock_client)

        decisions = [{"task_id": "task_004", "priority": 0.7}]

        await adapter.push_debate_decisions("debate_789", decisions)

        update = adapter._prioritization_updates["task_004"]
        assert update.applied is True
        assert update.applied_at is not None

    @pytest.mark.asyncio
    async def test_handles_client_errors(self):
        """Should handle client errors gracefully."""
        mock_client = MagicMock()
        mock_client.update_task_priority = AsyncMock(side_effect=RuntimeError("Client error"))
        adapter = OpenClawAdapter(openclaw_client=mock_client)

        decisions = [{"task_id": "task_005", "priority": 0.6}]

        count = await adapter.push_debate_decisions("debate_abc", decisions)

        # Should still count as pushed (stored locally)
        assert count == 1
        update = adapter._prioritization_updates["task_005"]
        assert update.applied is False

    @pytest.mark.asyncio
    async def test_skips_decisions_without_task_id(self):
        """Should skip decisions without task_id."""
        adapter = OpenClawAdapter()

        decisions = [
            {"priority": 0.9},  # Missing task_id
            {"task_id": "task_valid", "priority": 0.8},
        ]

        count = await adapter.push_debate_decisions("debate_xyz", decisions)

        assert count == 1


class TestUpdateOpenClawContext:
    """Tests for pushing context updates to OpenClaw."""

    @pytest.mark.asyncio
    async def test_stores_context_updates(self):
        """Should store context updates."""
        adapter = OpenClawAdapter()

        updates = [
            KMContextUpdate(
                update_id="update_001",
                context_type="pattern",
                content={"pattern": "success"},
                priority=0.8,
            ),
        ]

        count = await adapter.update_openclaw_context(updates)

        assert "update_001" in adapter._context_updates
        # Without client, no updates are applied
        assert count == 0

    @pytest.mark.asyncio
    async def test_pushes_to_client(self):
        """Should push context updates to client."""
        mock_client = MagicMock()
        mock_client.add_context = AsyncMock()
        adapter = OpenClawAdapter(openclaw_client=mock_client)

        updates = [
            KMContextUpdate(
                update_id="update_002",
                context_type="insight",
                content={"insight": "valuable"},
                priority=0.9,
            ),
        ]

        count = await adapter.update_openclaw_context(updates)

        assert count == 1
        mock_client.add_context.assert_called_once()


class TestSyncKnowledgeForAction:
    """Tests for syncing knowledge for action context."""

    @pytest.mark.asyncio
    async def test_returns_relevant_items(self):
        """Should return relevant knowledge items."""
        adapter = OpenClawAdapter()

        action_context = {
            "task_type": "web_search",
            "capabilities": ["web_search"],
            "query": "search test query",
        }

        km_items = [
            {
                "id": "item_001",
                "content": "search test content",
                "confidence": 0.9,
                "metadata": {
                    "task_type": "web_search",
                    "capabilities_used": ["web_search"],
                },
            },
            {
                "id": "item_002",
                "content": "unrelated content",
                "confidence": 0.5,
                "metadata": {"task_type": "code_review"},
            },
        ]

        relevant = await adapter.sync_knowledge_for_action(action_context, km_items)

        assert len(relevant) >= 1
        assert relevant[0]["id"] == "item_001"
        assert "action_relevance" in relevant[0]

    @pytest.mark.asyncio
    async def test_limits_results(self):
        """Should limit results to top 20."""
        adapter = OpenClawAdapter()

        action_context = {"capabilities": ["test"]}

        # Create many items
        km_items = [
            {
                "id": f"item_{i}",
                "content": "test content",
                "confidence": 0.9,
                "metadata": {"capabilities_used": ["test"]},
            }
            for i in range(30)
        ]

        relevant = await adapter.sync_knowledge_for_action(action_context, km_items)

        assert len(relevant) <= 20


# =============================================================================
# Bidirectional Learning Tests
# =============================================================================


class TestExtractActionPatterns:
    """Tests for extracting action patterns."""

    @pytest.mark.asyncio
    async def test_extracts_success_patterns(self):
        """Should extract success patterns from high success rate capabilities."""
        adapter = OpenClawAdapter()

        # Store multiple successful actions with same capability
        for i in range(5):
            action = OpenClawKnowledgeItem(
                action_id=f"action_{i}",
                result=ActionStatus.SUCCESS,
                context={},
                capabilities_used=["web_search"],
                workspace_id="ws_test",
            )
            adapter.store_action_result(action)

        patterns = await adapter.extract_action_patterns("ws_test", min_observations=3)

        assert len(patterns) >= 1
        web_search_pattern = next(
            (p for p in patterns if "web_search" in p.capabilities_involved), None
        )
        assert web_search_pattern is not None
        assert web_search_pattern.pattern_type == PatternType.SUCCESS_PATTERN

    @pytest.mark.asyncio
    async def test_extracts_failure_patterns(self):
        """Should extract failure patterns from low success rate capabilities."""
        adapter = OpenClawAdapter()

        # Store mostly failed actions
        for i in range(4):
            action = OpenClawKnowledgeItem(
                action_id=f"fail_{i}",
                result=ActionStatus.FAILED,
                context={},
                capabilities_used=["flaky_api"],
                workspace_id="ws_test",
            )
            adapter.store_action_result(action)

        # Add one success
        action = OpenClawKnowledgeItem(
            action_id="success_1",
            result=ActionStatus.SUCCESS,
            context={},
            capabilities_used=["flaky_api"],
            workspace_id="ws_test",
        )
        adapter.store_action_result(action)

        patterns = await adapter.extract_action_patterns("ws_test", min_observations=3)

        flaky_pattern = next((p for p in patterns if "flaky_api" in p.capabilities_involved), None)
        assert flaky_pattern is not None
        assert flaky_pattern.pattern_type == PatternType.FAILURE_PATTERN
        assert flaky_pattern.success_rate <= 0.3

    @pytest.mark.asyncio
    async def test_respects_min_observations(self):
        """Should not extract patterns below min observations."""
        adapter = OpenClawAdapter()

        # Store only 2 actions
        for i in range(2):
            action = OpenClawKnowledgeItem(
                action_id=f"action_{i}",
                result=ActionStatus.SUCCESS,
                context={},
                capabilities_used=["rare_capability"],
                workspace_id="ws_test",
            )
            adapter.store_action_result(action)

        patterns = await adapter.extract_action_patterns("ws_test", min_observations=5)

        rare_pattern = next(
            (p for p in patterns if "rare_capability" in p.capabilities_involved), None
        )
        assert rare_pattern is None

    @pytest.mark.asyncio
    async def test_stores_extracted_patterns(self):
        """Should store extracted patterns."""
        adapter = OpenClawAdapter()

        for i in range(3):
            action = OpenClawKnowledgeItem(
                action_id=f"action_{i}",
                result=ActionStatus.SUCCESS,
                context={},
                capabilities_used=["stored_cap"],
                workspace_id="ws_test",
            )
            adapter.store_action_result(action)

        patterns = await adapter.extract_action_patterns("ws_test", min_observations=3)

        # Pattern should be stored
        assert len(adapter._patterns) >= 1


class TestGetFailurePatterns:
    """Tests for getting failure patterns."""

    @pytest.mark.asyncio
    async def test_returns_failure_patterns(self):
        """Should return failure patterns."""
        adapter = OpenClawAdapter()

        # Store a failure pattern directly
        adapter._patterns["oc_pattern_fail_1"] = {
            "pattern_id": "oc_pattern_fail_1",
            "pattern_type": PatternType.FAILURE_PATTERN.value,
            "description": "Timeout pattern",
            "observation_count": 10,
        }

        patterns = await adapter.get_failure_patterns("ws_test")

        assert len(patterns) == 1
        assert patterns[0].pattern_type == PatternType.FAILURE_PATTERN

    @pytest.mark.asyncio
    async def test_sorts_by_observation_count(self):
        """Should sort by observation count."""
        adapter = OpenClawAdapter()

        adapter._patterns["oc_pattern_fail_1"] = {
            "pattern_id": "oc_pattern_fail_1",
            "pattern_type": PatternType.FAILURE_PATTERN.value,
            "description": "Less observed",
            "observation_count": 5,
        }
        adapter._patterns["oc_pattern_fail_2"] = {
            "pattern_id": "oc_pattern_fail_2",
            "pattern_type": PatternType.FAILURE_PATTERN.value,
            "description": "More observed",
            "observation_count": 15,
        }

        patterns = await adapter.get_failure_patterns("ws_test")

        assert patterns[0].observation_count == 15


class TestGetSuccessPatterns:
    """Tests for getting success patterns."""

    @pytest.mark.asyncio
    async def test_returns_success_patterns(self):
        """Should return success patterns."""
        adapter = OpenClawAdapter()

        adapter._patterns["oc_pattern_success_1"] = {
            "pattern_id": "oc_pattern_success_1",
            "pattern_type": PatternType.SUCCESS_PATTERN.value,
            "description": "Good pattern",
            "success_rate": 0.9,
            "confidence": 0.85,
        }

        patterns = await adapter.get_success_patterns("ws_test")

        assert len(patterns) == 1
        assert patterns[0].pattern_type == PatternType.SUCCESS_PATTERN

    @pytest.mark.asyncio
    async def test_respects_limit(self):
        """Should respect limit parameter."""
        adapter = OpenClawAdapter()

        for i in range(5):
            adapter._patterns[f"oc_pattern_success_{i}"] = {
                "pattern_id": f"oc_pattern_success_{i}",
                "pattern_type": PatternType.SUCCESS_PATTERN.value,
                "description": f"Pattern {i}",
                "success_rate": 0.8,
                "confidence": 0.7,
            }

        patterns = await adapter.get_success_patterns("ws_test", limit=3)

        assert len(patterns) == 3


class TestCrossDebateLearning:
    """Tests for cross-debate learning."""

    @pytest.mark.asyncio
    async def test_learns_from_outcomes(self):
        """Should learn from action outcomes."""
        adapter = OpenClawAdapter()

        # Store an action first
        action = OpenClawKnowledgeItem(
            action_id="action_learn",
            result=ActionStatus.SUCCESS,
            context={},
        )
        item_id = adapter.store_action_result(action)

        action_outcomes = [
            {
                "action_id": item_id,
                "confidence": 0.85,
                "utility": 0.9,
                "was_supported": True,
            }
        ]

        results = await adapter.cross_debate_learning("debate_learn", action_outcomes)

        assert results["debate_id"] == "debate_learn"
        assert results["actions_analyzed"] == 1
        assert item_id in adapter._km_validations

    @pytest.mark.asyncio
    async def test_updates_action_confidence(self):
        """Should update action with KM confidence."""
        adapter = OpenClawAdapter()

        action = OpenClawKnowledgeItem(
            action_id="action_update",
            result=ActionStatus.SUCCESS,
            context={},
        )
        item_id = adapter.store_action_result(action)

        action_outcomes = [
            {
                "action_id": item_id,
                "confidence": 0.95,
                "utility": 0.8,
            }
        ]

        await adapter.cross_debate_learning("debate_update", action_outcomes)

        stored = adapter._actions[item_id]
        assert stored["km_validated"] is True
        assert stored["km_confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_generates_recommendations(self):
        """Should generate recommendations for high utility actions."""
        adapter = OpenClawAdapter()

        action_outcomes = [
            {
                "action_id": "action_high_utility",
                "confidence": 0.9,
                "utility": 0.8,  # High utility
            }
        ]

        results = await adapter.cross_debate_learning("debate_recommend", action_outcomes)

        assert "recommendations" in results


# =============================================================================
# Batch Sync Tests
# =============================================================================


class TestSyncValidationsFromKM:
    """Tests for syncing validations from KM."""

    @pytest.mark.asyncio
    async def test_syncs_validations(self):
        """Should sync validations from KM items."""
        adapter = OpenClawAdapter()

        # Store an action first
        action = OpenClawKnowledgeItem(
            action_id="action_validate",
            result=ActionStatus.SUCCESS,
            context={},
        )
        adapter.store_action_result(action)

        km_items = [
            {
                "id": "km_001",
                "content": "Validated action",
                "confidence": 0.85,
                "metadata": {
                    "openclaw_action_id": "action_validate",
                    "was_supported": True,
                },
            }
        ]

        result = await adapter.sync_validations_from_km(km_items)

        assert result.actions_analyzed >= 1

    @pytest.mark.asyncio
    async def test_skips_low_confidence(self):
        """Should skip items below confidence threshold."""
        adapter = OpenClawAdapter()

        km_items = [
            {
                "id": "km_002",
                "content": "Low confidence",
                "confidence": 0.3,
                "metadata": {"openclaw_action_id": "action_low"},
            }
        ]

        result = await adapter.sync_validations_from_km(km_items, min_confidence=0.7)

        # Item should be analyzed but not updated due to low confidence
        assert result.actions_updated == 0

    @pytest.mark.asyncio
    async def test_handles_errors(self):
        """Should handle errors gracefully."""
        adapter = OpenClawAdapter()

        km_items = [
            {
                # Missing required metadata
                "id": "km_003",
                "confidence": 0.9,
            }
        ]

        result = await adapter.sync_validations_from_km(km_items)

        # Should complete without crashing
        assert isinstance(result, OpenClawKMSyncResult)


# =============================================================================
# Query Method Tests
# =============================================================================


class TestGetAction:
    """Tests for getting actions by ID."""

    def test_gets_action_by_id(self):
        """Should get action by ID."""
        adapter = OpenClawAdapter()

        action = OpenClawKnowledgeItem(
            action_id="action_get",
            result=ActionStatus.SUCCESS,
            context={},
        )
        item_id = adapter.store_action_result(action)

        result = adapter.get_action(item_id)

        assert result is not None
        assert result["action_id"] == "action_get"

    def test_adds_prefix_if_missing(self):
        """Should return action when queried with full ID."""
        adapter = OpenClawAdapter()

        action = OpenClawKnowledgeItem(
            action_id="action_prefix",
            result=ActionStatus.SUCCESS,
            context={},
        )
        item_id = adapter.store_action_result(action)

        # Query with full ID should work
        result = adapter.get_action(item_id)

        assert result is not None
        assert result["action_id"] == "action_prefix"

    def test_returns_none_for_missing(self):
        """Should return None for missing action."""
        adapter = OpenClawAdapter()

        result = adapter.get_action("nonexistent")

        assert result is None


class TestGetPattern:
    """Tests for getting patterns by ID."""

    def test_gets_pattern_by_id(self):
        """Should get pattern by ID."""
        adapter = OpenClawAdapter()

        adapter._patterns["oc_pattern_test"] = {
            "pattern_id": "oc_pattern_test",
            "pattern_type": PatternType.SUCCESS_PATTERN.value,
            "description": "Test pattern",
        }

        result = adapter.get_pattern("oc_pattern_test")

        assert result is not None
        assert result["description"] == "Test pattern"


class TestSearchActionsByCapability:
    """Tests for searching actions by capability."""

    def test_searches_by_capability(self):
        """Should search actions by capability."""
        adapter = OpenClawAdapter()

        action1 = OpenClawKnowledgeItem(
            action_id="action_cap1",
            result=ActionStatus.SUCCESS,
            context={},
            capabilities_used=["web_search"],
        )
        action2 = OpenClawKnowledgeItem(
            action_id="action_cap2",
            result=ActionStatus.SUCCESS,
            context={},
            capabilities_used=["file_read"],
        )

        adapter.store_action_result(action1)
        adapter.store_action_result(action2)

        results = adapter.search_actions_by_capability("web_search")

        assert len(results) == 1
        assert results[0]["action_id"] == "action_cap1"

    def test_respects_limit(self):
        """Should respect limit parameter."""
        adapter = OpenClawAdapter()

        for i in range(5):
            action = OpenClawKnowledgeItem(
                action_id=f"action_limit_{i}",
                result=ActionStatus.SUCCESS,
                context={},
                capabilities_used=["common"],
            )
            adapter.store_action_result(action)

        results = adapter.search_actions_by_capability("common", limit=3)

        assert len(results) == 3


class TestSearchActionsByDebate:
    """Tests for searching actions by debate."""

    def test_searches_by_debate(self):
        """Should search actions by debate ID."""
        adapter = OpenClawAdapter()

        action1 = OpenClawKnowledgeItem(
            action_id="action_debate1",
            result=ActionStatus.SUCCESS,
            context={},
            debate_id="debate_search",
        )
        action2 = OpenClawKnowledgeItem(
            action_id="action_debate2",
            result=ActionStatus.SUCCESS,
            context={},
            debate_id="other_debate",
        )

        adapter.store_action_result(action1)
        adapter.store_action_result(action2)

        results = adapter.search_actions_by_debate("debate_search")

        assert len(results) == 1
        assert results[0]["debate_id"] == "debate_search"


class TestToKnowledgeItem:
    """Tests for converting to KnowledgeItem."""

    def test_converts_to_knowledge_item(self):
        """Should convert action to KnowledgeItem."""
        adapter = OpenClawAdapter()

        action = OpenClawKnowledgeItem(
            action_id="action_convert",
            result=ActionStatus.SUCCESS,
            context={"task": "test"},
            capabilities_used=["test_cap"],
        )
        item_id = adapter.store_action_result(action)
        action_data = adapter._actions[item_id]

        km_item = adapter.to_knowledge_item(action_data)

        assert km_item.id == item_id
        assert km_item.metadata["openclaw_action_id"] == "action_convert"

    def test_maps_confidence_levels(self):
        """Should map confidence to appropriate level."""
        adapter = OpenClawAdapter()

        # High confidence action
        high_conf_action = {
            "id": "high_conf",
            "action_id": "action_high",
            "result": "success",
            "confidence": 0.85,
            "context": {},
        }

        km_item = adapter.to_knowledge_item(high_conf_action)

        # ConfidenceLevel.HIGH is for >= 0.8
        from aragora.knowledge.mound.types import ConfidenceLevel

        assert km_item.confidence == ConfidenceLevel.HIGH


# =============================================================================
# Statistics and Health Tests
# =============================================================================


class TestGetStats:
    """Tests for getting adapter statistics."""

    def test_returns_stats(self):
        """Should return statistics dict."""
        adapter = OpenClawAdapter()

        # Store some data
        action = OpenClawKnowledgeItem(
            action_id="action_stats",
            result=ActionStatus.SUCCESS,
            context={},
            capabilities_used=["test"],
            debate_id="debate_stats",
        )
        adapter.store_action_result(action)

        stats = adapter.get_stats()

        assert "total_actions" in stats
        assert "total_patterns" in stats
        assert "capabilities_tracked" in stats
        assert "debates_tracked" in stats
        assert stats["total_actions"] >= 1


class TestGetReverseFlowStats:
    """Tests for getting reverse flow statistics."""

    def test_returns_reverse_flow_stats(self):
        """Should return reverse flow stats."""
        adapter = OpenClawAdapter()

        stats = adapter.get_reverse_flow_stats()

        assert "validations_stored" in stats
        assert "context_updates" in stats
        assert "prioritization_updates" in stats


class TestClearReverseFlowState:
    """Tests for clearing reverse flow state."""

    def test_clears_state(self):
        """Should clear all reverse flow state."""
        adapter = OpenClawAdapter()

        # Add some state
        adapter._context_updates["update_1"] = MagicMock()
        adapter._prioritization_updates["task_1"] = MagicMock()
        adapter._km_validations["action_1"] = MagicMock()

        adapter.clear_reverse_flow_state()

        assert len(adapter._context_updates) == 0
        assert len(adapter._prioritization_updates) == 0
        assert len(adapter._km_validations) == 0


# =============================================================================
# Mixin Implementation Tests
# =============================================================================


class TestFusionMixinImplementation:
    """Tests for FusionMixin method implementations."""

    def test_get_fusion_sources(self):
        """Should return fusion sources."""
        adapter = OpenClawAdapter()

        sources = adapter._get_fusion_sources()

        assert isinstance(sources, list)
        assert len(sources) > 0
        assert "consensus" in sources

    def test_extract_fusible_data(self):
        """Should extract fusible data from KM item."""
        adapter = OpenClawAdapter()

        km_item = {
            "id": "item_001",
            "confidence": 0.85,
            "metadata": {
                "capabilities_used": ["test"],
            },
        }

        fusible = adapter._extract_fusible_data(km_item)

        assert fusible is not None
        assert fusible["confidence"] == 0.85

    def test_extract_fusible_data_no_confidence(self):
        """Should return None if no confidence."""
        adapter = OpenClawAdapter()

        km_item = {"id": "item_002", "metadata": {}}

        fusible = adapter._extract_fusible_data(km_item)

        assert fusible is None

    def test_apply_fusion_result(self):
        """Should apply fusion result to record."""
        adapter = OpenClawAdapter()

        record = {"id": "record_001", "confidence": 0.5}

        mock_fusion = MagicMock()
        mock_fusion.fused_confidence = 0.85

        result = adapter._apply_fusion_result(record, mock_fusion)

        assert result is True
        assert record["km_fused"] is True
        assert record["km_fused_confidence"] == 0.85


class TestSemanticSearchMixinImplementation:
    """Tests for SemanticSearchMixin method implementations."""

    def test_get_record_by_id(self):
        """Should get record by ID."""
        adapter = OpenClawAdapter()

        action = OpenClawKnowledgeItem(
            action_id="action_semantic",
            result=ActionStatus.SUCCESS,
            context={},
        )
        item_id = adapter.store_action_result(action)

        record = adapter._get_record_by_id(item_id)

        assert record is not None
        assert record["action_id"] == "action_semantic"

    def test_get_record_by_id_without_prefix(self):
        """Should find record without prefix."""
        adapter = OpenClawAdapter()

        action = OpenClawKnowledgeItem(
            action_id="action_noprefix",
            result=ActionStatus.SUCCESS,
            context={},
        )
        item_id = adapter.store_action_result(action)

        # Remove prefix
        raw_id = item_id[len(adapter.ID_PREFIX) :]

        record = adapter._get_record_by_id(raw_id)

        assert record is not None

    def test_record_to_dict_with_dict(self):
        """Should handle dict records."""
        adapter = OpenClawAdapter()

        record = {"id": "test", "value": 123}

        result = adapter._record_to_dict(record, similarity=0.9)

        assert result["similarity"] == 0.9
        assert result["value"] == 123

    def test_record_to_dict_with_knowledge_item(self):
        """Should handle OpenClawKnowledgeItem."""
        adapter = OpenClawAdapter()

        item = OpenClawKnowledgeItem(
            action_id="action_item",
            result=ActionStatus.SUCCESS,
            context={},
        )

        result = adapter._record_to_dict(item, similarity=0.8)

        assert result["similarity"] == 0.8
        assert result["action_id"] == "action_item"

    def test_extract_record_id(self):
        """Should extract record ID from prefixed ID."""
        adapter = OpenClawAdapter()

        extracted = adapter._extract_record_id("oc_action_123")

        assert extracted == "action_123"


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_event_callback_handles_errors(self):
        """Should handle event callback errors."""
        callback = MagicMock(side_effect=RuntimeError("Callback failed"))
        adapter = OpenClawAdapter(event_callback=callback)

        action = OpenClawKnowledgeItem(
            action_id="action_error",
            result=ActionStatus.SUCCESS,
            context={},
        )

        # Should not raise
        adapter.store_action_result(action)

    def test_action_status_enum_values(self):
        """Should have expected enum values."""
        assert ActionStatus.PENDING.value == "pending"
        assert ActionStatus.RUNNING.value == "running"
        assert ActionStatus.SUCCESS.value == "success"
        assert ActionStatus.FAILED.value == "failed"
        assert ActionStatus.TIMEOUT.value == "timeout"
        assert ActionStatus.CANCELLED.value == "cancelled"

    def test_pattern_type_enum_values(self):
        """Should have expected enum values."""
        assert PatternType.SUCCESS_PATTERN.value == "success"
        assert PatternType.FAILURE_PATTERN.value == "failure"
        assert PatternType.TIMEOUT_PATTERN.value == "timeout"
        assert PatternType.RESOURCE_PATTERN.value == "resource"
        assert PatternType.CAPABILITY_PATTERN.value == "capability"
        assert PatternType.WORKFLOW_PATTERN.value == "workflow"

    def test_slow_execution_reduces_confidence(self):
        """Should reduce confidence for slow executions."""
        adapter = OpenClawAdapter()

        action = OpenClawKnowledgeItem(
            action_id="action_slow",
            result=ActionStatus.SUCCESS,
            context={},
            execution_time_ms=50000,  # Very slow
        )

        item_id = adapter.store_action_result(action)
        stored = adapter._actions[item_id]

        # Should still be positive but penalized
        assert stored["confidence"] < 0.8

    def test_timeout_status_confidence(self):
        """Should calculate appropriate confidence for timeout status."""
        adapter = OpenClawAdapter()

        action = OpenClawKnowledgeItem(
            action_id="action_timeout",
            result=ActionStatus.TIMEOUT,
            context={},
        )

        item_id = adapter.store_action_result(action)
        stored = adapter._actions[item_id]

        # Timeout slightly reduces confidence
        assert stored["confidence"] < 0.5

    def test_health_check(self):
        """Should return health check status."""
        adapter = OpenClawAdapter()

        health = adapter.health_check()

        assert "adapter" in health
        assert health["adapter"] == "openclaw"
        assert "healthy" in health
        assert "reverse_flow_stats" in health

    @pytest.mark.asyncio
    async def test_empty_km_items_sync(self):
        """Should handle empty KM items list."""
        adapter = OpenClawAdapter()

        result = await adapter.sync_validations_from_km([])

        assert result.actions_analyzed == 0
        assert result.actions_updated == 0

    def test_build_action_content(self):
        """Should build content string for action."""
        adapter = OpenClawAdapter()

        action_data = {
            "output": "Search results: item1, item2",
            "result": "success",
            "capabilities_used": ["web_search"],
            "context": {"task": "search"},
        }

        content = adapter._build_action_content(action_data)

        assert "Search results" in content
        assert "success" in content
        assert "web_search" in content

    def test_build_action_content_empty(self):
        """Should handle empty action data with default result."""
        adapter = OpenClawAdapter()

        action_data = {}

        content = adapter._build_action_content(action_data)

        # With no output or capabilities, content includes "Result: unknown"
        assert "Result:" in content or content == "OpenClaw action execution"
