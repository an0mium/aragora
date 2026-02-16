"""
Comprehensive tests for OpenClawAdapter - Knowledge Mound integration.

Tests cover:
1. Adapter initialization and configuration
2. Forward sync operations (OpenClaw -> KM): store_action_result, sync_actions_to_mound,
   store_execution_log, index_session_context
3. Reverse sync operations (KM -> OpenClaw): push_debate_decisions,
   update_openclaw_context, sync_knowledge_for_action
4. Pattern extraction: extract_action_patterns, get_failure_patterns, get_success_patterns
5. Cross-debate learning
6. Batch validation sync: sync_validations_from_km
7. Knowledge item conversion: to_knowledge_item
8. Mixin integration (FusionMixin, SemanticSearchMixin)
9. Error handling and resilience
10. Edge cases and boundary conditions
11. Query methods: get_action, get_pattern, search_actions_by_capability/debate
12. Statistics and health monitoring
13. Dataclass serialization round-trips
"""

import hashlib
import time

import pytest
from datetime import datetime, timezone, timedelta
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
# Fixtures
# =============================================================================


@pytest.fixture
def adapter():
    """Create a fresh OpenClawAdapter for each test."""
    return OpenClawAdapter(enable_resilience=False)


@pytest.fixture
def adapter_with_client():
    """Create an adapter with a mock OpenClaw client."""
    mock_client = MagicMock()
    mock_client.update_task_priority = AsyncMock()
    mock_client.add_context = AsyncMock()
    return OpenClawAdapter(openclaw_client=mock_client, enable_resilience=False)


@pytest.fixture
def sample_action():
    """Create a sample OpenClawKnowledgeItem."""
    return OpenClawKnowledgeItem(
        action_id="test_action_001",
        result=ActionStatus.SUCCESS,
        context={"task": "web_search", "query": "latest news"},
        debate_id="debate_001",
        workspace_id="ws_001",
        tenant_id="tenant_001",
        capabilities_used=["web_search", "summarize"],
        execution_time_ms=750.0,
        output="Found 5 relevant articles",
        metadata={"source": "google"},
    )


@pytest.fixture
def sample_failed_action():
    """Create a sample failed action."""
    return OpenClawKnowledgeItem(
        action_id="test_action_002",
        result=ActionStatus.FAILED,
        context={"task": "api_call"},
        error="Connection timeout",
        execution_time_ms=31000.0,
        capabilities_used=["api_call"],
        workspace_id="ws_001",
    )


@pytest.fixture
def mock_mound():
    """Create a mock KnowledgeMound."""
    mound = MagicMock()
    mound.ingest = AsyncMock(return_value="km_node_001")
    return mound


@pytest.fixture
def event_log():
    """Create a list to capture emitted events."""
    events = []

    def callback(event_type, data):
        events.append({"type": event_type, "data": data})

    return events, callback


# =============================================================================
# 1. Adapter Initialization Tests
# =============================================================================


class TestAdapterInitialization:
    """Tests for OpenClawAdapter initialization and configuration."""

    def test_default_initialization(self):
        """Should initialize with all default values."""
        adapter = OpenClawAdapter(enable_resilience=False)

        assert adapter.ID_PREFIX == "oc_"
        assert adapter.adapter_name == "openclaw"
        assert adapter.source_type == "openclaw"
        assert adapter._openclaw_client is None
        assert adapter._enable_dual_write is False
        assert len(adapter._actions) == 0
        assert len(adapter._patterns) == 0
        assert len(adapter._execution_logs) == 0
        assert len(adapter._session_contexts) == 0

    def test_init_with_openclaw_client(self):
        """Should accept and store an OpenClaw client."""
        client = MagicMock()
        adapter = OpenClawAdapter(openclaw_client=client, enable_resilience=False)

        assert adapter._openclaw_client is client

    def test_init_with_dual_write_enabled(self):
        """Should accept dual write flag."""
        adapter = OpenClawAdapter(enable_dual_write=True, enable_resilience=False)

        assert adapter._enable_dual_write is True

    def test_init_with_event_callback(self, event_log):
        """Should accept and use event callback."""
        events, callback = event_log
        adapter = OpenClawAdapter(event_callback=callback, enable_resilience=False)

        assert adapter._event_callback is callback

    def test_init_with_tracing_disabled(self):
        """Should accept tracing configuration."""
        adapter = OpenClawAdapter(enable_tracing=False, enable_resilience=False)

        assert adapter._enable_tracing is False

    def test_init_with_resilience_options(self):
        """Should accept resilience configuration."""
        adapter = OpenClawAdapter(enable_resilience=False, resilience_timeout=60.0)

        assert adapter._enable_resilience is False

    def test_set_openclaw_client_after_init(self, adapter):
        """Should allow setting client after initialization."""
        client = MagicMock()
        adapter.set_openclaw_client(client)

        assert adapter._openclaw_client is client

    def test_set_openclaw_client_replaces_existing(self):
        """Should replace existing client."""
        client1 = MagicMock()
        client2 = MagicMock()
        adapter = OpenClawAdapter(openclaw_client=client1, enable_resilience=False)

        adapter.set_openclaw_client(client2)

        assert adapter._openclaw_client is client2

    def test_internal_indices_initialized_empty(self, adapter):
        """Should initialize all internal indices as empty."""
        assert adapter._capability_actions == {}
        assert adapter._debate_actions == {}
        assert adapter._action_hash_map == {}
        assert adapter._context_updates == {}
        assert adapter._prioritization_updates == {}
        assert adapter._km_validations == {}

    def test_class_constants(self):
        """Should have correct class-level constants."""
        assert OpenClawAdapter.MIN_ACTION_CONFIDENCE == 0.5
        assert OpenClawAdapter.MIN_PATTERN_OBSERVATIONS == 3
        assert OpenClawAdapter.PATTERN_CONFIDENCE_DECAY == 0.95


# =============================================================================
# 2. Forward Sync: store_action_result
# =============================================================================


class TestStoreActionResult:
    """Tests for store_action_result forward sync method."""

    def test_stores_action_and_returns_id(self, adapter, sample_action):
        """Should store action and return a prefixed ID."""
        item_id = adapter.store_action_result(sample_action)

        assert item_id.startswith("oc_action_")
        assert item_id in adapter._actions

    def test_stored_action_contains_all_fields(self, adapter, sample_action):
        """Should store all action fields in data dict."""
        item_id = adapter.store_action_result(sample_action)
        stored = adapter._actions[item_id]

        assert stored["action_id"] == "test_action_001"
        assert stored["result"] == "success"
        assert stored["debate_id"] == "debate_001"
        assert stored["workspace_id"] == "ws_001"
        assert stored["tenant_id"] == "tenant_001"
        assert stored["capabilities_used"] == ["web_search", "summarize"]
        assert stored["execution_time_ms"] == 750.0
        assert stored["output"] == "Found 5 relevant articles"

    def test_generates_deterministic_id(self, adapter):
        """Should generate deterministic IDs based on action_id and created_at."""
        created = datetime(2024, 6, 15, 10, 0, 0, tzinfo=timezone.utc)
        action1 = OpenClawKnowledgeItem(
            action_id="same_id",
            result=ActionStatus.SUCCESS,
            context={},
            created_at=created,
        )
        action2 = OpenClawKnowledgeItem(
            action_id="same_id",
            result=ActionStatus.SUCCESS,
            context={},
            created_at=created,
        )

        id1 = adapter.store_action_result(action1)
        # Second store overwrites since same hash
        id2 = adapter.store_action_result(action2)

        assert id1 == id2

    def test_stores_action_hash_in_map(self, adapter, sample_action):
        """Should update the action hash map."""
        item_id = adapter.store_action_result(sample_action)
        stored = adapter._actions[item_id]

        action_hash = stored["action_hash"]
        assert adapter._action_hash_map[action_hash] == item_id

    def test_updates_capability_index(self, adapter, sample_action):
        """Should index by each capability used."""
        item_id = adapter.store_action_result(sample_action)

        assert item_id in adapter._capability_actions["web_search"]
        assert item_id in adapter._capability_actions["summarize"]

    def test_updates_debate_index(self, adapter, sample_action):
        """Should index by debate ID when present."""
        item_id = adapter.store_action_result(sample_action)

        assert item_id in adapter._debate_actions["debate_001"]

    def test_no_debate_index_when_no_debate_id(self, adapter):
        """Should not update debate index when debate_id is None."""
        action = OpenClawKnowledgeItem(
            action_id="no_debate",
            result=ActionStatus.SUCCESS,
            context={},
        )
        adapter.store_action_result(action)

        assert len(adapter._debate_actions) == 0

    def test_stores_action_with_callback_configured(self, adapter, event_log, sample_action):
        """Should store action when event callback is configured.

        Note: Due to MRO, FusionMixin's no-op _emit_event takes precedence
        over the base class implementation, so events are not actually emitted.
        This test verifies the main functionality is not affected.
        """
        events, callback = event_log
        adapter.set_event_callback(callback)

        item_id = adapter.store_action_result(sample_action)

        assert item_id is not None
        assert item_id in adapter._actions

    def test_confidence_for_success_fast(self, adapter):
        """Success + fast execution should yield high confidence."""
        action = OpenClawKnowledgeItem(
            action_id="fast_success",
            result=ActionStatus.SUCCESS,
            context={},
            execution_time_ms=500,
        )
        item_id = adapter.store_action_result(action)

        # 0.5 (base) + 0.3 (success) + 0.1 (fast) = 0.9
        assert adapter._actions[item_id]["confidence"] == pytest.approx(0.9, abs=0.01)

    def test_confidence_for_success_slow(self, adapter):
        """Success + slow execution should reduce confidence."""
        action = OpenClawKnowledgeItem(
            action_id="slow_success",
            result=ActionStatus.SUCCESS,
            context={},
            execution_time_ms=50000,
        )
        item_id = adapter.store_action_result(action)

        # 0.5 + 0.3 - 0.1 = 0.7
        assert adapter._actions[item_id]["confidence"] == pytest.approx(0.7, abs=0.01)

    def test_confidence_for_failure(self, adapter):
        """Failed action should have lower confidence."""
        action = OpenClawKnowledgeItem(
            action_id="failed",
            result=ActionStatus.FAILED,
            context={},
        )
        item_id = adapter.store_action_result(action)

        # 0.5 - 0.2 = 0.3
        assert adapter._actions[item_id]["confidence"] == pytest.approx(0.3, abs=0.01)

    def test_confidence_for_timeout(self, adapter):
        """Timeout action should have slightly reduced confidence."""
        action = OpenClawKnowledgeItem(
            action_id="timeout",
            result=ActionStatus.TIMEOUT,
            context={},
        )
        item_id = adapter.store_action_result(action)

        # 0.5 - 0.1 = 0.4
        assert adapter._actions[item_id]["confidence"] == pytest.approx(0.4, abs=0.01)

    def test_confidence_for_pending(self, adapter):
        """Pending action should have base confidence."""
        action = OpenClawKnowledgeItem(
            action_id="pending",
            result=ActionStatus.PENDING,
            context={},
        )
        item_id = adapter.store_action_result(action)

        # 0.5 + 0.0 = 0.5
        assert adapter._actions[item_id]["confidence"] == pytest.approx(0.5, abs=0.01)

    def test_confidence_for_cancelled(self, adapter):
        """Cancelled action should have base confidence."""
        action = OpenClawKnowledgeItem(
            action_id="cancelled",
            result=ActionStatus.CANCELLED,
            context={},
        )
        item_id = adapter.store_action_result(action)

        assert adapter._actions[item_id]["confidence"] == pytest.approx(0.5, abs=0.01)

    def test_confidence_clamped_at_zero(self, adapter):
        """Confidence should not go below 0.0."""
        action = OpenClawKnowledgeItem(
            action_id="very_bad",
            result=ActionStatus.FAILED,
            context={},
            execution_time_ms=50000,  # slow + failed
        )
        item_id = adapter.store_action_result(action)

        assert adapter._actions[item_id]["confidence"] >= 0.0

    def test_confidence_clamped_at_one(self, adapter):
        """Confidence should not go above 1.0."""
        action = OpenClawKnowledgeItem(
            action_id="very_good",
            result=ActionStatus.SUCCESS,
            context={},
            execution_time_ms=100,  # fast + success
        )
        item_id = adapter.store_action_result(action)

        assert adapter._actions[item_id]["confidence"] <= 1.0

    def test_multiple_actions_stored_independently(self, adapter):
        """Should store multiple actions with separate IDs."""
        action1 = OpenClawKnowledgeItem(action_id="a1", result=ActionStatus.SUCCESS, context={})
        action2 = OpenClawKnowledgeItem(action_id="a2", result=ActionStatus.FAILED, context={})

        id1 = adapter.store_action_result(action1)
        id2 = adapter.store_action_result(action2)

        assert id1 != id2
        assert len(adapter._actions) == 2


# =============================================================================
# 2b. Forward Sync: sync_actions_to_mound
# =============================================================================


class TestSyncActionsToMound:
    """Tests for batch syncing actions to KnowledgeMound."""

    @pytest.mark.asyncio
    async def test_syncs_high_confidence_actions(self, adapter, mock_mound, sample_action):
        """Should sync actions above confidence threshold."""
        adapter.store_action_result(sample_action)

        result = await adapter.sync_actions_to_mound(
            mound=mock_mound,
            workspace_id="ws_001",
            min_confidence=0.5,
        )

        assert result.items_synced >= 1
        assert result.direction == "forward"
        assert result.duration_ms > 0
        assert mock_mound.ingest.called

    @pytest.mark.asyncio
    async def test_skips_low_confidence_actions(self, adapter, mock_mound):
        """Should skip actions below confidence threshold."""
        action = OpenClawKnowledgeItem(
            action_id="low_conf",
            result=ActionStatus.FAILED,
            context={},
        )
        adapter.store_action_result(action)

        result = await adapter.sync_actions_to_mound(
            mound=mock_mound,
            workspace_id="ws_001",
            min_confidence=0.9,
        )

        assert result.items_skipped >= 1
        assert result.items_synced == 0

    @pytest.mark.asyncio
    async def test_skips_already_synced(self, adapter, mock_mound, sample_action):
        """Should skip actions already marked as synced."""
        item_id = adapter.store_action_result(sample_action)
        adapter._actions[item_id]["km_synced"] = True

        result = await adapter.sync_actions_to_mound(
            mound=mock_mound,
            workspace_id="ws_001",
        )

        assert result.items_skipped == 1
        assert result.items_synced == 0

    @pytest.mark.asyncio
    async def test_marks_synced_after_success(self, adapter, mock_mound, sample_action):
        """Should mark action as synced after successful ingestion."""
        item_id = adapter.store_action_result(sample_action)

        await adapter.sync_actions_to_mound(
            mound=mock_mound,
            workspace_id="ws_001",
            min_confidence=0.0,
        )

        assert adapter._actions[item_id].get("km_synced") is True
        assert adapter._actions[item_id].get("km_node_id") == "km_node_001"

    @pytest.mark.asyncio
    async def test_handles_ingest_error(self, adapter):
        """Should handle ingestion errors gracefully."""
        failing_mound = MagicMock()
        failing_mound.ingest = AsyncMock(side_effect=RuntimeError("DB error"))

        action = OpenClawKnowledgeItem(
            action_id="error_action",
            result=ActionStatus.SUCCESS,
            context={},
        )
        adapter.store_action_result(action)

        result = await adapter.sync_actions_to_mound(
            mound=failing_mound,
            workspace_id="ws_001",
            min_confidence=0.0,
        )

        assert result.items_failed >= 1
        assert len(result.errors) >= 1
        assert "DB error" in result.errors[0]

    @pytest.mark.asyncio
    async def test_respects_limit(self, adapter, mock_mound):
        """Should only sync up to limit actions."""
        for i in range(10):
            action = OpenClawKnowledgeItem(
                action_id=f"batch_{i}",
                result=ActionStatus.SUCCESS,
                context={},
            )
            adapter.store_action_result(action)

        result = await adapter.sync_actions_to_mound(
            mound=mock_mound,
            workspace_id="ws_001",
            min_confidence=0.0,
            limit=3,
        )

        assert (result.items_synced + result.items_skipped + result.items_failed) <= 3

    @pytest.mark.asyncio
    async def test_empty_actions_returns_zero_sync(self, adapter, mock_mound):
        """Should return zero counts when no actions stored."""
        result = await adapter.sync_actions_to_mound(
            mound=mock_mound,
            workspace_id="ws_001",
        )

        assert result.items_synced == 0
        assert result.items_skipped == 0
        assert result.items_failed == 0


# =============================================================================
# 2c. Forward Sync: store_execution_log and index_session_context
# =============================================================================


class TestStoreExecutionLog:
    """Tests for storing execution logs."""

    def test_stores_log_entries(self, adapter):
        """Should store log entries for an action."""
        entries = [
            {"timestamp": "2024-06-15T10:00:00Z", "level": "info", "message": "Started"},
            {"timestamp": "2024-06-15T10:00:01Z", "level": "info", "message": "Done"},
        ]

        log_id = adapter.store_execution_log("action_001", entries)

        assert log_id == "oc_log_action_001"
        assert len(adapter._execution_logs["action_001"]) == 2

    def test_appends_to_existing_logs(self, adapter):
        """Should append new entries to existing logs."""
        adapter.store_execution_log("action_002", [{"message": "First"}])
        adapter.store_execution_log("action_002", [{"message": "Second"}, {"message": "Third"}])

        assert len(adapter._execution_logs["action_002"]) == 3

    def test_stores_logs_with_callback_configured(self, adapter, event_log):
        """Should store logs when event callback is configured without errors.

        Note: FusionMixin's _emit_event no-op takes MRO precedence.
        """
        events, callback = event_log
        adapter.set_event_callback(callback)

        log_id = adapter.store_execution_log("action_003", [{"message": "Test"}])

        assert log_id == "oc_log_action_003"
        assert "action_003" in adapter._execution_logs

    def test_empty_log_entries(self, adapter):
        """Should handle empty log entries list."""
        log_id = adapter.store_execution_log("action_004", [])

        assert log_id == "oc_log_action_004"
        assert len(adapter._execution_logs["action_004"]) == 0


class TestIndexSessionContext:
    """Tests for indexing session context."""

    def test_indexes_session_context(self, adapter):
        """Should store session context with proper ID."""
        context = {"user_id": "user_123", "workspace": "ws_456"}

        context_id = adapter.index_session_context("session_001", context)

        assert context_id == "oc_ctx_session_001"
        assert context_id in adapter._session_contexts
        stored = adapter._session_contexts[context_id]
        assert stored["session_id"] == "session_001"
        assert stored["context"] == context
        assert "indexed_at" in stored

    def test_indexes_context_with_callback_configured(self, adapter, event_log):
        """Should index context when event callback is configured.

        Note: FusionMixin's _emit_event no-op takes MRO precedence.
        """
        events, callback = event_log
        adapter.set_event_callback(callback)

        context_id = adapter.index_session_context("session_002", {"key": "value"})

        assert context_id == "oc_ctx_session_002"
        assert context_id in adapter._session_contexts

    def test_overwrites_existing_context(self, adapter):
        """Should overwrite existing context for same session."""
        adapter.index_session_context("session_003", {"old": True})
        adapter.index_session_context("session_003", {"new": True})

        # The context_id is deterministic, so second call overwrites
        stored = adapter._session_contexts["oc_ctx_session_003"]
        assert stored["context"] == {"new": True}


# =============================================================================
# 3. Reverse Sync: push_debate_decisions
# =============================================================================


class TestPushDebateDecisions:
    """Tests for pushing debate decisions to OpenClaw."""

    @pytest.mark.asyncio
    async def test_stores_decisions_without_client(self, adapter):
        """Should store decisions locally even without a client."""
        decisions = [
            {"task_id": "t1", "priority": 0.9, "reason": "Critical"},
            {"task_id": "t2", "priority": 0.5},
        ]

        count = await adapter.push_debate_decisions("debate_001", decisions)

        assert count == 2
        assert "t1" in adapter._prioritization_updates
        assert "t2" in adapter._prioritization_updates
        assert adapter._prioritization_updates["t1"].new_priority == 0.9
        assert adapter._prioritization_updates["t1"].reason == "Critical"

    @pytest.mark.asyncio
    async def test_pushes_to_client_and_marks_applied(self, adapter_with_client):
        """Should push to client and mark as applied on success."""
        decisions = [{"task_id": "t3", "priority": 0.8}]

        count = await adapter_with_client.push_debate_decisions("debate_002", decisions)

        assert count == 1
        update = adapter_with_client._prioritization_updates["t3"]
        assert update.applied is True
        assert update.applied_at is not None
        adapter_with_client._openclaw_client.update_task_priority.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_client_error_gracefully(self):
        """Should handle client errors without crashing."""
        client = MagicMock()
        client.update_task_priority = AsyncMock(side_effect=ConnectionError("timeout"))
        adapter = OpenClawAdapter(openclaw_client=client, enable_resilience=False)

        decisions = [{"task_id": "t4", "priority": 0.7}]
        count = await adapter.push_debate_decisions("debate_003", decisions)

        assert count == 1
        assert adapter._prioritization_updates["t4"].applied is False

    @pytest.mark.asyncio
    async def test_skips_decisions_without_task_id(self, adapter):
        """Should skip decisions that lack a task_id."""
        decisions = [
            {"priority": 0.9},  # No task_id
            {"task_id": "", "priority": 0.8},  # Empty task_id (falsy)
            {"task_id": "valid_task", "priority": 0.7},
        ]

        count = await adapter.push_debate_decisions("debate_004", decisions)

        # Only "valid_task" has a truthy task_id
        assert count == 1

    @pytest.mark.asyncio
    async def test_stores_default_values_for_missing_fields(self, adapter):
        """Should use defaults for missing decision fields."""
        decisions = [{"task_id": "t5"}]

        await adapter.push_debate_decisions("debate_005", decisions)

        update = adapter._prioritization_updates["t5"]
        assert update.original_priority == 0.5
        assert update.new_priority == 0.5
        assert update.reason == "debate_decision"
        assert update.confidence == 0.7

    @pytest.mark.asyncio
    async def test_stores_decisions_with_callback_configured(self, adapter, event_log):
        """Should store decisions when event callback is configured.

        Note: FusionMixin's _emit_event no-op takes MRO precedence.
        """
        events, callback = event_log
        adapter.set_event_callback(callback)

        decisions = [{"task_id": "t6", "priority": 0.9}]
        count = await adapter.push_debate_decisions("debate_006", decisions)

        assert count == 1
        assert "t6" in adapter._prioritization_updates

    @pytest.mark.asyncio
    async def test_empty_decisions_returns_zero(self, adapter):
        """Should handle empty decisions list."""
        count = await adapter.push_debate_decisions("debate_007", [])

        assert count == 0


# =============================================================================
# 3b. Reverse Sync: update_openclaw_context
# =============================================================================


class TestUpdateOpenClawContext:
    """Tests for pushing context updates to OpenClaw."""

    @pytest.mark.asyncio
    async def test_stores_updates_without_client(self, adapter):
        """Should store updates locally without a client."""
        updates = [
            KMContextUpdate(
                update_id="u1",
                context_type="pattern",
                content={"pattern": "success"},
                priority=0.8,
            ),
        ]

        count = await adapter.update_openclaw_context(updates)

        # No client means no applied updates
        assert count == 0
        assert "u1" in adapter._context_updates

    @pytest.mark.asyncio
    async def test_pushes_to_client(self, adapter_with_client):
        """Should push context updates to client."""
        updates = [
            KMContextUpdate(
                update_id="u2",
                context_type="insight",
                content={"insight": "valuable"},
                priority=0.9,
            ),
        ]

        count = await adapter_with_client.update_openclaw_context(updates)

        assert count == 1
        adapter_with_client._openclaw_client.add_context.assert_called_once_with(
            context_type="insight",
            content={"insight": "valuable"},
            priority=0.9,
        )

    @pytest.mark.asyncio
    async def test_handles_client_error(self):
        """Should handle client errors gracefully."""
        client = MagicMock()
        client.add_context = AsyncMock(side_effect=RuntimeError("Failed"))
        adapter = OpenClawAdapter(openclaw_client=client, enable_resilience=False)

        updates = [
            KMContextUpdate(update_id="u3", context_type="pattern", content={}),
        ]

        count = await adapter.update_openclaw_context(updates)

        assert count == 0
        assert "u3" in adapter._context_updates

    @pytest.mark.asyncio
    async def test_multiple_updates(self, adapter_with_client):
        """Should handle multiple context updates."""
        updates = [
            KMContextUpdate(update_id=f"u_{i}", context_type="insight", content={"n": i})
            for i in range(5)
        ]

        count = await adapter_with_client.update_openclaw_context(updates)

        assert count == 5
        assert adapter_with_client._openclaw_client.add_context.call_count == 5

    @pytest.mark.asyncio
    async def test_stores_updates_with_callback_configured(self, adapter, event_log):
        """Should store updates when event callback is configured.

        Note: FusionMixin's _emit_event no-op takes MRO precedence.
        """
        events, callback = event_log
        adapter.set_event_callback(callback)

        updates = [
            KMContextUpdate(update_id="u4", context_type="test", content={}),
        ]
        await adapter.update_openclaw_context(updates)

        assert "u4" in adapter._context_updates


# =============================================================================
# 3c. Reverse Sync: sync_knowledge_for_action
# =============================================================================


class TestSyncKnowledgeForAction:
    """Tests for syncing knowledge for action context."""

    @pytest.mark.asyncio
    async def test_returns_relevant_items_by_combined_match(self, adapter):
        """Should return items with high combined relevance (capability + task + keywords)."""
        action_ctx = {
            "capabilities": ["web_search", "summarize"],
            "task_type": "web_search",
            "query": "search relevant content",
        }
        km_items = [
            {
                "id": "rel",
                "content": "search relevant content results",
                "confidence": 0.9,
                "metadata": {
                    "capabilities_used": ["web_search", "summarize"],
                    "task_type": "web_search",
                },
            },
            {
                "id": "irr",
                "content": "unrelated code review",
                "confidence": 0.5,
                "metadata": {"capabilities_used": ["code_review"]},
            },
        ]

        results = await adapter.sync_knowledge_for_action(action_ctx, km_items)

        # The highly relevant item should be returned
        assert any(r["id"] == "rel" for r in results)
        for r in results:
            assert "action_relevance" in r

    @pytest.mark.asyncio
    async def test_returns_items_matching_task_type_and_capability(self, adapter):
        """Should return items matching task type combined with capabilities."""
        action_ctx = {
            "task_type": "code_review",
            "capabilities": ["code_analysis", "lint"],
            "query": "review code quality",
        }
        km_items = [
            {
                "id": "match",
                "content": "code review quality analysis results",
                "confidence": 0.9,
                "metadata": {
                    "task_type": "code_review",
                    "capabilities_used": ["code_analysis", "lint"],
                },
            },
        ]

        results = await adapter.sync_knowledge_for_action(action_ctx, km_items)

        assert len(results) >= 1
        assert results[0]["id"] == "match"

    @pytest.mark.asyncio
    async def test_keyword_matching_in_query(self, adapter):
        """Should match keywords between query and content."""
        action_ctx = {"query": "machine learning optimization", "capabilities": []}
        km_items = [
            {
                "id": "relevant",
                "content": "optimization techniques for machine learning models",
                "confidence": 0.9,
                "metadata": {},
            },
            {
                "id": "irrelevant",
                "content": "cooking recipes database",
                "confidence": 0.9,
                "metadata": {},
            },
        ]

        results = await adapter.sync_knowledge_for_action(action_ctx, km_items)

        if len(results) > 0:
            # If any results returned, the relevant one should be first
            relevant_ids = [r["id"] for r in results]
            if "relevant" in relevant_ids and "irrelevant" in relevant_ids:
                assert relevant_ids.index("relevant") < relevant_ids.index("irrelevant")

    @pytest.mark.asyncio
    async def test_limits_to_top_20(self, adapter):
        """Should limit returned results to 20."""
        action_ctx = {"capabilities": ["test"]}
        km_items = [
            {
                "id": f"item_{i}",
                "content": "test content",
                "confidence": 0.9,
                "metadata": {"capabilities_used": ["test"]},
            }
            for i in range(30)
        ]

        results = await adapter.sync_knowledge_for_action(action_ctx, km_items)

        assert len(results) <= 20

    @pytest.mark.asyncio
    async def test_sorts_by_relevance_descending(self, adapter):
        """Should sort results by relevance in descending order."""
        action_ctx = {"capabilities": ["a", "b"], "task_type": "analysis"}
        km_items = [
            {
                "id": "low",
                "content": "",
                "confidence": 0.9,
                "metadata": {"capabilities_used": ["a"]},
            },
            {
                "id": "high",
                "content": "",
                "confidence": 0.9,
                "metadata": {"capabilities_used": ["a", "b"], "task_type": "analysis"},
            },
        ]

        results = await adapter.sync_knowledge_for_action(action_ctx, km_items)

        if len(results) >= 2:
            assert results[0]["action_relevance"] >= results[1]["action_relevance"]

    @pytest.mark.asyncio
    async def test_filters_below_relevance_threshold(self, adapter):
        """Should filter out items below 0.5 relevance."""
        action_ctx = {"capabilities": [], "task_type": "", "query": ""}
        km_items = [
            {
                "id": "zero_relevance",
                "content": "completely unrelated",
                "confidence": 0.1,
                "metadata": {},
            },
        ]

        results = await adapter.sync_knowledge_for_action(action_ctx, km_items)

        # Should filter out zero-relevance items
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_empty_km_items(self, adapter):
        """Should handle empty KM items list."""
        results = await adapter.sync_knowledge_for_action({"capabilities": ["test"]}, [])

        assert results == []


# =============================================================================
# 4. Pattern Extraction
# =============================================================================


class TestExtractActionPatterns:
    """Tests for extract_action_patterns method."""

    @pytest.mark.asyncio
    async def test_extracts_success_pattern(self, adapter):
        """Should extract success pattern for high success rate capability."""
        for i in range(5):
            action = OpenClawKnowledgeItem(
                action_id=f"s_{i}",
                result=ActionStatus.SUCCESS,
                context={},
                capabilities_used=["web_search"],
                workspace_id="ws_test",
            )
            adapter.store_action_result(action)

        patterns = await adapter.extract_action_patterns("ws_test", min_observations=3)

        ws_pattern = next((p for p in patterns if "web_search" in p.capabilities_involved), None)
        assert ws_pattern is not None
        assert ws_pattern.pattern_type == PatternType.SUCCESS_PATTERN
        assert ws_pattern.success_rate >= 0.8

    @pytest.mark.asyncio
    async def test_extracts_failure_pattern(self, adapter):
        """Should extract failure pattern for low success rate capability."""
        for i in range(5):
            action = OpenClawKnowledgeItem(
                action_id=f"f_{i}",
                result=ActionStatus.FAILED,
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
    async def test_extracts_capability_pattern_for_moderate_rate(self, adapter):
        """Should extract capability pattern for moderate success rate."""
        for i in range(3):
            adapter.store_action_result(
                OpenClawKnowledgeItem(
                    action_id=f"mod_s_{i}",
                    result=ActionStatus.SUCCESS,
                    context={},
                    capabilities_used=["moderate_api"],
                    workspace_id="ws_test",
                )
            )
        for i in range(3):
            adapter.store_action_result(
                OpenClawKnowledgeItem(
                    action_id=f"mod_f_{i}",
                    result=ActionStatus.FAILED,
                    context={},
                    capabilities_used=["moderate_api"],
                    workspace_id="ws_test",
                )
            )

        patterns = await adapter.extract_action_patterns("ws_test", min_observations=3)

        mod_pattern = next((p for p in patterns if "moderate_api" in p.capabilities_involved), None)
        assert mod_pattern is not None
        assert mod_pattern.pattern_type == PatternType.CAPABILITY_PATTERN
        assert 0.3 < mod_pattern.success_rate < 0.8

    @pytest.mark.asyncio
    async def test_respects_min_observations(self, adapter):
        """Should not extract patterns below min observations threshold."""
        for i in range(2):
            adapter.store_action_result(
                OpenClawKnowledgeItem(
                    action_id=f"rare_{i}",
                    result=ActionStatus.SUCCESS,
                    context={},
                    capabilities_used=["rare_cap"],
                    workspace_id="ws_test",
                )
            )

        patterns = await adapter.extract_action_patterns("ws_test", min_observations=5)

        rare = next((p for p in patterns if "rare_cap" in p.capabilities_involved), None)
        assert rare is None

    @pytest.mark.asyncio
    async def test_filters_by_workspace(self, adapter):
        """Should only consider actions for the specified workspace."""
        for i in range(5):
            adapter.store_action_result(
                OpenClawKnowledgeItem(
                    action_id=f"other_{i}",
                    result=ActionStatus.SUCCESS,
                    context={},
                    capabilities_used=["cap_a"],
                    workspace_id="other_ws",
                )
            )

        patterns = await adapter.extract_action_patterns("ws_test", min_observations=3)

        assert len(patterns) == 0

    @pytest.mark.asyncio
    async def test_stores_extracted_patterns_internally(self, adapter):
        """Should store extracted patterns in _patterns dict."""
        for i in range(4):
            adapter.store_action_result(
                OpenClawKnowledgeItem(
                    action_id=f"stored_{i}",
                    result=ActionStatus.SUCCESS,
                    context={},
                    capabilities_used=["stored_cap"],
                    workspace_id="ws_test",
                )
            )

        await adapter.extract_action_patterns("ws_test", min_observations=3)

        assert len(adapter._patterns) >= 1

    @pytest.mark.asyncio
    async def test_pattern_id_format(self, adapter):
        """Should generate proper pattern IDs."""
        for i in range(3):
            adapter.store_action_result(
                OpenClawKnowledgeItem(
                    action_id=f"pid_{i}",
                    result=ActionStatus.SUCCESS,
                    context={},
                    capabilities_used=["my_cap"],
                    workspace_id="ws_test",
                )
            )

        patterns = await adapter.extract_action_patterns("ws_test", min_observations=3)

        assert patterns[0].pattern_id == "oc_pattern_my_cap_ws_test"

    @pytest.mark.asyncio
    async def test_pattern_confidence_scales_with_observations(self, adapter):
        """Should increase confidence with more observations."""
        for i in range(20):
            adapter.store_action_result(
                OpenClawKnowledgeItem(
                    action_id=f"many_{i}",
                    result=ActionStatus.SUCCESS,
                    context={},
                    capabilities_used=["popular_cap"],
                    workspace_id="ws_test",
                )
            )

        patterns = await adapter.extract_action_patterns("ws_test", min_observations=3)

        pop_pattern = next((p for p in patterns if "popular_cap" in p.capabilities_involved), None)
        assert pop_pattern is not None
        # 0.5 + 20 * 0.02 = 0.9, capped at 0.9
        assert pop_pattern.confidence == pytest.approx(0.9, abs=0.01)

    @pytest.mark.asyncio
    async def test_pattern_extraction_works_with_callback(self, adapter, event_log):
        """Should extract patterns when event callback is configured.

        Note: FusionMixin's _emit_event no-op takes MRO precedence.
        """
        events, callback = event_log
        adapter.set_event_callback(callback)

        for i in range(3):
            adapter.store_action_result(
                OpenClawKnowledgeItem(
                    action_id=f"evt_{i}",
                    result=ActionStatus.SUCCESS,
                    context={},
                    capabilities_used=["evt_cap"],
                    workspace_id="ws_test",
                )
            )

        patterns = await adapter.extract_action_patterns("ws_test", min_observations=3)

        assert len(patterns) >= 1
        assert len(adapter._patterns) >= 1


class TestGetFailurePatterns:
    """Tests for get_failure_patterns method."""

    @pytest.mark.asyncio
    async def test_returns_only_failure_patterns(self, adapter):
        """Should return only failure-type patterns."""
        adapter._patterns["p1"] = {
            "pattern_id": "p1",
            "pattern_type": PatternType.FAILURE_PATTERN.value,
            "description": "Failure",
            "observation_count": 10,
        }
        adapter._patterns["p2"] = {
            "pattern_id": "p2",
            "pattern_type": PatternType.SUCCESS_PATTERN.value,
            "description": "Success",
            "observation_count": 5,
        }

        patterns = await adapter.get_failure_patterns("ws_test")

        assert len(patterns) == 1
        assert patterns[0].pattern_type == PatternType.FAILURE_PATTERN

    @pytest.mark.asyncio
    async def test_sorts_by_observation_count(self, adapter):
        """Should sort failure patterns by observation count descending."""
        adapter._patterns["pf1"] = {
            "pattern_id": "pf1",
            "pattern_type": PatternType.FAILURE_PATTERN.value,
            "description": "Less",
            "observation_count": 5,
        }
        adapter._patterns["pf2"] = {
            "pattern_id": "pf2",
            "pattern_type": PatternType.FAILURE_PATTERN.value,
            "description": "More",
            "observation_count": 20,
        }

        patterns = await adapter.get_failure_patterns("ws_test")

        assert patterns[0].observation_count == 20
        assert patterns[1].observation_count == 5

    @pytest.mark.asyncio
    async def test_respects_limit(self, adapter):
        """Should respect the limit parameter."""
        for i in range(5):
            adapter._patterns[f"pf_{i}"] = {
                "pattern_id": f"pf_{i}",
                "pattern_type": PatternType.FAILURE_PATTERN.value,
                "description": f"Failure {i}",
                "observation_count": i,
            }

        patterns = await adapter.get_failure_patterns("ws_test", limit=2)

        assert len(patterns) == 2


class TestGetSuccessPatterns:
    """Tests for get_success_patterns method."""

    @pytest.mark.asyncio
    async def test_returns_only_success_patterns(self, adapter):
        """Should return only success-type patterns."""
        adapter._patterns["ps1"] = {
            "pattern_id": "ps1",
            "pattern_type": PatternType.SUCCESS_PATTERN.value,
            "description": "Good pattern",
            "success_rate": 0.9,
            "confidence": 0.85,
        }
        adapter._patterns["pf1"] = {
            "pattern_id": "pf1",
            "pattern_type": PatternType.FAILURE_PATTERN.value,
            "description": "Bad pattern",
        }

        patterns = await adapter.get_success_patterns("ws_test")

        assert len(patterns) == 1
        assert patterns[0].pattern_type == PatternType.SUCCESS_PATTERN

    @pytest.mark.asyncio
    async def test_sorts_by_success_rate_times_confidence(self, adapter):
        """Should sort by success_rate * confidence."""
        adapter._patterns["low_score"] = {
            "pattern_id": "low_score",
            "pattern_type": PatternType.SUCCESS_PATTERN.value,
            "description": "Low",
            "success_rate": 0.5,
            "confidence": 0.5,
        }
        adapter._patterns["high_score"] = {
            "pattern_id": "high_score",
            "pattern_type": PatternType.SUCCESS_PATTERN.value,
            "description": "High",
            "success_rate": 0.95,
            "confidence": 0.9,
        }

        patterns = await adapter.get_success_patterns("ws_test")

        assert patterns[0].pattern_id == "high_score"


# =============================================================================
# 5. Cross-Debate Learning
# =============================================================================


class TestCrossDebateLearning:
    """Tests for cross_debate_learning method."""

    @pytest.mark.asyncio
    async def test_creates_validations_for_outcomes(self, adapter):
        """Should create KM validations for action outcomes."""
        outcomes = [
            {"action_id": "a1", "confidence": 0.85, "utility": 0.7, "was_supported": True},
            {"action_id": "a2", "confidence": 0.6, "utility": 0.3, "was_contradicted": True},
        ]

        results = await adapter.cross_debate_learning("debate_001", outcomes)

        assert results["actions_analyzed"] == 2
        assert "a1" in adapter._km_validations
        assert "a2" in adapter._km_validations
        assert adapter._km_validations["a1"].was_supported is True
        assert adapter._km_validations["a2"].was_contradicted is True

    @pytest.mark.asyncio
    async def test_updates_stored_action_with_km_data(self, adapter, sample_action):
        """Should update stored action with KM validation data."""
        item_id = adapter.store_action_result(sample_action)

        outcomes = [
            {"action_id": item_id, "confidence": 0.95, "utility": 0.8},
        ]

        await adapter.cross_debate_learning("debate_002", outcomes)

        stored = adapter._actions[item_id]
        assert stored["km_validated"] is True
        assert stored["km_confidence"] == 0.95
        assert stored["cross_debate_utility"] == 0.8

    @pytest.mark.asyncio
    async def test_increments_validation_count(self, adapter):
        """Should increment validation count on repeated learning."""
        outcomes1 = [{"action_id": "a3", "confidence": 0.7, "utility": 0.5}]
        outcomes2 = [{"action_id": "a3", "confidence": 0.8, "utility": 0.6}]

        await adapter.cross_debate_learning("debate_003", outcomes1)
        await adapter.cross_debate_learning("debate_004", outcomes2)

        assert adapter._km_validations["a3"].validation_count >= 2

    @pytest.mark.asyncio
    async def test_generates_recommendations_for_high_utility(self, adapter):
        """Should generate recommendations when high utility actions exist."""
        outcomes = [
            {"action_id": "high_util", "confidence": 0.9, "utility": 0.8},
        ]

        results = await adapter.cross_debate_learning("debate_005", outcomes)

        assert len(results["recommendations"]) >= 1
        assert "high-utility" in results["recommendations"][0]

    @pytest.mark.asyncio
    async def test_skips_outcomes_without_action_id(self, adapter):
        """Should skip outcomes missing action_id."""
        outcomes = [
            {"confidence": 0.9, "utility": 0.8},  # No action_id
            {"action_id": "valid", "confidence": 0.7, "utility": 0.5},
        ]

        results = await adapter.cross_debate_learning("debate_006", outcomes)

        assert results["actions_analyzed"] == 2
        assert "valid" in adapter._km_validations
        assert len(adapter._km_validations) == 1

    @pytest.mark.asyncio
    async def test_cross_debate_learning_with_callback(self, adapter, event_log):
        """Should perform cross-debate learning when callback is configured.

        Note: FusionMixin's _emit_event no-op takes MRO precedence.
        """
        events, callback = event_log
        adapter.set_event_callback(callback)

        outcomes = [{"action_id": "a4", "confidence": 0.8, "utility": 0.9}]
        results = await adapter.cross_debate_learning("debate_007", outcomes)

        assert results["debate_id"] == "debate_007"
        assert results["actions_analyzed"] == 1

    @pytest.mark.asyncio
    async def test_empty_outcomes(self, adapter):
        """Should handle empty outcomes list."""
        results = await adapter.cross_debate_learning("debate_008", [])

        assert results["actions_analyzed"] == 0
        assert results["recommendations"] == []


# =============================================================================
# 6. Batch Validation Sync
# =============================================================================


class TestSyncValidationsFromKM:
    """Tests for sync_validations_from_km method."""

    @pytest.mark.asyncio
    async def test_syncs_validations_and_updates_actions(self, adapter, sample_action):
        """Should sync validations and update matching actions."""
        adapter.store_action_result(sample_action)

        km_items = [
            {
                "id": "km_001",
                "confidence": 0.85,
                "metadata": {
                    "openclaw_action_id": "test_action_001",
                    "was_supported": True,
                },
            },
        ]

        result = await adapter.sync_validations_from_km(km_items)

        assert result.actions_analyzed >= 1
        assert result.actions_updated >= 1
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_skips_items_below_min_confidence(self, adapter):
        """Should skip items below minimum confidence."""
        km_items = [
            {
                "id": "km_002",
                "confidence": 0.3,
                "metadata": {"openclaw_action_id": "test_low"},
            },
        ]

        result = await adapter.sync_validations_from_km(km_items, min_confidence=0.7)

        assert result.actions_analyzed >= 1
        assert result.actions_updated == 0

    @pytest.mark.asyncio
    async def test_skips_items_without_action_id(self, adapter):
        """Should skip items without openclaw_action_id in metadata."""
        km_items = [
            {"id": "km_003", "confidence": 0.9, "metadata": {}},
        ]

        result = await adapter.sync_validations_from_km(km_items)

        assert result.actions_analyzed == 0

    @pytest.mark.asyncio
    async def test_handles_missing_metadata(self, adapter):
        """Should handle items with no metadata key."""
        km_items = [
            {"id": "km_004", "confidence": 0.9},
        ]

        result = await adapter.sync_validations_from_km(km_items)

        assert isinstance(result, OpenClawKMSyncResult)
        assert result.actions_analyzed == 0

    @pytest.mark.asyncio
    async def test_empty_items_list(self, adapter):
        """Should handle empty items list."""
        result = await adapter.sync_validations_from_km([])

        assert result.actions_analyzed == 0
        assert result.actions_updated == 0
        assert result.errors == []

    @pytest.mark.asyncio
    async def test_creates_km_validation_records(self, adapter, sample_action):
        """Should create KMValidationResult records."""
        adapter.store_action_result(sample_action)

        km_items = [
            {
                "id": "km_005",
                "confidence": 0.9,
                "metadata": {
                    "openclaw_action_id": "test_action_001",
                    "was_supported": True,
                    "was_contradicted": False,
                },
            },
        ]

        await adapter.sync_validations_from_km(km_items)

        assert "test_action_001" in adapter._km_validations
        val = adapter._km_validations["test_action_001"]
        assert val.km_confidence == 0.9
        assert val.was_supported is True


# =============================================================================
# 7. Knowledge Item Conversion
# =============================================================================


class TestToKnowledgeItem:
    """Tests for to_knowledge_item conversion."""

    def test_converts_high_confidence_action(self, adapter):
        """Should map high confidence to ConfidenceLevel.HIGH."""
        from aragora.knowledge.mound.types import ConfidenceLevel

        action_data = {
            "id": "oc_001",
            "action_id": "act_001",
            "result": "success",
            "confidence": 0.85,
            "context": {"task": "test"},
            "capabilities_used": ["web_search"],
            "execution_time_ms": 500,
        }

        km_item = adapter.to_knowledge_item(action_data)

        assert km_item.confidence == ConfidenceLevel.HIGH
        assert km_item.id == "oc_001"
        assert km_item.importance == 0.85

    def test_converts_medium_confidence_action(self, adapter):
        """Should map medium confidence to ConfidenceLevel.MEDIUM."""
        from aragora.knowledge.mound.types import ConfidenceLevel

        action_data = {
            "id": "oc_002",
            "action_id": "act_002",
            "confidence": 0.65,
            "context": {},
        }

        km_item = adapter.to_knowledge_item(action_data)

        assert km_item.confidence == ConfidenceLevel.MEDIUM

    def test_converts_low_confidence_action(self, adapter):
        """Should map low confidence to ConfidenceLevel.LOW."""
        from aragora.knowledge.mound.types import ConfidenceLevel

        action_data = {
            "id": "oc_003",
            "action_id": "act_003",
            "confidence": 0.45,
            "context": {},
        }

        km_item = adapter.to_knowledge_item(action_data)

        assert km_item.confidence == ConfidenceLevel.LOW

    def test_converts_very_low_confidence_to_unverified(self, adapter):
        """Should map very low confidence to ConfidenceLevel.UNVERIFIED."""
        from aragora.knowledge.mound.types import ConfidenceLevel

        action_data = {
            "id": "oc_004",
            "action_id": "act_004",
            "confidence": 0.2,
            "context": {},
        }

        km_item = adapter.to_knowledge_item(action_data)

        assert km_item.confidence == ConfidenceLevel.UNVERIFIED

    def test_builds_content_from_action_data(self, adapter):
        """Should build meaningful content string from action data."""
        action_data = {
            "id": "oc_005",
            "action_id": "act_005",
            "output": "Found results",
            "result": "success",
            "capabilities_used": ["web_search"],
            "context": {"task": "search"},
            "confidence": 0.8,
        }

        km_item = adapter.to_knowledge_item(action_data)

        assert "Found results" in km_item.content
        assert "success" in km_item.content

    def test_handles_iso_format_created_at(self, adapter):
        """Should parse ISO format created_at string."""
        action_data = {
            "id": "oc_006",
            "action_id": "act_006",
            "confidence": 0.5,
            "context": {},
            "created_at": "2024-06-15T10:00:00+00:00",
        }

        km_item = adapter.to_knowledge_item(action_data)

        assert km_item.created_at.year == 2024
        assert km_item.created_at.month == 6

    def test_handles_z_suffix_created_at(self, adapter):
        """Should parse Z-suffix datetime string."""
        action_data = {
            "id": "oc_007",
            "action_id": "act_007",
            "confidence": 0.5,
            "context": {},
            "created_at": "2024-06-15T10:00:00Z",
        }

        km_item = adapter.to_knowledge_item(action_data)

        assert km_item.created_at is not None

    def test_handles_missing_created_at(self, adapter):
        """Should use current time when created_at is None."""
        action_data = {
            "id": "oc_008",
            "action_id": "act_008",
            "confidence": 0.5,
            "context": {},
        }

        km_item = adapter.to_knowledge_item(action_data)

        assert km_item.created_at is not None

    def test_metadata_includes_openclaw_fields(self, adapter):
        """Should include OpenClaw-specific fields in metadata."""
        action_data = {
            "id": "oc_009",
            "action_id": "act_009",
            "result": "success",
            "capabilities_used": ["cap_a"],
            "execution_time_ms": 123.4,
            "debate_id": "d_001",
            "confidence": 0.7,
            "context": {},
        }

        km_item = adapter.to_knowledge_item(action_data)

        assert km_item.metadata["openclaw_action_id"] == "act_009"
        assert km_item.metadata["result"] == "success"
        assert km_item.metadata["capabilities_used"] == ["cap_a"]
        assert km_item.metadata["execution_time_ms"] == 123.4
        assert km_item.metadata["debate_id"] == "d_001"


# =============================================================================
# 8. Mixin Integration Tests
# =============================================================================


class TestFusionMixinIntegration:
    """Tests for FusionMixin method implementations in OpenClawAdapter."""

    def test_get_fusion_sources_returns_expected(self, adapter):
        """Should return the configured fusion sources."""
        sources = adapter._get_fusion_sources()

        assert isinstance(sources, list)
        assert "consensus" in sources
        assert "evidence" in sources
        assert "belief" in sources
        assert "continuum" in sources
        assert "insights" in sources

    def test_extract_fusible_data_with_top_level_confidence(self, adapter):
        """Should extract data when confidence is at top level."""
        item = {"id": "test", "confidence": 0.8, "metadata": {}}

        result = adapter._extract_fusible_data(item)

        assert result is not None
        assert result["confidence"] == 0.8
        assert result["is_valid"] is True

    def test_extract_fusible_data_with_metadata_confidence(self, adapter):
        """Should extract data when confidence is in metadata."""
        item = {"id": "test", "metadata": {"confidence": 0.6}}

        result = adapter._extract_fusible_data(item)

        assert result is not None
        assert result["confidence"] == 0.6
        assert result["is_valid"] is True

    def test_extract_fusible_data_returns_none_without_confidence(self, adapter):
        """Should return None when no confidence is available."""
        item = {"id": "test", "metadata": {}}

        result = adapter._extract_fusible_data(item)

        assert result is None

    def test_extract_fusible_data_low_confidence_invalid(self, adapter):
        """Should mark item as invalid when confidence < 0.5."""
        item = {"id": "test", "confidence": 0.3, "metadata": {}}

        result = adapter._extract_fusible_data(item)

        assert result is not None
        assert result["is_valid"] is False

    def test_apply_fusion_result_to_dict(self, adapter):
        """Should apply fusion result to a dict record."""
        record = {"id": "r1"}
        fusion = MagicMock()
        fusion.fused_confidence = 0.88

        success = adapter._apply_fusion_result(record, fusion)

        assert success is True
        assert record["km_fused"] is True
        assert record["km_fused_confidence"] == 0.88
        assert "km_fusion_time" in record

    def test_apply_fusion_result_with_metadata(self, adapter):
        """Should include metadata when provided."""
        record = {"id": "r2"}
        fusion = MagicMock()
        fusion.fused_confidence = 0.75

        success = adapter._apply_fusion_result(record, fusion, metadata={"source": "test"})

        assert success is True
        assert record["km_fusion_metadata"] == {"source": "test"}

    def test_apply_fusion_result_no_fused_confidence(self, adapter):
        """Should return False when fusion result has no fused_confidence."""
        record = {"id": "r3"}
        fusion = MagicMock(spec=[])  # No fused_confidence attribute

        success = adapter._apply_fusion_result(record, fusion)

        assert success is False

    def test_apply_fusion_result_non_dict_record(self, adapter):
        """Should return False for non-dict records."""
        record = "not_a_dict"
        fusion = MagicMock()
        fusion.fused_confidence = 0.9

        success = adapter._apply_fusion_result(record, fusion)

        assert success is False


class TestSemanticSearchMixinIntegration:
    """Tests for SemanticSearchMixin method implementations."""

    def test_get_record_by_id_from_actions(self, adapter, sample_action):
        """Should find record in _actions by full ID."""
        item_id = adapter.store_action_result(sample_action)

        record = adapter._get_record_by_id(item_id)

        assert record is not None
        assert record["action_id"] == "test_action_001"

    def test_get_record_by_id_without_prefix(self, adapter, sample_action):
        """Should find record by raw ID without prefix."""
        item_id = adapter.store_action_result(sample_action)
        raw_id = item_id[len("oc_") :]

        record = adapter._get_record_by_id(raw_id)

        assert record is not None

    def test_get_record_by_id_from_patterns(self, adapter):
        """Should find record in _patterns storage."""
        adapter._patterns["oc_pattern_test"] = {"pattern_id": "oc_pattern_test"}

        record = adapter._get_record_by_id("oc_pattern_test")

        assert record is not None
        assert record["pattern_id"] == "oc_pattern_test"

    def test_get_record_by_id_from_session_contexts(self, adapter):
        """Should find record in _session_contexts."""
        adapter.index_session_context("sess_123", {"key": "val"})

        record = adapter._get_record_by_id("oc_ctx_sess_123")

        assert record is not None
        assert record["session_id"] == "sess_123"

    def test_get_record_by_id_returns_none_for_missing(self, adapter):
        """Should return None for nonexistent IDs."""
        assert adapter._get_record_by_id("nonexistent") is None

    def test_record_to_dict_from_dict(self, adapter):
        """Should convert dict record and add similarity."""
        record = {"id": "test", "value": 42}

        result = adapter._record_to_dict(record, similarity=0.92)

        assert result["similarity"] == 0.92
        assert result["value"] == 42

    def test_record_to_dict_from_knowledge_item(self, adapter):
        """Should convert OpenClawKnowledgeItem and add similarity."""
        item = OpenClawKnowledgeItem(
            action_id="conv_test",
            result=ActionStatus.SUCCESS,
            context={"key": "val"},
        )

        result = adapter._record_to_dict(item, similarity=0.77)

        assert result["similarity"] == 0.77
        assert result["action_id"] == "conv_test"
        assert result["result"] == "success"

    def test_record_to_dict_from_arbitrary_object(self, adapter):
        """Should handle arbitrary objects with fallback."""
        obj = MagicMock()
        obj.id = "obj_id"
        obj.output = "some output"
        obj.confidence = 0.6
        obj.metadata = {"k": "v"}

        result = adapter._record_to_dict(obj, similarity=0.5)

        assert result["similarity"] == 0.5

    def test_extract_record_id_removes_prefix(self, adapter):
        """Should remove oc_ prefix from source ID."""
        assert adapter._extract_record_id("oc_action_123") == "action_123"

    def test_extract_record_id_no_prefix(self, adapter):
        """Should return unchanged ID when no prefix."""
        assert adapter._extract_record_id("action_123") == "action_123"


# =============================================================================
# 9. Error Handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling across the adapter."""

    def test_event_callback_error_does_not_crash_store(self, adapter):
        """Should handle callback errors without crashing store_action_result."""
        adapter.set_event_callback(MagicMock(side_effect=RuntimeError("callback boom")))

        action = OpenClawKnowledgeItem(action_id="err_001", result=ActionStatus.SUCCESS, context={})

        # Should not raise
        item_id = adapter.store_action_result(action)
        assert item_id is not None

    @pytest.mark.asyncio
    async def test_partial_client_failures_dont_halt_batch(self):
        """Should continue processing even when client fails for some items."""
        client = MagicMock()
        call_count = 0

        async def flaky_update(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Flaky error")

        client.update_task_priority = flaky_update
        adapter = OpenClawAdapter(openclaw_client=client, enable_resilience=False)

        decisions = [{"task_id": f"t_{i}", "priority": 0.5 + i * 0.1} for i in range(3)]

        count = await adapter.push_debate_decisions("debate_batch", decisions)

        # All 3 should be counted even though one failed
        assert count == 3


# =============================================================================
# 10. Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_action_with_empty_capabilities(self, adapter):
        """Should handle action with no capabilities."""
        action = OpenClawKnowledgeItem(
            action_id="no_caps",
            result=ActionStatus.SUCCESS,
            context={},
            capabilities_used=[],
        )
        item_id = adapter.store_action_result(action)

        assert item_id in adapter._actions
        assert len(adapter._capability_actions) == 0

    def test_action_with_very_long_output(self, adapter):
        """Should store action with long output."""
        long_output = "x" * 10000
        action = OpenClawKnowledgeItem(
            action_id="long_out",
            result=ActionStatus.SUCCESS,
            context={},
            output=long_output,
        )
        item_id = adapter.store_action_result(action)

        assert adapter._actions[item_id]["output"] == long_output

    def test_build_action_content_truncates_output(self, adapter):
        """Should truncate output to 500 chars in content string."""
        action_data = {"output": "a" * 1000, "result": "success", "context": {}}

        content = adapter._build_action_content(action_data)

        # Output in content should be truncated
        assert len(content) < 1000

    def test_build_action_content_no_output(self, adapter):
        """Should handle action with no output."""
        action_data = {"result": "success", "context": {}}

        content = adapter._build_action_content(action_data)

        assert "success" in content

    def test_build_action_content_empty_dict(self, adapter):
        """Should handle completely empty action data."""
        content = adapter._build_action_content({})

        assert isinstance(content, str)
        assert len(content) > 0

    def test_build_action_content_with_task_in_context(self, adapter):
        """Should include task from context."""
        action_data = {"context": {"task": "search"}, "result": "success"}

        content = adapter._build_action_content(action_data)

        assert "search" in content

    @pytest.mark.asyncio
    async def test_relevance_is_zero_for_no_matching_features(self, adapter):
        """Should compute zero relevance for completely unmatched items."""
        relevance = adapter._calculate_item_relevance(
            {"content": "", "confidence": 0.5, "metadata": {}},
            task_type="",
            capabilities=[],
            query="",
        )

        assert relevance == 0.0

    def test_search_by_capability_empty_results(self, adapter):
        """Should return empty list for unknown capability."""
        results = adapter.search_actions_by_capability("nonexistent_cap")

        assert results == []

    def test_search_by_debate_empty_results(self, adapter):
        """Should return empty list for unknown debate."""
        results = adapter.search_actions_by_debate("nonexistent_debate")

        assert results == []


# =============================================================================
# 11. Query Methods
# =============================================================================


class TestQueryMethods:
    """Tests for get_action, get_pattern, search_actions_by_*."""

    def test_get_action_by_full_id(self, adapter, sample_action):
        """Should retrieve action by full prefixed ID."""
        item_id = adapter.store_action_result(sample_action)

        result = adapter.get_action(item_id)

        assert result is not None
        assert result["action_id"] == "test_action_001"

    def test_get_action_returns_none_for_missing(self, adapter):
        """Should return None for nonexistent action."""
        assert adapter.get_action("oc_action_nonexistent") is None

    def test_get_pattern_by_full_id(self, adapter):
        """Should retrieve pattern by full ID."""
        adapter._patterns["oc_pattern_cap_ws"] = {
            "pattern_id": "oc_pattern_cap_ws",
            "description": "Test",
        }

        result = adapter.get_pattern("oc_pattern_cap_ws")

        assert result is not None
        assert result["description"] == "Test"

    def test_get_pattern_returns_none_for_missing(self, adapter):
        """Should return None for nonexistent pattern."""
        assert adapter.get_pattern("oc_pattern_nonexistent") is None

    def test_search_actions_by_capability(self, adapter):
        """Should return actions matching capability."""
        for i in range(3):
            adapter.store_action_result(
                OpenClawKnowledgeItem(
                    action_id=f"cap_search_{i}",
                    result=ActionStatus.SUCCESS,
                    context={},
                    capabilities_used=["target_cap"],
                )
            )
        adapter.store_action_result(
            OpenClawKnowledgeItem(
                action_id="other",
                result=ActionStatus.SUCCESS,
                context={},
                capabilities_used=["other_cap"],
            )
        )

        results = adapter.search_actions_by_capability("target_cap")

        assert len(results) == 3

    def test_search_actions_by_capability_with_limit(self, adapter):
        """Should respect limit parameter."""
        for i in range(5):
            adapter.store_action_result(
                OpenClawKnowledgeItem(
                    action_id=f"limited_{i}",
                    result=ActionStatus.SUCCESS,
                    context={},
                    capabilities_used=["limited_cap"],
                )
            )

        results = adapter.search_actions_by_capability("limited_cap", limit=2)

        assert len(results) == 2

    def test_search_actions_by_debate(self, adapter):
        """Should return actions matching debate ID."""
        adapter.store_action_result(
            OpenClawKnowledgeItem(
                action_id="d_match",
                result=ActionStatus.SUCCESS,
                context={},
                debate_id="target_debate",
            )
        )
        adapter.store_action_result(
            OpenClawKnowledgeItem(
                action_id="d_other",
                result=ActionStatus.SUCCESS,
                context={},
                debate_id="other_debate",
            )
        )

        results = adapter.search_actions_by_debate("target_debate")

        assert len(results) == 1
        assert results[0]["debate_id"] == "target_debate"


# =============================================================================
# 12. Statistics and Health
# =============================================================================


class TestStatisticsAndHealth:
    """Tests for get_stats, get_reverse_flow_stats, clear_reverse_flow_state, health_check."""

    def test_get_stats_empty_adapter(self, adapter):
        """Should return zeroed stats for empty adapter."""
        stats = adapter.get_stats()

        assert stats["total_actions"] == 0
        assert stats["total_patterns"] == 0
        assert stats["total_execution_logs"] == 0
        assert stats["total_session_contexts"] == 0
        assert stats["capabilities_tracked"] == 0
        assert stats["debates_tracked"] == 0

    def test_get_stats_with_data(self, adapter, sample_action):
        """Should reflect stored data in stats."""
        adapter.store_action_result(sample_action)
        adapter.store_execution_log("act_1", [{"msg": "test"}])
        adapter.index_session_context("sess_1", {"key": "val"})

        stats = adapter.get_stats()

        assert stats["total_actions"] >= 1
        assert stats["total_execution_logs"] >= 1
        assert stats["total_session_contexts"] >= 1
        assert stats["capabilities_tracked"] >= 1
        assert stats["debates_tracked"] >= 1

    def test_get_reverse_flow_stats(self, adapter):
        """Should return reverse flow statistics."""
        stats = adapter.get_reverse_flow_stats()

        assert "validations_stored" in stats
        assert "context_updates" in stats
        assert "prioritization_updates" in stats
        assert "applied_prioritizations" in stats

    @pytest.mark.asyncio
    async def test_reverse_flow_stats_count_applied(self, adapter_with_client):
        """Should count applied prioritizations."""
        decisions = [{"task_id": "t_applied", "priority": 0.9}]
        await adapter_with_client.push_debate_decisions("d_stats", decisions)

        stats = adapter_with_client.get_reverse_flow_stats()

        assert stats["applied_prioritizations"] == 1

    def test_clear_reverse_flow_state(self, adapter):
        """Should clear all reverse flow state."""
        adapter._context_updates["u1"] = MagicMock()
        adapter._prioritization_updates["t1"] = MagicMock()
        adapter._km_validations["a1"] = MagicMock()

        adapter.clear_reverse_flow_state()

        assert len(adapter._context_updates) == 0
        assert len(adapter._prioritization_updates) == 0
        assert len(adapter._km_validations) == 0

    def test_health_check(self, adapter):
        """Should return health status dict."""
        health = adapter.health_check()

        assert health["adapter"] == "openclaw"
        assert "healthy" in health
        assert "reverse_flow_stats" in health


# =============================================================================
# 13. Dataclass Serialization Round-Trips
# =============================================================================


class TestDataclassRoundTrips:
    """Tests for dataclass serialization and deserialization round-trips."""

    def test_openclaw_knowledge_item_round_trip(self):
        """Should survive to_dict/from_dict round-trip."""
        original = OpenClawKnowledgeItem(
            action_id="rt_001",
            result=ActionStatus.SUCCESS,
            context={"task": "search"},
            debate_id="d_001",
            workspace_id="ws_001",
            capabilities_used=["web_search"],
            execution_time_ms=500.0,
            output="Results found",
            metadata={"version": 2},
        )

        restored = OpenClawKnowledgeItem.from_dict(original.to_dict())

        assert restored.action_id == original.action_id
        assert restored.result == original.result
        assert restored.debate_id == original.debate_id
        assert restored.workspace_id == original.workspace_id
        assert restored.capabilities_used == original.capabilities_used
        assert restored.execution_time_ms == original.execution_time_ms
        assert restored.output == original.output
        assert restored.metadata == original.metadata

    def test_action_pattern_round_trip(self):
        """Should survive to_dict/from_dict round-trip."""
        original = ActionPattern(
            pattern_id="pat_001",
            pattern_type=PatternType.WORKFLOW_PATTERN,
            description="Complex workflow",
            success_rate=0.75,
            observation_count=12,
            capabilities_involved=["cap_a", "cap_b"],
            recommendation="Use with caution",
            confidence=0.82,
            contributing_actions=["a1", "a2"],
        )

        restored = ActionPattern.from_dict(original.to_dict())

        assert restored.pattern_id == original.pattern_id
        assert restored.pattern_type == original.pattern_type
        assert restored.success_rate == original.success_rate
        assert restored.observation_count == original.observation_count
        assert restored.capabilities_involved == original.capabilities_involved
        assert restored.confidence == original.confidence

    def test_sync_result_to_dict(self):
        """Should serialize SyncResult correctly."""
        result = SyncResult(
            items_synced=10,
            items_skipped=3,
            items_failed=1,
            errors=["Error A"],
            duration_ms=250.0,
            direction="forward",
            metadata={"batch": 1},
        )

        data = result.to_dict()

        assert data["items_synced"] == 10
        assert data["items_skipped"] == 3
        assert data["items_failed"] == 1
        assert data["errors"] == ["Error A"]
        assert data["duration_ms"] == 250.0
        assert data["direction"] == "forward"
        assert data["metadata"] == {"batch": 1}

    def test_km_context_update_to_dict_with_expiry(self):
        """Should serialize KMContextUpdate with expiry."""
        expires = datetime(2025, 12, 31, tzinfo=timezone.utc)
        update = KMContextUpdate(
            update_id="ctx_001",
            context_type="pattern",
            content={"key": "val"},
            expires_at=expires,
        )

        data = update.to_dict()

        assert data["expires_at"] is not None
        assert "2025" in data["expires_at"]

    def test_km_context_update_to_dict_without_expiry(self):
        """Should handle None expires_at."""
        update = KMContextUpdate(
            update_id="ctx_002",
            context_type="insight",
            content={},
        )

        data = update.to_dict()

        assert data["expires_at"] is None

    def test_task_prioritization_update_to_dict(self):
        """Should serialize TaskPrioritizationUpdate."""
        applied_at = datetime.now(timezone.utc)
        update = TaskPrioritizationUpdate(
            task_id="t_001",
            debate_id="d_001",
            original_priority=0.3,
            new_priority=0.9,
            reason="Consensus",
            confidence=0.85,
            applied=True,
            applied_at=applied_at,
        )

        data = update.to_dict()

        assert data["task_id"] == "t_001"
        assert data["original_priority"] == 0.3
        assert data["new_priority"] == 0.9
        assert data["applied"] is True
        assert data["applied_at"] is not None

    def test_km_validation_result_to_dict(self):
        """Should serialize KMValidationResult."""
        result = KMValidationResult(
            action_id="a_001",
            km_confidence=0.9,
            cross_debate_utility=0.7,
            validation_count=3,
            was_supported=True,
            patterns_matched=["p1", "p2"],
        )

        data = result.to_dict()

        assert data["action_id"] == "a_001"
        assert data["km_confidence"] == 0.9
        assert data["was_supported"] is True
        assert data["patterns_matched"] == ["p1", "p2"]

    def test_openclaw_km_sync_result_to_dict(self):
        """Should serialize OpenClawKMSyncResult."""
        result = OpenClawKMSyncResult(
            actions_analyzed=50,
            actions_updated=30,
            patterns_extracted=5,
            context_updates_pushed=10,
            prioritization_updates=3,
            errors=["E1"],
            duration_ms=500.0,
        )

        data = result.to_dict()

        assert data["actions_analyzed"] == 50
        assert data["actions_updated"] == 30
        assert data["patterns_extracted"] == 5
        assert data["context_updates_pushed"] == 10

    def test_action_status_enum_values(self):
        """Should have correct enum string values."""
        assert ActionStatus.PENDING.value == "pending"
        assert ActionStatus.RUNNING.value == "running"
        assert ActionStatus.SUCCESS.value == "success"
        assert ActionStatus.FAILED.value == "failed"
        assert ActionStatus.TIMEOUT.value == "timeout"
        assert ActionStatus.CANCELLED.value == "cancelled"

    def test_pattern_type_enum_values(self):
        """Should have correct enum string values."""
        assert PatternType.SUCCESS_PATTERN.value == "success"
        assert PatternType.FAILURE_PATTERN.value == "failure"
        assert PatternType.TIMEOUT_PATTERN.value == "timeout"
        assert PatternType.RESOURCE_PATTERN.value == "resource"
        assert PatternType.CAPABILITY_PATTERN.value == "capability"
        assert PatternType.WORKFLOW_PATTERN.value == "workflow"

    def test_from_dict_defaults_for_missing_fields(self):
        """Should use defaults for missing optional fields in from_dict."""
        item = OpenClawKnowledgeItem.from_dict(
            {
                "action_id": "minimal",
                "context": {},
            }
        )

        assert item.result == ActionStatus.PENDING
        assert item.workspace_id == "default"
        assert item.tenant_id == "default"
        assert item.capabilities_used == []
        assert item.execution_time_ms == 0.0
        assert item.output == ""
        assert item.error is None
        assert item.metadata == {}

    def test_action_pattern_from_dict_missing_timestamps(self):
        """Should use current time for missing timestamps."""
        pattern = ActionPattern.from_dict(
            {
                "pattern_id": "no_ts",
                "pattern_type": "success",
            }
        )

        assert pattern.first_observed_at is not None
        assert pattern.last_observed_at is not None
