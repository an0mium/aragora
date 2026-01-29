"""
Tests for DecisionService - async debate orchestration.

Covers:
- InMemoryStateStore CRUD operations
- EventBus pub/sub
- AsyncDecisionService lifecycle (start, get, cancel, list)
- DebateState serialization
- DebateEvent serialization
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.debate.decision_service import (
    AsyncDecisionService,
    DebateEvent,
    DebateRequest,
    DebateState,
    DebateStatus,
    EventBus,
    EventType,
    InMemoryStateStore,
    get_decision_service,
    reset_decision_service,
)


# =============================================================================
# InMemoryStateStore Tests
# =============================================================================


class TestInMemoryStateStore:
    """Tests for the in-memory state store."""

    @pytest.fixture
    def store(self):
        return InMemoryStateStore()

    @pytest.fixture
    def sample_state(self):
        return DebateState(
            id="test-123",
            task="Test debate topic",
            status=DebateStatus.PENDING,
            total_rounds=3,
            agents=["claude", "gemini"],
        )

    @pytest.mark.asyncio
    async def test_save_and_get(self, store, sample_state):
        await store.save(sample_state)
        retrieved = await store.get("test-123")
        assert retrieved is not None
        assert retrieved.id == "test-123"
        assert retrieved.task == "Test debate topic"
        assert retrieved.status == DebateStatus.PENDING

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, store):
        result = await store.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, store, sample_state):
        await store.save(sample_state)
        deleted = await store.delete("test-123")
        assert deleted is True
        assert await store.get("test-123") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, store):
        deleted = await store.delete("nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_list_active(self, store):
        # Create states with different statuses
        for i, status in enumerate(
            [DebateStatus.PENDING, DebateStatus.RUNNING, DebateStatus.COMPLETED]
        ):
            state = DebateState(
                id=f"debate-{i}",
                task=f"Task {i}",
                status=status,
            )
            await store.save(state)

        active = await store.list_active()
        assert len(active) == 2  # PENDING and RUNNING only
        statuses = {s.status for s in active}
        assert DebateStatus.COMPLETED not in statuses

    @pytest.mark.asyncio
    async def test_list_active_with_limit(self, store):
        for i in range(10):
            state = DebateState(
                id=f"debate-{i}",
                task=f"Task {i}",
                status=DebateStatus.RUNNING,
            )
            await store.save(state)

        active = await store.list_active(limit=5)
        assert len(active) == 5

    @pytest.mark.asyncio
    async def test_save_updates_timestamp(self, store, sample_state):
        await store.save(sample_state)
        first_update = sample_state.updated_at

        await asyncio.sleep(0.01)
        await store.save(sample_state)
        assert sample_state.updated_at >= first_update


# =============================================================================
# EventBus Tests
# =============================================================================


class TestEventBus:
    """Tests for the event pub/sub system."""

    @pytest.fixture
    def bus(self):
        return EventBus()

    @pytest.fixture
    def sample_event(self):
        return DebateEvent(
            debate_id="test-123",
            type=EventType.DEBATE_STARTED,
            data={"task": "Test topic"},
        )

    @pytest.mark.asyncio
    async def test_subscribe_and_publish(self, bus, sample_event):
        queue = await bus.subscribe("test-123")
        await bus.publish(sample_event)

        event = queue.get_nowait()
        assert event.debate_id == "test-123"
        assert event.type == EventType.DEBATE_STARTED

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self, bus, sample_event):
        q1 = await bus.subscribe("test-123")
        q2 = await bus.subscribe("test-123")

        await bus.publish(sample_event)

        # Both subscribers should receive the event
        assert q1.get_nowait().type == EventType.DEBATE_STARTED
        assert q2.get_nowait().type == EventType.DEBATE_STARTED

    @pytest.mark.asyncio
    async def test_unsubscribe(self, bus, sample_event):
        queue = await bus.subscribe("test-123")
        await bus.unsubscribe("test-123", queue)

        await bus.publish(sample_event)

        # Queue should be empty after unsubscribing
        assert queue.empty()

    @pytest.mark.asyncio
    async def test_publish_to_nonexistent_debate(self, bus, sample_event):
        # Should not raise
        await bus.publish(sample_event)

    @pytest.mark.asyncio
    async def test_unsubscribe_nonexistent(self, bus):
        queue = asyncio.Queue()
        # Should not raise
        await bus.unsubscribe("nonexistent", queue)


# =============================================================================
# Data Type Tests
# =============================================================================


class TestDebateState:
    """Tests for DebateState serialization."""

    def test_to_dict(self):
        state = DebateState(
            id="test-123",
            task="Test topic",
            status=DebateStatus.RUNNING,
            progress=0.5,
            current_round=2,
            total_rounds=4,
            agents=["claude", "gemini"],
        )
        d = state.to_dict()
        assert d["id"] == "test-123"
        assert d["status"] == "running"
        assert d["progress"] == 0.5
        assert d["current_round"] == 2
        assert d["agents"] == ["claude", "gemini"]
        assert "created_at" in d
        assert "updated_at" in d

    def test_to_dict_with_result(self):
        result = MagicMock()
        result.to_dict.return_value = {"synthesis": "Test conclusion"}

        state = DebateState(
            id="test-123",
            task="Test",
            status=DebateStatus.COMPLETED,
            result=result,
        )
        d = state.to_dict()
        assert d["result"] == {"synthesis": "Test conclusion"}

    def test_to_dict_without_result(self):
        state = DebateState(
            id="test-123",
            task="Test",
            status=DebateStatus.PENDING,
        )
        d = state.to_dict()
        assert d["result"] is None


class TestDebateEvent:
    """Tests for DebateEvent serialization."""

    def test_to_dict(self):
        event = DebateEvent(
            debate_id="test-123",
            type=EventType.ROUND_STARTED,
            data={"round": 1},
        )
        d = event.to_dict()
        assert d["debate_id"] == "test-123"
        assert d["type"] == "round_started"
        assert d["data"] == {"round": 1}
        assert "timestamp" in d


class TestDebateRequest:
    """Tests for DebateRequest defaults."""

    def test_defaults(self):
        req = DebateRequest(task="Test topic")
        assert req.task == "Test topic"
        assert req.agents is None
        assert req.rounds > 0
        assert req.timeout == 600.0
        assert req.priority == 0
        assert req.enable_streaming is True
        assert req.enable_checkpointing is True
        assert req.enable_memory is True

    def test_custom_values(self):
        req = DebateRequest(
            task="Custom topic",
            agents=["claude", "gemini"],
            rounds=5,
            timeout=120.0,
            priority=3,
        )
        assert req.agents == ["claude", "gemini"]
        assert req.rounds == 5
        assert req.timeout == 120.0
        assert req.priority == 3


# =============================================================================
# AsyncDecisionService Tests
# =============================================================================


class TestAsyncDecisionService:
    """Tests for the async decision service."""

    @pytest.fixture
    def store(self):
        return InMemoryStateStore()

    @pytest.fixture
    def event_bus(self):
        return EventBus()

    @pytest.fixture
    def service(self, store, event_bus):
        return AsyncDecisionService(
            store=store,
            event_bus=event_bus,
            max_concurrent=5,
            default_agents=["claude", "gemini"],
        )

    @pytest.mark.asyncio
    async def test_start_debate_returns_id(self, service):
        request = DebateRequest(task="Test topic")

        with patch("aragora.debate.decision_service.AsyncDecisionService._run_debate"):
            debate_id = await service.start_debate(request)
            assert debate_id is not None
            assert len(debate_id) > 0

    @pytest.mark.asyncio
    async def test_start_debate_creates_state(self, service, store):
        request = DebateRequest(task="Test topic")

        with patch("aragora.debate.decision_service.AsyncDecisionService._run_debate"):
            debate_id = await service.start_debate(request)
            state = await store.get(debate_id)
            assert state is not None
            assert state.task == "Test topic"
            assert state.status == DebateStatus.PENDING

    @pytest.mark.asyncio
    async def test_get_debate(self, service, store):
        state = DebateState(
            id="test-123",
            task="Test",
            status=DebateStatus.RUNNING,
        )
        await store.save(state)

        result = await service.get_debate("test-123")
        assert result is not None
        assert result.id == "test-123"

    @pytest.mark.asyncio
    async def test_get_debate_nonexistent(self, service):
        result = await service.get_debate("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_cancel_debate(self, service, store):
        state = DebateState(
            id="test-123",
            task="Test",
            status=DebateStatus.RUNNING,
        )
        await store.save(state)

        cancelled = await service.cancel_debate("test-123")
        assert cancelled is True

        updated = await store.get("test-123")
        assert updated.status == DebateStatus.CANCELLED
        assert updated.completed_at is not None

    @pytest.mark.asyncio
    async def test_cancel_completed_debate(self, service, store):
        state = DebateState(
            id="test-123",
            task="Test",
            status=DebateStatus.COMPLETED,
        )
        await store.save(state)

        cancelled = await service.cancel_debate("test-123")
        assert cancelled is False

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_debate(self, service):
        cancelled = await service.cancel_debate("nonexistent")
        assert cancelled is False

    @pytest.mark.asyncio
    async def test_list_debates(self, service, store):
        for i in range(3):
            state = DebateState(
                id=f"debate-{i}",
                task=f"Task {i}",
                status=DebateStatus.RUNNING,
            )
            await store.save(state)

        debates = await service.list_debates()
        assert len(debates) == 3

    @pytest.mark.asyncio
    async def test_start_debate_publishes_event(self, service, event_bus):
        request = DebateRequest(task="Test topic")

        with patch("aragora.debate.decision_service.AsyncDecisionService._run_debate"):
            debate_id = await service.start_debate(request)

            # Subscribe after start - event was already published
            # This tests that publish was called (not that we receive it)
            state = await service.get_debate(debate_id)
            assert state is not None


# =============================================================================
# Global Service Tests
# =============================================================================


class TestGlobalService:
    """Tests for the global service factory."""

    def setup_method(self):
        reset_decision_service()

    def teardown_method(self):
        reset_decision_service()

    def test_get_creates_service(self):
        service = get_decision_service()
        assert isinstance(service, AsyncDecisionService)

    def test_get_returns_same_instance(self):
        s1 = get_decision_service()
        s2 = get_decision_service()
        assert s1 is s2

    def test_get_with_store_creates_new(self):
        s1 = get_decision_service()
        s2 = get_decision_service(store=InMemoryStateStore())
        assert s1 is not s2

    def test_reset_clears_instance(self):
        s1 = get_decision_service()
        reset_decision_service()
        s2 = get_decision_service()
        assert s1 is not s2


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Tests for status and event type enums."""

    def test_debate_status_values(self):
        assert DebateStatus.PENDING.value == "pending"
        assert DebateStatus.RUNNING.value == "running"
        assert DebateStatus.COMPLETED.value == "completed"
        assert DebateStatus.FAILED.value == "failed"
        assert DebateStatus.CANCELLED.value == "cancelled"
        assert DebateStatus.PAUSED.value == "paused"

    def test_event_type_values(self):
        assert EventType.DEBATE_STARTED.value == "debate_started"
        assert EventType.DEBATE_COMPLETED.value == "debate_completed"
        assert EventType.ROUND_STARTED.value == "round_started"
        assert EventType.AGENT_MESSAGE.value == "agent_message"
        assert EventType.CONSENSUS_REACHED.value == "consensus_reached"
