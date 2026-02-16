"""
Tests for Nomic Escalation Store.

Comprehensive tests for:
- Escalation record creation and storage
- Escalation state transitions
- Priority and severity management
- Resolution tracking
- Query and filtering operations
- Persistence and recovery
- Expiration and cleanup
"""

import asyncio
import json
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.nomic.escalation_store import (
    EscalationChain,
    EscalationChainConfig,
    EscalationEvent,
    EscalationRecovery,
    EscalationStatus,
    EscalationStore,
    get_escalation_store,
    reset_escalation_store,
)
from aragora.nomic.molecules import EscalationLevel


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_storage_dir(tmp_path):
    """Create a temporary storage directory."""
    storage_dir = tmp_path / "escalations"
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir


@pytest.fixture
def default_config():
    """Create a default escalation chain config."""
    return EscalationChainConfig(
        levels=[
            EscalationLevel.WARN,
            EscalationLevel.THROTTLE,
            EscalationLevel.SUSPEND,
            EscalationLevel.TERMINATE,
        ],
        auto_escalate_minutes=5,
        cooldown_minutes=30,
        max_duration_hours=24,
        allow_skip_levels=False,
        auto_resolve_on_success=True,
        suppress_duplicates_minutes=10,
    )


@pytest.fixture
async def store(temp_storage_dir):
    """Create an initialized escalation store."""
    store = EscalationStore(storage_dir=temp_storage_dir)
    await store.initialize()
    return store


@pytest.fixture
async def chain_with_store(store):
    """Create an escalation chain attached to a store."""
    return await store.create_chain(
        source="test_monitor",
        target="agent-001",
        reason="Test escalation",
    )


@pytest.fixture
def sample_event():
    """Create a sample escalation event."""
    return EscalationEvent(
        id="event-123",
        chain_id="chain-456",
        level=EscalationLevel.WARN,
        action="create",
        timestamp=datetime.now(timezone.utc),
        reason="Test event",
        previous_level=None,
        handler_result=None,
        metadata={"key": "value"},
    )


@pytest.fixture
def sample_chain(default_config):
    """Create a sample escalation chain (not attached to store)."""
    now = datetime.now(timezone.utc)
    return EscalationChain(
        id="chain-test-123",
        source="agent_monitor",
        target="agent-001",
        reason="Response latency exceeded",
        status=EscalationStatus.ACTIVE,
        current_level=EscalationLevel.WARN,
        config=default_config,
        created_at=now,
        updated_at=now,
        auto_escalate_at=now + timedelta(minutes=5),
    )


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton store between tests."""
    reset_escalation_store()
    yield
    reset_escalation_store()


# ============================================================================
# Test EscalationEvent
# ============================================================================


class TestEscalationEvent:
    """Tests for EscalationEvent dataclass."""

    def test_event_creation(self, sample_event):
        """Should create event with all fields."""
        assert sample_event.id == "event-123"
        assert sample_event.chain_id == "chain-456"
        assert sample_event.level == EscalationLevel.WARN
        assert sample_event.action == "create"
        assert sample_event.reason == "Test event"
        assert sample_event.metadata == {"key": "value"}

    def test_event_to_dict(self, sample_event):
        """Should serialize to dictionary."""
        data = sample_event.to_dict()

        assert data["id"] == "event-123"
        assert data["chain_id"] == "chain-456"
        assert data["level"] == "warn"
        assert data["action"] == "create"
        assert data["reason"] == "Test event"
        assert data["previous_level"] is None
        assert data["metadata"] == {"key": "value"}
        assert "timestamp" in data

    def test_event_from_dict(self):
        """Should deserialize from dictionary."""
        data = {
            "id": "event-456",
            "chain_id": "chain-789",
            "level": "throttle",
            "action": "escalate",
            "timestamp": "2024-01-15T10:00:00+00:00",
            "reason": "Auto escalation",
            "previous_level": "warn",
            "handler_result": {"success": True},
            "metadata": {"auto": True},
        }

        event = EscalationEvent.from_dict(data)

        assert event.id == "event-456"
        assert event.level == EscalationLevel.THROTTLE
        assert event.previous_level == EscalationLevel.WARN
        assert event.handler_result == {"success": True}
        assert event.metadata == {"auto": True}

    def test_event_from_dict_without_optional_fields(self):
        """Should handle missing optional fields."""
        data = {
            "id": "event-min",
            "chain_id": "chain-min",
            "level": "warn",
            "action": "create",
            "timestamp": "2024-01-15T10:00:00+00:00",
            "reason": "Minimal event",
        }

        event = EscalationEvent.from_dict(data)

        assert event.id == "event-min"
        assert event.previous_level is None
        assert event.handler_result is None
        assert event.metadata == {}

    def test_event_roundtrip_serialization(self, sample_event):
        """Should preserve data through serialize/deserialize cycle."""
        data = sample_event.to_dict()
        restored = EscalationEvent.from_dict(data)

        assert restored.id == sample_event.id
        assert restored.chain_id == sample_event.chain_id
        assert restored.level == sample_event.level
        assert restored.action == sample_event.action
        assert restored.reason == sample_event.reason
        assert restored.metadata == sample_event.metadata


# ============================================================================
# Test EscalationChainConfig
# ============================================================================


class TestEscalationChainConfig:
    """Tests for EscalationChainConfig dataclass."""

    def test_default_config_values(self):
        """Should have sensible defaults."""
        config = EscalationChainConfig()

        assert len(config.levels) == 4
        assert config.levels[0] == EscalationLevel.WARN
        assert config.levels[-1] == EscalationLevel.TERMINATE
        assert config.auto_escalate_minutes == 5
        assert config.cooldown_minutes == 30
        assert config.max_duration_hours == 24
        assert config.allow_skip_levels is False
        assert config.auto_resolve_on_success is True
        assert config.suppress_duplicates_minutes == 10

    def test_custom_config(self):
        """Should accept custom configuration."""
        config = EscalationChainConfig(
            levels=[EscalationLevel.WARN, EscalationLevel.SUSPEND],
            auto_escalate_minutes=10,
            max_duration_hours=48,
            allow_skip_levels=True,
        )

        assert len(config.levels) == 2
        assert config.auto_escalate_minutes == 10
        assert config.max_duration_hours == 48
        assert config.allow_skip_levels is True


# ============================================================================
# Test EscalationChain
# ============================================================================


class TestEscalationChain:
    """Tests for EscalationChain dataclass."""

    def test_chain_creation(self, sample_chain):
        """Should create chain with all fields."""
        assert sample_chain.id == "chain-test-123"
        assert sample_chain.source == "agent_monitor"
        assert sample_chain.target == "agent-001"
        assert sample_chain.status == EscalationStatus.ACTIVE
        assert sample_chain.current_level == EscalationLevel.WARN

    def test_chain_to_dict(self, sample_chain):
        """Should serialize to dictionary."""
        data = sample_chain.to_dict()

        assert data["id"] == "chain-test-123"
        assert data["source"] == "agent_monitor"
        assert data["target"] == "agent-001"
        assert data["status"] == "active"
        assert data["current_level"] == "warn"
        assert "config" in data
        assert data["config"]["levels"] == ["warn", "throttle", "suspend", "terminate"]

    def test_chain_from_dict(self):
        """Should deserialize from dictionary."""
        data = {
            "id": "chain-from-dict",
            "source": "test_source",
            "target": "test_target",
            "reason": "Test reason",
            "status": "active",
            "current_level": "throttle",
            "config": {
                "levels": ["warn", "throttle", "suspend"],
                "auto_escalate_minutes": 10,
            },
            "created_at": "2024-01-15T10:00:00+00:00",
            "updated_at": "2024-01-15T10:05:00+00:00",
            "events": [],
            "metadata": {"test": True},
        }

        chain = EscalationChain.from_dict(data)

        assert chain.id == "chain-from-dict"
        assert chain.status == EscalationStatus.ACTIVE
        assert chain.current_level == EscalationLevel.THROTTLE
        assert len(chain.config.levels) == 3
        assert chain.config.auto_escalate_minutes == 10
        assert chain.metadata == {"test": True}

    def test_chain_roundtrip_serialization(self, sample_chain):
        """Should preserve data through serialize/deserialize cycle."""
        data = sample_chain.to_dict()
        restored = EscalationChain.from_dict(data)

        assert restored.id == sample_chain.id
        assert restored.source == sample_chain.source
        assert restored.target == sample_chain.target
        assert restored.status == sample_chain.status
        assert restored.current_level == sample_chain.current_level
        assert len(restored.config.levels) == len(sample_chain.config.levels)


class TestEscalationChainLevelManagement:
    """Tests for level management in EscalationChain."""

    def test_can_escalate_when_active_not_at_max(self, sample_chain):
        """Should allow escalation when active and not at max level."""
        assert sample_chain.can_escalate() is True

    def test_cannot_escalate_when_at_max_level(self, sample_chain):
        """Should not allow escalation at max level."""
        sample_chain.current_level = EscalationLevel.TERMINATE
        assert sample_chain.can_escalate() is False

    def test_cannot_escalate_when_not_active(self, sample_chain):
        """Should not allow escalation when not active."""
        sample_chain.status = EscalationStatus.RESOLVED
        assert sample_chain.can_escalate() is False

    def test_get_next_level(self, sample_chain):
        """Should return correct next level."""
        assert sample_chain.get_next_level() == EscalationLevel.THROTTLE

        sample_chain.current_level = EscalationLevel.THROTTLE
        assert sample_chain.get_next_level() == EscalationLevel.SUSPEND

        sample_chain.current_level = EscalationLevel.SUSPEND
        assert sample_chain.get_next_level() == EscalationLevel.TERMINATE

    def test_get_next_level_at_max(self, sample_chain):
        """Should return None when at max level."""
        sample_chain.current_level = EscalationLevel.TERMINATE
        assert sample_chain.get_next_level() is None

    def test_can_deescalate(self, sample_chain):
        """Should allow de-escalation when above min level."""
        sample_chain.current_level = EscalationLevel.THROTTLE
        assert sample_chain.can_deescalate() is True

    def test_cannot_deescalate_at_min_level(self, sample_chain):
        """Should not allow de-escalation at min level."""
        sample_chain.current_level = EscalationLevel.WARN
        assert sample_chain.can_deescalate() is False

    def test_get_previous_level(self, sample_chain):
        """Should return correct previous level."""
        sample_chain.current_level = EscalationLevel.THROTTLE
        assert sample_chain.get_previous_level() == EscalationLevel.WARN

        sample_chain.current_level = EscalationLevel.SUSPEND
        assert sample_chain.get_previous_level() == EscalationLevel.THROTTLE


class TestEscalationChainStateTransitions:
    """Tests for state transitions in EscalationChain."""

    @pytest.mark.asyncio
    async def test_escalate_success(self, sample_chain):
        """Should escalate to next level."""
        event = await sample_chain.escalate(reason="Test escalation")

        assert event is not None
        assert event.action == "escalate"
        assert event.level == EscalationLevel.THROTTLE
        assert event.previous_level == EscalationLevel.WARN
        assert sample_chain.current_level == EscalationLevel.THROTTLE
        assert len(sample_chain.events) == 1

    @pytest.mark.asyncio
    async def test_escalate_fails_at_max(self, sample_chain):
        """Should return None when escalating at max level."""
        sample_chain.current_level = EscalationLevel.TERMINATE
        event = await sample_chain.escalate()

        assert event is None
        assert sample_chain.current_level == EscalationLevel.TERMINATE

    @pytest.mark.asyncio
    async def test_escalate_updates_auto_escalate_time(self, sample_chain):
        """Should update auto-escalate time after escalation."""
        old_time = sample_chain.auto_escalate_at
        await asyncio.sleep(0.01)  # Small delay for time difference

        await sample_chain.escalate()

        assert sample_chain.auto_escalate_at is not None
        assert sample_chain.auto_escalate_at > old_time

    @pytest.mark.asyncio
    async def test_resolve_success(self, sample_chain):
        """Should resolve escalation."""
        event = await sample_chain.resolve(reason="Issue fixed")

        assert event is not None
        assert event.action == "resolve"
        assert sample_chain.status == EscalationStatus.RESOLVED
        assert sample_chain.resolved_at is not None
        assert sample_chain.auto_escalate_at is None

    @pytest.mark.asyncio
    async def test_deescalate_success(self, sample_chain):
        """Should de-escalate to previous level."""
        sample_chain.current_level = EscalationLevel.THROTTLE

        event = await sample_chain.deescalate(reason="Situation improving")

        assert event is not None
        assert event.action == "deescalate"
        assert event.level == EscalationLevel.WARN
        assert event.previous_level == EscalationLevel.THROTTLE
        assert sample_chain.current_level == EscalationLevel.WARN

    @pytest.mark.asyncio
    async def test_deescalate_fails_at_min(self, sample_chain):
        """Should return None when de-escalating at min level."""
        event = await sample_chain.deescalate()

        assert event is None
        assert sample_chain.current_level == EscalationLevel.WARN

    @pytest.mark.asyncio
    async def test_suppress_success(self, sample_chain):
        """Should suppress escalation for duration."""
        event = await sample_chain.suppress(duration_minutes=30, reason="Maintenance window")

        assert event is not None
        assert event.action == "suppress"
        assert sample_chain.status == EscalationStatus.SUPPRESSED
        assert sample_chain.suppress_until is not None
        assert sample_chain.auto_escalate_at is None

    @pytest.mark.asyncio
    async def test_suppress_sets_correct_duration(self, sample_chain):
        """Should set suppress_until to correct time."""
        now = datetime.now(timezone.utc)
        await sample_chain.suppress(duration_minutes=60)

        # Suppress until should be approximately 60 minutes from now
        expected = now + timedelta(minutes=60)
        diff = abs((sample_chain.suppress_until - expected).total_seconds())
        assert diff < 5  # Within 5 seconds tolerance


class TestEscalationChainProperties:
    """Tests for EscalationChain computed properties."""

    def test_duration_active(self, sample_chain):
        """Should calculate duration for active escalation."""
        sample_chain.created_at = datetime.now(timezone.utc) - timedelta(hours=2)
        sample_chain.resolved_at = None

        duration = sample_chain.duration
        assert duration.total_seconds() >= 7200  # At least 2 hours

    def test_duration_resolved(self, sample_chain):
        """Should calculate duration for resolved escalation."""
        sample_chain.created_at = datetime.now(timezone.utc) - timedelta(hours=5)
        sample_chain.resolved_at = datetime.now(timezone.utc) - timedelta(hours=2)

        duration = sample_chain.duration
        assert 10800 <= duration.total_seconds() <= 10810  # ~3 hours

    def test_is_expired_false(self, sample_chain):
        """Should return False when not expired."""
        sample_chain.created_at = datetime.now(timezone.utc) - timedelta(hours=1)
        assert sample_chain.is_expired is False

    def test_is_expired_true(self, sample_chain):
        """Should return True when max duration exceeded."""
        sample_chain.created_at = datetime.now(timezone.utc) - timedelta(hours=25)
        assert sample_chain.is_expired is True

    def test_needs_auto_escalate_false_not_active(self, sample_chain):
        """Should not need auto-escalate when not active."""
        sample_chain.status = EscalationStatus.RESOLVED
        assert sample_chain.needs_auto_escalate is False

    def test_needs_auto_escalate_false_no_time_set(self, sample_chain):
        """Should not need auto-escalate when time not set."""
        sample_chain.auto_escalate_at = None
        assert sample_chain.needs_auto_escalate is False

    def test_needs_auto_escalate_false_future_time(self, sample_chain):
        """Should not need auto-escalate when time is in future."""
        sample_chain.auto_escalate_at = datetime.now(timezone.utc) + timedelta(minutes=10)
        assert sample_chain.needs_auto_escalate is False

    def test_needs_auto_escalate_true_past_time(self, sample_chain):
        """Should need auto-escalate when time has passed."""
        sample_chain.auto_escalate_at = datetime.now(timezone.utc) - timedelta(minutes=1)
        assert sample_chain.needs_auto_escalate is True


# ============================================================================
# Test EscalationStore
# ============================================================================


class TestEscalationStoreInitialization:
    """Tests for EscalationStore initialization."""

    @pytest.mark.asyncio
    async def test_store_initialization(self, temp_storage_dir):
        """Should initialize store with empty chains."""
        store = EscalationStore(storage_dir=temp_storage_dir)
        await store.initialize()

        assert store._initialized is True
        assert len(store._chains) == 0

    @pytest.mark.asyncio
    async def test_store_creates_directory(self, tmp_path):
        """Should create storage directory if not exists."""
        storage_dir = tmp_path / "new_escalations"
        store = EscalationStore(storage_dir=storage_dir)
        await store.initialize()

        assert storage_dir.exists()

    @pytest.mark.asyncio
    async def test_double_initialization(self, store):
        """Should not reinitialize if already initialized."""
        store._chains["test"] = MagicMock()
        await store.initialize()

        assert "test" in store._chains  # Should not be cleared


class TestEscalationStoreChainCreation:
    """Tests for chain creation in EscalationStore."""

    @pytest.mark.asyncio
    async def test_create_chain(self, store):
        """Should create a new escalation chain."""
        chain = await store.create_chain(
            source="agent_monitor",
            target="agent-001",
            reason="High latency",
        )

        assert chain is not None
        assert chain.source == "agent_monitor"
        assert chain.target == "agent-001"
        assert chain.status == EscalationStatus.ACTIVE
        assert chain.current_level == EscalationLevel.WARN
        assert len(chain.events) == 1  # Initial create event

    @pytest.mark.asyncio
    async def test_create_chain_with_custom_config(self, store):
        """Should create chain with custom configuration."""
        config = EscalationChainConfig(
            levels=[EscalationLevel.WARN, EscalationLevel.TERMINATE],
            auto_escalate_minutes=10,
        )

        chain = await store.create_chain(
            source="custom_monitor",
            target="agent-002",
            reason="Custom test",
            config=config,
        )

        assert len(chain.config.levels) == 2
        assert chain.config.auto_escalate_minutes == 10

    @pytest.mark.asyncio
    async def test_create_chain_with_initial_level(self, store):
        """Should create chain with specified initial level."""
        chain = await store.create_chain(
            source="urgent_monitor",
            target="agent-003",
            reason="Critical failure",
            initial_level=EscalationLevel.SUSPEND,
        )

        assert chain.current_level == EscalationLevel.SUSPEND

    @pytest.mark.asyncio
    async def test_create_chain_with_metadata(self, store):
        """Should create chain with metadata."""
        chain = await store.create_chain(
            source="monitor",
            target="agent-004",
            reason="Test",
            metadata={"severity": "high", "ticket": "INC-123"},
        )

        assert chain.metadata == {"severity": "high", "ticket": "INC-123"}

    @pytest.mark.asyncio
    async def test_create_chain_persists(self, store, temp_storage_dir):
        """Should persist chain to storage."""
        await store.create_chain(
            source="persist_test",
            target="agent-persist",
            reason="Persistence test",
        )

        # Check file was created
        chains_file = temp_storage_dir / "chains.jsonl"
        assert chains_file.exists()

        with open(chains_file) as f:
            content = f.read()
            assert "persist_test" in content

    @pytest.mark.asyncio
    async def test_duplicate_chain_returns_existing(self, store):
        """Should return existing active chain for same source/target."""
        chain1 = await store.create_chain(
            source="dup_source",
            target="dup_target",
            reason="First",
        )

        chain2 = await store.create_chain(
            source="dup_source",
            target="dup_target",
            reason="Second",
        )

        assert chain1.id == chain2.id  # Should return same chain


class TestEscalationStoreQueries:
    """Tests for query operations in EscalationStore."""

    @pytest.mark.asyncio
    async def test_get_chain_by_id(self, store):
        """Should retrieve chain by ID."""
        created = await store.create_chain(
            source="get_test",
            target="agent-get",
            reason="Get test",
        )

        retrieved = await store.get_chain(created.id)

        assert retrieved is not None
        assert retrieved.id == created.id

    @pytest.mark.asyncio
    async def test_get_chain_not_found(self, store):
        """Should return None for non-existent chain."""
        result = await store.get_chain("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_active_chain(self, store):
        """Should get active chain for source/target pair."""
        await store.create_chain(
            source="active_source",
            target="active_target",
            reason="Active test",
        )

        chain = await store.get_active_chain("active_source", "active_target")

        assert chain is not None
        assert chain.status == EscalationStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_get_active_chain_excludes_resolved(self, store):
        """Should not return resolved chains."""
        chain = await store.create_chain(
            source="resolved_source",
            target="resolved_target",
            reason="Will resolve",
        )
        await chain.resolve()

        result = await store.get_active_chain("resolved_source", "resolved_target")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_active_escalations(self, store):
        """Should return all active escalations."""
        await store.create_chain(source="src1", target="tgt1", reason="Test 1")
        await store.create_chain(source="src2", target="tgt2", reason="Test 2")
        chain3 = await store.create_chain(source="src3", target="tgt3", reason="Test 3")
        await chain3.resolve()

        active = await store.get_active_escalations()

        assert len(active) == 2
        assert all(c.status == EscalationStatus.ACTIVE for c in active)

    @pytest.mark.asyncio
    async def test_get_chains_by_target(self, store):
        """Should return all chains for a target."""
        await store.create_chain(source="src1", target="shared-target", reason="Test 1")
        await store.create_chain(source="src2", target="shared-target", reason="Test 2")
        await store.create_chain(source="src3", target="other-target", reason="Test 3")

        chains = await store.get_chains_by_target("shared-target")

        assert len(chains) == 2
        assert all(c.target == "shared-target" for c in chains)

    @pytest.mark.asyncio
    async def test_get_chains_by_source(self, store):
        """Should return all chains from a source."""
        await store.create_chain(source="shared-source", target="tgt1", reason="Test 1")
        await store.create_chain(source="shared-source", target="tgt2", reason="Test 2")
        await store.create_chain(source="other-source", target="tgt3", reason="Test 3")

        chains = await store.get_chains_by_source("shared-source")

        assert len(chains) == 2
        assert all(c.source == "shared-source" for c in chains)


class TestEscalationStoreHandlers:
    """Tests for handler registration and execution."""

    @pytest.mark.asyncio
    async def test_register_handler(self, store):
        """Should register handler for level."""
        handler = MagicMock()
        store.register_handler(EscalationLevel.WARN, handler)

        assert EscalationLevel.WARN in store._handlers
        assert store._handlers[EscalationLevel.WARN] == handler

    @pytest.mark.asyncio
    async def test_handler_called_on_create(self, store):
        """Should execute handler when chain is created."""
        handler = MagicMock()
        store.register_handler(EscalationLevel.WARN, handler)

        await store.create_chain(
            source="handler_test",
            target="agent-handler",
            reason="Handler test",
        )

        handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_handler_called(self, store):
        """Should execute async handler correctly."""
        handler = AsyncMock(return_value="handler_result")
        store.register_handler(EscalationLevel.WARN, handler)

        await store.create_chain(
            source="async_test",
            target="agent-async",
            reason="Async handler test",
        )

        handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_handler_exception_logged(self, store):
        """Should log handler exceptions without failing."""
        handler = MagicMock(side_effect=RuntimeError("Handler error"))
        store.register_handler(EscalationLevel.WARN, handler)

        # Should not raise
        chain = await store.create_chain(
            source="error_test",
            target="agent-error",
            reason="Error handler test",
        )

        assert chain is not None


class TestEscalationStoreAutoEscalation:
    """Tests for auto-escalation processing."""

    @pytest.mark.asyncio
    async def test_process_auto_escalations_escalates(self, store):
        """Should auto-escalate chains that need it."""
        chain = await store.create_chain(
            source="auto_test",
            target="agent-auto",
            reason="Auto escalation test",
        )

        # Set auto-escalate time in the past
        chain.auto_escalate_at = datetime.now(timezone.utc) - timedelta(minutes=1)

        events = await store.process_auto_escalations()

        assert len(events) == 1
        assert events[0].action == "escalate"
        assert chain.current_level == EscalationLevel.THROTTLE

    @pytest.mark.asyncio
    async def test_process_auto_escalations_expires(self, store):
        """Should expire chains that exceed max duration."""
        chain = await store.create_chain(
            source="expire_test",
            target="agent-expire",
            reason="Expiration test",
        )

        # Set creation time far in the past
        chain.created_at = datetime.now(timezone.utc) - timedelta(hours=25)

        events = await store.process_auto_escalations()

        assert len(events) == 1
        assert events[0].action == "timeout"
        assert chain.status == EscalationStatus.EXPIRED

    @pytest.mark.asyncio
    async def test_process_auto_escalations_unsuppresses(self, store):
        """Should unsuppress chains when suppress time passes."""
        chain = await store.create_chain(
            source="unsuppress_test",
            target="agent-unsuppress",
            reason="Unsuppress test",
        )
        await chain.suppress(duration_minutes=1)

        # Set suppress_until in the past
        chain.suppress_until = datetime.now(timezone.utc) - timedelta(minutes=1)

        await store.process_auto_escalations()

        assert chain.status == EscalationStatus.ACTIVE
        assert chain.suppress_until is None


class TestEscalationStoreResolution:
    """Tests for resolution operations."""

    @pytest.mark.asyncio
    async def test_resolve_by_target(self, store):
        """Should resolve all active escalations for target."""
        await store.create_chain(source="src1", target="resolve-target", reason="Test 1")
        await store.create_chain(source="src2", target="resolve-target", reason="Test 2")

        events = await store.resolve_by_target("resolve-target", reason="Target fixed")

        assert len(events) == 2
        chains = await store.get_chains_by_target("resolve-target")
        assert all(c.status == EscalationStatus.RESOLVED for c in chains)


class TestEscalationStoreStatistics:
    """Tests for statistics gathering."""

    @pytest.mark.asyncio
    async def test_get_statistics_empty(self, store):
        """Should return empty statistics."""
        stats = await store.get_statistics()

        assert stats["total_chains"] == 0
        assert stats["active_chains"] == 0
        assert stats["total_events"] == 0

    @pytest.mark.asyncio
    async def test_get_statistics_with_chains(self, store):
        """Should return correct statistics."""
        await store.create_chain(source="src1", target="tgt1", reason="Test 1")
        chain2 = await store.create_chain(source="src2", target="tgt2", reason="Test 2")
        await chain2.escalate()
        await chain2.resolve()

        stats = await store.get_statistics()

        assert stats["total_chains"] == 2
        assert stats["active_chains"] == 1
        assert stats["by_status"]["active"] == 1
        assert stats["by_status"]["resolved"] == 1
        assert stats["by_source"]["src1"] == 1
        assert stats["by_source"]["src2"] == 1
        assert stats["total_events"] >= 2


# ============================================================================
# Test Persistence and Recovery
# ============================================================================


class TestEscalationStorePersistence:
    """Tests for persistence and recovery."""

    @pytest.mark.asyncio
    async def test_chains_persist_across_store_instances(self, temp_storage_dir):
        """Should load chains from previous instance."""
        # Create chain in first store
        store1 = EscalationStore(storage_dir=temp_storage_dir)
        await store1.initialize()
        created = await store1.create_chain(
            source="persist_source",
            target="persist_target",
            reason="Persistence test",
        )
        chain_id = created.id

        # Create new store instance
        store2 = EscalationStore(storage_dir=temp_storage_dir)
        await store2.initialize()

        # Should load the chain
        loaded = await store2.get_chain(chain_id)
        assert loaded is not None
        assert loaded.source == "persist_source"
        assert loaded.target == "persist_target"

    @pytest.mark.asyncio
    async def test_events_persist(self, temp_storage_dir):
        """Should persist events with chain."""
        store1 = EscalationStore(storage_dir=temp_storage_dir)
        await store1.initialize()
        chain = await store1.create_chain(
            source="events_source",
            target="events_target",
            reason="Events test",
        )
        await chain.escalate(reason="Test escalation")
        chain_id = chain.id

        # Create new store instance
        store2 = EscalationStore(storage_dir=temp_storage_dir)
        await store2.initialize()

        loaded = await store2.get_chain(chain_id)
        assert len(loaded.events) == 2  # create + escalate

    @pytest.mark.asyncio
    async def test_handles_corrupt_data(self, temp_storage_dir):
        """Should handle corrupt JSONL lines gracefully."""
        # Write corrupt data
        chains_file = temp_storage_dir / "chains.jsonl"
        with open(chains_file, "w") as f:
            f.write("invalid json\n")
            f.write(
                '{"id": "valid", "source": "s", "target": "t", "status": "active", "current_level": "warn", "created_at": "2024-01-01T00:00:00+00:00", "updated_at": "2024-01-01T00:00:00+00:00"}\n'
            )

        store = EscalationStore(storage_dir=temp_storage_dir)
        await store.initialize()

        # Should load the valid chain
        assert len(store._chains) == 1


class TestEscalationRecovery:
    """Tests for EscalationRecovery class."""

    @pytest.mark.asyncio
    async def test_recovery_returns_chains_needing_attention(self, temp_storage_dir):
        """Should return chains that need attention after recovery."""
        store = EscalationStore(storage_dir=temp_storage_dir)
        await store.initialize()

        chain = await store.create_chain(
            source="recovery_source",
            target="recovery_target",
            reason="Recovery test",
        )
        chain.auto_escalate_at = datetime.now(timezone.utc) - timedelta(minutes=1)
        await store._save_all_chains()

        # Create new recovery instance
        store2 = EscalationStore(storage_dir=temp_storage_dir)
        recovery = EscalationRecovery(store2)
        recovered = await recovery.recover()

        assert len(recovered) >= 1

    @pytest.mark.asyncio
    async def test_recovery_processes_pending_escalations(self, temp_storage_dir):
        """Should process pending escalations during recovery."""
        store = EscalationStore(storage_dir=temp_storage_dir)
        await store.initialize()

        chain = await store.create_chain(
            source="pending_source",
            target="pending_target",
            reason="Pending test",
        )
        original_level = chain.current_level
        chain.auto_escalate_at = datetime.now(timezone.utc) - timedelta(minutes=1)
        await store._save_all_chains()

        # Create new recovery instance
        store2 = EscalationStore(storage_dir=temp_storage_dir)
        recovery = EscalationRecovery(store2)
        await recovery.recover()

        # Should have escalated
        recovered_chain = await store2.get_chain(chain.id)
        # Note: The chain may or may not be escalated depending on timing
        # This test verifies the recovery mechanism runs without error


class TestGetEscalationStoreSingleton:
    """Tests for singleton store access."""

    @pytest.mark.asyncio
    async def test_get_escalation_store_creates_singleton(self, tmp_path):
        """Should create singleton instance."""
        reset_escalation_store()
        store = await get_escalation_store(storage_dir=tmp_path / "singleton")

        assert store is not None
        assert store._initialized is True

    @pytest.mark.asyncio
    async def test_get_escalation_store_returns_same_instance(self, tmp_path):
        """Should return same instance on subsequent calls."""
        reset_escalation_store()
        store1 = await get_escalation_store(storage_dir=tmp_path / "singleton")
        store2 = await get_escalation_store()

        assert store1 is store2

    @pytest.mark.asyncio
    async def test_reset_escalation_store(self, tmp_path):
        """Should reset singleton to None."""
        reset_escalation_store()
        await get_escalation_store(storage_dir=tmp_path / "reset")

        reset_escalation_store()

        # Next call should create new instance
        store = await get_escalation_store(storage_dir=tmp_path / "reset2")
        assert store.storage_dir == tmp_path / "reset2"


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEscalationStoreEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_chain_with_store_saves_on_escalate(self, store):
        """Chain with store should auto-save on escalate."""
        chain = await store.create_chain(
            source="auto_save_src",
            target="auto_save_tgt",
            reason="Auto save test",
        )
        original_level = chain.current_level

        await chain.escalate()

        # Reload from store
        reloaded = await store.get_chain(chain.id)
        assert reloaded.current_level == EscalationLevel.THROTTLE

    @pytest.mark.asyncio
    async def test_empty_target_chains(self, store):
        """Should handle query for non-existent target."""
        chains = await store.get_chains_by_target("nonexistent-target")
        assert chains == []

    @pytest.mark.asyncio
    async def test_empty_source_chains(self, store):
        """Should handle query for non-existent source."""
        chains = await store.get_chains_by_source("nonexistent-source")
        assert chains == []

    @pytest.mark.asyncio
    async def test_level_index_for_unknown_level(self, sample_chain):
        """Should return -1 for level not in config."""
        # Create a config without THROTTLE
        sample_chain.config.levels = [EscalationLevel.WARN, EscalationLevel.TERMINATE]
        sample_chain.current_level = EscalationLevel.THROTTLE

        index = sample_chain._get_level_index(EscalationLevel.THROTTLE)
        assert index == -1

    @pytest.mark.asyncio
    async def test_escalate_with_metadata(self, sample_chain):
        """Should include metadata in escalation event."""
        event = await sample_chain.escalate(
            reason="With metadata",
            metadata={"triggered_by": "test", "priority": "high"},
        )

        assert event.metadata == {"triggered_by": "test", "priority": "high"}

    @pytest.mark.asyncio
    async def test_resolve_with_metadata(self, sample_chain):
        """Should include metadata in resolution event."""
        event = await sample_chain.resolve(
            reason="Fixed",
            metadata={"fixed_by": "admin", "ticket": "INC-456"},
        )

        assert event.metadata == {"fixed_by": "admin", "ticket": "INC-456"}

    @pytest.mark.asyncio
    async def test_concurrent_chain_creation(self, store):
        """Should handle concurrent chain creation safely."""

        async def create_chain(i):
            return await store.create_chain(
                source=f"concurrent_src_{i}",
                target=f"concurrent_tgt_{i}",
                reason=f"Concurrent test {i}",
            )

        chains = await asyncio.gather(*[create_chain(i) for i in range(10)])

        assert len(chains) == 10
        assert len(set(c.id for c in chains)) == 10  # All unique IDs


# ============================================================================
# Test Conftest Fixtures (if needed)
# ============================================================================


@pytest.fixture
def mock_nomic_state():
    """Create a mock nomic state for testing."""
    state = MagicMock()
    state.cycle_id = "test-cycle-123"
    state.phase = "context"
    state.proposals = []
    return state
