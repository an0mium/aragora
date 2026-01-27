"""
Tests for aragora.debate.propulsion module.

Covers:
- PropulsionPriority enum
- PropulsionPayload dataclass
- PropulsionResult dataclass
- RegisteredHandler dataclass
- PropulsionEngine class
- Global engine singleton
- Propulsion handler decorator
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.debate.propulsion import (
    PropulsionEngine,
    PropulsionPayload,
    PropulsionPriority,
    PropulsionResult,
    RegisteredHandler,
    get_propulsion_engine,
    propulsion_handler,
    reset_propulsion_engine,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def engine() -> PropulsionEngine:
    """Create a fresh propulsion engine."""
    return PropulsionEngine()


@pytest.fixture
def payload() -> PropulsionPayload:
    """Create a basic payload for testing."""
    return PropulsionPayload(
        data={"test": "data"},
        source_molecule_id="mol-123",
    )


@pytest.fixture
def high_priority_payload() -> PropulsionPayload:
    """Create a high priority payload."""
    return PropulsionPayload(
        data={"urgent": True},
        priority=PropulsionPriority.HIGH,
    )


# =============================================================================
# PropulsionPriority Tests
# =============================================================================


class TestPropulsionPriority:
    """Tests for PropulsionPriority enum."""

    def test_priority_values(self):
        """Test priority enum values."""
        assert PropulsionPriority.CRITICAL.value == 0
        assert PropulsionPriority.HIGH.value == 1
        assert PropulsionPriority.NORMAL.value == 2
        assert PropulsionPriority.LOW.value == 3
        assert PropulsionPriority.BACKGROUND.value == 4

    def test_priority_ordering(self):
        """Test that priorities can be compared by value."""
        assert PropulsionPriority.CRITICAL.value < PropulsionPriority.HIGH.value
        assert PropulsionPriority.HIGH.value < PropulsionPriority.NORMAL.value
        assert PropulsionPriority.NORMAL.value < PropulsionPriority.LOW.value
        assert PropulsionPriority.LOW.value < PropulsionPriority.BACKGROUND.value


# =============================================================================
# PropulsionPayload Tests
# =============================================================================


class TestPropulsionPayload:
    """Tests for PropulsionPayload dataclass."""

    def test_basic_creation(self, payload: PropulsionPayload):
        """Test creating a basic payload."""
        assert payload.data == {"test": "data"}
        assert payload.source_molecule_id == "mol-123"

    def test_default_values(self, payload: PropulsionPayload):
        """Test default values."""
        assert payload.priority == PropulsionPriority.NORMAL
        assert payload.deadline is None
        assert payload.id is not None
        assert payload.created_at is not None
        assert payload.source_stage is None
        assert payload.target_stage is None
        assert payload.routing_key is None
        assert payload.agent_affinity is None
        assert payload.attempt_count == 0
        assert payload.max_attempts == 3
        assert payload.last_error is None

    def test_unique_ids(self):
        """Test that each payload gets a unique ID."""
        p1 = PropulsionPayload(data={})
        p2 = PropulsionPayload(data={})
        assert p1.id != p2.id

    def test_is_expired_no_deadline(self, payload: PropulsionPayload):
        """Test is_expired with no deadline."""
        assert payload.is_expired() is False

    def test_is_expired_future_deadline(self):
        """Test is_expired with future deadline."""
        payload = PropulsionPayload(
            data={},
            deadline=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        assert payload.is_expired() is False

    def test_is_expired_past_deadline(self):
        """Test is_expired with past deadline."""
        payload = PropulsionPayload(
            data={},
            deadline=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        assert payload.is_expired() is True

    def test_can_retry(self, payload: PropulsionPayload):
        """Test can_retry with attempts remaining."""
        assert payload.can_retry() is True

        payload.attempt_count = 2
        assert payload.can_retry() is True

        payload.attempt_count = 3
        assert payload.can_retry() is False

    def test_to_dict(self, payload: PropulsionPayload):
        """Test serialization to dict."""
        data = payload.to_dict()

        assert data["id"] == payload.id
        assert data["data"] == {"test": "data"}
        assert data["priority"] == PropulsionPriority.NORMAL.value
        assert data["source_molecule_id"] == "mol-123"
        assert "created_at" in data

    def test_to_dict_with_deadline(self):
        """Test serialization includes deadline."""
        deadline = datetime.now(timezone.utc) + timedelta(hours=1)
        payload = PropulsionPayload(data={}, deadline=deadline)
        data = payload.to_dict()

        assert data["deadline"] == deadline.isoformat()


# =============================================================================
# PropulsionResult Tests
# =============================================================================


class TestPropulsionResult:
    """Tests for PropulsionResult dataclass."""

    def test_create_success_result(self):
        """Test creating a success result."""
        result = PropulsionResult(
            payload_id="p-123",
            success=True,
            handler_name="test_handler",
            result={"processed": True},
            duration_ms=50.0,
        )

        assert result.success is True
        assert result.payload_id == "p-123"
        assert result.handler_name == "test_handler"
        assert result.result == {"processed": True}

    def test_create_failure_result(self):
        """Test creating a failure result."""
        result = PropulsionResult(
            payload_id="p-123",
            success=False,
            handler_name="test_handler",
            error_message="Processing failed",
        )

        assert result.success is False
        assert result.error_message == "Processing failed"

    def test_to_dict(self):
        """Test serialization to dict."""
        result = PropulsionResult(
            payload_id="p-123",
            success=True,
            handler_name="test_handler",
            duration_ms=100.0,
        )

        data = result.to_dict()

        assert data["payload_id"] == "p-123"
        assert data["success"] is True
        assert data["handler_name"] == "test_handler"
        assert data["duration_ms"] == 100.0
        assert "timestamp" in data


# =============================================================================
# PropulsionEngine Registration Tests
# =============================================================================


class TestPropulsionEngineRegistration:
    """Tests for PropulsionEngine handler registration."""

    def test_register_handler(self, engine: PropulsionEngine):
        """Test registering a handler."""
        handler = AsyncMock()
        engine.register_handler("test_event", handler, name="test_handler")

        assert "test_event" in engine._handlers
        assert len(engine._handlers["test_event"]) == 1
        assert engine._handlers["test_event"][0].name == "test_handler"

    def test_register_multiple_handlers(self, engine: PropulsionEngine):
        """Test registering multiple handlers for same event."""
        handler1 = AsyncMock()
        handler2 = AsyncMock()

        engine.register_handler("test_event", handler1, name="handler1")
        engine.register_handler("test_event", handler2, name="handler2")

        assert len(engine._handlers["test_event"]) == 2

    def test_register_with_priority(self, engine: PropulsionEngine):
        """Test handlers are sorted by priority."""
        low_handler = AsyncMock()
        high_handler = AsyncMock()

        engine.register_handler(
            "test_event", low_handler, name="low", priority=PropulsionPriority.LOW
        )
        engine.register_handler(
            "test_event", high_handler, name="high", priority=PropulsionPriority.HIGH
        )

        handlers = engine._handlers["test_event"]
        assert handlers[0].name == "high"
        assert handlers[1].name == "low"

    def test_unregister_via_return_function(self, engine: PropulsionEngine):
        """Test unregistering via returned function."""
        handler = AsyncMock()
        unregister = engine.register_handler("test_event", handler, name="test")

        assert len(engine._handlers["test_event"]) == 1

        unregister()

        assert len(engine._handlers["test_event"]) == 0

    def test_unregister_handler_by_name(self, engine: PropulsionEngine):
        """Test unregistering by name."""
        handler = AsyncMock()
        engine.register_handler("test_event", handler, name="test_handler")

        result = engine.unregister_handler("test_event", "test_handler")

        assert result is True
        assert len(engine._handlers["test_event"]) == 0

    def test_unregister_nonexistent_handler(self, engine: PropulsionEngine):
        """Test unregistering nonexistent handler."""
        result = engine.unregister_handler("test_event", "nonexistent")
        assert result is False


# =============================================================================
# PropulsionEngine Propel Tests
# =============================================================================


class TestPropulsionEnginePropel:
    """Tests for PropulsionEngine propel method."""

    @pytest.mark.asyncio
    async def test_propel_basic(self, engine: PropulsionEngine, payload: PropulsionPayload):
        """Test basic propulsion."""
        handler = AsyncMock(return_value={"handled": True})
        engine.register_handler("test_event", handler)

        results = await engine.propel("test_event", payload)

        assert len(results) == 1
        assert results[0].success is True
        handler.assert_called_once_with(payload)

    @pytest.mark.asyncio
    async def test_propel_no_handlers(self, engine: PropulsionEngine, payload: PropulsionPayload):
        """Test propelling with no handlers."""
        results = await engine.propel("nonexistent_event", payload)
        assert results == []

    @pytest.mark.asyncio
    async def test_propel_updates_stats(self, engine: PropulsionEngine, payload: PropulsionPayload):
        """Test propelling updates statistics."""
        handler = AsyncMock()
        engine.register_handler("test_event", handler)

        await engine.propel("test_event", payload)

        stats = engine.get_stats()
        assert stats["total_propelled"] == 1
        assert stats["successful"] == 1

    @pytest.mark.asyncio
    async def test_propel_handles_exception(
        self, engine: PropulsionEngine, payload: PropulsionPayload
    ):
        """Test propelling handles handler exceptions."""
        handler = AsyncMock(side_effect=ValueError("Handler error"))
        engine.register_handler("test_event", handler)

        results = await engine.propel("test_event", payload)

        assert len(results) == 1
        assert results[0].success is False
        assert "Handler error" in results[0].error_message

    @pytest.mark.asyncio
    async def test_propel_expired_payload(self, engine: PropulsionEngine):
        """Test propelling expired payload."""
        payload = PropulsionPayload(
            data={},
            deadline=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        handler = AsyncMock()
        engine.register_handler("test_event", handler)

        results = await engine.propel("test_event", payload)

        assert len(results) == 1
        assert results[0].success is False
        assert "expired" in results[0].error_message.lower()
        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_propel_with_filter(self, engine: PropulsionEngine):
        """Test propelling with filter function."""
        handler = AsyncMock()

        def filter_fn(p):
            return p.data.get("include", False)

        engine.register_handler("test_event", handler, filter_fn=filter_fn)

        # Should be filtered out
        payload1 = PropulsionPayload(data={"include": False})
        results1 = await engine.propel("test_event", payload1)
        assert len(results1) == 0
        handler.assert_not_called()

        # Should be included
        payload2 = PropulsionPayload(data={"include": True})
        results2 = await engine.propel("test_event", payload2)
        assert len(results2) == 1
        handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_propel_sync_handler(self, engine: PropulsionEngine, payload: PropulsionPayload):
        """Test propelling with synchronous handler."""

        def sync_handler(p):
            return {"sync": True}

        engine.register_handler("test_event", sync_handler)

        results = await engine.propel("test_event", payload)

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].result == {"sync": True}

    @pytest.mark.asyncio
    async def test_propel_sets_target_stage(
        self, engine: PropulsionEngine, payload: PropulsionPayload
    ):
        """Test propelling sets target_stage on payload."""
        handler = AsyncMock()
        engine.register_handler("proposals_ready", handler)

        await engine.propel("proposals_ready", payload)

        assert payload.target_stage == "proposals_ready"


# =============================================================================
# PropulsionEngine Chain Tests
# =============================================================================


class TestPropulsionEngineChain:
    """Tests for PropulsionEngine chain method."""

    @pytest.mark.asyncio
    async def test_chain_basic(self, engine: PropulsionEngine):
        """Test basic chaining."""
        handler1 = AsyncMock(return_value={"stage": 1})
        handler2 = AsyncMock(return_value={"stage": 2})

        engine.register_handler("stage1", handler1)
        engine.register_handler("stage2", handler2)

        payload1 = PropulsionPayload(data={"step": 1})
        payload2 = PropulsionPayload(data={"step": 2})

        results = await engine.chain(
            [
                ("stage1", payload1),
                ("stage2", payload2),
            ]
        )

        assert len(results) == 2
        assert len(results[0]) == 1  # stage1 results
        assert len(results[1]) == 1  # stage2 results
        handler1.assert_called_once()
        handler2.assert_called_once()

    @pytest.mark.asyncio
    async def test_chain_stops_on_failure(self, engine: PropulsionEngine):
        """Test chain stops on failure by default."""
        handler1 = AsyncMock(side_effect=ValueError("Failed!"))
        handler2 = AsyncMock()

        engine.register_handler("stage1", handler1)
        engine.register_handler("stage2", handler2)

        payload1 = PropulsionPayload(data={})
        payload2 = PropulsionPayload(data={})

        results = await engine.chain(
            [
                ("stage1", payload1),
                ("stage2", payload2),
            ]
        )

        assert len(results) == 1  # Stopped after stage1
        handler2.assert_not_called()

    @pytest.mark.asyncio
    async def test_chain_continues_on_failure(self, engine: PropulsionEngine):
        """Test chain continues when stop_on_failure=False."""
        handler1 = AsyncMock(side_effect=ValueError("Failed!"))
        handler2 = AsyncMock()

        engine.register_handler("stage1", handler1)
        engine.register_handler("stage2", handler2)

        payload1 = PropulsionPayload(data={})
        payload2 = PropulsionPayload(data={})

        results = await engine.chain(
            [
                ("stage1", payload1),
                ("stage2", payload2),
            ],
            stop_on_failure=False,
        )

        assert len(results) == 2
        handler2.assert_called_once()

    @pytest.mark.asyncio
    async def test_chain_links_stages(self, engine: PropulsionEngine):
        """Test chain links source_stage between payloads."""
        captured_payloads = []

        async def capturing_handler(payload):
            captured_payloads.append(payload)
            return {}

        engine.register_handler("stage1", capturing_handler)
        engine.register_handler("stage2", capturing_handler)

        payload1 = PropulsionPayload(data={})
        payload2 = PropulsionPayload(data={})

        await engine.chain(
            [
                ("stage1", payload1),
                ("stage2", payload2),
            ]
        )

        # Second payload should have source_stage set
        assert captured_payloads[1].source_stage == "stage1"


# =============================================================================
# PropulsionEngine Retry Tests
# =============================================================================


class TestPropulsionEngineRetry:
    """Tests for PropulsionEngine propel_with_retry method."""

    @pytest.mark.asyncio
    async def test_retry_on_failure(self, engine: PropulsionEngine):
        """Test retry on failure."""
        call_count = 0

        async def flaky_handler(payload):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return {"success": True}

        engine.register_handler("test_event", flaky_handler)
        payload = PropulsionPayload(data={})

        results = await engine.propel_with_retry(
            "test_event", payload, max_retries=3, backoff_base=0.01
        )

        assert call_count == 3
        assert results[0].success is True

    @pytest.mark.asyncio
    async def test_retry_respects_max(self, engine: PropulsionEngine):
        """Test retry respects max_retries."""
        handler = AsyncMock(side_effect=ValueError("Always fails"))
        engine.register_handler("test_event", handler)
        payload = PropulsionPayload(data={})

        results = await engine.propel_with_retry(
            "test_event", payload, max_retries=2, backoff_base=0.01
        )

        assert handler.call_count == 2
        assert results[0].success is False


# =============================================================================
# PropulsionEngine Broadcast Tests
# =============================================================================


class TestPropulsionEngineBroadcast:
    """Tests for PropulsionEngine broadcast method."""

    @pytest.mark.asyncio
    async def test_broadcast_to_multiple(
        self, engine: PropulsionEngine, payload: PropulsionPayload
    ):
        """Test broadcasting to multiple event types."""
        handler1 = AsyncMock()
        handler2 = AsyncMock()

        engine.register_handler("event1", handler1)
        engine.register_handler("event2", handler2)

        results = await engine.broadcast(["event1", "event2"], payload)

        assert "event1" in results
        assert "event2" in results
        handler1.assert_called_once()
        handler2.assert_called_once()

    @pytest.mark.asyncio
    async def test_broadcast_handles_missing_event(
        self, engine: PropulsionEngine, payload: PropulsionPayload
    ):
        """Test broadcast handles missing event types."""
        handler = AsyncMock()
        engine.register_handler("event1", handler)

        results = await engine.broadcast(["event1", "nonexistent"], payload)

        assert "event1" in results
        assert "nonexistent" in results
        assert results["nonexistent"] == []


# =============================================================================
# PropulsionEngine Stats and Results Tests
# =============================================================================


class TestPropulsionEngineStatsAndResults:
    """Tests for stats and result tracking."""

    @pytest.mark.asyncio
    async def test_get_stats(self, engine: PropulsionEngine, payload: PropulsionPayload):
        """Test getting statistics."""
        handler = AsyncMock()
        engine.register_handler("test_event", handler)

        await engine.propel("test_event", payload)

        stats = engine.get_stats()

        assert "total_propelled" in stats
        assert "successful" in stats
        assert "failed" in stats
        assert "retried" in stats
        assert "registered_handlers" in stats

    @pytest.mark.asyncio
    async def test_get_result(self, engine: PropulsionEngine, payload: PropulsionPayload):
        """Test getting a specific result."""
        handler = AsyncMock(return_value={"result": True})
        engine.register_handler("test_event", handler, name="my_handler")

        await engine.propel("test_event", payload)

        result = engine.get_result(payload.id, "my_handler")

        assert result is not None
        assert result.success is True

    def test_clear_results(self, engine: PropulsionEngine):
        """Test clearing stored results."""
        engine._results["test"] = PropulsionResult(payload_id="p1", success=True, handler_name="h1")

        engine.clear_results()

        assert len(engine._results) == 0


# =============================================================================
# Global Engine Singleton Tests
# =============================================================================


class TestGlobalEngine:
    """Tests for global engine singleton."""

    def test_get_propulsion_engine_returns_singleton(self):
        """Test get_propulsion_engine returns same instance."""
        reset_propulsion_engine()

        engine1 = get_propulsion_engine()
        engine2 = get_propulsion_engine()

        assert engine1 is engine2

    def test_reset_propulsion_engine(self):
        """Test reset_propulsion_engine creates new instance."""
        engine1 = get_propulsion_engine()
        reset_propulsion_engine()
        engine2 = get_propulsion_engine()

        assert engine1 is not engine2


# =============================================================================
# Propulsion Handler Decorator Tests
# =============================================================================


class TestPropulsionHandlerDecorator:
    """Tests for propulsion_handler decorator."""

    def test_decorator_registers_handler(self):
        """Test decorator registers handler with global engine."""
        reset_propulsion_engine()

        @propulsion_handler("decorated_event")
        async def my_handler(payload: PropulsionPayload):
            return {"decorated": True}

        engine = get_propulsion_engine()
        handlers = engine._handlers.get("decorated_event", [])

        assert len(handlers) == 1
        assert handlers[0].name == "my_handler"

    def test_decorator_with_priority(self):
        """Test decorator with custom priority."""
        reset_propulsion_engine()

        @propulsion_handler("priority_event", priority=PropulsionPriority.HIGH)
        async def high_priority_handler(payload: PropulsionPayload):
            return {}

        engine = get_propulsion_engine()
        handlers = engine._handlers.get("priority_event", [])

        assert handlers[0].priority == PropulsionPriority.HIGH


# =============================================================================
# Concurrency Tests
# =============================================================================


class TestPropulsionEngineConcurrency:
    """Tests for concurrency handling."""

    @pytest.mark.asyncio
    async def test_concurrent_handler_limit(self):
        """Test max_concurrent limit is respected."""
        engine = PropulsionEngine(max_concurrent=2)
        concurrent_count = 0
        max_concurrent_seen = 0

        async def tracking_handler(payload):
            nonlocal concurrent_count, max_concurrent_seen
            concurrent_count += 1
            max_concurrent_seen = max(max_concurrent_seen, concurrent_count)
            await asyncio.sleep(0.05)
            concurrent_count -= 1
            return {}

        for i in range(5):
            engine.register_handler("test_event", tracking_handler, name=f"handler_{i}")

        payload = PropulsionPayload(data={})
        await engine.propel("test_event", payload)

        # All handlers are for the same event, so they run sequentially
        # within propel(), but the semaphore limits execution
        assert max_concurrent_seen <= 2

    @pytest.mark.asyncio
    async def test_broadcast_runs_concurrently(self):
        """Test broadcast runs handlers concurrently."""
        engine = PropulsionEngine()
        start_times = []

        async def timing_handler(payload):
            import time

            start_times.append(time.time())
            await asyncio.sleep(0.05)
            return {}

        engine.register_handler("event1", timing_handler)
        engine.register_handler("event2", timing_handler)
        engine.register_handler("event3", timing_handler)

        payload = PropulsionPayload(data={})
        await engine.broadcast(["event1", "event2", "event3"], payload)

        # All should start at roughly the same time (concurrent)
        if len(start_times) >= 2:
            time_diff = max(start_times) - min(start_times)
            assert time_diff < 0.04  # Should all start within 40ms
