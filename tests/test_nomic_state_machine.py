"""
Tests for the Nomic Loop State Machine.

Tests cover:
- State transitions
- Event processing
- Checkpoint persistence
- Error recovery
- Circuit breakers
"""

import asyncio
import json
import os
import tempfile
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.nomic.states import (
    NomicState,
    StateContext,
    VALID_TRANSITIONS,
    is_valid_transition,
    get_state_config,
)
from aragora.nomic.events import (
    Event,
    EventType,
    EventLog,
    start_event,
    stop_event,
    error_event,
    phase_complete_event,
)
from aragora.nomic.state_machine import (
    NomicStateMachine,
    TransitionError,
    create_nomic_state_machine,
)
from aragora.nomic.checkpoints import (
    CheckpointManager,
    save_checkpoint,
    load_checkpoint,
    load_latest_checkpoint,
    list_checkpoints,
    cleanup_old_checkpoints,
)
from aragora.nomic.metrics import NOMIC_CYCLES_IN_PROGRESS, NOMIC_CYCLES_TOTAL, NOMIC_PHASE_TRANSITIONS
from aragora.nomic.recovery import (
    RecoveryStrategy,
    RecoveryDecision,
    RecoveryManager,
    CircuitBreaker,
    CircuitBreakerRegistry,
    calculate_backoff,
)


# =============================================================================
# State Tests
# =============================================================================


class TestNomicState:
    """Tests for NomicState enum and transitions."""

    def test_all_states_defined(self):
        """Verify all expected states are defined."""
        expected = {
            "IDLE",
            "CONTEXT",
            "DEBATE",
            "DESIGN",
            "IMPLEMENT",
            "VERIFY",
            "COMMIT",
            "RECOVERY",
            "COMPLETED",
            "FAILED",
            "PAUSED",
        }
        actual = {s.name for s in NomicState}
        assert actual == expected

    def test_valid_transitions_exist_for_all_states(self):
        """Verify transition rules exist for all states."""
        for state in NomicState:
            assert state in VALID_TRANSITIONS, f"Missing transitions for {state}"

    def test_idle_to_context_valid(self):
        """Test IDLE -> CONTEXT is valid."""
        assert is_valid_transition(NomicState.IDLE, NomicState.CONTEXT)

    def test_idle_to_debate_invalid(self):
        """Test IDLE -> DEBATE is invalid (must go through CONTEXT)."""
        assert not is_valid_transition(NomicState.IDLE, NomicState.DEBATE)

    def test_all_states_can_reach_recovery(self):
        """Test that active states can transition to RECOVERY."""
        active_states = [
            NomicState.CONTEXT,
            NomicState.DEBATE,
            NomicState.DESIGN,
            NomicState.IMPLEMENT,
            NomicState.VERIFY,
            NomicState.COMMIT,
        ]
        for state in active_states:
            assert is_valid_transition(
                state, NomicState.RECOVERY
            ), f"{state} should be able to transition to RECOVERY"

    def test_recovery_can_go_anywhere(self):
        """Test RECOVERY can transition to most states."""
        recovery_targets = [
            NomicState.IDLE,
            NomicState.CONTEXT,
            NomicState.DEBATE,
            NomicState.FAILED,
            NomicState.PAUSED,
        ]
        for target in recovery_targets:
            assert is_valid_transition(NomicState.RECOVERY, target)


class TestStateContext:
    """Tests for StateContext."""

    def test_default_values(self):
        """Test StateContext default values."""
        ctx = StateContext()
        assert ctx.current_state == NomicState.IDLE
        assert ctx.cycle_id == ""
        assert ctx.errors == []
        assert ctx.retry_counts == {}

    def test_serialization_roundtrip(self):
        """Test StateContext serialization and deserialization."""
        ctx = StateContext(
            cycle_id="test123",
            started_at=datetime.utcnow(),
            current_state=NomicState.DEBATE,
            context_result={"files": ["a.py", "b.py"]},
            errors=[{"msg": "test error"}],
        )
        data = ctx.to_dict()
        restored = StateContext.from_dict(data)

        assert restored.cycle_id == ctx.cycle_id
        assert restored.current_state == ctx.current_state
        assert restored.context_result == ctx.context_result
        assert restored.errors == ctx.errors


# =============================================================================
# Event Tests
# =============================================================================


class TestEvents:
    """Tests for events."""

    def test_start_event_creation(self):
        """Test creating a start event."""
        event = start_event(trigger="test", config={"key": "value"})
        assert event.event_type == EventType.START
        assert event.source == "test"
        assert event.data["config"]["key"] == "value"

    def test_error_event_creation(self):
        """Test creating an error event."""
        error = ValueError("test error")
        event = error_event("debate", error, recoverable=True)
        assert event.event_type == EventType.ERROR
        assert event.source == "debate"
        assert "test error" in event.error_message
        assert event.error_type == "ValueError"
        assert event.recoverable is True

    def test_phase_complete_event(self):
        """Test creating a phase complete event."""
        event = phase_complete_event(
            phase="context",
            result={"improvements": ["a", "b"]},
            duration_seconds=30.5,
            tokens_used=1000,
        )
        assert event.event_type == EventType.CONTEXT_COMPLETE
        assert event.source == "context"
        assert event.data["duration_seconds"] == 30.5
        assert event.data["tokens_used"] == 1000

    def test_event_serialization(self):
        """Test event serialization roundtrip."""
        event = start_event(trigger="test")
        data = event.to_dict()
        restored = Event.from_dict(data)

        assert restored.event_type == event.event_type
        assert restored.source == event.source


class TestEventLog:
    """Tests for EventLog."""

    def test_append_and_retrieve(self):
        """Test appending and retrieving events."""
        log = EventLog(cycle_id="test")
        log.append(start_event())
        log.append(error_event("test", ValueError("oops")))

        assert len(log.events) == 2
        errors = log.get_errors()
        assert len(errors) == 1

    def test_summary(self):
        """Test event log summary."""
        log = EventLog(cycle_id="test")
        log.append(start_event())
        log.append(phase_complete_event("context", {}))
        log.append(error_event("debate", ValueError("oops")))

        summary = log.summary()
        assert summary["total_events"] == 3
        assert summary["errors"] == 1


# =============================================================================
# State Machine Tests
# =============================================================================


class TestStateMachine:
    """Tests for NomicStateMachine."""

    def test_create_machine(self):
        """Test creating a state machine."""
        machine = create_nomic_state_machine()
        assert machine.current_state == NomicState.IDLE
        assert not machine.running

    @pytest.mark.asyncio
    async def test_start_transitions_to_context(self):
        """Test that start() transitions to CONTEXT."""
        machine = NomicStateMachine(enable_checkpoints=False)

        # Register a simple context handler
        async def context_handler(ctx, event):
            return NomicState.DEBATE, {"files": ["test.py"]}

        machine.register_handler(NomicState.CONTEXT, context_handler)

        # Register debate handler to prevent error
        async def debate_handler(ctx, event):
            return NomicState.DESIGN, {}

        machine.register_handler(NomicState.DEBATE, debate_handler)

        await machine.start()

        # Machine should have progressed through handlers
        assert machine.cycle_id != ""
        assert len(machine.event_log.events) > 0

    @pytest.mark.asyncio
    async def test_pause_and_resume(self):
        """Test pause and resume functionality."""
        machine = NomicStateMachine(enable_checkpoints=False)
        machine.context.current_state = NomicState.DEBATE
        machine.running = True

        await machine.pause("test pause")
        assert machine.current_state == NomicState.PAUSED

    @pytest.mark.asyncio
    async def test_error_triggers_recovery(self):
        """Test that errors trigger recovery state."""
        machine = NomicStateMachine(enable_checkpoints=False)

        # Register a handler that raises an error
        async def failing_handler(ctx, event):
            raise ValueError("test failure")

        machine.register_handler(NomicState.CONTEXT, failing_handler)

        # Register recovery handler
        async def recovery_handler(ctx, event):
            return NomicState.FAILED, {"recovered": False}

        machine.register_handler(NomicState.RECOVERY, recovery_handler)

        await machine.start()

        # Should have ended in FAILED
        assert machine.current_state == NomicState.FAILED

    def test_get_metrics(self):
        """Test getting machine metrics."""
        machine = NomicStateMachine()
        metrics = machine.get_metrics()

        assert "current_state" in metrics
        assert "running" in metrics
        assert metrics["total_cycles"] == 0


# =============================================================================
# Checkpoint Tests
# =============================================================================


class TestCheckpoints:
    """Tests for checkpoint persistence."""

    def test_save_and_load_checkpoint(self):
        """Test saving and loading a checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {
                "context": {"cycle_id": "test123", "current_state": "DEBATE"},
                "event_log": {"events": []},
            }

            path = save_checkpoint(data, tmpdir, "test123")
            assert Path(path).exists()

            loaded = load_checkpoint(path)
            assert loaded["context"]["cycle_id"] == "test123"

    def test_load_latest_checkpoint(self):
        """Test loading the most recent checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save multiple checkpoints
            for i in range(3):
                data = {"context": {"cycle_id": f"cycle{i}"}}
                save_checkpoint(data, tmpdir, f"cycle{i}")

            latest = load_latest_checkpoint(tmpdir)
            assert latest is not None
            # Latest should be from 'latest.json'
            assert "cycle_id" in latest["context"]

    def test_list_checkpoints(self):
        """Test listing available checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                data = {"context": {"cycle_id": f"cycle{i}", "current_state": "DEBATE"}}
                save_checkpoint(data, tmpdir, f"cycle{i}")

            checkpoints = list_checkpoints(tmpdir)
            assert len(checkpoints) == 3

    def test_cleanup_old_checkpoints(self):
        """Test cleaning up old checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create many checkpoints
            for i in range(15):
                data = {"context": {"cycle_id": f"cycle{i}"}}
                save_checkpoint(data, tmpdir, f"cycle{i}")

            # Cleanup, keeping only 5
            deleted = cleanup_old_checkpoints(tmpdir, keep_count=5, keep_days=0)

            # Should have deleted some
            remaining = list_checkpoints(tmpdir)
            assert len(remaining) <= 5


class TestCheckpointManager:
    """Tests for CheckpointManager."""

    def test_manager_save_and_load(self):
        """Test CheckpointManager save and load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=tmpdir)

            data = {"context": {"cycle_id": "test"}}
            manager.save(data, "test", "DEBATE")

            loaded = manager.load_latest()
            assert loaded is not None
            assert loaded["context"]["cycle_id"] == "test"

    def test_recovery_options(self):
        """Test getting recovery options."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(checkpoint_dir=tmpdir)

            # No checkpoints - should recommend fresh start
            options = manager.get_recovery_options()
            assert any(o["option"] == "start_fresh" for o in options)

            # Add a checkpoint
            data = {"context": {"cycle_id": "test", "current_state": "DEBATE"}}
            manager.save(data, "test", "DEBATE")

            options = manager.get_recovery_options()
            assert any(o["option"] == "resume_latest" for o in options)


# =============================================================================
# Recovery Tests
# =============================================================================


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    def test_initial_state(self):
        """Test circuit breaker initial state."""
        cb = CircuitBreaker("test", failure_threshold=3)
        assert not cb.is_open

    def test_opens_after_threshold(self):
        """Test circuit opens after failure threshold."""
        cb = CircuitBreaker("test", failure_threshold=3)

        cb.record_failure()
        cb.record_failure()
        assert not cb.is_open

        cb.record_failure()
        assert cb.is_open

    def test_success_resets(self):
        """Test success resets failure count."""
        cb = CircuitBreaker("test", failure_threshold=3)

        cb.record_failure()
        cb.record_failure()
        cb.record_success()

        assert not cb.is_open
        # Should need 3 more failures to open
        cb.record_failure()
        cb.record_failure()
        assert not cb.is_open


class TestCircuitBreakerRegistry:
    """Tests for CircuitBreakerRegistry."""

    def test_get_or_create(self):
        """Test getting or creating circuit breakers."""
        registry = CircuitBreakerRegistry()

        cb1 = registry.get_or_create("agent1")
        cb2 = registry.get_or_create("agent1")
        assert cb1 is cb2

    def test_all_open(self):
        """Test getting all open circuit breakers."""
        registry = CircuitBreakerRegistry()

        cb1 = registry.get_or_create("agent1", failure_threshold=1)
        cb2 = registry.get_or_create("agent2", failure_threshold=1)

        cb1.record_failure()
        assert registry.all_open() == ["agent1"]

        cb2.record_failure()
        assert set(registry.all_open()) == {"agent1", "agent2"}


class TestRecoveryManager:
    """Tests for RecoveryManager."""

    def test_transient_error_retries(self):
        """Test that transient errors trigger retries."""
        manager = RecoveryManager()
        ctx = StateContext()

        error = TimeoutError("Connection timed out")
        decision = manager.decide_recovery(NomicState.DEBATE, error, ctx)

        assert decision.strategy == RecoveryStrategy.RETRY

    def test_max_retries_exceeded(self):
        """Test behavior when max retries exceeded."""
        manager = RecoveryManager()
        ctx = StateContext()
        ctx.retry_counts[NomicState.DEBATE.name] = 10  # Many retries already

        error = ValueError("Persistent error")
        decision = manager.decide_recovery(NomicState.DEBATE, error, ctx)

        # Critical state should fail
        assert decision.strategy in (RecoveryStrategy.FAIL, RecoveryStrategy.PAUSE)


class TestBackoff:
    """Tests for backoff calculation."""

    def test_exponential_growth(self):
        """Test delays grow exponentially."""
        delays = [calculate_backoff(i, jitter=False) for i in range(1, 5)]
        assert delays[1] > delays[0]
        assert delays[2] > delays[1]

    def test_max_delay_cap(self):
        """Test delays are capped at max."""
        delay = calculate_backoff(100, max_delay=60, jitter=False)
        assert delay == 60

    def test_jitter_adds_variation(self):
        """Test jitter adds randomness."""
        delays = [calculate_backoff(3, jitter=True) for _ in range(10)]
        # With jitter, not all should be exactly the same
        assert len(set(delays)) > 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestStateMachineIntegration:
    """Integration tests for the full state machine."""

    @pytest.mark.asyncio
    async def test_full_cycle_success(self):
        """Test a complete successful cycle."""
        machine = NomicStateMachine(enable_checkpoints=False)

        # Register handlers for all states
        async def context_handler(ctx, event):
            return NomicState.DEBATE, {"files": ["a.py"]}

        async def debate_handler(ctx, event):
            return NomicState.DESIGN, {"improvements": ["fix bug"]}

        async def design_handler(ctx, event):
            return NomicState.IMPLEMENT, {"design": "simple fix"}

        async def implement_handler(ctx, event):
            return NomicState.VERIFY, {"changes": ["a.py"]}

        async def verify_handler(ctx, event):
            return NomicState.COMMIT, {"tests_passed": True}

        async def commit_handler(ctx, event):
            return NomicState.COMPLETED, {"commit_sha": "abc123"}

        machine.register_handler(NomicState.CONTEXT, context_handler)
        machine.register_handler(NomicState.DEBATE, debate_handler)
        machine.register_handler(NomicState.DESIGN, design_handler)
        machine.register_handler(NomicState.IMPLEMENT, implement_handler)
        machine.register_handler(NomicState.VERIFY, verify_handler)
        machine.register_handler(NomicState.COMMIT, commit_handler)

        await machine.start()

        # Should have completed successfully
        assert machine.current_state == NomicState.COMPLETED
        assert not machine.running
        metrics = machine.get_metrics()
        assert metrics["successful_cycles"] == 1

    @pytest.mark.asyncio
    async def test_metrics_track_transitions(self):
        """Ensure metrics update on transitions and cycle completion."""
        initial_transitions = NOMIC_PHASE_TRANSITIONS.get(from_phase="idle", to_phase="context")
        initial_cycles = NOMIC_CYCLES_TOTAL.get(outcome="success")
        initial_in_progress = NOMIC_CYCLES_IN_PROGRESS.get()

        machine = NomicStateMachine(enable_checkpoints=False, enable_metrics=True)

        async def context_handler(ctx, event):
            return NomicState.DEBATE, {}

        async def debate_handler(ctx, event):
            return NomicState.DESIGN, {}

        async def design_handler(ctx, event):
            return NomicState.IMPLEMENT, {}

        async def implement_handler(ctx, event):
            return NomicState.VERIFY, {}

        async def verify_handler(ctx, event):
            return NomicState.COMMIT, {}

        async def commit_handler(ctx, event):
            return NomicState.COMPLETED, {}

        machine.register_handler(NomicState.CONTEXT, context_handler)
        machine.register_handler(NomicState.DEBATE, debate_handler)
        machine.register_handler(NomicState.DESIGN, design_handler)
        machine.register_handler(NomicState.IMPLEMENT, implement_handler)
        machine.register_handler(NomicState.VERIFY, verify_handler)
        machine.register_handler(NomicState.COMMIT, commit_handler)

        await machine.start()

        assert (
            NOMIC_PHASE_TRANSITIONS.get(from_phase="idle", to_phase="context")
            == initial_transitions + 1
        )
        assert NOMIC_CYCLES_TOTAL.get(outcome="success") == initial_cycles + 1
        assert NOMIC_CYCLES_IN_PROGRESS.get() == initial_in_progress

    @pytest.mark.asyncio
    async def test_cycle_with_recovery(self):
        """Test cycle that requires recovery."""
        machine = NomicStateMachine(enable_checkpoints=False)
        attempt_count = [0]

        async def context_handler(ctx, event):
            return NomicState.DEBATE, {}

        async def debate_handler(ctx, event):
            attempt_count[0] += 1
            if attempt_count[0] < 2:
                raise ValueError("First attempt fails")
            return NomicState.DESIGN, {}

        async def design_handler(ctx, event):
            return NomicState.IMPLEMENT, {}

        async def implement_handler(ctx, event):
            return NomicState.VERIFY, {}

        async def verify_handler(ctx, event):
            return NomicState.COMMIT, {}

        async def commit_handler(ctx, event):
            return NomicState.COMPLETED, {}

        async def recovery_handler(ctx, event):
            # Retry the failed state
            return NomicState.DEBATE, {}

        machine.register_handler(NomicState.CONTEXT, context_handler)
        machine.register_handler(NomicState.DEBATE, debate_handler)
        machine.register_handler(NomicState.DESIGN, design_handler)
        machine.register_handler(NomicState.IMPLEMENT, implement_handler)
        machine.register_handler(NomicState.VERIFY, verify_handler)
        machine.register_handler(NomicState.COMMIT, commit_handler)
        machine.register_handler(NomicState.RECOVERY, recovery_handler)

        await machine.start()

        # Should have recovered and completed
        assert machine.current_state == NomicState.COMPLETED
        assert attempt_count[0] == 2  # Retried once
