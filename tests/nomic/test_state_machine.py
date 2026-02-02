"""
Tests for Nomic Loop State Machine.

State machine functionality for autonomous self-improvement:
- State transitions (valid and invalid)
- State machine initialization
- Event handling
- Guard conditions
- Error states and recovery
- Edge cases
"""

import asyncio
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.nomic.events import (
    Event,
    EventLog,
    EventType,
    error_event,
    pause_event,
    phase_complete_event,
    resume_event,
    start_event,
    stop_event,
    timeout_event,
)
from aragora.nomic.state_machine import (
    NomicStateMachine,
    StateTimeoutError,
    TransitionError,
    create_nomic_state_machine,
)
from aragora.nomic.states import (
    NomicState,
    StateContext,
    VALID_TRANSITIONS,
    get_state_config,
    is_valid_transition,
)


class TestStateMachineInitialization:
    """Tests for state machine initialization."""

    def test_init_with_defaults(self, tmp_path):
        """Should initialize with default configuration."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_metrics=False,
        )

        assert machine.running is False
        assert machine.current_state == NomicState.IDLE
        assert machine.enable_checkpoints is True
        assert machine._handlers == {}
        assert machine._on_transition_callbacks == []
        assert machine._on_error_callbacks == []

    def test_init_with_custom_checkpoint_dir(self, tmp_path):
        """Should accept custom checkpoint directory."""
        checkpoint_dir = str(tmp_path / "checkpoints")
        machine = NomicStateMachine(
            checkpoint_dir=checkpoint_dir,
            enable_metrics=False,
        )

        assert machine.checkpoint_dir == checkpoint_dir

    def test_init_with_checkpoints_disabled(self, tmp_path):
        """Should allow disabling checkpoints."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_checkpoints=False,
            enable_metrics=False,
        )

        assert machine.enable_checkpoints is False

    def test_init_creates_fresh_context(self, tmp_path):
        """Should create a fresh StateContext on init."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_metrics=False,
        )

        assert isinstance(machine.context, StateContext)
        assert machine.context.current_state == NomicState.IDLE
        assert machine.context.cycle_id == ""

    def test_init_creates_event_log(self, tmp_path):
        """Should create an EventLog on init."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_metrics=False,
        )

        assert isinstance(machine.event_log, EventLog)
        assert len(machine.event_log.events) == 0

    def test_init_metrics_disabled(self, tmp_path):
        """Should not register metrics callback when disabled."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_metrics=False,
        )

        # No metrics callback should be registered
        assert len(machine._on_transition_callbacks) == 0


class TestStateMachineHandlerRegistration:
    """Tests for handler registration."""

    def test_register_handler(self, tmp_path):
        """Should register a handler for a state."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_metrics=False,
        )

        async def context_handler(ctx, event):
            return NomicState.DEBATE, {"result": "ok"}

        machine.register_handler(NomicState.CONTEXT, context_handler)

        assert NomicState.CONTEXT in machine._handlers
        assert machine._handlers[NomicState.CONTEXT] is context_handler

    def test_register_multiple_handlers(self, tmp_path):
        """Should register handlers for multiple states."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_metrics=False,
        )

        async def context_handler(ctx, event):
            return NomicState.DEBATE, {}

        async def debate_handler(ctx, event):
            return NomicState.DESIGN, {}

        machine.register_handler(NomicState.CONTEXT, context_handler)
        machine.register_handler(NomicState.DEBATE, debate_handler)

        assert len(machine._handlers) == 2
        assert NomicState.CONTEXT in machine._handlers
        assert NomicState.DEBATE in machine._handlers

    def test_overwrite_handler(self, tmp_path):
        """Should overwrite existing handler when re-registering."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_metrics=False,
        )

        async def handler1(ctx, event):
            return NomicState.DEBATE, {"v": 1}

        async def handler2(ctx, event):
            return NomicState.DEBATE, {"v": 2}

        machine.register_handler(NomicState.CONTEXT, handler1)
        machine.register_handler(NomicState.CONTEXT, handler2)

        assert machine._handlers[NomicState.CONTEXT] is handler2


class TestStateMachineCallbackRegistration:
    """Tests for callback registration."""

    def test_on_transition_callback(self, tmp_path):
        """Should register transition callback."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_metrics=False,
        )

        def callback(from_state, to_state, event):
            pass

        machine.on_transition(callback)

        assert callback in machine._on_transition_callbacks

    def test_on_error_callback(self, tmp_path):
        """Should register error callback."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_metrics=False,
        )

        def error_callback(state, error):
            pass

        machine.on_error(error_callback)

        assert error_callback in machine._on_error_callbacks

    def test_multiple_callbacks(self, tmp_path):
        """Should register multiple callbacks."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_metrics=False,
        )

        def cb1(f, t, e):
            pass

        def cb2(f, t, e):
            pass

        def cb3(f, t, e):
            pass

        machine.on_transition(cb1)
        machine.on_transition(cb2)
        machine.on_transition(cb3)

        assert len(machine._on_transition_callbacks) == 3


class TestStateMachineStart:
    """Tests for starting the state machine."""

    @pytest.mark.asyncio
    async def test_start_initializes_cycle(self, tmp_path):
        """Should initialize a new cycle on start."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_checkpoints=False,
            enable_metrics=False,
        )

        # Register handlers for all states to keep machine running after CONTEXT
        async def context_handler(ctx, event):
            # Return result but don't auto-advance
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
            # Return COMPLETED to finish the cycle
            return NomicState.COMPLETED, {}

        machine.register_handler(NomicState.CONTEXT, context_handler)
        machine.register_handler(NomicState.DEBATE, debate_handler)
        machine.register_handler(NomicState.DESIGN, design_handler)
        machine.register_handler(NomicState.IMPLEMENT, implement_handler)
        machine.register_handler(NomicState.VERIFY, verify_handler)
        machine.register_handler(NomicState.COMMIT, commit_handler)

        # Patch _save_checkpoint to avoid file I/O
        with patch.object(machine, "_save_checkpoint", new_callable=AsyncMock):
            await machine.start()

        # After completing the full cycle, running should be False
        # but cycle_id should be set
        assert machine.context.cycle_id != ""
        assert machine.context.started_at is not None
        assert machine._total_cycles == 1

    @pytest.mark.asyncio
    async def test_start_sets_running_flag(self, tmp_path):
        """Should set running flag to True initially during processing."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_checkpoints=False,
            enable_metrics=False,
        )

        running_during_handler = []

        async def handler(ctx, event):
            # Capture running state during handler execution
            running_during_handler.append(machine.running)
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

        machine.register_handler(NomicState.CONTEXT, handler)
        machine.register_handler(NomicState.DEBATE, debate_handler)
        machine.register_handler(NomicState.DESIGN, design_handler)
        machine.register_handler(NomicState.IMPLEMENT, implement_handler)
        machine.register_handler(NomicState.VERIFY, verify_handler)
        machine.register_handler(NomicState.COMMIT, commit_handler)

        with patch.object(machine, "_save_checkpoint", new_callable=AsyncMock):
            await machine.start()

        # Running should have been True during handler execution
        assert running_during_handler[0] is True

    @pytest.mark.asyncio
    async def test_start_processes_start_event(self, tmp_path):
        """Should process START event and transition to CONTEXT."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_checkpoints=False,
            enable_metrics=False,
        )

        async def handler(ctx, event):
            return NomicState.DEBATE, {}

        machine.register_handler(NomicState.CONTEXT, handler)

        with patch.object(machine, "_save_checkpoint", new_callable=AsyncMock):
            await machine.start()

        # After start, should be in CONTEXT or past it
        assert machine.current_state != NomicState.IDLE

    @pytest.mark.asyncio
    async def test_start_with_config(self, tmp_path):
        """Should pass config to start event."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_checkpoints=False,
            enable_metrics=False,
        )

        config = {"goal": "improve tests", "max_changes": 5}
        events_received = []

        async def handler(ctx, event):
            events_received.append(event)
            return NomicState.DEBATE, {}

        machine.register_handler(NomicState.CONTEXT, handler)

        with patch.object(machine, "_save_checkpoint", new_callable=AsyncMock):
            await machine.start(config=config)

        # The handler should have received an event
        assert len(events_received) > 0

    @pytest.mark.asyncio
    async def test_start_when_already_running(self, tmp_path):
        """Should not start if already running."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_checkpoints=False,
            enable_metrics=False,
        )
        machine.running = True

        initial_cycles = machine._total_cycles

        await machine.start()

        # Should not have incremented cycle count
        assert machine._total_cycles == initial_cycles


class TestStateMachineStop:
    """Tests for stopping the state machine."""

    @pytest.mark.asyncio
    async def test_stop_sets_running_false(self, tmp_path):
        """Should set running to False on stop."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_checkpoints=False,
            enable_metrics=False,
        )
        machine.running = True
        # Use RECOVERY state which can transition to FAILED
        machine.context.current_state = NomicState.RECOVERY

        with patch.object(machine, "_save_checkpoint", new_callable=AsyncMock):
            await machine.stop()

        assert machine.running is False

    @pytest.mark.asyncio
    async def test_stop_transitions_to_failed(self, tmp_path):
        """Should transition to FAILED when stopping mid-cycle from RECOVERY."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_checkpoints=False,
            enable_metrics=False,
        )
        machine.running = True
        # RECOVERY can transition to FAILED
        machine.context.current_state = NomicState.RECOVERY

        with patch.object(machine, "_save_checkpoint", new_callable=AsyncMock):
            await machine.stop(reason="user_cancelled")

        assert machine.current_state == NomicState.FAILED
        assert machine._failed_cycles == 1

    @pytest.mark.asyncio
    async def test_stop_from_paused_does_nothing_special(self, tmp_path):
        """Stop from IDLE/COMPLETED should not transition to FAILED."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_checkpoints=False,
            enable_metrics=False,
        )
        machine.running = True
        machine.context.current_state = NomicState.IDLE

        await machine.stop()

        # Should still be running=False but state unchanged
        assert machine.running is False

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self, tmp_path):
        """Should do nothing when not running."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_checkpoints=False,
            enable_metrics=False,
        )
        machine.running = False

        await machine.stop()

        # Should still be not running
        assert machine.running is False


class TestStateMachinePauseResume:
    """Tests for pause and resume functionality."""

    @pytest.mark.asyncio
    async def test_pause_transitions_to_paused(self, tmp_path):
        """Should transition to PAUSED state."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_checkpoints=False,
            enable_metrics=False,
        )
        machine.running = True
        machine.context.current_state = NomicState.DEBATE

        with patch.object(machine, "_save_checkpoint", new_callable=AsyncMock):
            await machine.pause()

        assert machine.current_state == NomicState.PAUSED

    @pytest.mark.asyncio
    async def test_resume_from_paused(self, tmp_path):
        """Should resume from PAUSED to previous state."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_checkpoints=False,
            enable_metrics=False,
        )
        machine.running = True
        machine.context.current_state = NomicState.PAUSED
        machine.context.previous_state = NomicState.DEBATE

        resumed_to_debate = []

        async def debate_handler(ctx, event):
            resumed_to_debate.append(True)
            return NomicState.DESIGN, {}

        async def design_handler(ctx, event):
            return NomicState.IMPLEMENT, {}

        async def implement_handler(ctx, event):
            return NomicState.VERIFY, {}

        async def verify_handler(ctx, event):
            return NomicState.COMMIT, {}

        async def commit_handler(ctx, event):
            return NomicState.COMPLETED, {}

        machine.register_handler(NomicState.DEBATE, debate_handler)
        machine.register_handler(NomicState.DESIGN, design_handler)
        machine.register_handler(NomicState.IMPLEMENT, implement_handler)
        machine.register_handler(NomicState.VERIFY, verify_handler)
        machine.register_handler(NomicState.COMMIT, commit_handler)

        with patch.object(machine, "_save_checkpoint", new_callable=AsyncMock):
            await machine.resume()

        # Should have resumed to DEBATE (handler was called)
        assert len(resumed_to_debate) > 0

    @pytest.mark.asyncio
    async def test_resume_when_not_paused(self, tmp_path):
        """Should not resume when not in PAUSED state."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_checkpoints=False,
            enable_metrics=False,
        )
        machine.context.current_state = NomicState.DEBATE

        await machine.resume()

        # Should still be in DEBATE
        assert machine.current_state == NomicState.DEBATE


class TestStateTransitionsValid:
    """Tests for valid state transitions."""

    @pytest.mark.asyncio
    async def test_idle_to_context(self, tmp_path):
        """Should transition from IDLE to CONTEXT on START."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_checkpoints=False,
            enable_metrics=False,
        )

        transitioned_to_context = []

        async def handler(ctx, event):
            transitioned_to_context.append(ctx.current_state)
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

        machine.register_handler(NomicState.CONTEXT, handler)
        machine.register_handler(NomicState.DEBATE, debate_handler)
        machine.register_handler(NomicState.DESIGN, design_handler)
        machine.register_handler(NomicState.IMPLEMENT, implement_handler)
        machine.register_handler(NomicState.VERIFY, verify_handler)
        machine.register_handler(NomicState.COMMIT, commit_handler)

        with patch.object(machine, "_save_checkpoint", new_callable=AsyncMock):
            await machine.start()

        # Should have transitioned through CONTEXT
        assert NomicState.CONTEXT in transitioned_to_context

    @pytest.mark.asyncio
    async def test_context_to_debate(self, tmp_path):
        """Should transition from CONTEXT to DEBATE."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_checkpoints=False,
            enable_metrics=False,
        )
        machine.context.current_state = NomicState.CONTEXT

        event = phase_complete_event("context", {"result": "ok"})
        next_state = machine._determine_next_state(event)

        assert next_state == NomicState.DEBATE

    @pytest.mark.asyncio
    async def test_debate_to_design(self, tmp_path):
        """Should transition from DEBATE to DESIGN."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_checkpoints=False,
            enable_metrics=False,
        )
        machine.context.current_state = NomicState.DEBATE

        event = phase_complete_event("debate", {"consensus": True})
        next_state = machine._determine_next_state(event)

        assert next_state == NomicState.DESIGN

    @pytest.mark.asyncio
    async def test_design_to_implement(self, tmp_path):
        """Should transition from DESIGN to IMPLEMENT."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_checkpoints=False,
            enable_metrics=False,
        )
        machine.context.current_state = NomicState.DESIGN

        event = phase_complete_event("design", {"approved": True})
        next_state = machine._determine_next_state(event)

        assert next_state == NomicState.IMPLEMENT

    @pytest.mark.asyncio
    async def test_implement_to_verify(self, tmp_path):
        """Should transition from IMPLEMENT to VERIFY."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_checkpoints=False,
            enable_metrics=False,
        )
        machine.context.current_state = NomicState.IMPLEMENT

        event = phase_complete_event("implement", {"success": True})
        next_state = machine._determine_next_state(event)

        assert next_state == NomicState.VERIFY

    @pytest.mark.asyncio
    async def test_verify_to_commit(self, tmp_path):
        """Should transition from VERIFY to COMMIT."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_checkpoints=False,
            enable_metrics=False,
        )
        machine.context.current_state = NomicState.VERIFY

        event = phase_complete_event("verify", {"passed": True})
        next_state = machine._determine_next_state(event)

        assert next_state == NomicState.COMMIT

    @pytest.mark.asyncio
    async def test_commit_to_completed(self, tmp_path):
        """Should transition from COMMIT to COMPLETED."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_checkpoints=False,
            enable_metrics=False,
        )
        machine.context.current_state = NomicState.COMMIT

        event = phase_complete_event("commit", {"committed": True})
        next_state = machine._determine_next_state(event)

        assert next_state == NomicState.COMPLETED


class TestStateTransitionsInvalid:
    """Tests for invalid state transitions."""

    @pytest.mark.asyncio
    async def test_invalid_transition_raises_error(self, tmp_path):
        """Should raise TransitionError for invalid transition."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_checkpoints=False,
            enable_metrics=False,
        )
        machine.context.current_state = NomicState.IDLE
        machine.running = True

        # Try to transition directly to COMMIT (invalid from IDLE)
        with pytest.raises(TransitionError) as exc_info:
            event = Event(event_type=EventType.START)
            await machine._transition_to(NomicState.COMMIT, event)

        assert "Invalid transition" in str(exc_info.value)

    def test_is_valid_transition_helper(self):
        """Should correctly validate transitions."""
        # Valid transitions
        assert is_valid_transition(NomicState.IDLE, NomicState.CONTEXT) is True
        assert is_valid_transition(NomicState.CONTEXT, NomicState.DEBATE) is True
        assert is_valid_transition(NomicState.DEBATE, NomicState.DESIGN) is True

        # Invalid transitions
        assert is_valid_transition(NomicState.IDLE, NomicState.COMMIT) is False
        assert is_valid_transition(NomicState.COMPLETED, NomicState.DEBATE) is False
        assert is_valid_transition(NomicState.FAILED, NomicState.IMPLEMENT) is False

    def test_all_states_have_valid_transitions(self):
        """Every state should have defined transitions."""
        for state in NomicState:
            assert state in VALID_TRANSITIONS
            # Each state should have at least one valid transition (except terminal)
            if state not in (NomicState.COMPLETED, NomicState.FAILED):
                assert len(VALID_TRANSITIONS[state]) > 0


class TestEventHandling:
    """Tests for event handling."""

    def test_determine_next_state_start_event(self, tmp_path):
        """START event should lead to CONTEXT."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_metrics=False,
        )
        machine.context.current_state = NomicState.IDLE

        event = start_event()
        next_state = machine._determine_next_state(event)

        assert next_state == NomicState.CONTEXT

    def test_determine_next_state_stop_event(self, tmp_path):
        """STOP event should lead to FAILED when active."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_metrics=False,
        )
        machine.context.current_state = NomicState.DEBATE

        event = stop_event()
        next_state = machine._determine_next_state(event)

        assert next_state == NomicState.FAILED

    def test_determine_next_state_pause_event(self, tmp_path):
        """PAUSE event should lead to PAUSED."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_metrics=False,
        )
        machine.context.current_state = NomicState.IMPLEMENT

        event = pause_event()
        next_state = machine._determine_next_state(event)

        assert next_state == NomicState.PAUSED

    def test_determine_next_state_error_recoverable(self, tmp_path):
        """Recoverable ERROR should lead to RECOVERY."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_metrics=False,
        )
        machine.context.current_state = NomicState.IMPLEMENT

        event = error_event("implement", Exception("test"), recoverable=True)
        next_state = machine._determine_next_state(event)

        assert next_state == NomicState.RECOVERY

    def test_determine_next_state_error_not_recoverable(self, tmp_path):
        """Non-recoverable ERROR should lead to FAILED."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_metrics=False,
        )
        machine.context.current_state = NomicState.IMPLEMENT

        event = error_event("implement", Exception("critical"), recoverable=False)
        next_state = machine._determine_next_state(event)

        assert next_state == NomicState.FAILED

    def test_determine_next_state_timeout(self, tmp_path):
        """TIMEOUT event should lead to RECOVERY."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_metrics=False,
        )
        machine.context.current_state = NomicState.DEBATE

        event = timeout_event("debate", 3600)
        next_state = machine._determine_next_state(event)

        assert next_state == NomicState.RECOVERY

    def test_event_log_records_events(self, tmp_path):
        """Events should be logged to event log."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_metrics=False,
        )

        event = start_event()
        machine.event_log.append(event)

        assert len(machine.event_log.events) == 1
        assert machine.event_log.events[0].event_type == EventType.START


class TestGuardConditionsAndCallbacks:
    """Tests for guard conditions and callbacks."""

    @pytest.mark.asyncio
    async def test_transition_callback_called(self, tmp_path):
        """Transition callback should be called on state change."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_checkpoints=False,
            enable_metrics=False,
        )

        callback_calls = []

        def callback(from_state, to_state, event):
            callback_calls.append((from_state, to_state))

        machine.on_transition(callback)
        machine.context.current_state = NomicState.IDLE

        event = start_event()
        with patch.object(machine, "_run_state_handler", new_callable=AsyncMock):
            await machine._transition_to(NomicState.CONTEXT, event)

        assert len(callback_calls) == 1
        assert callback_calls[0] == (NomicState.IDLE, NomicState.CONTEXT)

    @pytest.mark.asyncio
    async def test_async_transition_callback(self, tmp_path):
        """Async transition callback should work."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_checkpoints=False,
            enable_metrics=False,
        )

        callback_calls = []

        async def async_callback(from_state, to_state, event):
            callback_calls.append((from_state, to_state))

        machine.on_transition(async_callback)
        machine.context.current_state = NomicState.IDLE

        event = start_event()
        with patch.object(machine, "_run_state_handler", new_callable=AsyncMock):
            await machine._transition_to(NomicState.CONTEXT, event)

        assert len(callback_calls) == 1

    @pytest.mark.asyncio
    async def test_callback_with_metrics_args(self, tmp_path):
        """Callback supporting metrics args should receive duration and cycle_id."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_checkpoints=False,
            enable_metrics=False,
        )

        callback_args = []

        def callback(from_state, to_state, event, duration, cycle_id):
            callback_args.append(
                {
                    "from": from_state,
                    "to": to_state,
                    "duration": duration,
                    "cycle_id": cycle_id,
                }
            )

        machine.on_transition(callback)
        machine.context.current_state = NomicState.IDLE
        machine.context.cycle_id = "test-123"
        machine._state_entry_time = time.time() - 1.0  # 1 second ago

        event = start_event()
        with patch.object(machine, "_run_state_handler", new_callable=AsyncMock):
            await machine._transition_to(NomicState.CONTEXT, event)

        assert len(callback_args) == 1
        assert callback_args[0]["cycle_id"] == "test-123"
        assert callback_args[0]["duration"] >= 1.0

    @pytest.mark.asyncio
    async def test_error_callback_called_on_handler_error(self, tmp_path):
        """Error callback should be called when handler fails."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_checkpoints=False,
            enable_metrics=False,
        )

        error_calls = []

        def error_callback(state, error):
            error_calls.append((state, str(error)))

        machine.on_error(error_callback)

        async def failing_handler(ctx, event):
            raise ValueError("Handler failed")

        machine.register_handler(NomicState.CONTEXT, failing_handler)
        machine.context.current_state = NomicState.CONTEXT
        machine.running = True

        event = start_event()
        with patch.object(machine, "_save_checkpoint", new_callable=AsyncMock):
            await machine._run_state_handler(NomicState.CONTEXT, event)

        # Error callback should have been called after max retries
        assert len(error_calls) > 0

    @pytest.mark.asyncio
    async def test_callback_exception_does_not_stop_transition(self, tmp_path):
        """Callback exception should not prevent transition."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_checkpoints=False,
            enable_metrics=False,
        )

        def failing_callback(from_state, to_state, event):
            raise Exception("Callback failed")

        machine.on_transition(failing_callback)
        machine.context.current_state = NomicState.IDLE

        event = start_event()
        with patch.object(machine, "_run_state_handler", new_callable=AsyncMock):
            # Should not raise
            await machine._transition_to(NomicState.CONTEXT, event)

        assert machine.current_state == NomicState.CONTEXT


class TestErrorStatesAndRecovery:
    """Tests for error handling and recovery."""

    @pytest.mark.asyncio
    async def test_handler_timeout_triggers_recovery(self, tmp_path):
        """Handler timeout should trigger RECOVERY state."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_checkpoints=False,
            enable_metrics=False,
        )

        async def slow_handler(ctx, event):
            await asyncio.sleep(10)  # Longer than timeout
            return NomicState.DEBATE, {}

        machine.register_handler(NomicState.CONTEXT, slow_handler)
        machine.context.current_state = NomicState.CONTEXT
        machine.running = True

        # Patch get_state_config to return short timeout
        with patch("aragora.nomic.state_machine.get_state_config") as mock_config:
            mock_metadata = MagicMock()
            mock_metadata.timeout_seconds = 0.1  # Very short timeout
            mock_metadata.max_retries = 0
            mock_metadata.is_critical = False
            mock_metadata.requires_checkpoint = False
            mock_config.return_value = mock_metadata

            with patch.object(machine, "_save_checkpoint", new_callable=AsyncMock):
                event = start_event()
                await machine._run_state_handler(NomicState.CONTEXT, event)

        # Should have transitioned to RECOVERY due to timeout
        assert machine.current_state == NomicState.RECOVERY

    @pytest.mark.asyncio
    async def test_handler_error_with_retries(self, tmp_path):
        """Handler errors should be retried up to max_retries."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_checkpoints=False,
            enable_metrics=False,
        )

        call_count = [0]

        async def flaky_handler(ctx, event):
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("Temporary failure")
            return NomicState.DEBATE, {}

        machine.register_handler(NomicState.CONTEXT, flaky_handler)
        machine.context.current_state = NomicState.CONTEXT
        machine.running = True

        with patch("aragora.nomic.state_machine.get_state_config") as mock_config:
            mock_metadata = MagicMock()
            mock_metadata.timeout_seconds = 0
            mock_metadata.max_retries = 3
            mock_metadata.is_critical = False
            mock_metadata.requires_checkpoint = False
            mock_config.return_value = mock_metadata

            # Patch sleep to speed up test
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with patch.object(machine, "_save_checkpoint", new_callable=AsyncMock):
                    event = start_event()
                    await machine._run_state_handler(NomicState.CONTEXT, event)

        # Handler should have been called 3 times (2 failures + 1 success)
        assert call_count[0] == 3

    @pytest.mark.asyncio
    async def test_max_retries_exceeded_transitions_to_recovery(self, tmp_path):
        """Exceeding max retries on non-critical phase should transition to RECOVERY."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_checkpoints=False,
            enable_metrics=False,
        )

        async def always_failing_handler(ctx, event):
            raise ValueError("Always fails")

        # Use CONTEXT which can transition to RECOVERY
        machine.register_handler(NomicState.CONTEXT, always_failing_handler)
        machine.context.current_state = NomicState.CONTEXT
        machine.running = True

        with patch("aragora.nomic.state_machine.get_state_config") as mock_config:
            mock_metadata = MagicMock()
            mock_metadata.timeout_seconds = 0
            mock_metadata.max_retries = 1
            mock_metadata.is_critical = False  # Non-critical phase
            mock_metadata.requires_checkpoint = False
            mock_config.return_value = mock_metadata

            with patch("asyncio.sleep", new_callable=AsyncMock):
                with patch.object(machine, "_save_checkpoint", new_callable=AsyncMock):
                    event = start_event()
                    await machine._run_state_handler(NomicState.CONTEXT, event)

        # Should have transitioned to RECOVERY (recoverable for non-critical)
        assert machine.current_state == NomicState.RECOVERY

    @pytest.mark.asyncio
    async def test_recovery_handler_can_redirect(self, tmp_path):
        """Recovery handler should be able to redirect to any valid state."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_checkpoints=False,
            enable_metrics=False,
        )

        redirected_to_context = []

        async def recovery_handler(ctx, event):
            # Decide to go back to CONTEXT (valid from RECOVERY)
            return NomicState.CONTEXT, {"recovery_action": "retry_context"}

        async def context_handler(ctx, event):
            redirected_to_context.append(True)
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

        machine.register_handler(NomicState.RECOVERY, recovery_handler)
        machine.register_handler(NomicState.CONTEXT, context_handler)
        machine.register_handler(NomicState.DEBATE, debate_handler)
        machine.register_handler(NomicState.DESIGN, design_handler)
        machine.register_handler(NomicState.IMPLEMENT, implement_handler)
        machine.register_handler(NomicState.VERIFY, verify_handler)
        machine.register_handler(NomicState.COMMIT, commit_handler)
        machine.context.current_state = NomicState.RECOVERY
        machine.running = True

        with patch.object(machine, "_save_checkpoint", new_callable=AsyncMock):
            event = Event(event_type=EventType.RECOVER)
            await machine._run_state_handler(NomicState.RECOVERY, event)

        # Should have redirected to CONTEXT via recovery handler
        assert len(redirected_to_context) > 0


class TestTerminalStates:
    """Tests for terminal states (COMPLETED, FAILED)."""

    @pytest.mark.asyncio
    async def test_completed_state_stops_machine(self, tmp_path):
        """COMPLETED state should stop the machine."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_checkpoints=False,
            enable_metrics=False,
        )
        machine.running = True
        machine.context.current_state = NomicState.COMMIT

        event = phase_complete_event("commit", {"success": True})
        with patch.object(machine, "_save_checkpoint", new_callable=AsyncMock):
            await machine._transition_to(NomicState.COMPLETED, event)

        assert machine.running is False
        assert machine.current_state == NomicState.COMPLETED
        assert machine._successful_cycles == 1

    @pytest.mark.asyncio
    async def test_failed_state_stops_machine(self, tmp_path):
        """FAILED state should stop the machine."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_checkpoints=False,
            enable_metrics=False,
        )
        machine.running = True
        # RECOVERY can transition to FAILED
        machine.context.current_state = NomicState.RECOVERY

        event = stop_event()
        with patch.object(machine, "_save_checkpoint", new_callable=AsyncMock):
            await machine._transition_to(NomicState.FAILED, event)

        assert machine.running is False
        assert machine.current_state == NomicState.FAILED
        assert machine._failed_cycles == 1


class TestAutoAdvance:
    """Tests for auto-advance when no handler is registered."""

    @pytest.mark.asyncio
    async def test_auto_advance_to_next_state(self, tmp_path):
        """Should auto-advance when no handler is registered."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_checkpoints=False,
            enable_metrics=False,
        )
        machine.context.current_state = NomicState.CONTEXT
        machine.running = True

        # No handler registered for CONTEXT - will auto-advance through all states
        event = start_event()
        with patch.object(machine, "_save_checkpoint", new_callable=AsyncMock):
            await machine._run_state_handler(NomicState.CONTEXT, event)

        # Should have auto-advanced (possibly all the way to COMPLETED)
        assert machine.current_state != NomicState.CONTEXT

    @pytest.mark.asyncio
    async def test_auto_advance_through_multiple_states(self, tmp_path):
        """Should auto-advance through all states if no handlers."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_checkpoints=False,
            enable_metrics=False,
        )

        with patch.object(machine, "_save_checkpoint", new_callable=AsyncMock):
            await machine.start()

        # Should have auto-advanced to COMPLETED
        assert machine.current_state == NomicState.COMPLETED
        assert machine.running is False


class TestCheckpointing:
    """Tests for checkpoint functionality."""

    @pytest.mark.asyncio
    async def test_checkpoint_saved_on_transition(self, tmp_path):
        """Checkpoint should be saved on state transitions that require it."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_checkpoints=True,
            enable_metrics=False,
        )
        machine.context.current_state = NomicState.IDLE
        machine.running = True

        with patch.object(machine, "_save_checkpoint", new_callable=AsyncMock) as mock_save:
            with patch.object(machine, "_run_state_handler", new_callable=AsyncMock):
                event = start_event()
                await machine._transition_to(NomicState.CONTEXT, event)

        # CONTEXT requires checkpoint
        mock_save.assert_called()

    @pytest.mark.asyncio
    async def test_checkpoint_not_saved_when_disabled(self, tmp_path):
        """Checkpoint should not be saved when disabled."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_checkpoints=False,
            enable_metrics=False,
        )
        machine.context.current_state = NomicState.IDLE
        machine.running = True

        with patch.object(machine, "_save_checkpoint", new_callable=AsyncMock) as mock_save:
            with patch.object(machine, "_run_state_handler", new_callable=AsyncMock):
                event = start_event()
                await machine._transition_to(NomicState.CONTEXT, event)

        mock_save.assert_not_called()

    @pytest.mark.asyncio
    async def test_resume_from_checkpoint(self, tmp_path):
        """Should resume from checkpoint."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_checkpoints=True,
            enable_metrics=False,
        )

        checkpoint_data = {
            "context": {
                "cycle_id": "resume-test",
                "current_state": "DEBATE",
                "previous_state": "CONTEXT",
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
            "event_log": {"cycle_id": "resume-test", "events": []},
        }

        resumed_to_debate = []

        async def debate_handler(ctx, event):
            resumed_to_debate.append(True)
            return NomicState.DESIGN, {}

        async def design_handler(ctx, event):
            return NomicState.IMPLEMENT, {}

        async def implement_handler(ctx, event):
            return NomicState.VERIFY, {}

        async def verify_handler(ctx, event):
            return NomicState.COMMIT, {}

        async def commit_handler(ctx, event):
            return NomicState.COMPLETED, {}

        machine.register_handler(NomicState.DEBATE, debate_handler)
        machine.register_handler(NomicState.DESIGN, design_handler)
        machine.register_handler(NomicState.IMPLEMENT, implement_handler)
        machine.register_handler(NomicState.VERIFY, verify_handler)
        machine.register_handler(NomicState.COMMIT, commit_handler)

        with patch(
            "aragora.nomic.checkpoints.load_checkpoint",
            return_value=checkpoint_data,
        ):
            with patch.object(machine, "_save_checkpoint", new_callable=AsyncMock):
                await machine.resume_from_checkpoint("/fake/path")

        assert machine.context.cycle_id == "resume-test"
        # The handler should have been called
        assert len(resumed_to_debate) > 0

    @pytest.mark.asyncio
    async def test_resume_from_invalid_checkpoint_raises(self, tmp_path):
        """Should raise when checkpoint is invalid."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_checkpoints=True,
            enable_metrics=False,
        )

        with patch(
            "aragora.nomic.checkpoints.load_checkpoint",
            return_value=None,
        ):
            with pytest.raises(ValueError) as exc_info:
                await machine.resume_from_checkpoint("/fake/path")

        assert "Could not load checkpoint" in str(exc_info.value)


class TestMetrics:
    """Tests for metrics functionality."""

    def test_get_metrics_returns_expected_fields(self, tmp_path):
        """Metrics should contain expected fields."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_metrics=False,
        )
        machine.context.cycle_id = "metrics-test"
        machine._total_cycles = 10
        machine._successful_cycles = 8
        machine._failed_cycles = 2

        metrics = machine.get_metrics()

        assert metrics["current_state"] == "IDLE"
        assert metrics["cycle_id"] == "metrics-test"
        assert metrics["running"] is False
        assert metrics["total_cycles"] == 10
        assert metrics["successful_cycles"] == 8
        assert metrics["failed_cycles"] == 2
        assert metrics["success_rate"] == 0.8

    def test_get_metrics_with_zero_cycles(self, tmp_path):
        """Success rate should be 0 with no cycles."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_metrics=False,
        )

        metrics = machine.get_metrics()

        assert metrics["success_rate"] == 0


class TestStateContextSerialization:
    """Tests for StateContext serialization."""

    def test_context_to_dict(self):
        """StateContext should serialize to dict."""
        ctx = StateContext(
            cycle_id="ser-test",
            started_at=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            current_state=NomicState.DEBATE,
            previous_state=NomicState.CONTEXT,
        )
        ctx.context_result = {"data": "test"}

        data = ctx.to_dict()

        assert data["cycle_id"] == "ser-test"
        assert data["current_state"] == "DEBATE"
        assert data["previous_state"] == "CONTEXT"
        assert data["context_result"] == {"data": "test"}

    def test_context_from_dict(self):
        """StateContext should deserialize from dict."""
        data = {
            "cycle_id": "deser-test",
            "started_at": "2024-01-01T12:00:00+00:00",
            "current_state": "IMPLEMENT",
            "previous_state": "DESIGN",
            "context_result": {"gathered": True},
            "errors": [{"msg": "error1"}],
        }

        ctx = StateContext.from_dict(data)

        assert ctx.cycle_id == "deser-test"
        assert ctx.current_state == NomicState.IMPLEMENT
        assert ctx.previous_state == NomicState.DESIGN
        assert ctx.context_result == {"gathered": True}
        assert len(ctx.errors) == 1


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_handler_returns_none_result(self, tmp_path):
        """Handler returning None result should not crash."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_checkpoints=False,
            enable_metrics=False,
        )

        async def handler_with_none(ctx, event):
            return NomicState.DEBATE, None

        machine.register_handler(NomicState.CONTEXT, handler_with_none)
        machine.context.current_state = NomicState.CONTEXT
        machine.running = True

        with patch.object(machine, "_save_checkpoint", new_callable=AsyncMock):
            event = start_event()
            # Should not raise
            await machine._run_state_handler(NomicState.CONTEXT, event)

    @pytest.mark.asyncio
    async def test_rapid_start_stop_cycles(self, tmp_path):
        """Should handle rapid start/stop cycles."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_checkpoints=False,
            enable_metrics=False,
        )

        # When no handlers registered, auto-advance runs through to COMPLETED
        # so each start() completes the cycle immediately
        for _ in range(5):
            await machine.start()

        assert machine._total_cycles == 5
        # All cycles complete via auto-advance
        assert machine._successful_cycles == 5

    @pytest.mark.asyncio
    async def test_empty_event_data(self, tmp_path):
        """Should handle events with empty data."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_metrics=False,
        )

        event = Event(event_type=EventType.START, data={})
        machine.event_log.append(event)

        assert len(machine.event_log.events) == 1

    def test_state_config_for_all_states(self):
        """All states should have valid configuration."""
        for state in NomicState:
            config = get_state_config(state)
            assert config is not None
            assert hasattr(config, "timeout_seconds")
            assert hasattr(config, "max_retries")
            assert hasattr(config, "is_critical")
            assert hasattr(config, "requires_checkpoint")

    @pytest.mark.asyncio
    async def test_state_durations_tracked(self, tmp_path):
        """State durations should be tracked."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_checkpoints=False,
            enable_metrics=False,
        )
        machine.context.current_state = NomicState.IDLE
        machine._state_entry_time = time.time() - 1.0  # 1 second ago

        with patch.object(machine, "_run_state_handler", new_callable=AsyncMock):
            event = start_event()
            await machine._transition_to(NomicState.CONTEXT, event)

        assert NomicState.IDLE.name in machine.context.state_durations
        assert machine.context.state_durations[NomicState.IDLE.name] >= 1.0


class TestCreateNomicStateMachine:
    """Tests for the factory function."""

    def test_create_with_defaults(self, tmp_path):
        """Factory function should create machine with defaults."""
        machine = create_nomic_state_machine(
            checkpoint_dir=str(tmp_path),
            enable_metrics=False,
        )

        assert isinstance(machine, NomicStateMachine)
        assert machine.enable_checkpoints is True

    def test_create_with_custom_config(self, tmp_path):
        """Factory function should accept custom configuration."""
        machine = create_nomic_state_machine(
            checkpoint_dir=str(tmp_path),
            enable_checkpoints=False,
            enable_metrics=False,
        )

        assert machine.checkpoint_dir == str(tmp_path)
        assert machine.enable_checkpoints is False


class TestCycleIdProperty:
    """Tests for cycle_id property."""

    def test_cycle_id_property(self, tmp_path):
        """cycle_id property should return context cycle_id."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_metrics=False,
        )
        machine.context.cycle_id = "prop-test"

        assert machine.cycle_id == "prop-test"

    def test_current_state_property(self, tmp_path):
        """current_state property should return context current_state."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_metrics=False,
        )
        machine.context.current_state = NomicState.VERIFY

        assert machine.current_state == NomicState.VERIFY


class TestEventLogIntegration:
    """Tests for EventLog integration."""

    @pytest.mark.asyncio
    async def test_events_logged_during_cycle(self, tmp_path):
        """Events should be logged during cycle execution."""
        machine = NomicStateMachine(
            checkpoint_dir=str(tmp_path),
            enable_checkpoints=False,
            enable_metrics=False,
        )

        with patch.object(machine, "_save_checkpoint", new_callable=AsyncMock):
            await machine.start()

        # Should have logged events
        assert len(machine.event_log.events) > 0

        # Should have START event
        start_events = machine.event_log.get_events_by_type(EventType.START)
        assert len(start_events) >= 1

    def test_event_log_summary(self):
        """EventLog should provide summary."""
        log = EventLog(cycle_id="summary-test")
        log.append(start_event())
        log.append(phase_complete_event("context", {}))
        log.append(error_event("debate", Exception("test")))

        summary = log.summary()

        assert summary["total_events"] == 3
        assert summary["errors"] == 1
        assert EventType.START.name in summary["event_types"]
