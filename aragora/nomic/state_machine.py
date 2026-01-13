"""
Nomic Loop State Machine.

Event-driven state machine for reliable autonomous self-improvement.
Replaces the fragile phase-based architecture with a robust, recoverable system.

Key Features:
- Idempotent state handlers (safe to retry)
- Checkpoint after every transition
- Circuit breakers per agent
- Exponential backoff on failures
- Full event sourcing for audit trail
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, Optional

from .events import (
    Event,
    EventLog,
    EventType,
    checkpoint_loaded_event,
    error_event,
    pause_event,
    phase_complete_event,
    retry_event,
    start_event,
    stop_event,
    timeout_event,
)
from .states import (
    NomicState,
    StateContext,
    get_state_config,
    is_valid_transition,
)

logger = logging.getLogger(__name__)


class TransitionError(Exception):
    """Raised when an invalid state transition is attempted."""

    pass


class StateTimeoutError(Exception):
    """Raised when a state handler times out."""

    pass


# Type alias for state handlers
StateHandler = Callable[[StateContext, Event], Awaitable[tuple[NomicState, Dict[str, Any]]]]


class NomicStateMachine:
    """
    Event-driven state machine for the nomic loop.

    The state machine processes events and transitions between states.
    Each state has a handler that performs the actual work.

    Usage:
        machine = NomicStateMachine()
        machine.register_handler(NomicState.CONTEXT, context_handler)
        machine.register_handler(NomicState.DEBATE, debate_handler)
        # ... register other handlers

        await machine.start()  # Begin processing
    """

    def __init__(
        self,
        checkpoint_dir: str = ".nomic/checkpoints",
        enable_checkpoints: bool = True,
    ):
        """
        Initialize the state machine.

        Args:
            checkpoint_dir: Directory to store checkpoints
            enable_checkpoints: Whether to checkpoint after transitions
        """
        self.checkpoint_dir = checkpoint_dir
        self.enable_checkpoints = enable_checkpoints

        # State
        self.context = StateContext()
        self.event_log = EventLog()
        self.running = False

        # Handlers
        self._handlers: Dict[NomicState, StateHandler] = {}
        self._on_transition_callbacks: list[Callable] = []
        self._on_error_callbacks: list[Callable] = []

        # Metrics
        self._state_entry_time: Optional[float] = None
        self._total_cycles = 0
        self._successful_cycles = 0
        self._failed_cycles = 0

    def register_handler(self, state: NomicState, handler: StateHandler) -> None:
        """
        Register a handler for a state.

        The handler should be an async function that takes (context, event)
        and returns (next_state, result_data).
        """
        self._handlers[state] = handler

    def on_transition(self, callback: Callable) -> None:
        """Register a callback to be called on every transition."""
        self._on_transition_callbacks.append(callback)

    def on_error(self, callback: Callable) -> None:
        """Register a callback to be called on errors."""
        self._on_error_callbacks.append(callback)

    @property
    def current_state(self) -> NomicState:
        """Get the current state."""
        return self.context.current_state

    @property
    def cycle_id(self) -> str:
        """Get the current cycle ID."""
        return self.context.cycle_id

    async def start(self, config: Optional[Dict] = None) -> None:
        """
        Start a new nomic loop cycle.

        Args:
            config: Optional configuration for this cycle
        """
        if self.running:
            logger.warning("State machine already running")
            return

        # Initialize new cycle
        self.context = StateContext(
            cycle_id=str(uuid.uuid4())[:8],
            started_at=datetime.utcnow(),
            current_state=NomicState.IDLE,
        )
        self.event_log = EventLog(cycle_id=self.context.cycle_id)
        self.running = True
        self._total_cycles += 1

        logger.info(f"Starting nomic cycle {self.context.cycle_id}")

        # Send start event
        event = start_event(trigger="start", config=config)
        await self._process_event(event)

    async def stop(self, reason: str = "manual") -> None:
        """Stop the current cycle."""
        if not self.running:
            return

        event = stop_event(reason=reason)
        await self._process_event(event)
        self.running = False

    async def pause(self, reason: str = "manual") -> None:
        """Pause the current cycle."""
        event = pause_event(reason=reason)
        await self._process_event(event)

    async def resume(self) -> None:
        """Resume a paused cycle."""
        if self.context.current_state != NomicState.PAUSED:
            logger.warning("Cannot resume - not in PAUSED state")
            return

        from .events import resume_event

        event = resume_event()
        await self._process_event(event)

    async def resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """
        Resume from a saved checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file
        """
        from .checkpoints import load_checkpoint

        checkpoint_data = load_checkpoint(checkpoint_path)
        if not checkpoint_data:
            raise ValueError(f"Could not load checkpoint from {checkpoint_path}")

        self.context = StateContext.from_dict(checkpoint_data["context"])
        self.event_log = EventLog.from_dict(checkpoint_data.get("event_log", {}))
        self.running = True

        logger.info(f"Resumed from checkpoint, state: {self.context.current_state.name}")

        # Log the checkpoint load
        event = checkpoint_loaded_event(checkpoint_path, self.context.current_state.name)
        self.event_log.append(event)

        # Continue processing from current state
        await self._continue_from_state()

    async def _process_event(self, event: Event) -> None:
        """
        Process an event and potentially transition to a new state.

        This is the core event loop of the state machine.
        """
        self.event_log.append(event)
        logger.debug(
            f"Processing event: {event.event_type.name} in state {self.current_state.name}"
        )

        # Determine next state based on event
        next_state = self._determine_next_state(event)

        if next_state and next_state != self.current_state:
            await self._transition_to(next_state, event)

    def _determine_next_state(self, event: Event) -> Optional[NomicState]:
        """Determine the next state based on current state and event."""
        current = self.current_state

        # Event-based transitions
        if event.event_type == EventType.START:
            return NomicState.CONTEXT

        elif event.event_type == EventType.STOP:
            return (
                NomicState.FAILED
                if current not in (NomicState.IDLE, NomicState.COMPLETED)
                else None
            )

        elif event.event_type == EventType.PAUSE:
            return NomicState.PAUSED

        elif event.event_type == EventType.RESUME:
            # Resume to the state we were in before pause
            return self.context.previous_state or NomicState.CONTEXT

        elif event.event_type == EventType.ERROR:
            if event.recoverable:
                return NomicState.RECOVERY
            else:
                return NomicState.FAILED

        elif event.event_type == EventType.TIMEOUT:
            return NomicState.RECOVERY

        # Phase completion transitions
        elif event.event_type == EventType.CONTEXT_COMPLETE:
            return NomicState.DEBATE

        elif event.event_type == EventType.DEBATE_COMPLETE:
            return NomicState.DESIGN

        elif event.event_type == EventType.DESIGN_COMPLETE:
            return NomicState.IMPLEMENT

        elif event.event_type == EventType.IMPLEMENT_COMPLETE:
            return NomicState.VERIFY

        elif event.event_type == EventType.VERIFY_COMPLETE:
            return NomicState.COMMIT

        elif event.event_type == EventType.COMMIT_COMPLETE:
            return NomicState.COMPLETED

        elif event.event_type == EventType.RECOVER:
            # Recovery handler determines where to go
            return None

        elif event.event_type == EventType.ROLLBACK:
            return NomicState.RECOVERY

        elif event.event_type == EventType.ABORT:
            return NomicState.FAILED

        return None

    async def _transition_to(self, next_state: NomicState, trigger_event: Event) -> None:
        """
        Transition to a new state.

        This validates the transition, updates state, checkpoints,
        and runs the state handler.
        """
        current = self.current_state

        # Validate transition
        if not is_valid_transition(current, next_state):
            raise TransitionError(f"Invalid transition from {current.name} to {next_state.name}")

        # Record state duration
        if self._state_entry_time:
            duration = time.time() - self._state_entry_time
            self.context.state_durations[current.name] = duration

        # Update context
        self.context.previous_state = current
        self.context.current_state = next_state
        self._state_entry_time = time.time()

        logger.info(f"Transition: {current.name} -> {next_state.name}")

        # Call transition callbacks
        for callback in self._on_transition_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(current, next_state, trigger_event)
                else:
                    callback(current, next_state, trigger_event)
            except Exception as e:
                logger.error(f"Transition callback error: {e}")

        # Checkpoint if required
        state_config = get_state_config(next_state)
        if self.enable_checkpoints and state_config.requires_checkpoint:
            await self._save_checkpoint()

        # Handle terminal states
        if next_state == NomicState.COMPLETED:
            self._successful_cycles += 1
            self.running = False
            logger.info(f"Cycle {self.cycle_id} completed successfully")
            return

        elif next_state == NomicState.FAILED:
            self._failed_cycles += 1
            self.running = False
            logger.error(f"Cycle {self.cycle_id} failed")
            return

        elif next_state == NomicState.PAUSED:
            logger.info(f"Cycle {self.cycle_id} paused")
            return

        # Run state handler
        await self._run_state_handler(next_state, trigger_event)

    async def _run_state_handler(self, state: NomicState, trigger_event: Event) -> None:
        """
        Run the handler for a state with timeout and error handling.
        """
        handler = self._handlers.get(state)
        if not handler:
            logger.warning(f"No handler registered for state {state.name}")
            # Auto-advance to next logical state for unhandled states
            await self._auto_advance(state)
            return

        state_config = get_state_config(state)
        timeout = state_config.timeout_seconds

        try:
            # Run handler with timeout
            if timeout > 0:
                next_state, result = await asyncio.wait_for(
                    handler(self.context, trigger_event), timeout=timeout
                )
            else:
                next_state, result = await handler(self.context, trigger_event)

            # Store result in context
            result_attr = f"{state.name.lower()}_result"
            if hasattr(self.context, result_attr):
                setattr(self.context, result_attr, result)

            # For RECOVERY state, use the handler's returned next_state directly
            # This allows recovery handlers to dictate where to go next
            if state == NomicState.RECOVERY and next_state:
                logger.info(f"Recovery handler returned transition to {next_state.name}")
                await self._transition_to(next_state, trigger_event)
                return

            # Generate completion event for normal phases
            duration = time.time() - (self._state_entry_time or time.time())
            event = phase_complete_event(
                phase=state.name.lower(),
                result=result,
                duration_seconds=duration,
                tokens_used=result.get("tokens_used", 0) if result else 0,
            )
            await self._process_event(event)

        except asyncio.TimeoutError:
            logger.error(f"State {state.name} timed out after {timeout}s")
            event = timeout_event(state.name.lower(), timeout)
            await self._process_event(event)

        except Exception as e:
            logger.error(f"Error in state {state.name}: {e}")
            # Check if we can retry
            retry_count = self.context.retry_counts.get(state.name, 0)
            if retry_count < state_config.max_retries:
                self.context.retry_counts[state.name] = retry_count + 1
                event = retry_event(state.name.lower(), retry_count + 1, state_config.max_retries)
                self.event_log.append(event)
                # Exponential backoff
                await asyncio.sleep(min(2**retry_count, 60))
                await self._run_state_handler(state, trigger_event)
            else:
                # Max retries exceeded
                self.context.errors.append(
                    {
                        "state": state.name,
                        "error": str(e),
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )
                for callback in self._on_error_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(state, e)
                        else:
                            callback(state, e)
                    except Exception as cb_err:
                        logger.error(f"Error callback failed: {cb_err}")

                event = error_event(
                    phase=state.name.lower(),
                    error=e,
                    recoverable=not state_config.is_critical,
                )
                await self._process_event(event)

    async def _auto_advance(self, state: NomicState) -> None:
        """Auto-advance to next state when no handler is registered."""
        # Map states to their completion events
        auto_events = {
            NomicState.CONTEXT: EventType.CONTEXT_COMPLETE,
            NomicState.DEBATE: EventType.DEBATE_COMPLETE,
            NomicState.DESIGN: EventType.DESIGN_COMPLETE,
            NomicState.IMPLEMENT: EventType.IMPLEMENT_COMPLETE,
            NomicState.VERIFY: EventType.VERIFY_COMPLETE,
            NomicState.COMMIT: EventType.COMMIT_COMPLETE,
        }
        if state in auto_events:
            event = Event(event_type=auto_events[state], source="auto_advance")
            await self._process_event(event)

    async def _continue_from_state(self) -> None:
        """Continue processing from the current state after resume."""
        state = self.current_state

        if state in (NomicState.IDLE, NomicState.COMPLETED, NomicState.FAILED):
            return

        # Create a synthetic trigger event
        event = Event(
            event_type=EventType.RESUME,
            source="checkpoint_resume",
        )
        await self._run_state_handler(state, event)

    async def _save_checkpoint(self) -> None:
        """Save current state to checkpoint."""
        from .checkpoints import save_checkpoint

        checkpoint_data = {
            "context": self.context.to_dict(),
            "event_log": self.event_log.to_dict(),
            "timestamp": datetime.utcnow().isoformat(),
        }
        save_checkpoint(
            checkpoint_data,
            self.checkpoint_dir,
            self.context.cycle_id,
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get state machine metrics."""
        return {
            "current_state": self.current_state.name,
            "cycle_id": self.cycle_id,
            "running": self.running,
            "total_cycles": self._total_cycles,
            "successful_cycles": self._successful_cycles,
            "failed_cycles": self._failed_cycles,
            "success_rate": (
                self._successful_cycles / self._total_cycles if self._total_cycles > 0 else 0
            ),
            "event_count": len(self.event_log.events),
            "error_count": len(self.event_log.get_errors()),
            "state_durations": self.context.state_durations,
        }


# Convenience function to create a pre-configured state machine
def create_nomic_state_machine(
    checkpoint_dir: str = ".nomic/checkpoints",
    enable_checkpoints: bool = True,
) -> NomicStateMachine:
    """
    Create a nomic state machine with default configuration.

    Returns a state machine ready for handler registration.
    """
    return NomicStateMachine(
        checkpoint_dir=checkpoint_dir,
        enable_checkpoints=enable_checkpoints,
    )
