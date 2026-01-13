"""
Phase Executor for Aragora Debates.

Coordinates the execution of debate phases in sequence.
Extracted from Arena to enable cleaner phase management and testing.

Usage:
    from aragora.debate.phase_executor import PhaseExecutor, PhaseConfig

    # Create executor with phases
    executor = PhaseExecutor(phases, config)

    # Execute debate
    result = await executor.execute(context)
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol

from aragora.observability.tracing import trace_debate_phase

logger = logging.getLogger(__name__)


class PhaseStatus(Enum):
    """Status of a phase execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class PhaseResult:
    """Result from a single phase execution."""

    phase_name: str
    status: PhaseStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: float = 0.0
    output: Any = None
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if phase completed successfully."""
        return self.status in (PhaseStatus.COMPLETED, PhaseStatus.SKIPPED)


@dataclass
class ExecutionResult:
    """Result from full phase execution."""

    debate_id: str
    success: bool
    phases: List[PhaseResult]
    total_duration_ms: float
    final_output: Any = None
    error: Optional[str] = None

    def get_phase_result(self, name: str) -> Optional[PhaseResult]:
        """Get result for a specific phase."""
        for phase in self.phases:
            if phase.phase_name == name:
                return phase
        return None


class Phase(Protocol):
    """Protocol for debate phases."""

    @property
    def name(self) -> str:
        """Phase name."""
        ...

    async def execute(self, context: Any) -> Any:
        """Execute the phase with given context."""
        ...


@dataclass
class PhaseConfig:
    """Configuration for phase execution."""

    # Timeout settings
    total_timeout_seconds: float = 300.0  # 5 minutes default
    phase_timeout_seconds: float = 60.0  # Per-phase timeout

    # Execution behavior
    stop_on_failure: bool = True
    skip_optional_on_timeout: bool = True

    # Tracing
    enable_tracing: bool = True
    trace_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None

    # Metrics
    metrics_callback: Optional[Callable[[str, float], None]] = None


# Phase ordering for standard debate flow
STANDARD_PHASE_ORDER = [
    "context_initializer",  # Phase 0: Gather context
    "proposal",  # Phase 1: Initial proposals
    "debate_rounds",  # Phase 2: Critique/revise cycles
    "consensus",  # Phase 3: Voting and agreement
    "analytics",  # Phases 4-6: Analytics and learning
    "feedback",  # Phase 7: Memory storage
]

# Optional phases that can be skipped
OPTIONAL_PHASES = {"analytics", "feedback"}


class PhaseExecutor:
    """
    Executes debate phases in sequence.

    Features:
    - Sequential phase execution with timeouts
    - Phase-level error handling and recovery
    - Metrics collection and tracing
    - Optional phase skipping
    - Early termination support
    """

    def __init__(
        self,
        phases: Dict[str, Phase],
        config: Optional[PhaseConfig] = None,
    ):
        """
        Initialize the phase executor.

        Args:
            phases: Dictionary mapping phase names to phase objects
            config: Optional execution configuration
        """
        self._phases = phases
        self._config = config or PhaseConfig()

        # Execution state
        self._current_phase: Optional[str] = None
        self._should_terminate: bool = False
        self._termination_reason: Optional[str] = None

        # Results tracking
        self._results: List[PhaseResult] = []

    # =========================================================================
    # Main Execution
    # =========================================================================

    async def execute(
        self,
        context: Any,
        debate_id: str = "",
        phase_order: Optional[List[str]] = None,
    ) -> ExecutionResult:
        """
        Execute all phases in sequence.

        Extracted from Arena._run_inner().

        Args:
            context: Debate context object
            debate_id: ID of the debate for tracking
            phase_order: Optional custom phase order

        Returns:
            ExecutionResult with all phase results
        """
        start_time = time.time()
        self._results = []
        self._should_terminate = False
        self._termination_reason = None

        # Determine phase order
        order = phase_order or STANDARD_PHASE_ORDER

        # Filter to available phases
        order = [p for p in order if p in self._phases]

        logger.info(f"Starting phase execution for debate {debate_id}: {order}")

        # Execute with overall timeout
        try:
            final_output = await asyncio.wait_for(
                self._execute_phases(context, order, debate_id),
                timeout=self._config.total_timeout_seconds,
            )
            success = all(r.success for r in self._results)
            error = None
        except asyncio.TimeoutError:
            logger.error(f"Phase execution timed out after {self._config.total_timeout_seconds}s")
            success = False
            error = f"Execution timed out after {self._config.total_timeout_seconds}s"
            final_output = None
        except Exception as e:
            logger.exception(f"Phase execution failed: {e}")
            success = False
            error = str(e)
            final_output = None

        total_duration = (time.time() - start_time) * 1000

        return ExecutionResult(
            debate_id=debate_id,
            success=success,
            phases=self._results.copy(),
            total_duration_ms=total_duration,
            final_output=final_output,
            error=error,
        )

    async def _execute_phases(
        self,
        context: Any,
        phase_order: List[str],
        debate_id: str,
    ) -> Any:
        """Execute phases in order."""
        final_output = None

        for phase_name in phase_order:
            # Check for early termination
            if self._should_terminate:
                logger.info(f"Early termination: {self._termination_reason}")
                break

            # Execute phase
            result = await self._execute_single_phase(phase_name, context, debate_id)
            self._results.append(result)

            # Track final output from consensus phase
            if phase_name == "consensus" and result.output is not None:
                final_output = result.output

            # Handle failure
            if not result.success:
                if self._config.stop_on_failure:
                    if phase_name not in OPTIONAL_PHASES:
                        logger.error(f"Required phase '{phase_name}' failed, stopping")
                        break
                    else:
                        logger.warning(f"Optional phase '{phase_name}' failed, continuing")

        return final_output

    async def _execute_single_phase(
        self,
        phase_name: str,
        context: Any,
        debate_id: str,
    ) -> PhaseResult:
        """Execute a single phase with error handling and OpenTelemetry tracing."""
        self._current_phase = phase_name
        phase = self._phases.get(phase_name)

        if phase is None:
            return PhaseResult(
                phase_name=phase_name,
                status=PhaseStatus.SKIPPED,
                error=f"Phase '{phase_name}' not found",
            )

        started_at = datetime.now(timezone.utc)
        start_time = time.time()

        logger.debug(f"Starting phase: {phase_name}")

        # Use OpenTelemetry tracing when enabled
        if self._config.enable_tracing:
            return await self._execute_with_tracing(
                phase, phase_name, context, debate_id, started_at, start_time
            )
        else:
            return await self._execute_without_tracing(
                phase, phase_name, context, started_at, start_time
            )

    async def _execute_with_tracing(
        self,
        phase: Phase,
        phase_name: str,
        context: Any,
        debate_id: str,
        started_at: datetime,
        start_time: float,
    ) -> PhaseResult:
        """Execute phase with OpenTelemetry tracing."""
        with trace_debate_phase(phase_name, debate_id) as span:
            try:
                output = await asyncio.wait_for(
                    phase.execute(context),
                    timeout=self._config.phase_timeout_seconds,
                )

                duration_ms = (time.time() - start_time) * 1000
                logger.debug(f"Completed phase '{phase_name}' in {duration_ms:.1f}ms")

                # Add phase-specific attributes to span
                self._add_phase_span_attributes(span, phase_name, context)

                # Report metrics
                if self._config.metrics_callback:
                    self._config.metrics_callback(f"phase_{phase_name}_duration_ms", duration_ms)

                return PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.COMPLETED,
                    started_at=started_at,
                    completed_at=datetime.now(timezone.utc),
                    duration_ms=duration_ms,
                    output=output,
                )

            except asyncio.TimeoutError:
                duration_ms = (time.time() - start_time) * 1000
                span.set_attribute("phase.timeout", True)

                if phase_name in OPTIONAL_PHASES and self._config.skip_optional_on_timeout:
                    logger.warning(f"Optional phase '{phase_name}' timed out, skipping")
                    return PhaseResult(
                        phase_name=phase_name,
                        status=PhaseStatus.SKIPPED,
                        started_at=started_at,
                        completed_at=datetime.now(timezone.utc),
                        duration_ms=duration_ms,
                        error=f"Timed out after {self._config.phase_timeout_seconds}s",
                    )
                else:
                    logger.error(f"Phase '{phase_name}' timed out")
                    return PhaseResult(
                        phase_name=phase_name,
                        status=PhaseStatus.FAILED,
                        started_at=started_at,
                        completed_at=datetime.now(timezone.utc),
                        duration_ms=duration_ms,
                        error=f"Timed out after {self._config.phase_timeout_seconds}s",
                    )

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.exception(f"Phase '{phase_name}' failed: {e}")
                span.record_exception(e)

                return PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.FAILED,
                    started_at=started_at,
                    completed_at=datetime.now(timezone.utc),
                    duration_ms=duration_ms,
                    error=str(e),
                )

            finally:
                self._current_phase = None

    async def _execute_without_tracing(
        self,
        phase: Phase,
        phase_name: str,
        context: Any,
        started_at: datetime,
        start_time: float,
    ) -> PhaseResult:
        """Execute phase without tracing (for testing or when tracing disabled)."""
        try:
            output = await asyncio.wait_for(
                phase.execute(context),
                timeout=self._config.phase_timeout_seconds,
            )

            duration_ms = (time.time() - start_time) * 1000
            logger.debug(f"Completed phase '{phase_name}' in {duration_ms:.1f}ms")

            if self._config.metrics_callback:
                self._config.metrics_callback(f"phase_{phase_name}_duration_ms", duration_ms)

            return PhaseResult(
                phase_name=phase_name,
                status=PhaseStatus.COMPLETED,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                duration_ms=duration_ms,
                output=output,
            )

        except asyncio.TimeoutError:
            duration_ms = (time.time() - start_time) * 1000

            if phase_name in OPTIONAL_PHASES and self._config.skip_optional_on_timeout:
                logger.warning(f"Optional phase '{phase_name}' timed out, skipping")
                return PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.SKIPPED,
                    started_at=started_at,
                    completed_at=datetime.now(timezone.utc),
                    duration_ms=duration_ms,
                    error=f"Timed out after {self._config.phase_timeout_seconds}s",
                )
            else:
                logger.error(f"Phase '{phase_name}' timed out")
                return PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.FAILED,
                    started_at=started_at,
                    completed_at=datetime.now(timezone.utc),
                    duration_ms=duration_ms,
                    error=f"Timed out after {self._config.phase_timeout_seconds}s",
                )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.exception(f"Phase '{phase_name}' failed: {e}")

            return PhaseResult(
                phase_name=phase_name,
                status=PhaseStatus.FAILED,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                duration_ms=duration_ms,
                error=str(e),
            )

        finally:
            self._current_phase = None

    def _add_phase_span_attributes(self, span: Any, phase_name: str, context: Any) -> None:
        """Add phase-specific attributes to the tracing span."""
        result = getattr(context, "result", None)
        if result is None:
            return

        if phase_name == "debate_rounds":
            rounds_used = getattr(result, "rounds_used", None)
            if rounds_used is not None:
                span.set_attribute("debate.rounds_used", rounds_used)
        elif phase_name == "consensus":
            consensus_reached = getattr(result, "consensus_reached", None)
            if consensus_reached is not None:
                span.set_attribute("debate.consensus_reached", consensus_reached)

    # =========================================================================
    # Termination Control
    # =========================================================================

    def request_termination(self, reason: str = "Requested") -> None:
        """
        Request early termination of phase execution.

        Args:
            reason: Reason for termination
        """
        self._should_terminate = True
        self._termination_reason = reason
        logger.info(f"Termination requested: {reason}")

    def check_termination(self) -> tuple[bool, Optional[str]]:
        """
        Check if termination has been requested.

        Returns:
            Tuple of (should_terminate, reason)
        """
        return self._should_terminate, self._termination_reason

    # =========================================================================
    # Phase Management
    # =========================================================================

    def add_phase(self, name: str, phase: Phase) -> None:
        """
        Add or replace a phase.

        Args:
            name: Phase name
            phase: Phase object
        """
        self._phases[name] = phase

    def remove_phase(self, name: str) -> bool:
        """
        Remove a phase.

        Args:
            name: Phase name

        Returns:
            True if phase was removed
        """
        if name in self._phases:
            del self._phases[name]
            return True
        return False

    def get_phase(self, name: str) -> Optional[Phase]:
        """Get a phase by name."""
        return self._phases.get(name)

    @property
    def phase_names(self) -> List[str]:
        """Get list of available phase names."""
        return list(self._phases.keys())

    @property
    def current_phase(self) -> Optional[str]:
        """Get currently executing phase name."""
        return self._current_phase

    # =========================================================================
    # Results & Metrics
    # =========================================================================

    def get_results(self) -> List[PhaseResult]:
        """Get all phase results from last execution."""
        return self._results.copy()

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get execution metrics.

        Returns:
            Dictionary with execution statistics
        """
        total_duration = sum(r.duration_ms for r in self._results)
        completed = sum(1 for r in self._results if r.status == PhaseStatus.COMPLETED)
        failed = sum(1 for r in self._results if r.status == PhaseStatus.FAILED)
        skipped = sum(1 for r in self._results if r.status == PhaseStatus.SKIPPED)

        return {
            "total_phases": len(self._results),
            "completed_phases": completed,
            "failed_phases": failed,
            "skipped_phases": skipped,
            "total_duration_ms": total_duration,
            "phase_durations": {r.phase_name: r.duration_ms for r in self._results},
            "current_phase": self._current_phase,
            "terminated_early": self._should_terminate,
            "termination_reason": self._termination_reason,
        }


__all__ = [
    "PhaseExecutor",
    "PhaseConfig",
    "PhaseResult",
    "PhaseStatus",
    "ExecutionResult",
    "Phase",
    "STANDARD_PHASE_ORDER",
    "OPTIONAL_PHASES",
]
