"""
Registry and factory functions for molecules.

This module contains escalation support and factory functions:
- EscalationLevel: Severity levels for escalation
- EscalationContext: Context for escalation step execution
- EscalationStepExecutor: Execute escalation steps
- Factory functions for creating molecules and getting the engine
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from aragora.nomic.beads import BeadStore
from aragora.nomic.molecules.base import Molecule, MoleculeStep
from aragora.nomic.molecules.proposal import StepExecutor

logger = logging.getLogger(__name__)


# =============================================================================
# Escalation Support (Gastown pattern)
# =============================================================================


class EscalationLevel(str, Enum):
    """Standard escalation severity levels."""

    WARN = "warn"  # Log warning, notify observers
    THROTTLE = "throttle"  # Reduce throughput/rate limit
    SUSPEND = "suspend"  # Pause agent/operation
    TERMINATE = "terminate"  # Full stop with cleanup


@dataclass
class EscalationContext:
    """Context for escalation step execution."""

    level: EscalationLevel
    source: str  # What triggered the escalation
    reason: str  # Why escalation was triggered
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    previous_level: EscalationLevel | None = None
    auto_escalate_at: datetime | None = None


class EscalationStepExecutor(StepExecutor):
    """Execute escalation steps with severity-aware handlers."""

    def __init__(
        self,
        handlers: dict[str, Any],
        auto_escalate_seconds: float = 300.0,
    ):
        """
        Initialize escalation executor.

        Args:
            handlers: Dict mapping severity level to handler callable
            auto_escalate_seconds: Time before auto-escalating to next level
        """
        self._handlers = handlers
        self._auto_escalate_seconds = auto_escalate_seconds
        self._current_level: EscalationLevel | None = None

    async def execute(self, step: MoleculeStep, context: dict[str, Any]) -> Any:
        """Execute escalation step."""
        level_str = step.config.get("level", "warn")
        try:
            level = EscalationLevel(level_str)
        except ValueError:
            level = EscalationLevel.WARN

        source = step.config.get("source", "unknown")
        reason = step.config.get("reason", "unspecified")
        metadata = step.config.get("metadata", {})

        escalation_ctx = EscalationContext(
            level=level,
            source=source,
            reason=reason,
            metadata=metadata,
            previous_level=self._current_level,
        )

        # Find and execute handler
        handler = self._handlers.get(level.value)
        if handler is None:
            logger.warning("No handler for escalation level: %s", level.value)
            return {"status": "no_handler", "level": level.value}

        logger.info("Executing escalation: level=%s source=%s", level.value, source)

        try:
            if asyncio.iscoroutinefunction(handler):
                result = await handler(escalation_ctx)
            else:
                result = handler(escalation_ctx)

            self._current_level = level

            return {
                "status": "executed",
                "level": level.value,
                "result": result,
                "previous_level": escalation_ctx.previous_level.value
                if escalation_ctx.previous_level
                else None,
            }
        except (RuntimeError, ValueError, OSError) as e:
            logger.error("Escalation handler failed: %s", e)
            raise


def create_escalation_molecule(
    name: str,
    severity_levels: list[str],
    handlers: dict[str, Any],
    source: str = "system",
    reason: str = "threshold_exceeded",
    auto_escalate_after_seconds: float = 300.0,
    metadata: dict[str, Any] | None = None,
) -> Molecule:
    """
    Create a molecule for escalation workflows (Gastown pattern).

    This creates a multi-step molecule where each step represents an
    escalation level. Steps are executed in sequence if the previous
    level doesn't resolve the issue.

    Args:
        name: Name for the escalation molecule
        severity_levels: List of severity levels in order (e.g., ["warn", "throttle", "suspend"])
        handlers: Dict mapping level names to handler callables
        source: Source of the escalation (for logging)
        reason: Reason for the escalation (for logging)
        auto_escalate_after_seconds: Time before auto-escalating to next level
        metadata: Additional metadata for the escalation

    Returns:
        Configured Molecule ready for execution

    Example:
        async def warn_handler(ctx):
            await send_alert(ctx.source, ctx.reason)

        async def throttle_handler(ctx):
            await apply_rate_limit(ctx.source, 0.5)

        async def suspend_handler(ctx):
            await pause_agent(ctx.source)

        molecule = create_escalation_molecule(
            name="agent_overload_escalation",
            severity_levels=["warn", "throttle", "suspend"],
            handlers={
                "warn": warn_handler,
                "throttle": throttle_handler,
                "suspend": suspend_handler,
            },
            source="agent_monitor",
            reason="response_latency_exceeded",
        )

        engine = await get_molecule_engine()
        engine.register_executor("escalation", EscalationStepExecutor(handlers))
        result = await engine.execute(molecule)
    """
    steps = []
    prev_step_id = None

    for i, level in enumerate(severity_levels):
        step = MoleculeStep.create(
            name=f"escalate_{level}",
            step_type="escalation",
            config={
                "level": level,
                "source": source,
                "reason": reason,
                "metadata": metadata or {},
                "auto_escalate_seconds": auto_escalate_after_seconds,
            },
            timeout_seconds=auto_escalate_after_seconds * 2,  # Allow time for resolution
            dependencies=[prev_step_id] if prev_step_id else [],
        )
        steps.append(step)
        prev_step_id = step.id

    return Molecule.create(
        name=name,
        steps=steps,
        description=f"Escalation workflow: {' -> '.join(severity_levels)}",
        metadata={
            "type": "escalation",
            "source": source,
            "reason": reason,
            "severity_levels": severity_levels,
            **(metadata or {}),
        },
    )


def create_conditional_escalation_molecule(
    name: str,
    check_fn: Any,  # Callable[[], bool] - condition to check
    severity_levels: list[str],
    handlers: dict[str, Any],
    check_interval_seconds: float = 60.0,
    max_checks_per_level: int = 5,
    source: str = "conditional_monitor",
    reason: str = "condition_triggered",
) -> Molecule:
    """
    Create an escalation molecule with condition checking.

    This molecule checks a condition at each level and only escalates
    if the condition remains true after multiple checks.

    Args:
        name: Name for the escalation molecule
        check_fn: Callable that returns True if escalation should continue
        severity_levels: List of severity levels in order
        handlers: Dict mapping level names to handler callables
        check_interval_seconds: Time between condition checks
        max_checks_per_level: Number of checks before escalating
        source: Source of the escalation
        reason: Reason for the escalation

    Returns:
        Configured Molecule with condition checking
    """
    steps = []
    prev_step_id = None

    for level in severity_levels:
        # Add check step
        check_step = MoleculeStep.create(
            name=f"check_before_{level}",
            step_type="conditional",
            config={
                "check_fn": check_fn,
                "max_checks": max_checks_per_level,
                "check_interval": check_interval_seconds,
            },
            timeout_seconds=check_interval_seconds * max_checks_per_level * 2,
            dependencies=[prev_step_id] if prev_step_id else [],
        )
        steps.append(check_step)

        # Add escalation step
        escalate_step = MoleculeStep.create(
            name=f"escalate_{level}",
            step_type="escalation",
            config={
                "level": level,
                "source": source,
                "reason": reason,
            },
            dependencies=[check_step.id],
        )
        steps.append(escalate_step)
        prev_step_id = escalate_step.id

    return Molecule.create(
        name=name,
        steps=steps,
        description=f"Conditional escalation: {' -> '.join(severity_levels)}",
        metadata={
            "type": "conditional_escalation",
            "source": source,
            "severity_levels": severity_levels,
        },
    )


# =============================================================================
# Singleton Engine Instance
# =============================================================================

# Import here to avoid circular imports
from aragora.nomic.molecules.execution import MoleculeEngine

# Singleton instance
_default_engine: MoleculeEngine | None = None


async def get_molecule_engine(
    bead_store: BeadStore | None = None,
    checkpoint_dir: Path | None = None,
) -> MoleculeEngine:
    """Get the default molecule engine instance."""
    global _default_engine
    if _default_engine is None:
        _default_engine = MoleculeEngine(bead_store, checkpoint_dir)
        await _default_engine.initialize()
    return _default_engine


def reset_molecule_engine() -> None:
    """Reset the default engine (for testing)."""
    global _default_engine
    _default_engine = None
