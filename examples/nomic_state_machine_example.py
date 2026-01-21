#!/usr/bin/env python3
"""
Example: Using the Nomic State Machine

This example demonstrates the new event-driven state machine for
autonomous self-improvement. The state machine provides:

- Checkpoint/resume capability
- Circuit breakers for agent failures
- Exponential backoff on errors
- Full event sourcing for audit trail

Usage:
    python examples/nomic_state_machine_example.py
"""

import asyncio
import logging
from pathlib import Path

from aragora.nomic import (
    NomicState,
    StateContext,
    Event,
    create_nomic_state_machine,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Custom Handlers
# =============================================================================


async def simple_context_handler(context: StateContext, event: Event):
    """
    Simple context gathering handler for demonstration.

    In production, this would use ContextPhase from scripts/nomic/phases/.
    """
    logger.info("CONTEXT: Gathering codebase understanding...")
    await asyncio.sleep(0.5)  # Simulate work

    return NomicState.DEBATE, {
        "codebase_summary": "Aragora is a multi-agent debate framework",
        "recent_changes": "Added state machine architecture",
        "agents_succeeded": 2,
    }


async def simple_debate_handler(context: StateContext, event: Event):
    """
    Simple debate handler for demonstration.
    """
    logger.info("DEBATE: Agents debating improvements...")
    await asyncio.sleep(0.5)

    return NomicState.DESIGN, {
        "improvement": "Add WebSocket message compression",
        "confidence": 0.85,
        "consensus_reached": True,
    }


async def simple_design_handler(context: StateContext, event: Event):
    """
    Simple design handler for demonstration.
    """
    logger.info("DESIGN: Creating implementation plan...")
    await asyncio.sleep(0.3)

    return NomicState.IMPLEMENT, {
        "design": "Use zlib compression for WebSocket messages > 1KB",
        "files_affected": ["aragora/server/stream.py"],
        "complexity_estimate": "low",
    }


async def simple_implement_handler(context: StateContext, event: Event):
    """
    Simple implement handler for demonstration.
    """
    logger.info("IMPLEMENT: Writing code changes...")
    await asyncio.sleep(0.5)

    return NomicState.VERIFY, {
        "files_modified": ["aragora/server/stream.py"],
        "diff_summary": "+15 lines, -2 lines",
    }


async def simple_verify_handler(context: StateContext, event: Event):
    """
    Simple verify handler for demonstration.
    """
    logger.info("VERIFY: Running tests and validation...")
    await asyncio.sleep(0.3)

    return NomicState.COMMIT, {
        "tests_passed": True,
        "syntax_valid": True,
        "test_output": "All 150 tests passed",
    }


async def simple_commit_handler(context: StateContext, event: Event):
    """
    Simple commit handler for demonstration.
    """
    logger.info("COMMIT: Committing changes...")
    await asyncio.sleep(0.2)

    return NomicState.COMPLETED, {
        "committed": True,
        "commit_hash": "abc1234",
        "message": "feat(stream): add WebSocket message compression",
    }


async def simple_recovery_handler(context: StateContext, event: Event):
    """
    Simple recovery handler for demonstration.
    """
    logger.info("RECOVERY: Handling error, deciding next action...")

    # In production, use RecoveryManager to make intelligent decisions
    # For demo, just retry the previous state once
    if context.previous_state and context.retry_counts.get(context.previous_state.name, 0) < 1:
        logger.info(f"RECOVERY: Retrying {context.previous_state.name}")
        return context.previous_state, {"action": "retry"}
    else:
        logger.info("RECOVERY: Max retries exceeded, failing")
        return NomicState.FAILED, {"action": "fail"}


# =============================================================================
# Main Example
# =============================================================================


async def run_example():
    """Run the state machine example."""

    # Create checkpoint directory
    checkpoint_dir = Path(".nomic/example_checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create the state machine
    machine = create_nomic_state_machine(
        checkpoint_dir=str(checkpoint_dir),
        enable_checkpoints=True,
    )

    # Register handlers
    machine.register_handler(NomicState.CONTEXT, simple_context_handler)
    machine.register_handler(NomicState.DEBATE, simple_debate_handler)
    machine.register_handler(NomicState.DESIGN, simple_design_handler)
    machine.register_handler(NomicState.IMPLEMENT, simple_implement_handler)
    machine.register_handler(NomicState.VERIFY, simple_verify_handler)
    machine.register_handler(NomicState.COMMIT, simple_commit_handler)
    machine.register_handler(NomicState.RECOVERY, simple_recovery_handler)

    # Register callbacks for visibility
    def on_transition(from_state, to_state, event):
        pass

    def on_error(state, error):
        pass

    machine.on_transition(on_transition)
    machine.on_error(on_error)

    # Start the cycle
    await machine.start()

    # Wait for completion (in real usage, this would be event-driven)
    while machine.running:
        await asyncio.sleep(0.1)

    # Show results

    metrics = machine.get_metrics()

    if metrics["state_durations"]:
        for state, duration in metrics["state_durations"].items():
            pass

    return machine


async def run_recovery_example():
    """Demonstrate error recovery."""

    machine = create_nomic_state_machine(enable_checkpoints=False)

    # Counter for failures
    failure_count = [0]

    async def failing_context_handler(context, event):
        failure_count[0] += 1
        if failure_count[0] <= 2:
            raise ValueError(f"Simulated failure #{failure_count[0]}")
        return NomicState.DEBATE, {"recovered": True}

    machine.register_handler(NomicState.CONTEXT, failing_context_handler)
    machine.register_handler(NomicState.DEBATE, simple_debate_handler)
    machine.register_handler(NomicState.DESIGN, simple_design_handler)
    machine.register_handler(NomicState.IMPLEMENT, simple_implement_handler)
    machine.register_handler(NomicState.VERIFY, simple_verify_handler)
    machine.register_handler(NomicState.COMMIT, simple_commit_handler)
    machine.register_handler(NomicState.RECOVERY, simple_recovery_handler)

    await machine.start()

    while machine.running:
        await asyncio.sleep(0.1)

    machine.get_metrics()

    return machine


if __name__ == "__main__":
    # Run main example
    asyncio.run(run_example())

    # Run recovery example
    asyncio.run(run_recovery_example())
