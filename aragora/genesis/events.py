"""
Genesis Events - Streaming event hooks for real-time visualization.

Provides hooks for streaming genesis events to the live dashboard:
- Fractal debate spawning and merging
- Agent birth, evolution, and death
- Lineage branching
- Population changes
"""

from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

from aragora.genesis.genome import AgentGenome


class GenesisStreamEventType(Enum):
    """Event types for streaming to dashboard."""

    # Fractal events
    FRACTAL_START = "fractal_start"
    FRACTAL_SPAWN = "fractal_spawn"
    FRACTAL_MERGE = "fractal_merge"
    FRACTAL_COMPLETE = "fractal_complete"

    # Agent events
    AGENT_BIRTH = "agent_birth"
    AGENT_EVOLUTION = "agent_evolution"
    AGENT_DEATH = "agent_death"
    LINEAGE_BRANCH = "lineage_branch"

    # Population events
    POPULATION_UPDATE = "population_update"
    GENERATION_ADVANCE = "generation_advance"

    # Tension events
    TENSION_DETECTED = "tension_detected"
    TENSION_RESOLVED = "tension_resolved"


def create_genesis_hooks(
    emitter: Any,  # SyncEventEmitter from aragora.server.stream
    ledger: Optional[Any] = None,  # GenesisLedger instance
) -> dict[str, Callable]:
    """
    Create hooks for streaming genesis events to the dashboard.

    Args:
        emitter: SyncEventEmitter instance for WebSocket streaming
        ledger: Optional GenesisLedger for recording events

    Returns:
        Dictionary of hook functions to pass to FractalOrchestrator
    """

    def emit_event(event_type: GenesisStreamEventType, **data):
        """Helper to emit an event."""
        event_data = {
            "type": event_type.value,
            "timestamp": datetime.now().isoformat(),
            **data,
        }

        if hasattr(emitter, "emit"):
            emitter.emit(event_type.value, event_data)
        elif hasattr(emitter, "emit_sync"):
            emitter.emit_sync(event_type.value, event_data)

    def on_fractal_start(
        debate_id: str,
        task: str,
        depth: int,
        parent_id: Optional[str] = None,
    ):
        """Called when a fractal debate starts."""
        emit_event(
            GenesisStreamEventType.FRACTAL_START,
            debate_id=debate_id,
            task=task[:200],
            depth=depth,
            parent_id=parent_id,
        )

        if ledger:
            ledger.record_debate_start(
                debate_id=debate_id,
                task=task,
                agents=[],  # Will be filled in later
                parent_debate_id=parent_id,
            )

    def on_fractal_spawn(
        debate_id: str,
        parent_id: str,
        tension: str,
        depth: int,
    ):
        """Called when a sub-debate is spawned."""
        emit_event(
            GenesisStreamEventType.FRACTAL_SPAWN,
            debate_id=debate_id,
            parent_id=parent_id,
            tension=tension[:200],
            depth=depth,
        )

        if ledger:
            ledger.record_debate_spawn(
                parent_id=parent_id,
                child_id=debate_id,
                trigger="unresolved_tension",
                tension_description=tension,
            )

    def on_fractal_merge(
        debate_id: str,
        parent_id: str,
        success: bool,
        resolution: str = "",
    ):
        """Called when a sub-debate is merged back."""
        emit_event(
            GenesisStreamEventType.FRACTAL_MERGE,
            debate_id=debate_id,
            parent_id=parent_id,
            success=success,
            resolution=resolution[:200],
        )

        if ledger:
            ledger.record_debate_merge(
                parent_id=parent_id,
                child_id=debate_id,
                success=success,
                resolution=resolution,
            )

    def on_fractal_complete(
        debate_id: str,
        depth: int,
        sub_debates: int,
        consensus_reached: bool,
    ):
        """Called when a fractal debate completes."""
        emit_event(
            GenesisStreamEventType.FRACTAL_COMPLETE,
            debate_id=debate_id,
            depth=depth,
            sub_debates=sub_debates,
            consensus_reached=consensus_reached,
        )

    def on_agent_birth(
        genome: AgentGenome,
        parents: list[str],
        birth_type: str = "crossover",
    ):
        """Called when a new agent genome is created."""
        emit_event(
            GenesisStreamEventType.AGENT_BIRTH,
            genome_id=genome.genome_id,
            name=genome.name,
            parents=parents,
            generation=genome.generation,
            birth_type=birth_type,
            traits=list(genome.traits.keys()),
            top_expertise=[k for k, _ in genome.get_top_expertise(3)],
        )

        if ledger:
            ledger.record_agent_birth(
                genome=genome,
                parents=parents,
                birth_type=birth_type,
            )

    def on_agent_evolution(
        genome_id: str,
        old_fitness: float,
        new_fitness: float,
        reason: str,
    ):
        """Called when an agent's fitness is updated."""
        emit_event(
            GenesisStreamEventType.AGENT_EVOLUTION,
            genome_id=genome_id,
            old_fitness=old_fitness,
            new_fitness=new_fitness,
            change=new_fitness - old_fitness,
            reason=reason,
        )

        if ledger:
            ledger.record_fitness_update(
                genome_id=genome_id,
                old_fitness=old_fitness,
                new_fitness=new_fitness,
                reason=reason,
            )

    def on_agent_death(
        genome_id: str,
        reason: str,
        final_fitness: float,
    ):
        """Called when an agent genome is culled."""
        emit_event(
            GenesisStreamEventType.AGENT_DEATH,
            genome_id=genome_id,
            reason=reason,
            final_fitness=final_fitness,
        )

        if ledger:
            ledger.record_agent_death(
                genome_id=genome_id,
                reason=reason,
                final_fitness=final_fitness,
            )

    def on_lineage_branch(
        parent_genome_id: str,
        child_genome_ids: list[str],
        branch_type: str,
    ):
        """Called when a genome lineage branches (multiple children)."""
        emit_event(
            GenesisStreamEventType.LINEAGE_BRANCH,
            parent_genome_id=parent_genome_id,
            child_genome_ids=child_genome_ids,
            branch_type=branch_type,
        )

    def on_population_update(
        population_id: str,
        size: int,
        generation: int,
        average_fitness: float,
    ):
        """Called when a population is updated."""
        emit_event(
            GenesisStreamEventType.POPULATION_UPDATE,
            population_id=population_id,
            size=size,
            generation=generation,
            average_fitness=average_fitness,
        )

    def on_generation_advance(
        population_id: str,
        old_generation: int,
        new_generation: int,
        culled: int,
        born: int,
    ):
        """Called when a population advances to a new generation."""
        emit_event(
            GenesisStreamEventType.GENERATION_ADVANCE,
            population_id=population_id,
            old_generation=old_generation,
            new_generation=new_generation,
            culled=culled,
            born=born,
        )

    def on_tension_detected(
        debate_id: str,
        tension_id: str,
        description: str,
        severity: float,
    ):
        """Called when a tension is detected in a debate."""
        emit_event(
            GenesisStreamEventType.TENSION_DETECTED,
            debate_id=debate_id,
            tension_id=tension_id,
            description=description[:200],
            severity=severity,
        )

    def on_tension_resolved(
        debate_id: str,
        tension_id: str,
        resolution: str,
        success: bool,
    ):
        """Called when a tension is resolved."""
        emit_event(
            GenesisStreamEventType.TENSION_RESOLVED,
            debate_id=debate_id,
            tension_id=tension_id,
            resolution=resolution[:200],
            success=success,
        )

    return {
        "on_fractal_start": on_fractal_start,
        "on_fractal_spawn": on_fractal_spawn,
        "on_fractal_merge": on_fractal_merge,
        "on_fractal_complete": on_fractal_complete,
        "on_agent_birth": on_agent_birth,
        "on_agent_evolution": on_agent_evolution,
        "on_agent_death": on_agent_death,
        "on_lineage_branch": on_lineage_branch,
        "on_population_update": on_population_update,
        "on_generation_advance": on_generation_advance,
        "on_tension_detected": on_tension_detected,
        "on_tension_resolved": on_tension_resolved,
    }


def create_logging_hooks(
    log_func: Callable[[str], None] = print,
) -> dict[str, Callable]:
    """
    Create simple logging hooks for debugging.

    Args:
        log_func: Function to use for logging (default: print)

    Returns:
        Dictionary of hook functions
    """

    def log(event_type: str, **data):
        """Helper to log an event."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        data_str = ", ".join(f"{k}={v}" for k, v in list(data.items())[:5])
        log_func(f"[{timestamp}] {event_type}: {data_str}")

    def on_fractal_start(debate_id, task, depth, parent_id=None):
        log("FRACTAL_START", debate_id=debate_id, depth=depth, parent_id=parent_id)

    def on_fractal_spawn(debate_id, parent_id, tension, depth):
        log("FRACTAL_SPAWN", debate_id=debate_id, depth=depth, tension=tension[:50])

    def on_fractal_merge(debate_id, parent_id, success, resolution=""):
        log("FRACTAL_MERGE", debate_id=debate_id, success=success)

    def on_fractal_complete(debate_id, depth, sub_debates, consensus_reached):
        log(
            "FRACTAL_COMPLETE",
            debate_id=debate_id,
            sub_debates=sub_debates,
            consensus=consensus_reached,
        )

    def on_agent_birth(genome, parents, birth_type="crossover"):
        log("AGENT_BIRTH", genome_id=genome.genome_id, name=genome.name, birth_type=birth_type)

    def on_agent_evolution(genome_id, old_fitness, new_fitness, reason):
        log("AGENT_EVOLUTION", genome_id=genome_id, change=f"{new_fitness - old_fitness:+.2f}")

    def on_agent_death(genome_id, reason, final_fitness):
        log("AGENT_DEATH", genome_id=genome_id, reason=reason)

    return {
        "on_fractal_start": on_fractal_start,
        "on_fractal_spawn": on_fractal_spawn,
        "on_fractal_merge": on_fractal_merge,
        "on_fractal_complete": on_fractal_complete,
        "on_agent_birth": on_agent_birth,
        "on_agent_evolution": on_agent_evolution,
        "on_agent_death": on_agent_death,
    }
