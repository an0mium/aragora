"""
ArenaFactory: Centralized Arena instantiation for the nomic loop.

Consolidates Arena creation patterns from multiple phases (debate, design,
tournament, scenario, fractal) into a single factory with dependency injection.

This module is part of Wave 3 extraction from nomic_loop.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from aragora.debate.orchestrator import Arena as ArenaType
    from aragora.core import Environment, DebateProtocol

# Lazy imports for optional dependencies
_Arena: Optional[type] = None
create_arena_hooks = None

try:
    from aragora.debate.orchestrator import Arena as ArenaClass

    _Arena = ArenaClass
except ImportError:
    pass

try:
    from aragora.server.stream.arena_hooks import create_arena_hooks as _create_hooks

    create_arena_hooks = _create_hooks
except ImportError:
    pass


@dataclass
class ArenaConfig:
    """Configuration for Arena creation.

    Controls which integrations are enabled for a particular Arena instance.
    Different phases may need different configurations (e.g., design phase
    may skip memory recording, tournament may skip calibration).
    """

    phase_name: str = "debate"
    use_airlock: bool = True
    include_memory: bool = True
    include_tracking: bool = True
    include_calibration: bool = True
    include_relationships: bool = True


@dataclass
class ArenaFactoryDependencies:
    """Container for all Arena dependencies.

    Groups dependencies by category for cleaner initialization.
    All dependencies are optional - the factory gracefully handles
    missing dependencies by passing None to Arena.
    """

    # Streaming
    stream_emitter: Optional[Any] = None
    loop_id: str = ""

    # Memory systems
    critique_store: Optional[Any] = None
    debate_embeddings: Optional[Any] = None
    insight_store: Optional[Any] = None
    continuum_memory: Optional[Any] = None

    # Tracking systems
    position_tracker: Optional[Any] = None
    position_ledger: Optional[Any] = None
    calibration_tracker: Optional[Any] = None
    elo_system: Optional[Any] = None

    # Agent enhancement
    persona_manager: Optional[Any] = None
    relationship_tracker: Optional[Any] = None
    moment_detector: Optional[Any] = None


class ArenaFactory:
    """Factory for creating Arena instances with consistent configuration.

    Centralizes Arena instantiation to:
    1. Reduce code duplication across phases
    2. Ensure consistent hook creation
    3. Allow easy testing via dependency injection
    4. Support phase-specific configuration

    Usage:
        factory = ArenaFactory(deps, log_fn=self._log)
        arena = factory.create(env, agents, protocol, config=ArenaConfig(phase_name="debate"))
    """

    def __init__(
        self,
        deps: ArenaFactoryDependencies,
        log_fn: Optional[Callable[[str], None]] = None,
    ):
        """Initialize factory with dependencies.

        Args:
            deps: Container holding all Arena dependencies
            log_fn: Optional logging callback (defaults to print)
        """
        self.deps = deps
        self._log = log_fn or (lambda msg: print(f"[arena-factory] {msg}"))

    def create(
        self,
        environment: Any,
        agents: list,
        protocol: Any,
        config: Optional[ArenaConfig] = None,
        agent_weights: Optional[dict] = None,
        **extra_kwargs,
    ) -> Optional[Any]:
        """Create an Arena instance with configured dependencies.

        Args:
            environment: Environment describing the debate context
            agents: List of agents participating in the debate
            protocol: DebateProtocol controlling debate flow
            config: Optional ArenaConfig for phase-specific settings
            agent_weights: Optional reliability weights per agent
            **extra_kwargs: Additional Arena constructor arguments

        Returns:
            Configured Arena instance, or None if Arena is unavailable
        """
        if _Arena is None:
            self._log("Arena not available - cannot create instance")
            return None

        config = config or ArenaConfig()

        # Build hooks for this phase
        event_hooks = self._create_hooks(config.phase_name)

        # Build Arena kwargs based on config
        kwargs = {
            "event_emitter": self.deps.stream_emitter,
            "loop_id": self.deps.loop_id,
            "event_hooks": event_hooks,
            "use_airlock": config.use_airlock,
        }

        # Add memory systems if enabled
        if config.include_memory:
            kwargs.update(
                {
                    "memory": self.deps.critique_store,
                    "debate_embeddings": self.deps.debate_embeddings,
                    "insight_store": self.deps.insight_store,
                    "continuum_memory": self.deps.continuum_memory,
                }
            )

        # Add tracking systems if enabled
        if config.include_tracking:
            kwargs.update(
                {
                    "position_tracker": self.deps.position_tracker,
                    "position_ledger": self.deps.position_ledger,
                    "elo_system": self.deps.elo_system,
                }
            )

        # Add calibration if enabled
        if config.include_calibration:
            kwargs["calibration_tracker"] = self.deps.calibration_tracker

        # Add relationship tracking if enabled
        if config.include_relationships:
            kwargs.update(
                {
                    "persona_manager": self.deps.persona_manager,
                    "relationship_tracker": self.deps.relationship_tracker,
                    "moment_detector": self.deps.moment_detector,
                }
            )

        # Add agent weights if provided
        if agent_weights:
            kwargs["agent_weights"] = agent_weights

        # Merge any extra kwargs
        kwargs.update(extra_kwargs)

        return _Arena(environment, agents, protocol, **kwargs)

    def _create_hooks(self, phase_name: str) -> dict:
        """Create event hooks for real-time Arena logging and streaming.

        Combines local logging with streaming hooks when available.

        Args:
            phase_name: Name of the phase (for log prefix)

        Returns:
            Dictionary of hook callbacks
        """
        # Get streaming hooks if available
        stream_hooks = {}
        if self.deps.stream_emitter and create_arena_hooks:
            try:
                stream_hooks = create_arena_hooks(self.deps.stream_emitter)
            except Exception as e:
                self._log(f"Failed to create streaming hooks: {e}")

        def make_combined_hook(log_fn: Callable, stream_hook_name: str) -> Callable:
            """Combine logging and streaming for a hook."""
            stream_fn = stream_hooks.get(stream_hook_name)

            def combined(*args, **kwargs):
                log_fn(*args, **kwargs)
                if stream_fn:
                    try:
                        stream_fn(*args, **kwargs)
                    except Exception:
                        pass  # Don't let streaming errors break the loop

            return combined

        return {
            "on_debate_start": make_combined_hook(
                lambda task, agents: self._log(
                    f"    [{phase_name}] Debate started: {len(agents)} agents"
                ),
                "on_debate_start",
            ),
            "on_message": make_combined_hook(
                lambda agent, content, role, round_num: self._log(
                    f"    [{phase_name}][{role}] {agent} (round {round_num}): {content}"
                ),
                "on_message",
            ),
            "on_critique": make_combined_hook(
                lambda agent, target, issues, severity, round_num, full_content=None: self._log(
                    f"    [{phase_name}][critique] {agent} -> {target}: {len(issues)} issues, severity {severity:.1f}"
                ),
                "on_critique",
            ),
            "on_round_start": make_combined_hook(
                lambda round_num: self._log(f"    [{phase_name}] --- Round {round_num} ---"),
                "on_round_start",
            ),
            "on_consensus": make_combined_hook(
                lambda result: self._log(
                    f"    [{phase_name}] Consensus reached: {result.consensus_reached}"
                ),
                "on_consensus",
            ),
            "on_vote": make_combined_hook(
                lambda agent, choice, reasoning: self._log(
                    f"    [{phase_name}][vote] {agent} -> {choice}"
                ),
                "on_vote",
            ),
        }

    def create_for_tournament(
        self,
        environment: Any,
        agents: list,
        protocol: Any,
    ) -> Optional[Any]:
        """Create Arena for tournament mode with minimal integrations.

        Tournament debates focus on agent comparison, so we skip
        memory recording and calibration to reduce overhead.
        """
        config = ArenaConfig(
            phase_name="tournament",
            use_airlock=True,
            include_memory=False,  # Tournament doesn't need memory
            include_tracking=True,  # Still track positions
            include_calibration=False,  # Skip calibration overhead
            include_relationships=False,  # Skip relationship tracking
        )
        return self.create(environment, agents, protocol, config)

    def create_for_scenario(
        self,
        environment: Any,
        agents: list,
        protocol: Any,
    ) -> Optional[Any]:
        """Create Arena for scenario debates (fractal sub-debates).

        Scenario debates are lightweight explorations that shouldn't
        pollute the main memory systems.
        """
        config = ArenaConfig(
            phase_name="scenario",
            use_airlock=True,
            include_memory=False,  # Don't record sub-debate memory
            include_tracking=False,  # Don't track positions
            include_calibration=False,  # Skip calibration
            include_relationships=False,  # Skip relationships
        )
        return self.create(environment, agents, protocol, config)
