"""
Factory for creating and configuring debate arenas.

Extracts agent creation and arena setup logic from unified_server.py
for better modularity and testability.
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

from aragora.config import ALLOWED_AGENT_TYPES, MAX_AGENTS_PER_DEBATE
from aragora.exceptions import ConfigurationError
from aragora.rlm.debate_integration import create_training_hook

logger = logging.getLogger(__name__)

# Import create_agent for agent creation
try:
    from aragora.agents.base import create_agent
except ImportError:
    create_agent = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from aragora.agents.grounded import MomentDetector
    from aragora.agents.personas import PersonaManager
    from aragora.agents.positions import PositionLedger
    from aragora.agents.truth_grounding import PositionTracker
    from aragora.debate.embeddings import DebateEmbeddingsDatabase as DebateEmbeddings
    from aragora.debate.orchestrator import Arena
    from aragora.insights.flip_detector import FlipDetector
    from aragora.memory.consensus import DissentRetriever
    from aragora.pulse.ingestor import TrendingTopic
    from aragora.ranking.elo import EloSystem
    from aragora.server.stream.emitter import SyncEventEmitter


# Import the unified AgentSpec from agents.spec
from aragora.agents.spec import AgentSpec


@dataclass
class AgentCreationResult:
    """Result of agent creation attempt."""

    agents: list = field(default_factory=list)
    failed: list = field(default_factory=list)  # List of (agent_type, error_msg)

    @property
    def success_count(self) -> int:
        return len(self.agents)

    @property
    def failure_count(self) -> int:
        return len(self.failed)

    @property
    def has_minimum(self) -> bool:
        """Check if minimum number of agents were created."""
        return self.success_count >= 2


@dataclass
class DebateConfig:
    """Configuration for debate creation."""

    question: str
    agents_str: str = "anthropic-api,openai-api"
    rounds: int = 8  # 9-round format (0-8), default for all debates
    consensus: str = "judge"  # Judge-based consensus for final decisions
    debate_format: str = "full"  # "light" (~5 min) or "full" (~30 min)
    debate_id: Optional[str] = None
    trending_topic: Optional["TrendingTopic"] = None  # TrendingTopic from pulse

    def parse_agent_specs(self) -> list[AgentSpec]:
        """Parse agent specifications from comma-separated string or list.

        Supports both new pipe-delimited format (provider|model|persona|role)
        and legacy colon format (provider:persona).

        Returns:
            List of AgentSpec objects

        Raises:
            ValueError: If agent count exceeds maximum or minimum
        """
        # Handle both string and list formats
        if isinstance(self.agents_str, list):
            # Join list items into comma-separated string
            agents_str = ",".join(
                s.strip() if isinstance(s, str) else str(s) for s in self.agents_str if s
            )
        else:
            agents_str = self.agents_str

        # Use unified AgentSpec.parse_list for parsing
        specs = AgentSpec.parse_list(agents_str)

        # Validate count
        if len(specs) > MAX_AGENTS_PER_DEBATE:
            raise ValueError(f"Too many agents. Maximum: {MAX_AGENTS_PER_DEBATE}")
        if len(specs) < 2:
            raise ValueError("At least 2 agents required for a debate")

        return specs


class DebateFactory:
    """
    Factory for creating and configuring debates.

    Handles agent creation, validation, and arena setup with all
    required subsystems (ELO, personas, embeddings, etc.).

    Usage:
        factory = DebateFactory(
            elo_system=elo_system,
            persona_manager=persona_manager,
            stream_emitter=emitter,
        )

        config = DebateConfig(
            question="What is the best sorting algorithm?",
            agents_str="anthropic-api,openai-api,gemini",
            rounds=3,
        )

        arena = factory.create_arena(config)
        result = await arena.run()
    """

    def __init__(
        self,
        elo_system: Optional["EloSystem"] = None,
        persona_manager: Optional["PersonaManager"] = None,
        debate_embeddings: Optional["DebateEmbeddings"] = None,
        position_tracker: Optional["PositionTracker"] = None,
        position_ledger: Optional["PositionLedger"] = None,
        flip_detector: Optional["FlipDetector"] = None,
        dissent_retriever: Optional["DissentRetriever"] = None,
        moment_detector: Optional["MomentDetector"] = None,
        stream_emitter: Optional["SyncEventEmitter"] = None,
    ):
        """Initialize the debate factory.

        Args:
            elo_system: ELO rating system for agent rankings
            persona_manager: Manager for agent personas
            debate_embeddings: Embedding store for semantic search
            position_tracker: Tracks agent positions during debate
            position_ledger: Historical position ledger
            flip_detector: Detects agent position flips
            dissent_retriever: Retrieves past dissent patterns
            moment_detector: Detects key debate moments
            stream_emitter: Event stream emitter for live updates
        """
        self.elo_system = elo_system
        self.persona_manager = persona_manager
        self.debate_embeddings = debate_embeddings
        self.position_tracker = position_tracker
        self.position_ledger = position_ledger
        self.flip_detector = flip_detector
        self.dissent_retriever = dissent_retriever
        self.moment_detector = moment_detector
        self.stream_emitter = stream_emitter

    def create_agents(
        self,
        specs: list[AgentSpec],
        stream_wrapper: Optional[Callable[..., Any]] = None,
        debate_id: Optional[str] = None,
    ) -> AgentCreationResult:
        """Create agents from specifications.

        Args:
            specs: List of agent specifications
            stream_wrapper: Optional function to wrap agents for streaming
            debate_id: Optional debate ID for error reporting

        Returns:
            AgentCreationResult with created agents and failures
        """
        if create_agent is None:
            raise ConfigurationError(
                component="DebateFactory",
                reason="create_agent not available - agents module failed to import",
            )

        result = AgentCreationResult()

        for i, spec in enumerate(specs):
            # Assign role based on position if not explicitly specified
            # This ensures diverse debate roles: proposer, critic(s), synthesizer
            role = spec.role
            if role is None:
                if i == 0:
                    role = "proposer"
                elif i == len(specs) - 1 and len(specs) > 1:
                    role = "synthesizer"
                else:
                    role = "critic"
            try:
                agent = create_agent(
                    model_type=spec.provider,  # type: ignore[arg-type]
                    name=spec.name,
                    role=role,
                    model=spec.model,  # Pass model from spec
                )

                # Apply persona as system prompt modifier if specified
                if spec.persona:
                    try:
                        from aragora.agents.personas import apply_persona_to_agent

                        apply_persona_to_agent(agent, spec.persona)
                    except ImportError:
                        pass  # Personas module not available

                # Validate API key for API-based agents
                if hasattr(agent, "api_key") and not agent.api_key:
                    raise ValueError(f"Missing API key for {spec.provider}")

                # Apply streaming wrapper if provided
                if stream_wrapper is not None:
                    agent = stream_wrapper(agent, self.stream_emitter, debate_id)

                result.agents.append(agent)
                logger.debug(f"Created agent {spec.provider} successfully")

            except Exception as e:
                error_msg = f"Failed to create agent {spec.provider}: {e}"
                logger.error(error_msg)
                result.failed.append((spec.provider, str(e)))

                # Emit error event if emitter available
                if self.stream_emitter and debate_id:
                    self._emit_agent_error(spec.provider, str(e), debate_id)

        return result

    def _emit_agent_error(self, agent_type: str, error: str, debate_id: str) -> None:
        """Emit an error event for agent creation failure."""
        try:
            from aragora.server.stream.events import StreamEvent, StreamEventType

            self.stream_emitter.emit(
                StreamEvent(
                    type=StreamEventType.ERROR,
                    data={
                        "agent": agent_type,
                        "error": error,
                        "phase": "initialization",
                    },
                    loop_id=debate_id,
                )
            )
        except Exception as e:
            logger.warning(f"Failed to emit agent error event: {e}")

    def _get_persona_prompt(self, persona: str) -> Optional[str]:
        """Get system prompt modifier for a persona.

        DEPRECATED: Use aragora.agents.personas.get_persona_prompt() instead.

        Args:
            persona: Persona name (e.g., 'philosopher', 'security_engineer')

        Returns:
            Persona-specific system prompt, or None if not found
        """
        import warnings
        warnings.warn(
            "_get_persona_prompt is deprecated. Use aragora.agents.personas.get_persona_prompt() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # First check DEFAULT_PERSONAS from aragora.agents.personas
        try:
            from aragora.agents.personas import DEFAULT_PERSONAS

            if persona in DEFAULT_PERSONAS:
                p = DEFAULT_PERSONAS[persona]
                # Generate system prompt from persona attributes
                traits_str = ", ".join(p.traits) if p.traits else "analytical"
                prompt = f"You are a {traits_str} agent. {p.description}"
                if p.top_expertise:
                    top_domains = [d for d, _ in p.top_expertise]
                    prompt += f" Your key areas of expertise: {', '.join(top_domains)}."
                return prompt
        except ImportError:
            logger.debug("Personas module not available")

        # Check PersonaManager if available
        if self.persona_manager:
            try:
                stored_persona = self.persona_manager.get_persona(persona)
                if stored_persona:
                    traits_str = ", ".join(stored_persona.traits) if stored_persona.traits else "analytical"
                    prompt = f"You are a {traits_str} agent. {stored_persona.description}"
                    return prompt
            except Exception as e:
                logger.debug(f"Failed to get persona from manager: {e}")

        # Fallback: use persona name as behavioral hint
        if persona:
            return f"You are a {persona} in this debate. Approach arguments from that perspective."

        return None

    def _apply_persona_params(self, agent: Any, persona: str) -> None:
        """Apply persona generation parameters (temperature, top_p, frequency_penalty) to agent.

        DEPRECATED: Use aragora.agents.personas.apply_persona_to_agent() instead.

        Args:
            agent: Agent instance to configure
            persona: Persona name to look up parameters from
        """
        import warnings
        warnings.warn(
            "_apply_persona_params is deprecated. Use aragora.agents.personas.apply_persona_to_agent() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Only apply if agent supports set_generation_params
        if not hasattr(agent, "set_generation_params"):
            return

        # Check DEFAULT_PERSONAS first
        try:
            from aragora.agents.personas import DEFAULT_PERSONAS

            if persona in DEFAULT_PERSONAS:
                p = DEFAULT_PERSONAS[persona]
                agent.set_generation_params(
                    temperature=p.temperature,
                    top_p=p.top_p,
                    frequency_penalty=p.frequency_penalty,
                )
                logger.debug(f"Applied persona params for {persona}: temp={p.temperature}")
                return
        except ImportError:
            pass

        # Check PersonaManager
        if self.persona_manager:
            try:
                stored_persona = self.persona_manager.get_persona(persona)
                if stored_persona:
                    agent.set_generation_params(
                        temperature=stored_persona.temperature,
                        top_p=stored_persona.top_p,
                        frequency_penalty=stored_persona.frequency_penalty,
                    )
                    logger.debug(f"Applied persona params from manager for {persona}")
                    return
            except Exception as e:
                logger.debug(f"Failed to get persona params: {e}")

    def create_arena(
        self,
        config: DebateConfig,
        event_hooks: Optional[dict] = None,
        stream_wrapper: Optional[Callable[..., Any]] = None,
        enable_rlm_training: Optional[bool] = None,
    ) -> "Arena":
        """Create a fully configured debate arena.

        Uses ArenaBuilder internally for cleaner configuration.

        Args:
            config: Debate configuration
            event_hooks: Optional event hooks for the arena
            stream_wrapper: Optional function to wrap agents for streaming
            enable_rlm_training: Whether to enable RLM training (None = use settings)

        Returns:
            Configured Arena ready to run

        Raises:
            ValueError: If not enough agents could be created
        """
        from aragora.config.settings import get_settings
        from aragora.core_types import Environment

        # Read from settings if not explicitly provided
        if enable_rlm_training is None:
            enable_rlm_training = get_settings().integration.rlm_training_enabled
        from aragora.debate.arena_builder import ArenaBuilder
        from aragora.debate.protocol import (
            ARAGORA_AI_LIGHT_PROTOCOL,
            ARAGORA_AI_PROTOCOL,
            DebateProtocol,
        )

        # Parse and create agents
        specs = config.parse_agent_specs()
        agent_result = self.create_agents(
            specs,
            stream_wrapper=stream_wrapper,
            debate_id=config.debate_id,
        )

        if not agent_result.has_minimum:
            failed_names = [a for a, _ in agent_result.failed]
            raise ValueError(
                f"Only {agent_result.success_count} agents initialized "
                f"(need at least 2). Failed: {', '.join(failed_names)}"
            )

        # Select protocol based on debate_format
        # - "light": 4-round quick format (~5 min) with minimal features
        # - "full": 9-round thorough format (~30 min) with all features
        if config.debate_format == "light":
            base_protocol = ARAGORA_AI_LIGHT_PROTOCOL
            max_rounds = 4
            logger.info("debate_format=light using 4-round quick protocol")
        else:
            base_protocol = ARAGORA_AI_PROTOCOL
            max_rounds = 9
            logger.info("debate_format=full using 9-round thorough protocol")

        # Create environment with appropriate round count
        env = Environment(
            task=config.question,
            context="",
            max_rounds=max_rounds,
        )

        # Create protocol from preset, allowing consensus override
        protocol = DebateProtocol(
            rounds=base_protocol.rounds,
            consensus=config.consensus or base_protocol.consensus,  # type: ignore[arg-type]
            proposer_count=len(agent_result.agents),
            topology=base_protocol.topology,
            use_structured_phases=base_protocol.use_structured_phases,
            round_phases=base_protocol.round_phases,
            early_stopping=base_protocol.early_stopping,
            early_stop_threshold=base_protocol.early_stop_threshold,
            min_rounds_before_early_stop=base_protocol.min_rounds_before_early_stop,
            convergence_detection=base_protocol.convergence_detection,
            convergence_threshold=base_protocol.convergence_threshold,
            enable_trickster=base_protocol.enable_trickster,
            trickster_sensitivity=base_protocol.trickster_sensitivity,
            enable_calibration=base_protocol.enable_calibration,
            enable_rhetorical_observer=base_protocol.enable_rhetorical_observer,
            enable_evolution=base_protocol.enable_evolution,
            enable_evidence_weighting=base_protocol.enable_evidence_weighting,
            verify_claims_during_consensus=base_protocol.verify_claims_during_consensus,
            enable_research=base_protocol.enable_research,
            role_rotation=base_protocol.role_rotation,
            role_matching=base_protocol.role_matching,
            timeout_seconds=base_protocol.timeout_seconds,
            round_timeout_seconds=base_protocol.round_timeout_seconds,
            debate_rounds_timeout_seconds=base_protocol.debate_rounds_timeout_seconds,
            enable_breakpoints=base_protocol.enable_breakpoints,
        )

        # Prepare event hooks with RLM training hook if enabled
        hooks = dict(event_hooks or {})
        if enable_rlm_training:
            training_hook = create_training_hook()
            # Add training hook (chain with existing on_debate_complete if present)
            existing_hook = hooks.get("on_debate_complete")
            if existing_hook:
                # Chain hooks together
                def chained_hook(result, ctx=None, _existing=existing_hook, _training=training_hook):
                    _existing(result, ctx)
                    _training(result, ctx)
                hooks["on_debate_complete"] = chained_hook
            else:
                hooks["on_debate_complete"] = training_hook
            logger.debug("RLM training hook enabled for debate trajectory collection")

        # Build arena using ArenaBuilder for cleaner configuration
        builder = (
            ArenaBuilder(env, agent_result.agents)
            .with_protocol(protocol)
            .with_event_hooks(hooks)
            .with_event_emitter(self.stream_emitter)
            .with_loop_id(config.debate_id or "")
            .with_strict_loop_scoping(True)  # Enable strict scoping for web debates
        )

        # Add all available subsystems for comprehensive 9-round debates
        if self.persona_manager:
            builder = builder.with_persona_manager(self.persona_manager)
        if self.debate_embeddings:
            builder = builder.with_debate_embeddings(self.debate_embeddings)
        if self.elo_system:
            builder = builder.with_elo_system(self.elo_system)
        if self.position_tracker:
            builder = builder.with_position_tracker(self.position_tracker)
        if self.position_ledger:
            builder = builder.with_position_ledger(self.position_ledger)
        if self.flip_detector:
            builder = builder.with_flip_detector(self.flip_detector)
        if self.dissent_retriever:
            builder = builder.with_dissent_retriever(self.dissent_retriever)
        if self.moment_detector:
            builder = builder.with_moment_detector(self.moment_detector)
        if config.trending_topic:
            builder = builder.with_trending_topic(config.trending_topic)

        # Enable position ledger auto-creation for truth grounding
        builder = builder.with_enable_position_ledger(True)

        return builder.build()

    def reset_circuit_breakers(self, arena: "Arena") -> None:
        """Reset circuit breakers for fresh debate.

        For ad-hoc debates, we want all agents to have a fresh start.

        Args:
            arena: The arena whose circuit breakers to reset
        """
        cb_status = arena.circuit_breaker.get_all_status()
        if cb_status:
            logger.debug(f"Agent status before debate: {cb_status}")
            open_circuits = [
                name for name, status in cb_status.items() if status["status"] == "open"
            ]
            if open_circuits:
                logger.debug(f"Resetting open circuits for: {open_circuits}")
                arena.circuit_breaker.reset()
