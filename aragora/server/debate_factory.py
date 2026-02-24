"""
Factory for creating and configuring debate arenas.

Extracts agent creation and arena setup logic from unified_server.py
for better modularity and testability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast
from collections.abc import Callable

import os

from aragora.config import (
    DEFAULT_AGENTS,
    DEFAULT_CONSENSUS,
    DEFAULT_ROUNDS,
    MAX_AGENTS_PER_DEBATE,
)
from aragora.exceptions import ConfigurationError
from aragora.rlm.debate_integration import create_training_hook

# Default vertical specialist injection (can be disabled via env)
DEFAULT_ENABLE_VERTICALS = os.environ.get("ARAGORA_ENABLE_VERTICALS", "true").lower() in (
    "true",
    "1",
    "yes",
)

# Pre-register vertical specialists at import time
try:
    import aragora.verticals.specialists  # noqa: F401

    VERTICALS_AVAILABLE = True
except ImportError:
    VERTICALS_AVAILABLE = False

# Import credential validator for auto-trim
try:
    from aragora.agents.credential_validator import (
        filter_available_agents,
        get_credential_status,
        log_credential_status,
    )

    CREDENTIAL_VALIDATOR_AVAILABLE = True
except ImportError:
    CREDENTIAL_VALIDATOR_AVAILABLE = False
    filter_available_agents = None
    get_credential_status = None
    log_credential_status = None

logger = logging.getLogger(__name__)

# Import create_agent for agent creation
create_agent: Any
try:
    from aragora.agents.base import create_agent
except ImportError:
    create_agent = None

if TYPE_CHECKING:
    from aragora.agents.base import AgentType
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
    agents_str: str = DEFAULT_AGENTS
    rounds: int = DEFAULT_ROUNDS  # 9-round format (0-8), default for all debates
    consensus: str = DEFAULT_CONSENSUS  # Default consensus for final decisions
    debate_format: str = "full"  # "light" (~5 min) or "full" (~30 min)
    debate_id: str | None = None
    trending_topic: TrendingTopic | None = None  # TrendingTopic from pulse
    metadata: dict | None = None  # Custom metadata (e.g., is_onboarding)
    documents: list[str] = field(default_factory=list)
    enable_verticals: bool = field(
        default_factory=lambda: DEFAULT_ENABLE_VERTICALS and VERTICALS_AVAILABLE
    )  # Enable vertical specialist injection (auto-injects domain expert)
    vertical_id: str | None = None  # Explicit vertical ID (optional, auto-detected if None)
    auto_trim_unavailable: bool = True  # Auto-remove agents without credentials
    context: str | None = None  # Optional context for the debate
    mode: str | None = None  # Optional request mode (metadata-level semantics)
    budget_limit_usd: float | None = None  # Per-debate budget cap (USD)
    enable_cartographer: bool | None = None  # Enable argument cartography
    enable_introspection: bool | None = None  # Enable agent introspection
    enable_auto_execution: bool | None = None  # Enable post-debate auto-execution

    def parse_agent_specs(self) -> list[AgentSpec]:
        """Parse agent specifications from comma-separated string or list.

        Supports both new pipe-delimited format (provider|model|persona|role)
        and legacy colon format (provider:persona).

        When auto_trim_unavailable is True, agents without valid credentials
        are automatically filtered out with a warning.

        Returns:
            List of AgentSpec objects

        Raises:
            ValueError: If agent count exceeds maximum or minimum (after filtering)
        """
        # Handle strings, lists of strings, lists of dicts, or AgentSpec objects
        specs = AgentSpec.coerce_list(self.agents_str, warn=False)
        if not specs:
            specs = AgentSpec.coerce_list(DEFAULT_AGENTS, warn=False)

        # Auto-trim unavailable agents if enabled and validator is available
        if self.auto_trim_unavailable and CREDENTIAL_VALIDATOR_AVAILABLE:
            try:
                specs, filtered = filter_available_agents(
                    specs,
                    log_filtered=True,
                    min_agents=2,
                )
                if filtered:
                    logger.info(
                        "Auto-trimmed %s agents without credentials: %s",
                        len(filtered),
                        ", ".join(f[0] for f in filtered),
                    )
            except ValueError as e:
                # Re-raise with more context
                raise ValueError(
                    f"Not enough agents with valid credentials. {e}. "
                    "Please configure the required API keys or use different agents."
                ) from e

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
            rounds=DEFAULT_ROUNDS,
        )

        arena = factory.create_arena(config)
        result = await arena.run()
    """

    def __init__(
        self,
        elo_system: EloSystem | None = None,
        persona_manager: PersonaManager | None = None,
        debate_embeddings: DebateEmbeddings | None = None,
        position_tracker: PositionTracker | None = None,
        position_ledger: PositionLedger | None = None,
        flip_detector: FlipDetector | None = None,
        dissent_retriever: DissentRetriever | None = None,
        moment_detector: MomentDetector | None = None,
        stream_emitter: SyncEventEmitter | None = None,
        document_store: Any | None = None,
        evidence_store: Any | None = None,
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
        self.document_store = document_store
        self.evidence_store = evidence_store

    def create_agents(
        self,
        specs: list[AgentSpec],
        stream_wrapper: Callable[..., Any] | None = None,
        debate_id: str | None = None,
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
                    model_type=cast("AgentType", spec.provider),
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
                logger.debug("Created agent %s successfully", spec.provider)

            except (
                ValueError,
                TypeError,
                KeyError,
                AttributeError,
                ImportError,
                RuntimeError,
                OSError,
            ) as e:
                error_msg = f"Failed to create agent {spec.provider}: {e}"
                logger.error(error_msg)
                safe_msg = f"Agent creation failed for {spec.provider}"
                result.failed.append((spec.provider, safe_msg))

                # Emit error event if emitter available
                if self.stream_emitter and debate_id:
                    self._emit_agent_error(spec.provider, safe_msg, debate_id)

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
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("Failed to emit agent error event: %s", e)

    def _get_persona_prompt(self, persona: str) -> str | None:
        """Get system prompt modifier for a persona.

        DEPRECATED: Use aragora.agents.personas.get_persona_prompt() instead.

        Args:
            persona: Persona name (e.g., 'philosopher', 'security_engineer')

        Returns:
            Persona-specific system prompt, or None if not found
        """
        import warnings

        warnings.warn(
            "_get_persona_prompt is deprecated. "
            "Use aragora.agents.personas.get_persona_prompt() instead.",
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
                    traits_str = (
                        ", ".join(stored_persona.traits) if stored_persona.traits else "analytical"
                    )
                    prompt = f"You are a {traits_str} agent. {stored_persona.description}"
                    return prompt
            except (ValueError, TypeError, KeyError, AttributeError) as e:
                logger.debug("Failed to get persona from manager: %s", e)

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
            "_apply_persona_params is deprecated. "
            "Use aragora.agents.personas.apply_persona_to_agent() instead.",
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
                logger.debug("Applied persona params for %s: temp=%s", persona, p.temperature)
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
                    logger.debug("Applied persona params from manager for %s", persona)
                    return
            except (ValueError, TypeError, KeyError, AttributeError) as e:
                logger.debug("Failed to get persona params: %s", e)

    def _maybe_add_vertical_specialist(
        self,
        config: DebateConfig,
        agent_result: AgentCreationResult,
    ) -> AgentCreationResult:
        """Optionally inject a vertical specialist agent."""
        if not config.enable_verticals:
            return agent_result

        try:
            from aragora.verticals.registry import VerticalRegistry
        except ImportError:
            logger.debug("Verticals registry not available; skipping specialist injection")
            return agent_result

        vertical_id = config.vertical_id or VerticalRegistry.get_for_task(config.question)
        if not vertical_id:
            logger.debug("No matching vertical found for task; skipping specialist injection")
            return agent_result

        # Avoid duplicates
        for agent in agent_result.agents:
            if getattr(agent, "vertical_id", None) == vertical_id:
                return agent_result

        if len(agent_result.agents) >= MAX_AGENTS_PER_DEBATE:
            logger.info(
                "Skipping vertical specialist (%s): max agents limit reached (%s)",
                vertical_id,
                MAX_AGENTS_PER_DEBATE,
            )
            return agent_result

        try:
            specialist = VerticalRegistry.create_specialist(
                vertical_id=vertical_id,
                name=f"{vertical_id}_specialist",
                role="critic",
            )
            # Ensure vertical prompt is applied
            try:
                specialist.system_prompt = specialist.build_system_prompt()
            except (AttributeError, TypeError, RuntimeError):
                logger.debug(
                    "Failed to build system prompt for specialist %s", vertical_id, exc_info=True
                )
            agent_result.agents.append(specialist)
            logger.info("Injected vertical specialist: %s", vertical_id)
        except (ValueError, TypeError, KeyError, RuntimeError, AttributeError) as e:
            logger.warning("Failed to create vertical specialist %s: %s", vertical_id, e)

        return agent_result

    def create_arena(
        self,
        config: DebateConfig,
        event_hooks: dict | None = None,
        stream_wrapper: Callable[..., Any] | None = None,
        enable_rlm_training: bool | None = None,
    ) -> Arena:
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

        agent_result = self._maybe_add_vertical_specialist(config, agent_result)

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
            default_rounds = base_protocol.rounds
            logger.info("debate_format=light using %s-round quick protocol", default_rounds)
        else:
            base_protocol = ARAGORA_AI_PROTOCOL
            default_rounds = base_protocol.rounds
            logger.info("debate_format=full using %s-round thorough protocol", default_rounds)

        # Respect requested rounds when explicitly set (defaults differ for light vs full)
        rounds = default_rounds
        if config.debate_format == "light":
            if config.rounds != DEFAULT_ROUNDS and config.rounds != default_rounds:
                rounds = config.rounds
        else:
            rounds = config.rounds or default_rounds

        # Avoid mismatched structured phase lengths when rounds are overridden
        use_structured_phases = base_protocol.use_structured_phases
        round_phases = base_protocol.round_phases
        if rounds != base_protocol.rounds:
            use_structured_phases = False
            round_phases = None

        # Create environment with appropriate round count
        env = Environment(
            task=config.question,
            context=config.context or "",
            max_rounds=rounds,
            documents=list(config.documents or []),
        )

        # Create protocol from preset, allowing consensus override
        consensus_type = cast(
            Literal[
                "majority",
                "unanimous",
                "judge",
                "none",
                "weighted",
                "supermajority",
                "any",
                "byzantine",
            ],
            config.consensus or base_protocol.consensus,
        )
        protocol = DebateProtocol(
            rounds=rounds,
            consensus=consensus_type,
            proposer_count=len(agent_result.agents),
            topology=base_protocol.topology,
            use_structured_phases=use_structured_phases,
            round_phases=round_phases,
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
                def chained_hook(
                    result, ctx=None, _existing=existing_hook, _training=training_hook
                ):
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
        if self.document_store:
            builder = builder.with_document_store(self.document_store)
        if self.evidence_store:
            builder = builder.with_evidence_store(self.evidence_store)

        # Enable position ledger auto-creation for truth grounding
        builder = builder.with_enable_position_ledger(True)

        # Pass feature flags from config if specified
        if any(
            v is not None
            for v in (
                config.enable_cartographer,
                config.enable_introspection,
                config.enable_auto_execution,
            )
        ):
            builder = builder.with_feature_flags(
                enable_cartographer=config.enable_cartographer,
                enable_introspection=config.enable_introspection,
                enable_auto_execution=config.enable_auto_execution,
            )

        arena = builder.build()

        # Apply per-debate budget cap if specified
        if config.budget_limit_usd and config.budget_limit_usd > 0:
            setattr(arena, "budget_limit_usd", config.budget_limit_usd)
            if hasattr(arena, "extensions") and arena.extensions is not None:
                arena.extensions.debate_budget_limit_usd = config.budget_limit_usd
                arena.extensions.enforce_budget_limit = True

        return arena

    def reset_circuit_breakers(self, arena: Arena) -> None:
        """Reset circuit breakers for fresh debate.

        For ad-hoc debates, we want all agents to have a fresh start.

        Args:
            arena: The arena whose circuit breakers to reset
        """
        cb_status = arena.circuit_breaker.get_all_status()
        if cb_status:
            logger.debug("Agent status before debate: %s", cb_status)
            open_circuits = [
                name for name, status in cb_status.items() if status["status"] == "open"
            ]
            if open_circuits:
                logger.debug("Resetting open circuits for: %s", open_circuits)
                arena.circuit_breaker.reset()
