"""
Debate Service - High-level API for running debates.

Provides a simplified interface for common debate operations while
maintaining full flexibility through optional configuration.

Usage:
    from aragora.debate.service import DebateService, get_debate_service

    # Quick debate with defaults
    service = get_debate_service()
    result = await service.run("What is the best testing strategy?")

    # With custom agents and options
    result = await service.run(
        task="Design a rate limiter",
        agents=["claude", "gemini"],
        rounds=3,
        consensus="supermajority",
    )

    # Full configuration
    result = await service.run(
        task="Security audit",
        agents=custom_agents,
        protocol=custom_protocol,
        memory=memory_system,
        timeout=300,
    )
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Union

from aragora.core import Agent, DebateResult, Environment
from aragora.debate.protocol import DebateProtocol

if TYPE_CHECKING:
    from aragora.memory.continuum import ContinuumMemory

logger = logging.getLogger(__name__)


@dataclass
class DebateOptions:
    """Configuration options for a debate.

    All fields are optional with sensible defaults.
    """

    # Protocol options
    rounds: int = 3
    consensus: Literal[
        "majority", "unanimous", "judge", "none", "weighted", "supermajority", "any", "byzantine"
    ] = "majority"
    topology: Literal["all-to-all", "sparse", "round-robin", "ring", "star", "random-graph"] = (
        "all-to-all"
    )
    enable_graph: bool = False

    # Execution options
    timeout: float = 300.0  # 5 minutes default
    enable_streaming: bool = False
    enable_checkpointing: bool = True  # Enable by default for debate resume support

    # Memory options
    enable_memory: bool = True
    enable_knowledge_retrieval: bool = False

    # ML options
    enable_ml_delegation: bool = False
    enable_quality_gates: bool = False
    enable_consensus_estimation: bool = False

    # Telemetry
    org_id: str = ""
    user_id: str = ""
    correlation_id: str = ""

    # Event hooks
    on_round_start: Optional[Callable[[int], None]] = None
    on_agent_message: Optional[Callable[[str, str], None]] = None
    on_consensus: Optional[Callable[[str, float], None]] = None

    def to_protocol(self) -> DebateProtocol:
        """Convert options to a DebateProtocol."""
        return DebateProtocol(  # type: ignore[call-arg,arg-type]
            rounds=self.rounds,
            consensus=self.consensus,
            topology=self.topology,
            enable_graph=self.enable_graph,
        )


class DebateService:
    """High-level service for running debates.

    Provides a simplified API for common debate operations while
    maintaining full flexibility through optional configuration.

    The service handles:
    - Agent resolution (names to Agent objects)
    - Protocol configuration
    - Arena construction
    - Timeout management
    - Error handling

    Example:
        service = DebateService()

        # Simple debate
        result = await service.run("What is the best approach?")

        # With options
        result = await service.run(
            task="Design a cache",
            agents=["claude", "gemini"],
            options=DebateOptions(rounds=5, consensus="supermajority"),
        )
    """

    def __init__(
        self,
        default_agents: Optional[list[Agent]] = None,
        default_options: Optional[DebateOptions] = None,
        memory: Optional["ContinuumMemory"] = None,
        agent_resolver: Optional[Callable[[str], Agent]] = None,
    ):
        """Initialize the debate service.

        Args:
            default_agents: Default agents to use if none specified
            default_options: Default options for all debates
            memory: Shared memory system for debates
            agent_resolver: Function to resolve agent names to Agent objects
        """
        self._default_agents = default_agents
        self._default_options = default_options or DebateOptions()
        self._memory = memory
        self._agent_resolver = agent_resolver

    async def run(
        self,
        task: str,
        agents: Optional[Union[list[Agent], list[str]]] = None,
        protocol: Optional[DebateProtocol] = None,
        options: Optional[DebateOptions] = None,
        memory: Optional["ContinuumMemory"] = None,
        **kwargs: Any,
    ) -> DebateResult:
        """Run a debate on the given task.

        Args:
            task: The topic or question to debate
            agents: List of Agent objects or agent names to resolve
            protocol: Custom protocol (overrides options)
            options: Debate options (merged with defaults)
            memory: Memory system (overrides service default)
            **kwargs: Additional Arena constructor arguments

        Returns:
            DebateResult with consensus, synthesis, and metadata

        Raises:
            ValueError: If no agents available
            asyncio.TimeoutError: If debate exceeds timeout
        """
        # Merge options with defaults
        opts = self._merge_options(options)

        # Resolve agents
        resolved_agents = self._resolve_agents(agents)
        if not resolved_agents:
            raise ValueError("No agents available. Provide agents or configure default_agents.")

        # Create environment
        env = Environment(task=task)

        # Create or use provided protocol
        debate_protocol = protocol or opts.to_protocol()

        # Get memory system
        debate_memory = memory or self._memory

        # Build Arena kwargs
        arena_kwargs: dict[str, Any] = {
            "enable_checkpointing": opts.enable_checkpointing,
            "enable_knowledge_retrieval": opts.enable_knowledge_retrieval,
            "enable_ml_delegation": opts.enable_ml_delegation,
            "enable_quality_gates": opts.enable_quality_gates,
            "enable_consensus_estimation": opts.enable_consensus_estimation,
        }

        # Add telemetry if provided
        if opts.org_id:
            arena_kwargs["org_id"] = opts.org_id
        if opts.user_id:
            arena_kwargs["user_id"] = opts.user_id

        # Add event hooks if provided
        event_hooks = self._build_event_hooks(opts)
        if event_hooks:
            arena_kwargs["event_hooks"] = event_hooks

        # Merge with any additional kwargs
        arena_kwargs.update(kwargs)

        # Import Arena here to avoid circular imports
        from aragora.debate.orchestrator import Arena

        # Create Arena
        arena = Arena(
            env,
            resolved_agents,
            debate_protocol,
            memory=debate_memory,
            **arena_kwargs,
        )

        # Run with timeout
        try:
            result = await asyncio.wait_for(arena.run(), timeout=opts.timeout)
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Debate timed out after {opts.timeout}s: {task[:50]}...")
            raise

    async def run_quick(
        self,
        task: str,
        rounds: int = 2,
        agents: Optional[Union[list[Agent], list[str]]] = None,
    ) -> DebateResult:
        """Run a quick debate with minimal configuration.

        Convenience method for simple debates.

        Args:
            task: The topic to debate
            rounds: Number of rounds (default 2)
            agents: Optional agents (uses defaults if not provided)

        Returns:
            DebateResult
        """
        return await self.run(
            task=task,
            agents=agents,
            options=DebateOptions(rounds=rounds),
        )

    async def run_deep(
        self,
        task: str,
        agents: Optional[Union[list[Agent], list[str]]] = None,
        rounds: int = 5,
    ) -> DebateResult:
        """Run a thorough debate with more rounds and stricter consensus.

        Convenience method for important decisions.

        Args:
            task: The topic to debate
            agents: Optional agents (uses defaults if not provided)
            rounds: Number of rounds (default 5)

        Returns:
            DebateResult
        """
        return await self.run(
            task=task,
            agents=agents,
            options=DebateOptions(
                rounds=rounds,
                consensus="supermajority",
                enable_quality_gates=True,
            ),
        )

    def _merge_options(self, options: Optional[DebateOptions]) -> DebateOptions:
        """Merge provided options with defaults."""
        if options is None:
            return self._default_options

        # Create new options with defaults, then override with provided values
        merged = DebateOptions(
            rounds=options.rounds or self._default_options.rounds,
            consensus=options.consensus or self._default_options.consensus,
            topology=options.topology or self._default_options.topology,
            enable_graph=options.enable_graph,
            timeout=options.timeout or self._default_options.timeout,
            enable_streaming=options.enable_streaming,
            enable_checkpointing=options.enable_checkpointing,
            enable_memory=options.enable_memory,
            enable_knowledge_retrieval=options.enable_knowledge_retrieval,
            enable_ml_delegation=options.enable_ml_delegation,
            enable_quality_gates=options.enable_quality_gates,
            enable_consensus_estimation=options.enable_consensus_estimation,
            org_id=options.org_id or self._default_options.org_id,
            user_id=options.user_id or self._default_options.user_id,
            correlation_id=options.correlation_id or self._default_options.correlation_id,
            on_round_start=options.on_round_start or self._default_options.on_round_start,
            on_agent_message=options.on_agent_message or self._default_options.on_agent_message,
            on_consensus=options.on_consensus or self._default_options.on_consensus,
        )
        return merged

    def _resolve_agents(self, agents: Optional[Union[list[Agent], list[str]]]) -> list[Agent]:
        """Resolve agent specifications to Agent objects."""
        if agents is None:
            return self._default_agents or []

        resolved: list[Agent] = []
        for agent in agents:
            if isinstance(agent, Agent):
                resolved.append(agent)
            elif isinstance(agent, str) and self._agent_resolver:
                try:
                    resolved.append(self._agent_resolver(agent))
                except (KeyError, ValueError) as e:
                    logger.warning(f"Failed to resolve agent '{agent}': {e}")
                except Exception as e:
                    logger.exception(f"Unexpected error resolving agent '{agent}': {e}")
            elif isinstance(agent, str):
                # Try to create a basic agent from the name
                try:
                    from aragora.agents import create_agent

                    resolved.append(create_agent(agent))  # type: ignore[arg-type]
                except ImportError:
                    logger.warning(f"Cannot resolve agent '{agent}' - no resolver configured")

        return resolved

    def _build_event_hooks(self, opts: DebateOptions) -> Optional[dict[str, Any]]:
        """Build event hooks dictionary from options."""
        hooks: dict[str, Any] = {}

        if opts.on_round_start:
            hooks["round_start"] = opts.on_round_start
        if opts.on_agent_message:
            hooks["agent_message"] = opts.on_agent_message
        if opts.on_consensus:
            hooks["consensus"] = opts.on_consensus

        return hooks if hooks else None


# Global service instance
_debate_service: Optional[DebateService] = None


def get_debate_service(
    default_agents: Optional[list[Agent]] = None,
    **kwargs: Any,
) -> DebateService:
    """Get the global debate service instance.

    Creates a new instance on first call or when default_agents is provided.

    Args:
        default_agents: Default agents for the service
        **kwargs: Additional DebateService constructor arguments

    Returns:
        DebateService instance
    """
    global _debate_service

    if _debate_service is None or default_agents is not None:
        _debate_service = DebateService(default_agents=default_agents, **kwargs)

    return _debate_service


def reset_debate_service() -> None:
    """Reset the global debate service instance.

    Useful for testing or reconfiguration.
    """
    global _debate_service
    _debate_service = None


__all__ = [
    "DebateService",
    "DebateOptions",
    "get_debate_service",
    "reset_debate_service",
]
