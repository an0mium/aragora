"""Factory methods for Arena construction.

Extracted from orchestrator.py to reduce its size. Contains the three
classmethod factories: from_config, from_configs, and create.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aragora.core_types import Agent
    from aragora.core import Environment
    from aragora.debate.protocol import DebateProtocol
    from aragora.debate.orchestrator import Arena


def from_config(
    cls: type[Arena],
    environment: Environment,
    agents: list[Agent],
    protocol: DebateProtocol | None = None,
    config: Any | None = None,
) -> Arena:
    """Create an Arena from an ArenaConfig for cleaner dependency injection.

    Args:
        cls: The Arena class (passed from the classmethod).
        environment: The debate environment (task, context).
        agents: List of participating agents.
        protocol: Optional debate protocol override.
        config: ArenaConfig instance.

    Returns:
        A fully initialized Arena instance.
    """
    from aragora.debate.arena_config import ArenaConfig
    from aragora.debate.feature_validator import validate_and_warn

    config = config or ArenaConfig()
    validate_and_warn(config)
    return cls(
        environment=environment, agents=agents, protocol=protocol, **config.to_arena_kwargs()
    )


def from_configs(
    cls: type[Arena],
    environment: Environment,
    agents: list[Agent],
    protocol: DebateProtocol | None = None,
    **kwargs: Any,
) -> Arena:
    """Create an Arena from grouped config objects.

    Accepts keyword arguments for debate_config, agent_config,
    memory_config, streaming_config, observability_config,
    knowledge_config, supermemory_config, evolution_config, ml_config.

    Args:
        cls: The Arena class (passed from the classmethod).
        environment: The debate environment (task, context).
        agents: List of participating agents.
        protocol: Optional debate protocol override.
        **kwargs: Config objects to pass through.

    Returns:
        A fully initialized Arena instance.
    """
    return cls(
        environment=environment,
        agents=agents,
        protocol=protocol,
        **kwargs,
    )


def create(
    cls: type[Arena],
    environment: Environment,
    agents: list[Agent],
    protocol: DebateProtocol | None = None,
    **kwargs: Any,
) -> Arena:
    """Create an Arena with a clean, consolidated interface.

    This is the recommended entry point for new code. It accepts at most
    10 parameters -- the three positional core args plus up to six optional
    config objects -- and delegates to ``__init__`` after unpacking.

    The ``config`` parameter (ArenaConfig) is a legacy catch-all.  When
    provided alongside the typed config objects, the typed objects win for
    any overlapping fields.

    Example::

        arena = Arena.create(
            environment=env,
            agents=agents,
            debate_config=DebateConfig(rounds=5),
            memory_config=MemoryConfig(enable_supermemory=True),
        )

    Args:
        cls: The Arena class (passed from the classmethod).
        environment: The debate environment (task, context).
        agents: List of participating agents.
        protocol: Optional debate protocol override.
        **kwargs: Accepts ``config`` (legacy ArenaConfig) plus typed config
            objects (debate_config, agent_config, memory_config,
            streaming_config, observability_config).

    Returns:
        A fully initialized Arena instance.
    """
    # Extract the legacy config param if present
    config = kwargs.pop("config", None)

    # Start from ArenaConfig kwargs (if provided) as the base layer
    base_kwargs: dict[str, Any] = {}
    if config is not None:
        base_kwargs = config.to_arena_kwargs()

    # Typed config objects override the flat ArenaConfig values
    for key in (
        "debate_config",
        "agent_config",
        "memory_config",
        "streaming_config",
        "observability_config",
    ):
        if key in kwargs:
            base_kwargs[key] = kwargs.pop(key)

    # Pass through any remaining kwargs
    base_kwargs.update(kwargs)

    return cls(
        environment=environment,
        agents=agents,
        protocol=protocol,
        **base_kwargs,
    )


__all__ = [
    "from_config",
    "from_configs",
    "create",
]
