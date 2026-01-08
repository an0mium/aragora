"""
Agent Registry - Factory pattern for agent creation.

Replaces the 18+ if/elif branches in create_agent() with a
registration-based approach that's extensible and testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from aragora.config import ALLOWED_AGENT_TYPES
from aragora.agents.types import T


@dataclass(frozen=True)
class AgentSpec:
    """Specification for a registered agent type."""

    name: str
    agent_class: type
    default_model: str | None
    default_name: str
    agent_type: str  # "CLI", "API", "API (OpenRouter)"
    requires: str | None
    env_vars: str | None
    description: str | None = None
    accepts_api_key: bool = False


class AgentRegistry:
    """
    Factory registry for agent creation.

    Usage:
        # Registration (done in agent modules)
        @AgentRegistry.register(
            "claude",
            default_model="claude-sonnet-4",
            agent_type="CLI",
            requires="claude CLI (npm install -g @anthropic-ai/claude-code)",
        )
        class ClaudeAgent(BaseCliAgent):
            ...

        # Creation
        agent = AgentRegistry.create("claude", name="claude-1", role="proposer")

        # Listing
        available = AgentRegistry.list_all()
    """

    _registry: dict[str, AgentSpec] = {}

    @classmethod
    def register(
        cls,
        type_name: str,
        *,
        default_model: str | None = None,
        default_name: str | None = None,
        agent_type: str = "API",
        requires: str | None = None,
        env_vars: str | None = None,
        description: str | None = None,
        accepts_api_key: bool = False,
    ) -> Callable[[type[T]], type[T]]:
        """
        Decorator to register an agent class.

        Args:
            type_name: The agent type identifier (e.g., "claude", "gemini")
            default_model: Default model string if not specified
            default_name: Default agent name if not specified (defaults to type_name)
            agent_type: Category ("CLI", "API", "API (OpenRouter)")
            requires: External dependency description
            env_vars: Required environment variables
            description: Human-readable description
            accepts_api_key: Whether create() should pass api_key

        Returns:
            Decorator function
        """
        def decorator(agent_cls: type[T]) -> type[T]:
            spec = AgentSpec(
                name=type_name,
                agent_class=agent_cls,
                default_model=default_model,
                default_name=default_name or type_name,
                agent_type=agent_type,
                requires=requires,
                env_vars=env_vars,
                description=description,
                accepts_api_key=accepts_api_key,
            )
            cls._registry[type_name] = spec
            return agent_cls

        return decorator

    @classmethod
    def create(
        cls,
        model_type: str,
        name: str | None = None,
        role: str = "proposer",
        model: str | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ):
        """
        Create an agent by registered type name.

        Args:
            model_type: Registered agent type
            name: Agent instance name
            role: Agent role ("proposer", "critic", "synthesizer")
            model: Model to use (overrides default)
            api_key: API key for API-based agents
            **kwargs: Additional arguments passed to agent constructor

        Returns:
            Agent instance

        Raises:
            ValueError: If model_type is not registered
        """
        if model_type not in cls._registry:
            valid_types = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(
                f"Unknown agent type: {model_type}. "
                f"Valid types: {valid_types}"
            )

        spec = cls._registry[model_type]

        # Build constructor arguments
        ctor_args: dict[str, Any] = {
            "name": name or spec.default_name,
            "role": role,
            **kwargs,
        }

        # Add model if the agent accepts it
        if spec.default_model is not None or model is not None:
            ctor_args["model"] = model or spec.default_model

        # Add api_key if applicable
        if spec.accepts_api_key and api_key is not None:
            ctor_args["api_key"] = api_key

        return spec.agent_class(**ctor_args)

    @classmethod
    def is_registered(cls, model_type: str) -> bool:
        """Check if a model type is registered."""
        return model_type in cls._registry

    @classmethod
    def get_spec(cls, model_type: str) -> AgentSpec | None:
        """Get the spec for a registered agent type."""
        return cls._registry.get(model_type)

    @classmethod
    def list_all(cls) -> dict[str, dict]:
        """
        List all registered agent types with their metadata.

        Returns:
            Dict mapping type names to their specifications.
        """
        return {
            type_name: {
                "type": spec.agent_type,
                "requires": spec.requires,
                "env_vars": spec.env_vars,
                "description": spec.description,
                "default_model": spec.default_model,
            }
            for type_name, spec in cls._registry.items()
        }

    @classmethod
    def get_registered_types(cls) -> list[str]:
        """Get list of all registered type names."""
        return list(cls._registry.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registrations (for testing)."""
        cls._registry.clear()

    @classmethod
    def validate_allowed(cls, model_type: str) -> bool:
        """
        Check if agent type is in the allowed list.

        Uses ALLOWED_AGENT_TYPES from config for security validation.
        """
        return model_type in ALLOWED_AGENT_TYPES


def register_all_agents() -> None:
    """
    Import all agent modules to trigger registration.

    This function should be called once at startup to ensure
    all agents are registered before create() is used.
    """
    # Import modules to trigger @register decorators
    # These imports are intentionally side-effect only
    try:
        from aragora.agents import cli_agents  # noqa: F401
    except ImportError:
        pass

    try:
        from aragora.agents import api_agents  # noqa: F401
    except ImportError:
        pass
