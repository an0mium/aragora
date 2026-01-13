"""
Agent Registry - Factory pattern for agent creation.

Replaces the 18+ if/elif branches in create_agent() with a
registration-based approach that's extensible and testable.

Includes LRU caching of agent instances to prevent repeated instantiation.

IMPORTANT: Cached agents must be stateless. If an agent mutates its internal
state (stance, system prompt, memory, conversation history), the cache should
be disabled by passing use_cache=False to create_agent(). Alternatively,
agents should implement a reset() method to clear state between uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, Union

from aragora.config import ALLOWED_AGENT_TYPES
from aragora.agents.types import T

try:
    from aragora.agents.local_llm_detector import LocalLLMDetector
except (ImportError, ModuleNotFoundError):
    LocalLLMDetector = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from aragora.agents.api_agents import APIAgent
    from aragora.agents.cli_agents import CLIAgent

    Agent = Union[APIAgent, CLIAgent]


# Module-level cache for agent instances
# Key: (model_type, name, role, model, api_key) -> Agent
_agent_cache: dict[tuple[str, str, str, str | None, str | None], "Agent"] = {}
_CACHE_MAX_SIZE = 32


def _run_async_in_thread(coro):
    """Run an async coroutine in a thread-safe manner.

    Creates a new event loop for the thread to avoid RuntimeError when
    asyncio.run() is called from within a ThreadPoolExecutor.
    """
    import asyncio

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


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
        use_cache: bool = False,
        **kwargs: Any,
    ) -> "Agent":
        """
        Create an agent by registered type name.

        Args:
            model_type: Registered agent type
            name: Agent instance name
            role: Agent role ("proposer", "critic", "synthesizer")
            model: Model to use (overrides default)
            api_key: API key for API-based agents
            use_cache: If True, return cached instance if available
            **kwargs: Additional arguments passed to agent constructor

        Returns:
            Agent instance

        Raises:
            ValueError: If model_type is not registered
        """
        if model_type not in cls._registry:
            valid_types = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(f"Unknown agent type: {model_type}. " f"Valid types: {valid_types}")

        spec = cls._registry[model_type]
        resolved_name = name or spec.default_name
        resolved_model = model or spec.default_model

        # Check cache if enabled (only for simple cases without kwargs)
        if use_cache and not kwargs:
            cache_key = (model_type, resolved_name, role, resolved_model, api_key)
            if cache_key in _agent_cache:
                return _agent_cache[cache_key]

        # Build constructor arguments
        ctor_args: dict[str, Any] = {
            "name": resolved_name,
            "role": role,
            **kwargs,
        }

        # Add model if the agent accepts it
        if spec.default_model is not None or model is not None:
            ctor_args["model"] = resolved_model

        # Add api_key if applicable
        if spec.accepts_api_key and api_key is not None:
            ctor_args["api_key"] = api_key

        agent = spec.agent_class(**ctor_args)

        # Store in cache if enabled
        if use_cache and not kwargs:
            cache_key = (model_type, resolved_name, role, resolved_model, api_key)
            # Evict oldest if at capacity
            if len(_agent_cache) >= _CACHE_MAX_SIZE:
                oldest_key = next(iter(_agent_cache))
                del _agent_cache[oldest_key]
            _agent_cache[cache_key] = agent

        return agent

    @classmethod
    def get_cached(
        cls,
        model_type: str,
        name: str | None = None,
        role: str = "proposer",
        model: str | None = None,
        api_key: str | None = None,
    ) -> "Agent":
        """
        Get or create a cached agent instance.

        This is a convenience method that always uses caching.
        Use this for agents that don't need custom kwargs.

        Args:
            model_type: Registered agent type
            name: Agent instance name
            role: Agent role
            model: Model to use
            api_key: API key for API-based agents

        Returns:
            Cached or newly created agent instance
        """
        return cls.create(
            model_type=model_type,
            name=name,
            role=role,
            model=model,
            api_key=api_key,
            use_cache=True,
        )

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
        """Clear all registrations and cache (for testing)."""
        cls._registry.clear()
        cls.clear_cache()

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the agent instance cache."""
        _agent_cache.clear()

    @classmethod
    def cache_stats(cls) -> dict[str, Any]:
        """Get cache statistics for monitoring.

        Note: API keys are masked in the output to prevent secret leakage.
        """

        def _mask_key(cache_key: tuple) -> tuple:
            """Mask API key (5th element) in cache key tuple."""
            # Cache key: (model_type, name, role, model, api_key)
            if len(cache_key) >= 5 and cache_key[4]:
                masked = cache_key[4][:8] + "..." if len(cache_key[4]) > 8 else "***"
                return (*cache_key[:4], masked)
            return cache_key

        return {
            "size": len(_agent_cache),
            "max_size": _CACHE_MAX_SIZE,
            "keys": [_mask_key(k) for k in _agent_cache.keys()],
        }

    @classmethod
    def validate_allowed(cls, model_type: str) -> bool:
        """
        Check if agent type is in the allowed list.

        Uses ALLOWED_AGENT_TYPES from config for security validation.
        """
        return model_type in ALLOWED_AGENT_TYPES

    @classmethod
    def detect_local_agents(cls) -> list[dict]:
        """
        Detect available local LLM servers.

        Probes Ollama, LM Studio, and other OpenAI-compatible servers
        to find available local models.

        Returns:
            List of dicts with server info:
            [{"name": "ollama", "models": ["llama3.1", ...], "available": True}, ...]
        """
        import asyncio

        if LocalLLMDetector is None:
            return []

        detector = LocalLLMDetector()

        # Run async detection
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already in async context - create new event loop in thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(_run_async_in_thread, detector.detect_all())
                status = future.result(timeout=10)
        else:
            status = asyncio.run(detector.detect_all())

        return [
            {
                "name": server.name,
                "base_url": server.base_url,
                "models": server.models,
                "available": server.available,
                "default_model": server.default_model,
                "version": server.version,
            }
            for server in status.servers
        ]

    @classmethod
    def get_local_status(cls) -> dict:
        """
        Get overall local LLM status with recommendations.

        Returns:
            Dict with availability status and recommendations
        """
        import asyncio

        if LocalLLMDetector is None:
            return {
                "any_available": False,
                "total_models": 0,
                "recommended_server": None,
                "recommended_model": None,
                "available_agents": [],
                "servers": [],
            }

        detector = LocalLLMDetector()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(_run_async_in_thread, detector.detect_all())
                status = future.result(timeout=10)
        else:
            status = asyncio.run(detector.detect_all())

        return {
            "any_available": status.any_available,
            "total_models": status.total_models,
            "recommended_server": status.recommended_server,
            "recommended_model": status.recommended_model,
            "available_agents": status.get_available_agents(),
            "servers": [
                {
                    "name": s.name,
                    "base_url": s.base_url,
                    "available": s.available,
                    "models": s.models,
                    "default_model": s.default_model,
                }
                for s in status.servers
            ],
        }


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
        from aragora.agents import demo_agent  # noqa: F401
    except ImportError:
        pass

    try:
        from aragora.agents import api_agents  # noqa: F401
    except ImportError:
        pass
