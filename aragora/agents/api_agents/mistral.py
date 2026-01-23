"""
Mistral AI API agent with direct API access.

Uses Mistral's native OpenAI-compatible API at api.mistral.ai.
"""

from aragora.agents.api_agents.base import APIAgent
from aragora.agents.api_agents.common import get_api_key
from aragora.agents.api_agents.openai_compatible import OpenAICompatibleMixin
from aragora.agents.registry import AgentRegistry


@AgentRegistry.register(
    "mistral-api",
    default_model="mistral-large-2512",
    default_name="mistral-api",
    agent_type="API",
    env_vars="MISTRAL_API_KEY",
    accepts_api_key=True,
    description="Mistral AI - direct API access to Mistral Large, Medium, and Small models",
)
class MistralAPIAgent(OpenAICompatibleMixin, APIAgent):  # type: ignore[misc]
    """Agent that uses Mistral AI API directly.

    Mistral provides high-quality models with excellent reasoning capabilities.
    Uses an OpenAI-compatible API format.

    Available models:
    - mistral-large-latest: Most capable, best for complex reasoning
    - mistral-medium-latest: Balanced performance/cost
    - mistral-small-latest: Fast and efficient
    - codestral-latest: Optimized for code generation
    - ministral-8b-latest: Small but capable
    - ministral-3b-latest: Fastest, for simple tasks
    """

    # OpenRouter fallback mapping (in case direct API fails)
    OPENROUTER_MODEL_MAP = {
        "mistral-large-2512": "mistralai/mistral-large-2411",  # Mistral Large 3
        "mistral-large-latest": "mistralai/mistral-large-2411",
        "mistral-large-2411": "mistralai/mistral-large-2411",
        "mistral-medium-latest": "mistralai/mistral-medium",
        "mistral-small-latest": "mistralai/mistral-small",
        "codestral-latest": "mistralai/codestral-2501",
        "ministral-8b-latest": "mistralai/ministral-8b",
        "ministral-3b-latest": "mistralai/ministral-3b",
    }
    DEFAULT_FALLBACK_MODEL = "mistralai/mistral-large-2411"

    def __init__(
        self,
        name: str = "mistral-api",
        model: str = "mistral-large-2512",
        role: str = "proposer",
        timeout: int = 180,  # Increased from 60s - allow more time for complex responses
        api_key: str | None = None,
        enable_fallback: bool | None = None,  # None = use config setting
        circuit_breaker_threshold: int = 5,  # Increased from 3 - less aggressive fallback
    ):
        super().__init__(
            name=name,
            model=model,
            role=role,
            timeout=timeout,
            api_key=api_key or get_api_key("MISTRAL_API_KEY"),
            base_url="https://api.mistral.ai/v1",
            circuit_breaker_threshold=circuit_breaker_threshold,
            circuit_breaker_cooldown=90.0,  # Standard cooldown (was 60s)
        )
        self.agent_type = "mistral"
        # Use config setting if not explicitly provided
        if enable_fallback is None:
            from aragora.agents.fallback import get_default_fallback_enabled

            self.enable_fallback = get_default_fallback_enabled()
        else:
            self.enable_fallback = enable_fallback
        self._fallback_agent = None


@AgentRegistry.register(
    "codestral",
    default_model="codestral-latest",
    default_name="codestral",
    agent_type="API",
    env_vars="MISTRAL_API_KEY",
    accepts_api_key=True,
    description="Codestral - Mistral's code-specialized model for programming tasks",
)
class CodestralAgent(MistralAPIAgent):
    """Codestral via Mistral API - specialized for code generation and analysis."""

    def __init__(
        self,
        name: str = "codestral",
        model: str = "codestral-latest",
        role: str = "proposer",
        timeout: int = 120,
        api_key: str | None = None,
    ):
        super().__init__(
            name=name,
            model=model,
            role=role,
            timeout=timeout,
            api_key=api_key,
            # Use config-based default (same as MistralAPIAgent)
        )
        self.agent_type = "codestral"


__all__ = ["MistralAPIAgent", "CodestralAgent"]
