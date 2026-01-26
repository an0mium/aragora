"""
Grok agent for xAI's Grok API.
"""

from aragora.agents.api_agents.base import APIAgent
from aragora.core_types import AgentRole
from aragora.agents.api_agents.common import get_api_key
from aragora.agents.api_agents.openai_compatible import OpenAICompatibleMixin
from aragora.agents.registry import AgentRegistry


@AgentRegistry.register(
    "grok",
    default_model="grok-3",
    agent_type="API",
    env_vars="XAI_API_KEY or GROK_API_KEY",
)
class GrokAgent(OpenAICompatibleMixin, APIAgent):  # type: ignore[misc]
    """Agent that uses xAI's Grok API (OpenAI-compatible).

    Uses the xAI API at https://api.x.ai/v1 with models like grok-3.

    Supports automatic fallback to OpenRouter when xAI API returns
    rate limit/quota errors.

    Uses OpenAICompatibleMixin for standard OpenAI-compatible API implementation.
    """

    OPENROUTER_MODEL_MAP = {
        "grok-4": "x-ai/grok-2-1212",
        "grok-3": "x-ai/grok-2-1212",
        "grok-2": "x-ai/grok-2-1212",
        "grok-2-1212": "x-ai/grok-2-1212",
        "grok-beta": "x-ai/grok-beta",
    }
    DEFAULT_FALLBACK_MODEL = "x-ai/grok-2-1212"

    def __init__(
        self,
        name: str = "grok",
        model: str = "grok-4",
        role: AgentRole = "proposer",
        timeout: int = 120,
        api_key: str | None = None,
        enable_fallback: bool | None = None,  # None = use config setting
    ) -> None:
        super().__init__(
            name=name,
            model=model,
            role=role,
            timeout=timeout,
            api_key=api_key or get_api_key("XAI_API_KEY", "GROK_API_KEY"),
            base_url="https://api.x.ai/v1",
        )
        self.agent_type = "grok"
        # Use config setting if not explicitly provided
        if enable_fallback is None:
            from aragora.agents.fallback import get_default_fallback_enabled

            self.enable_fallback = get_default_fallback_enabled()
        else:
            self.enable_fallback = enable_fallback
        self._fallback_agent = None


__all__ = ["GrokAgent"]
