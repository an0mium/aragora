"""
Mistral AI API agent with direct API access.

Uses Mistral's native OpenAI-compatible API at api.mistral.ai.
"""

from aragora.agents.api_agents.base import APIAgent
from aragora.core_types import AgentRole
from aragora.agents.api_agents.common import get_primary_api_key, upgrade_retired_model_id
from aragora.agents.api_agents.openai_compatible import OpenAICompatibleMixin
from aragora.agents.registry import AgentRegistry
from aragora.config.model_pins import MISTRAL_MEDIUM_DIRECT, MISTRAL_MEDIUM_VIA_OPENROUTER
from typing import ClassVar

# Frontier pick for the Mistral API agent (2026-09-04 frontier-model-refresh).
DEFAULT_MODEL = MISTRAL_MEDIUM_DIRECT


@AgentRegistry.register(
    "mistral-api",
    default_model=DEFAULT_MODEL,
    default_name="mistral-api",
    agent_type="API",
    env_vars="MISTRAL_API_KEY",
    accepts_api_key=True,
    description="Mistral AI - direct API access to Mistral Large, Medium, and Small models",
)
class MistralAPIAgent(OpenAICompatibleMixin, APIAgent):
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

    # No static OPENROUTER_MODEL_MAP: QuotaFallbackMixin.get_fallback_model()
    # (aragora/agents/fallback.py) resolves the current model through the
    # catalog/upgrade-map instead, so every legacy or retired Mistral
    # spelling (not just a hand-enumerated subset) transparently upgrades
    # to its frontier via OpenRouter.
    DEFAULT_FALLBACK_MODEL = MISTRAL_MEDIUM_VIA_OPENROUTER

    # Upgrade a retired/known-dead explicit model id at construction time
    # (finding O-P2a). CodestralAgent turns this OFF -- see the comment on
    # that subclass.
    UPGRADE_RETIRED_MODEL_ID: ClassVar[bool] = True

    def __init__(
        self,
        name: str = "mistral-api",
        model: str = DEFAULT_MODEL,
        role: AgentRole = "proposer",
        timeout: int = 180,  # Increased from 60s - allow more time for complex responses
        api_key: str | None = None,
        enable_fallback: bool | None = None,  # None = use config setting
        circuit_breaker_threshold: int = 5,  # Increased from 3 - less aggressive fallback
    ) -> None:
        # A retired or known-dead explicit id is upgraded before it can be
        # sent to the native endpoint (finding O-P2a); active and unknown
        # ids pass through untouched. See upgrade_retired_model_id.
        if self.UPGRADE_RETIRED_MODEL_ID:
            model = upgrade_retired_model_id(model)
        super().__init__(
            name=name,
            model=model,
            role=role,
            timeout=timeout,
            api_key=api_key
            or get_primary_api_key("MISTRAL_API_KEY", allow_openrouter_fallback=True),
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

    # Codestral is a LIVE, code-specialized SKU on the native Mistral
    # endpoint, so its id must reach that endpoint verbatim. It is an
    # UPGRADES key only because the catalog carries no Codestral row, which
    # makes ``mistral-medium-2604`` the right OPENROUTER FALLBACK target for
    # it (aragora/models/upgrade_map.py records exactly that reasoning) --
    # not a reason to retarget the primary call away from the code model the
    # caller asked for. Finding O-P2a is about ids that are DEAD on the wire;
    # this one is not.
    UPGRADE_RETIRED_MODEL_ID: ClassVar[bool] = False

    def __init__(
        self,
        name: str = "codestral",
        model: str = "codestral-latest",
        role: AgentRole = "proposer",
        timeout: int = 120,
        api_key: str | None = None,
    ) -> None:
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
