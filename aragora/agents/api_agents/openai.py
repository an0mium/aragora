"""
OpenAI API agent with OpenRouter fallback support.

Supports web search tool for web-capable responses when URLs
or web-related keywords are detected in the prompt.
"""

import asyncio
import logging
import re
import time

from aragora.agents.api_agents.base import APIAgent
from aragora.agents.api_agents.common import (
    AgentAPIError,
    AgentCircuitOpenError,
    get_primary_api_key,
)
from aragora.agents.api_agents.openai_compatible import OpenAICompatibleMixin
from aragora.agents.registry import AgentRegistry
from aragora.agents.transports.vibeproxy import (
    ModelTransportPolicy,
    OpenAIProtocol,
    TransportMode,
    VibeProxyConfigurationError,
    VibeProxyTimeoutError,
    VibeProxyUnavailableError,
)
from aragora.core import Message
from aragora.core_types import AgentRole
from aragora.observability.metrics.agents import (
    record_circuit_breaker_rejection,
    record_provider_call,
    record_provider_token_usage,
)

logger = logging.getLogger(__name__)

# Pre-compiled patterns that indicate web search would be helpful
# Compiled at module load time for performance (avoids recompilation on each call)
_WEB_SEARCH_PATTERNS = [
    re.compile(r"https?://", re.IGNORECASE),  # URLs
    re.compile(r"github\.com", re.IGNORECASE),  # GitHub repos
    re.compile(r"\brepo\b", re.IGNORECASE),  # Repository mentions
    re.compile(r"\bwebsite\b", re.IGNORECASE),  # Website mentions
    re.compile(r"\bweb\s*page\b", re.IGNORECASE),  # Web page mentions
    re.compile(r"\bonline\b", re.IGNORECASE),  # Online content
    re.compile(r"\blatest\s+(news|updates?|release|releases|version|versions)\b", re.IGNORECASE),
    re.compile(r"\bcurrent\s+(events|status|market|prices?|pricing)\b", re.IGNORECASE),
    re.compile(r"\brecent\s+(news|developments|changes|updates?|articles?)\b", re.IGNORECASE),
    re.compile(r"\bnews\b", re.IGNORECASE),  # News
    re.compile(r"\barticle\b", re.IGNORECASE),  # Articles
]


def _resolve_openai_base_url() -> str:
    """OPENAI_BASE_URL override for gateways/proxies (issue #9304)."""
    import os

    raw = os.environ.get("OPENAI_BASE_URL", "").strip().rstrip("/")
    if not raw:
        return "https://api.openai.com/v1"
    return raw if raw.endswith("/v1") else raw + "/v1"


@AgentRegistry.register(
    "openai-api",
    default_model="gpt-5.5",
    default_name="openai-api",
    agent_type="API",
    env_vars="OPENAI_API_KEY",
    accepts_api_key=True,
)
class OpenAIAPIAgent(OpenAICompatibleMixin, APIAgent):
    """Agent that uses OpenAI API directly.

    Includes automatic fallback to OpenRouter when OpenAI quota is exceeded (429 error).
    The fallback uses the same GPT model via OpenRouter's API.

    Supports web search tool for web-capable responses when URLs or web-related
    keywords are detected in the prompt.

    Uses OpenAICompatibleMixin for standard OpenAI API implementation.
    """

    # Every OpenAI ID maps to the current frontier (GPT-5.5) via OpenRouter
    # so weaker historical models are transparently upgraded and a missing
    # OPENAI_API_KEY never blocks a debate. Distinct OpenRouter model IDs
    # are kept only where the Pro tier is explicitly requested.
    OPENROUTER_MODEL_MAP = {
        "gpt-5.5": "openai/gpt-5.5",
        "gpt-5.4": "openai/gpt-5.5",
        "gpt-5.4-pro": "openai/gpt-5.5",
        "gpt-5.3": "openai/gpt-5.5",
        "gpt-5.3-chat-latest": "openai/gpt-5.5",
        "gpt-5.3-codex": "openai/gpt-5.5",
        "gpt-4.1": "openai/gpt-5.5",
        "gpt-4.1-mini": "openai/gpt-5.5",
        "gpt-4.1-nano": "openai/gpt-5.5",
        "gpt-4o": "openai/gpt-5.5",
        "gpt-4o-mini": "openai/gpt-5.5",
        "gpt-4-turbo": "openai/gpt-5.5",
        "gpt-4": "openai/gpt-5.5",
        "gpt-3.5-turbo": "openai/gpt-5.5",
        "gpt-4o-search-preview": "openai/gpt-5.5",
        "o3": "openai/gpt-5.5",
        "o3-mini": "openai/gpt-5.5",
        "o4-mini": "openai/gpt-5.5",
    }
    DEFAULT_FALLBACK_MODEL = "openai/gpt-5.5"

    def __init__(
        self,
        name: str = "openai-api",
        model: str = "gpt-5.5",
        role: AgentRole = "proposer",
        timeout: int = 120,
        api_key: str | None = None,
        enable_fallback: bool | None = None,  # None = use config setting
    ) -> None:
        import os

        self._uses_official_openai_endpoint = not bool(
            os.environ.get("OPENAI_BASE_URL", "").strip()
        )
        super().__init__(
            name=name,
            model=model,
            role=role,
            timeout=timeout,
            api_key=api_key
            or get_primary_api_key("OPENAI_API_KEY", allow_openrouter_fallback=True),
            # OPENAI_BASE_URL supports BYOK gateways/proxies (issue #9304).
            base_url=_resolve_openai_base_url(),
        )
        self.agent_type = "openai"
        # Use config setting if not explicitly provided
        if enable_fallback is None:
            from aragora.agents.fallback import get_default_fallback_enabled

            self.enable_fallback = get_default_fallback_enabled()
        else:
            self.enable_fallback = enable_fallback
        self._fallback_agent = None
        self.enable_web_search = True  # Enable web search tool by default
        self._current_prompt = ""  # Track current prompt for web search detection
        self._model_transport_policy = ModelTransportPolicy.from_env(
            default_mode=TransportMode.DIRECT
        )

    def _needs_web_search(self, prompt: str) -> bool:
        """Detect if the prompt would benefit from web search.

        Returns True if the prompt contains URLs, GitHub references,
        or keywords indicating need for current/web information.
        """
        if not self.enable_web_search:
            return False

        # Use pre-compiled patterns for performance
        for pattern in _WEB_SEARCH_PATTERNS:
            if pattern.search(prompt):
                return True
        return False

    def _build_messages(self, full_prompt: str) -> list[dict]:
        """Build messages and track prompt for web search detection."""
        # Store prompt for _build_extra_payload to use
        self._current_prompt = full_prompt
        return super()._build_messages(full_prompt)

    def _build_extra_payload(self) -> dict | None:
        """Add web search tool if prompt indicates web content is needed."""
        if self._needs_web_search(self._current_prompt):
            logger.info("[%s] Enabling web search tool for web content", self.name)
            return {
                "tools": [
                    {
                        "type": "web_search",
                        "web_search": {},
                    }
                ]
            }
        return None

    def _can_route_exact_chat(self, full_prompt: str) -> bool:
        """Return whether this request is inside the contract-tested proxy slice."""

        return (
            self._model_transport_policy.mode is not TransportMode.DIRECT
            and self._uses_official_openai_endpoint
            and not self._needs_web_search(full_prompt)
        )

    async def generate(self, prompt: str, context: list[Message] | None = None) -> str:
        """Route exact, non-streaming OpenAI Chat requests through VibeProxy.

        Web search, streaming, custom endpoints, and any request the policy does
        not resolve exactly continue through the established direct path.
        """

        full_prompt = prompt
        if context:
            full_prompt = self._build_context_prompt(context) + prompt
        if not self._can_route_exact_chat(full_prompt):
            return await super().generate(prompt, context)

        cb = getattr(self, "_circuit_breaker", None)
        if cb is not None and not cb.can_proceed():
            record_circuit_breaker_rejection(self.agent_type)
            raise AgentCircuitOpenError(
                f"Circuit breaker open for {self.name} - too many recent failures",
                agent_name=self.name,
            )

        messages = self._build_messages(full_prompt)
        payload = self._build_payload(messages, stream=False)
        # Defensively keep future tool-bearing extensions on the direct path.
        if payload.get("tools"):
            return await super().generate(prompt, context)

        estimated_budget_usd = self._estimate_budget_cost_usd(payload)
        from aragora.billing import budget_guard

        budget_guard.assert_within_budget(
            estimated_budget_usd,
            label=getattr(self, "name", None),
        )

        deadline = time.monotonic() + float(self.timeout)

        def remaining_timeout() -> float:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise VibeProxyTimeoutError("VibeProxy OpenAI request timed out")
            return remaining

        try:
            route = await asyncio.to_thread(
                self._model_transport_policy.resolve,
                "openai",
                self.model,
                ("chat",),
                timeout=remaining_timeout(),
            )
            if route.transport == "direct":
                return await super().generate(prompt, context)

            client = self._model_transport_policy.client
            if client is None:
                raise VibeProxyUnavailableError("VibeProxy client is not configured")
            proxy_payload = dict(payload)
            proxy_payload["model"] = route.resolved_model
            data = await asyncio.to_thread(
                client.openai_request,
                protocol=OpenAIProtocol.CHAT,
                model=route.resolved_model,
                payload=proxy_payload,
                timeout=remaining_timeout(),
            )
        except (VibeProxyConfigurationError, VibeProxyUnavailableError) as exc:
            if self._model_transport_policy.mode is TransportMode.PREFER:
                logger.info(
                    "[%s] VibeProxy OpenAI request unavailable; using direct path: %s",
                    self.name,
                    exc,
                )
                return await super().generate(prompt, context)
            raise AgentAPIError(
                f"required VibeProxy OpenAI request failed: {exc}",
                agent_name=self.name,
            ) from exc

        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0) if isinstance(usage, dict) else 0
        output_tokens = usage.get("completion_tokens", 0) if isinstance(usage, dict) else 0
        if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
            input_tokens = 0
            output_tokens = 0
        usage_has_tokens = bool(input_tokens or output_tokens)
        self._record_token_usage(tokens_in=input_tokens, tokens_out=output_tokens)

        content = self._parse_response(data)
        if not content or not content.strip():
            raise AgentAPIError(
                f"{self._get_error_prefix()} returned empty response",
                agent_name=self.name,
            )
        if not usage_has_tokens and estimated_budget_usd > 0:
            budget_guard.record_spend(estimated_budget_usd)
        if cb is not None:
            cb.record_success()
        record_provider_call(provider=self.agent_type, success=True, model=self.model)
        record_provider_token_usage(
            provider=self.agent_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        return content


__all__ = ["OpenAIAPIAgent"]
