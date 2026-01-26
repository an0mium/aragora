"""
OpenAI-compatible API agent mixin.

Provides common implementation for agents using OpenAI-compatible APIs:
- OpenAI
- Grok (xAI)
- OpenRouter (and its model variants)
- Any other OpenAI-compatible endpoint

This eliminates ~150 lines of duplicate code per agent.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, AsyncGenerator, Optional

from aragora.core import Critique, Message

if TYPE_CHECKING:
    from collections.abc import Callable


from aragora.agents.api_agents.common import (
    AgentAPIError,
    AgentCircuitOpenError,
    AgentConnectionError,
    AgentRateLimitError,
    AgentStreamError,
    AgentTimeoutError,
    _sanitize_error_message,
    create_client_session,
    create_openai_sse_parser,
    handle_agent_errors,
)
from aragora.agents.fallback import QuotaFallbackMixin

logger = logging.getLogger(__name__)


class OpenAICompatibleMixin(QuotaFallbackMixin):
    """
    Mixin providing OpenAI-compatible API implementation.

    Subclasses must define:
    - OPENROUTER_MODEL_MAP: dict mapping models to OpenRouter equivalents
    - DEFAULT_FALLBACK_MODEL: default OpenRouter model for fallback
    - agent_type: str identifying the agent type
    - base_url: API base URL
    - api_key: API key

    Optional overrides:
    - _build_extra_headers(): Add provider-specific headers
    - _build_extra_payload(): Add provider-specific payload fields
    - max_tokens: Default 4096
    """

    # Subclasses should define these
    OPENROUTER_MODEL_MAP: dict[str, str] = {}
    DEFAULT_FALLBACK_MODEL: str = "openai/gpt-4o"

    # Default max tokens (can be overridden)
    max_tokens: int = 4096

    # Expected from base class (APIAgent) - declared for type checking
    api_key: str | None
    base_url: str | None
    model: str
    name: str
    agent_type: str
    timeout: int

    def _record_token_usage(self, tokens_in: int, tokens_out: int) -> None:
        """Record token usage (delegates to APIAgent base class)."""
        super()._record_token_usage(tokens_in, tokens_out)  # type: ignore[misc]

    # Methods inherited from CritiqueMixin (via APIAgent) - delegate to parent
    def _build_context_prompt(
        self,
        context: Optional[list[Message]] = None,
        truncate: bool = False,
        sanitize_fn: Optional["Callable[[str], str]"] = None,
    ) -> str:
        """Build context from previous messages (delegates to CritiqueMixin)."""
        # Delegate to CritiqueMixin via APIAgent in the MRO
        return super()._build_context_prompt(context, truncate, sanitize_fn)  # type: ignore[misc]

    def _parse_critique(
        self,
        response: str,
        target_agent: str,
        target_content: str,
    ) -> Critique:
        """Parse critique response (delegates to CritiqueMixin)."""
        # Delegate to CritiqueMixin via APIAgent in the MRO
        return super()._parse_critique(response, target_agent, target_content)  # type: ignore[misc]

    def _build_headers(self) -> dict:
        """Build request headers. Override to add provider-specific headers."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        extra = self._build_extra_headers()
        if extra:
            headers.update(extra)
        return headers

    def _build_extra_headers(self) -> dict | None:
        """Override to add provider-specific headers."""
        return None

    def _build_messages(self, full_prompt: str) -> list[dict]:
        """Build messages array with optional system prompt."""
        messages = [{"role": "user", "content": full_prompt}]
        if hasattr(self, "system_prompt") and self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        return messages

    def _build_payload(self, messages: list[dict], stream: bool = False) -> dict:
        """Build request payload. Override to add provider-specific fields."""
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        if stream:
            payload["stream"] = True

        # Apply generation parameters from persona if set (from APIAgent)
        if hasattr(self, "temperature") and self.temperature is not None:
            payload["temperature"] = self.temperature
        if hasattr(self, "top_p") and self.top_p is not None:
            payload["top_p"] = self.top_p
        if hasattr(self, "frequency_penalty") and self.frequency_penalty is not None:
            payload["frequency_penalty"] = self.frequency_penalty

        extra = self._build_extra_payload()
        if extra:
            payload.update(extra)
        return payload

    def _build_extra_payload(self) -> dict | None:
        """Override to add provider-specific payload fields."""
        return None

    def _parse_response(self, data: dict) -> str:
        """Parse response content from API response."""
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            raise AgentAPIError(
                f"Unexpected {self.agent_type} response format: {data}",
                agent_name=self.name,
            )

    def _get_endpoint_url(self) -> str:
        """Get the chat completions endpoint URL."""
        return f"{self.base_url}/chat/completions"

    def _get_error_prefix(self) -> str:
        """Get error message prefix for this agent type."""
        return self.agent_type.title() if hasattr(self, "agent_type") else "API"

    @handle_agent_errors(
        max_retries=3,
        retry_delay=1.0,
        retry_backoff=2.0,
        retryable_exceptions=(AgentRateLimitError, AgentConnectionError, AgentTimeoutError),
    )
    async def generate(self, prompt: str, context: list[Message] | None = None) -> str:
        """Generate a response using the OpenAI-compatible API.

        Includes circuit breaker protection to prevent cascading failures.
        """
        # Check circuit breaker before attempting API call
        cb = getattr(self, "_circuit_breaker", None)
        if cb is not None and not cb.can_proceed():
            raise AgentCircuitOpenError(
                f"Circuit breaker open for {self.name} - too many recent failures",
                agent_name=self.name,
            )

        full_prompt = prompt
        if context:
            full_prompt = self._build_context_prompt(context) + prompt
        url = self._get_endpoint_url()
        headers = self._build_headers()
        messages = self._build_messages(full_prompt)
        payload = self._build_payload(messages, stream=False)

        try:
            # Use shared connection pool for better resource management
            async with create_client_session(timeout=self.timeout) as session:
                async with session.post(
                    url,
                    headers=headers,
                    json=payload,
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        sanitized = _sanitize_error_message(error_text)

                        # Record failure for circuit breaker (non-quota errors)
                        if cb is not None and not self.is_quota_error(response.status, error_text):
                            cb.record_failure()

                        # Check for quota/billing errors and fallback
                        if self.is_quota_error(response.status, error_text):
                            result = await self.fallback_generate(prompt, context, response.status)
                            if result is not None:
                                return result

                        raise AgentAPIError(
                            f"{self._get_error_prefix()} API error {response.status}: {sanitized}",
                            agent_name=self.name,
                            status_code=response.status,
                        )

                    data = await response.json()

                    # Record token usage for billing (OpenAI format)
                    usage = data.get("usage", {})
                    self._record_token_usage(
                        tokens_in=usage.get("prompt_tokens", 0),
                        tokens_out=usage.get("completion_tokens", 0),
                    )

                    content = self._parse_response(data)
                    if not content or not content.strip():
                        if cb is not None:
                            cb.record_failure()
                        raise AgentAPIError(
                            f"{self._get_error_prefix()} returned empty response",
                            agent_name=self.name,
                        )
                    # Record success for circuit breaker
                    if cb is not None:
                        cb.record_success()
                    return content
        except (AgentAPIError, AgentCircuitOpenError):
            raise  # Re-raise without double-recording
        except Exception:
            # Record failure for unexpected errors
            if cb is not None:
                cb.record_failure()
            raise

    async def generate_stream(
        self, prompt: str, context: list[Message] | None = None
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from the OpenAI-compatible API."""
        full_prompt = prompt
        if context:
            full_prompt = self._build_context_prompt(context) + prompt
        url = self._get_endpoint_url()
        headers = self._build_headers()
        messages = self._build_messages(full_prompt)
        payload = self._build_payload(messages, stream=True)

        # Use shared connection pool for better resource management
        async with create_client_session(timeout=self.timeout) as session:
            async with session.post(
                url,
                headers=headers,
                json=payload,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    sanitized = _sanitize_error_message(error_text)

                    # Check for quota errors and fallback
                    if self.is_quota_error(response.status, error_text):
                        async for chunk in self.fallback_generate_stream(
                            prompt, context, response.status
                        ):
                            yield chunk
                        return

                    raise AgentStreamError(
                        f"{self._get_error_prefix()} streaming API error {response.status}: {sanitized}",
                        agent_name=self.name,
                    )

                # Use SSE parser for consistent streaming
                try:
                    parser = create_openai_sse_parser()
                    async for content in parser.parse_stream(response.content, self.name):
                        yield content
                except RuntimeError as e:
                    raise AgentStreamError(str(e), agent_name=self.name)

    async def critique(
        self,
        proposal: str,
        task: str,
        context: list[Message] | None = None,
        target_agent: str | None = None,
    ) -> Critique:
        """Critique a proposal using the API."""
        target_desc = f" from {target_agent}" if target_agent else ""
        critique_prompt = f"""Critically analyze this proposal{target_desc}:

Task: {task}
Proposal: {proposal}

Format your response as:
ISSUES:
- issue 1
- issue 2

SUGGESTIONS:
- suggestion 1
- suggestion 2

SEVERITY: X.X
REASONING: explanation"""

        response = await self.generate(critique_prompt, context)
        return self._parse_critique(response, target_agent or "proposal", proposal)


__all__ = ["OpenAICompatibleMixin"]
